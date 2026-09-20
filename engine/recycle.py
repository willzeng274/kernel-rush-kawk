"""Token recycling: draft trees from the model's own top-k predictions.

Every forward pass already produces, for each verified row, the k most likely
next tokens; they are stored in an adjacency table ``M[token] -> k candidates``
(initialised from the prompt at prefill). The next draft is a fixed-shape tree
grown from the current token by following ``M``: node i holds
``M[token(parent(i))][rank(i)]``. Verification is exact, so the tree only
changes how many greedy tokens each pass yields, never which ones.

Reference: Luo et al., "Turning Trash into Treasure: Accelerating Inference of
Large Language Models with Token Recycling" (2024).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from kernels.accept import flat_children
from kernels.attention import ancestor_masks

#: Rough acceptance probability of a child by its rank among a node's k
#: candidates, used only to decide which tree nodes are worth a row.
RANK_PRIOR = [0.55, 0.16, 0.08, 0.05, 0.035, 0.025, 0.02, 0.015]


@dataclass
class TreeTemplate:
    parent: list[int]
    rank: list[int]
    children: list[list[int]]
    masks: list[int]

    @property
    def depth(self) -> list[int]:
        out = []
        for p in self.parent:
            out.append(0 if p < 0 else out[p] + 1)
        return out

    @property
    def spine(self) -> list[int]:
        """Nodes on the rank-0 path from the root (depth 1, 2, ...): the chain
        that an n-gram continuation overrides when it has a confident match."""
        nodes, cur = [], 0
        while True:
            nxt = [c for c in self.children[cur] if self.rank[c] == 0]
            if not nxt:
                return nodes
            cur = nxt[0]
            nodes.append(cur)

    @property
    def max_depth(self) -> int:
        """Longest root-to-leaf path, i.e. the most draft nodes one round can accept."""
        return max(self.depth)

    @property
    def size(self) -> int:
        return len(self.parent)

    @classmethod
    def build(cls, size: int, k: int) -> "TreeTemplate":
        """Greedy tree of ``size`` nodes: repeatedly add the frontier node with the
        highest product of rank priors along its path (nodes stay in BFS-compatible
        order: a child always has a larger index than its parent)."""
        parent, rank, value = [-1], [0], [1.0]
        frontier = [(RANK_PRIOR[r], 0, r) for r in range(min(k, len(RANK_PRIOR)))]
        while len(parent) < size and frontier:
            frontier.sort(reverse=True)
            v, p, r = frontier.pop(0)
            i = len(parent)
            parent.append(p)
            rank.append(r)
            value.append(v)
            frontier += [(v * RANK_PRIOR[rr], i, rr) for rr in range(min(k, len(RANK_PRIOR)))]
        children = [[] for _ in parent]
        for i, p in enumerate(parent):
            if p >= 0:
                children[p].append(i)
        return cls(parent, rank, children, ancestor_masks(parent))


@triton.jit
def _draft_kernel(root_ptr, table_ptr, parent_ptr, rank_ptr, spine_slot_ptr, spine_ptr,
                  nseen_ptr, anchor_ptr, blk_ptr,
                  K: tl.constexpr, R: tl.constexpr, S: tl.constexpr, SP: tl.constexpr):
    """Nodes are filled in index order (parents first). A node on the spine takes
    the host's n-gram token when one is present (>= 0); every other node takes
    table[token(parent)][rank], so branches always hang off the current token.

    The host writes the n-gram continuation a round late — it launches round
    t + 1 before it has read round t's tokens — so the pool it wrote is anchored
    ``off = nseen - anchor`` tokens behind this draft's root, and the spine reads
    ``pool[off + slot]``. That shift is only the right continuation if the model
    actually followed the pool over those tokens, which ``pool[off - 1] == root``
    checks for the last of them; a miss falls the whole spine back to the table.
    """
    b = tl.program_id(0)
    tok = tl.load(root_ptr + b)
    tl.store(blk_ptr + b * R, tok)
    off = tl.load(nseen_ptr + b) - tl.load(anchor_ptr + b)
    prev = tl.load(spine_ptr + b * SP + tl.minimum(tl.maximum(off - 1, 0), SP - 1))
    fresh = (off >= 0) & (off + S <= SP) & ((off == 0) | (prev == tok))
    for i in tl.static_range(1, R):
        p = tl.load(parent_ptr + i)
        r = tl.load(rank_ptr + i)
        ptok = tl.load(blk_ptr + b * R + p)
        cand = tl.maximum(tl.load(table_ptr + ptok * K + r), 0)
        slot = tl.load(spine_slot_ptr + i)
        sp = tl.load(spine_ptr + b * SP + tl.minimum(tl.maximum(off + slot, 0), SP - 1))
        use_spine = (slot >= 0) & (sp >= 0) & fresh
        tl.store(blk_ptr + b * R + i, tl.where(use_spine, sp, cand))


class Recycler:
    """Adjacency table plus the buffers the draft/verify graphs touch."""

    def __init__(self, vocab: int, B: int, R: int, k: int, device):
        self.B, self.R, self.k = B, R, k
        self.template = TreeTemplate.build(R, k)
        self.maxa = max(1, self.template.max_depth)
        self.table = torch.full((vocab, k), -1, dtype=torch.int32, device=device)
        self.parent = torch.tensor(self.template.parent, dtype=torch.int32, device=device)
        self.rank = torch.tensor(self.template.rank, dtype=torch.int32, device=device)
        self.masks = torch.tensor(self.template.masks, dtype=torch.int64, device=device)
        self.depth = torch.tensor(self.template.depth, dtype=torch.int32, device=device)
        start, flat, par = flat_children(self.template.children)
        self.child_start = torch.tensor(start, dtype=torch.int32, device=device)
        self.child_list = torch.tensor(flat or [0], dtype=torch.int32, device=device)
        self.child_par = torch.tensor(par or [0], dtype=torch.int32, device=device)
        self.root = torch.zeros((B,), dtype=torch.int64, device=device)
        self.blk = torch.zeros((B, R), dtype=torch.int64, device=device)
        spine_nodes = self.template.spine
        self.S = max(1, len(spine_nodes))
        slot = [-1] * R
        for j, node in enumerate(spine_nodes):
            slot[node] = j
        self.spine_slot = torch.tensor(slot, dtype=torch.int32, device=device)
        # Room for the whole spine plus every token one round can accept past
        # the anchor the host wrote the pool from.
        self.SP = self.S + self.maxa + 1
        self.spine = torch.full((B, self.SP), -1, dtype=torch.int64, device=device)
        self.spine_anchor = torch.zeros((B,), dtype=torch.int32, device=device)

    def update(self, tokens: torch.Tensor, logits: torch.Tensor) -> None:
        """Record the top-k next tokens predicted after each of ``tokens`` ([N] int64, logits [N, V])."""
        top = torch.topk(logits, self.k, dim=-1).indices.to(torch.int32)
        self.table.index_copy_(0, tokens, top)

    def draft(self, nseen: torch.Tensor) -> None:
        """Fill ``blk`` from ``root`` by walking the template through the table.
        ``nseen`` [B] int32 is the device's token counter; the spine pool is read
        at ``nseen - spine_anchor``."""
        _draft_kernel[(self.B,)](self.root, self.table, self.parent, self.rank, self.spine_slot, self.spine,
                                 nseen, self.spine_anchor, self.blk,
                                 K=self.k, R=self.R, S=self.S, SP=self.SP)

    def accept(self, blk_row: list[int], cand_row: list[int]) -> tuple[list[int], list[int]]:
        """Longest verified path: returns (accepted tokens, block indices of the accepted nodes).

        The decode loop runs this on device instead (``kernels/accept.py``), so
        this stays as the readable definition of the rule and the reference the
        kernel is tested against.
        """
        tokens, path, node = [cand_row[0]], [], 0
        while True:
            nxt = None
            for c in self.template.children[node]:
                if blk_row[c] == tokens[-1]:
                    nxt = c
                    break
            if nxt is None:
                return tokens, path
            path.append(nxt)
            tokens.append(cand_row[nxt])
            node = nxt
