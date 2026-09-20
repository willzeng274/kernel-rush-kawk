"""Split-K flash-decoding for grouped-query attention with a device-side length.

One query token per sequence, all ``G = HQ // HKV`` query heads of a KV head
share one program so K/V are read once per group. Keys are split into NSPLIT
ranges so short batches still fill the GPU; partial (m, l, acc) triples are
merged by ``_reduce_kernel``. The valid key count is ``pos[b] + 1``, read from
device memory so a CUDA graph can replay the kernel as the sequence grows.

Numerics follow FlashAttention-2, which is what SDPA runs for the reference:
scores and softmax statistics in fp32, probabilities rounded to bf16 before the
PV product, output accumulated in fp32 and normalised once at the end.
"""

import math

import torch
import triton
import triton.language as tl

import budget


@triton.jit
def _tree_tile(q, k_ptr, v_ptr, kv_base, n0, end, pos, L, tree, m, l, acc, scale,
               D: tl.constexpr, BLOCK_N: tl.constexpr, MASK_TREE: tl.constexpr):
    """One unchanged attention recurrence; the tree mask is compile-time optional."""
    d = tl.arange(0, D)
    n = n0 + tl.arange(0, BLOCK_N)
    kmask = n < end
    k = tl.load(k_ptr + (kv_base + n[:, None]) * D + d[None, :], mask=kmask[:, None], other=0.0)
    sc = tl.dot(q, tl.trans(k)) * scale
    if MASK_TREE:
        rel = n[None, :] - pos
        sh = tl.minimum(tl.maximum(rel, 0), 63).to(tl.int64)
        allowed = (rel < 0) | (((tree[:, None] >> sh) & 1) == 1)
        sc = tl.where(allowed & (n[None, :] < L), sc, float("-inf"))
    m_new = tl.maximum(m, tl.max(sc, axis=1))
    m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
    alpha = tl.exp(m - m_safe)
    p = tl.exp(sc - m_safe[:, None])
    l = l * alpha + tl.sum(p, axis=1)
    v = tl.load(v_ptr + (kv_base + n[:, None]) * D + d[None, :], mask=kmask[:, None], other=0.0)
    acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
    return m_new, l, acc


@triton.jit
def _split_kernel(
    q_ptr, k_ptr, v_ptr, pos_ptr, tree_ptr, o_part_ptr, m_part_ptr, l_part_ptr, o_ptr,
    CAP, scale, NSPLIT, SPLIT_LEN,
    HQ: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, GP: tl.constexpr, R: tl.constexpr,
    D: tl.constexpr, BLOCK_N: tl.constexpr,
    FINAL: tl.constexpr, TREE: tl.constexpr,
):
    """Query tile rows are (t, g): query row t of the sequence, head kh*G + g.
    Without TREE, row t may see keys 0 .. pos[b] + t (causal inside the block
    of R new tokens). With TREE, keys before pos are always visible and key
    pos + j is visible to row t iff bit j of tree[t] is set: an ancestor mask,
    so a whole draft tree is verified in one pass. The running max is guarded
    so rows with no valid key in a split stay at (m=-inf, l=0, acc=0)."""
    b = tl.program_id(0)
    kh = tl.program_id(1)
    s = tl.program_id(2) % NSPLIT
    rb = tl.program_id(2) // NSPLIT
    pos = tl.load(pos_ptr + b)
    L = pos + R
    start = s * SPLIT_LEN
    end = tl.minimum(start + SPLIT_LEN, L)

    rows = rb * GP + tl.arange(0, GP)
    t = rows // G
    g = rows % G
    head = kh * G + g
    d = tl.arange(0, D)
    row_mask = rows < G * R
    q = tl.load(
        q_ptr + (((b * HQ + head[:, None]) * R + t[:, None]) * D + d[None, :]),
        mask=row_mask[:, None], other=0.0,
    )
    kv_base = (b * HKV + kh) * CAP
    row_limit = pos + t
    if TREE:
        tree = tl.load(tree_ptr + t, mask=row_mask, other=0)

    m = tl.full([GP], float("-inf"), tl.float32)
    l = tl.zeros([GP], tl.float32)
    acc = tl.zeros([GP, D], tl.float32)
    if TREE:
        # Keep each loop straight-line so the compiler can pipeline its K/V
        # loads. Both bounds retain the original BLOCK_N-aligned tile order.
        prefix_end = tl.minimum(end, (pos // BLOCK_N) * BLOCK_N)
        for n0 in range(start, prefix_end, BLOCK_N):
            m, l, acc = _tree_tile(q, k_ptr, v_ptr, kv_base, n0, end, pos, L, tree, m, l, acc, scale,
                                    D, BLOCK_N, MASK_TREE=False)
        tail_start = tl.maximum(start, prefix_end)
        for n0 in range(tail_start, end, BLOCK_N):
            m, l, acc = _tree_tile(q, k_ptr, v_ptr, kv_base, n0, end, pos, L, tree, m, l, acc, scale,
                                    D, BLOCK_N, MASK_TREE=True)
    else:
        for n0 in range(start, end, BLOCK_N):
            n = n0 + tl.arange(0, BLOCK_N)
            kmask = n < end
            k = tl.load(k_ptr + (kv_base + n[:, None]) * D + d[None, :], mask=kmask[:, None], other=0.0)
            sc = tl.dot(q, tl.trans(k)) * scale
            if TREE:
                rel = n[None, :] - pos
                sh = tl.minimum(tl.maximum(rel, 0), 63).to(tl.int64)
                allowed = (rel < 0) | (((tree[:, None] >> sh) & 1) == 1)
                sc = tl.where(allowed & (n[None, :] < L), sc, float("-inf"))
            else:
                sc = tl.where(n[None, :] <= row_limit[:, None], sc, float("-inf"))
            m_new = tl.maximum(m, tl.max(sc, axis=1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.exp(m - m_safe)
            p = tl.exp(sc - m_safe[:, None])
            l = l * alpha + tl.sum(p, axis=1)
            v = tl.load(v_ptr + (kv_base + n[:, None]) * D + d[None, :], mask=kmask[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
            m = m_new

    if FINAL:
        out = acc / l[:, None]
        tl.store(o_ptr + ((b * R + t[:, None]) * HQ + head[:, None]) * D + d[None, :], out.to(tl.bfloat16), mask=row_mask[:, None])
    else:
        part = ((b * HQ + head) * R + t) * NSPLIT + s
        tl.store(o_part_ptr + part[:, None] * D + d[None, :], acc, mask=row_mask[:, None])
        tl.store(m_part_ptr + part, m, mask=row_mask)
        tl.store(l_part_ptr + part, l, mask=row_mask)


@triton.jit
def _reduce_kernel(
    o_part_ptr, m_part_ptr, l_part_ptr, o_ptr, NSPLIT,
    HQ: tl.constexpr, R: tl.constexpr, D: tl.constexpr, NSP: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    t = tl.program_id(2)
    s = tl.arange(0, NSP)
    smask = s < NSPLIT
    base = ((b * HQ + h) * R + t) * NSPLIT
    m = tl.load(m_part_ptr + base + s, mask=smask, other=float("-inf"))
    l = tl.load(l_part_ptr + base + s, mask=smask, other=0.0)
    M = tl.max(m, axis=0)
    w = tl.exp(m - M)
    L = tl.sum(w * l, axis=0)
    d = tl.arange(0, D)
    o = tl.load(o_part_ptr + (base + s[:, None]) * D + d[None, :], mask=smask[:, None], other=0.0)
    out = tl.sum(o * w[:, None], axis=0) / L
    tl.store(o_ptr + ((b * R + t) * HQ + h) * D + d, out.to(tl.bfloat16))


class DecodeAttention:
    """Workspace-owning wrapper; one instance per (B, HQ, cap) plan."""

    def __init__(self, B: int, HQ: int, HKV: int, D: int, cap: int, device, nsplit: int | None = None,
                 block_n: int = 64, num_warps: int = 4, num_stages: int = 2, R: int = 1, tree: bool = False,
                 maxlen: int | None = None):
        self.B, self.HQ, self.HKV, self.D, self.cap, self.R = B, HQ, HKV, D, cap, R
        # Splits cover the longest sequence this plan will actually see, not the
        # padded capacity, so no program is launched for slots that stay empty.
        work = min(cap, maxlen) if maxlen else cap
        self.tree = tree
        self.G = HQ // HKV
        # Query rows per program: the whole group for small R, 32-row blocks for
        # trees (each block re-reads K/V; 64-row blocks crashed on the H100).
        self.GP = max(16, min(32, triton.next_power_of_2(self.G * R)))
        self.row_blocks = triton.cdiv(self.G * R, self.GP)
        self.BLOCK_N = block_n
        self.num_warps, self.num_stages = num_warps, num_stages
        if nsplit is None:
            nsplit = max(1, min(16, (256 + B * HKV - 1) // (B * HKV)))
        blocks = triton.cdiv(work, self.BLOCK_N)
        nsplit = max(1, min(nsplit, blocks))
        self.SPLIT_LEN = triton.cdiv(blocks, nsplit) * self.BLOCK_N
        self.NSPLIT = triton.cdiv(work, self.SPLIT_LEN)
        self.NSP = triton.next_power_of_2(self.NSPLIT)
        self.scale = 1.0 / math.sqrt(D)
        self.o_part = torch.empty((B, HQ, R, self.NSPLIT, D), dtype=torch.float32, device=device)
        self.m_part = torch.empty((B, HQ, R, self.NSPLIT), dtype=torch.float32, device=device)
        self.l_part = torch.empty((B, HQ, R, self.NSPLIT), dtype=torch.float32, device=device)

    def __call__(self, q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, pos: torch.Tensor, out: torch.Tensor,
                 tree: torch.Tensor | None = None) -> None:
        """q [B, HQ, R, D] bf16; k/v_cache [B, HKV, cap, D]; pos [B] int32 (position of
        query row 0); out [B, R, HQ, D] bf16, i.e. rows (b, t) of [B*R, HQ*D];
        tree [R] int64 ancestor bitmasks when this instance was built with tree=True."""
        _split_kernel[(self.B, self.HKV, self.NSPLIT * self.row_blocks)](
            q, k_cache, v_cache, pos, tree if tree is not None else pos, self.o_part, self.m_part, self.l_part, out,
            self.cap, self.scale, self.NSPLIT, self.SPLIT_LEN,
            HQ=self.HQ, HKV=self.HKV, G=self.G, GP=self.GP, R=self.R, D=self.D,
            BLOCK_N=self.BLOCK_N,
            FINAL=self.NSPLIT == 1, TREE=self.tree,
            num_warps=self.num_warps, num_stages=self.num_stages,
        )
        if self.NSPLIT == 1:
            return
        _reduce_kernel[(self.B, self.HQ, self.R)](
            self.o_part, self.m_part, self.l_part, out, self.NSPLIT,
            HQ=self.HQ, R=self.R, D=self.D, NSP=self.NSP, num_warps=1,
        )


ATTN_CONFIGS = [
    dict(block_n=64, num_warps=4, num_stages=2),
    dict(block_n=128, num_warps=4, num_stages=2),
    dict(block_n=128, num_warps=8, num_stages=3),
    dict(block_n=128, num_warps=4, num_stages=3),
    dict(block_n=64, num_warps=4, num_stages=3),
    dict(block_n=64, num_warps=8, num_stages=3),
    dict(block_n=64, num_warps=8, num_stages=2),
    dict(block_n=32, num_warps=4, num_stages=3),
]


def _time(fn, iters: int = 30, rotate: list | None = None) -> float:
    """Average milliseconds per call, measured as a CUDA graph replay.

    Host launch overhead exceeds the kernel time for many of these skinny
    shapes, so a host-driven loop would report launch cost and favour whatever
    launches fewest kernels. Capturing ``iters`` calls into one graph and
    replaying it measures device time, which is how the deployed step runs.
    With ``rotate``, call ``fn(w)`` over a cycle of distinct weight tensors so
    consecutive iterations cannot be served from L2.
    """
    ws = rotate or [None]
    call = (lambda i: fn(ws[i % len(ws)])) if rotate else (lambda i: fn())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for i in range(3):
            call(i)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(iters):
            call(i)
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(3):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / (3 * iters)


def reference_attention(q, k, v, pos, R: int, scale: float, tree: torch.Tensor | None = None) -> torch.Tensor:
    """SDPA over the valid keys with the block-causal (or ancestor-tree) mask, as [B, R, HQ, D] float."""
    import torch.nn.functional as F

    B, HQ, _, D = q.shape
    HKV = k.shape[1]
    outs = []
    for b in range(B):
        p = int(pos[b])
        L = p + R
        kb = k[b, :, :L].repeat_interleave(HQ // HKV, dim=0)
        vb = v[b, :, :L].repeat_interleave(HQ // HKV, dim=0)
        key_idx = torch.arange(L, device=q.device)
        if tree is None:
            row_lim = p + torch.arange(R, device=q.device)
            mask = key_idx[None, :] <= row_lim[:, None]
        else:
            bits = torch.stack([(tree >> j) & 1 for j in range(R)], dim=1).bool()  # [R rows, R draft keys]
            mask = torch.cat([torch.ones(R, p, dtype=torch.bool, device=q.device), bits], dim=1)
        o = F.scaled_dot_product_attention(q[b], kb, vb, attn_mask=mask[None], scale=scale)
        outs.append(o.transpose(0, 1))
    return torch.stack(outs).float()


def ancestor_masks(parent: list[int]) -> list[int]:
    """Per-node int64 bitmask of the node itself and all its ancestors (row 63 wraps to -1)."""
    masks = []
    for i in range(len(parent)):
        m, n = 0, i
        while n >= 0:
            m |= 1 << n
            n = parent[n]
        masks.append(m - (1 << 64) if m >= (1 << 63) else m)
    return masks


def chain_tree(R: int, device) -> torch.Tensor:
    """Ancestor masks for a plain chain: row t sees draft rows 0..t."""
    return torch.tensor(ancestor_masks([t - 1 for t in range(R)]), dtype=torch.int64, device=device)


def pick_attention(B: int, HQ: int, HKV: int, D: int, cap: int, typical_len: int, device, log=None, R: int = 1,
                   tree: bool = False, maxlen: int | None = None) -> DecodeAttention:
    """Time every (config, split) pair on synthetic data at a typical sequence
    length and keep the fastest whose output matches SDPA; the split count
    trades parallelism against the extra reduce launch, so it is measured too."""
    import torch.nn.functional as F

    q = torch.randn((B, HQ, R, D), dtype=torch.bfloat16, device=device)
    # Enough distinct cache copies that a timing loop cannot be served from L2
    # (the decode step reads every layer's cache once per token).
    one = B * HKV * cap * D * 2 * 2
    copies = max(1, min(36, (192 << 20) // one + 1))
    ks = [torch.randn((B, HKV, cap, D), dtype=torch.bfloat16, device=device) for _ in range(copies)]
    vs = [torch.randn((B, HKV, cap, D), dtype=torch.bfloat16, device=device) for _ in range(copies)]
    k, v = ks[0], vs[0]
    typical_len = max(typical_len, R + 1)
    pos = torch.full((B,), typical_len - R, dtype=torch.int32, device=device)
    out = torch.empty((B, R, HQ, D), dtype=torch.bfloat16, device=device)
    L = typical_len
    tree_mask = chain_tree(R, device) if tree else None
    ref = reference_attention(q, k, v, pos, R, D ** -0.5, tree_mask)
    programs_wanted = 256
    base = max(1, (programs_wanted + B * HKV - 1) // (B * HKV))
    splits = sorted({1, 2, 3, max(1, base // 2), base, min(32, base * 2)})
    best, best_ms, best_name = None, float("inf"), ""
    for cfg in ATTN_CONFIGS:
        for nsplit in splits:
            if budget.expired() and best is not None:
                break
            try:
                attn = DecodeAttention(B, HQ, HKV, D, cap, device, nsplit=nsplit, R=R, tree=tree, maxlen=maxlen, **cfg)
                attn(q, k, v, pos, out, tree_mask)
                err = (out.float() - ref).abs().max().item()
                if err > 0.02 * ref.abs().max().item() + 1e-3:
                    if log:
                        log(f"attention {cfg} nsplit={nsplit} rejected: err {err:.4g}")
                    continue
                ms = _time(lambda i: attn(q, ks[i], vs[i], pos, out, tree_mask), rotate=list(range(copies)))
            except Exception as exc:
                if log:
                    log(f"attention {cfg} nsplit={nsplit} failed: {exc}")
                continue
            if ms < best_ms:
                best, best_ms, best_name = attn, ms, f"{cfg} nsplit={attn.NSPLIT}"
    if best is None:
        if log:
            log("no Triton decode attention configuration works here; using the torch fallback")
        return TorchDecodeAttention(B, HQ, HKV, D, cap, R, device, tree)
    del ks, vs
    if log:
        kv_bytes = 2 * B * HKV * L * D * 2
        log(f"attention B={B} R={R} len={L}: {best_name} {best_ms * 1000:.1f}us ({kv_bytes / best_ms / 1e6:.0f} GB/s)")
    return best


class TorchDecodeAttention:
    """Graph-capturable SDPA fallback with an explicit block-causal mask over the
    full cache capacity. Slower than the Triton kernel; used only if no Triton
    configuration compiles on the run's hardware."""

    def __init__(self, B: int, HQ: int, HKV: int, D: int, cap: int, R: int, device, tree: bool = False):
        self.B, self.HQ, self.HKV, self.D, self.cap, self.R, self.tree = B, HQ, HKV, D, cap, R, tree
        self.NSPLIT = 0
        self.scale = 1.0 / math.sqrt(D)
        self.keys = torch.arange(cap, device=device, dtype=torch.int32)
        self.rows = torch.arange(R, device=device, dtype=torch.int32)

    def __call__(self, q, k_cache, v_cache, pos, out, tree=None) -> None:
        import torch.nn.functional as F

        if self.tree:
            rel = self.keys[None, :] - pos[:, None]                               # [B, cap]
            sh = rel.clamp(0, 63).to(torch.int64)
            bits = ((tree[:, None, None] >> sh[None]) & 1) == 1                   # [R, B, cap]
            allowed = (rel[None] < 0) | (bits & (rel[None] < self.R))
            mask = allowed.permute(1, 0, 2)[:, None, :, :]                        # [B, 1, R, cap]
        else:
            mask = self.keys[None, None, None, :] <= (pos[:, None] + self.rows[None, :])[:, None, :, None]
        o = F.scaled_dot_product_attention(q, k_cache, v_cache, attn_mask=mask, scale=self.scale, enable_gqa=True)
        out.copy_(o.transpose(1, 2))
