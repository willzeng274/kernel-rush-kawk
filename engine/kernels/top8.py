"""Exact verifier-only BF16 top-eight, with one bounded warmup selection.

The partition/merge structure extends the earlier independent k2/k4 recycling
helper. BF16 key semantics follow PyTorch 2.5.1 SortingRadixSelect.cuh: NaNs
rank highest, +0 ranks above -0. Equal keys prefer the lowest vocabulary ID.
This produces proposals only; the caller's separate torch.argmax is untouched.
"""

from __future__ import annotations

import sys
import time

import torch
import triton
import triton.language as tl

import budget

# A parent may abandon a plan after capture fails. Keep failed-drain graph
# owners and external inputs alive even if that parent drops its references.
_FAILED_TRIALS = []


@triton.jit
def _bf16_key(value):
    bits = value.to(tl.int16, bitcast=True).to(tl.int32) & 65535
    mask = tl.where((bits & 32768) != 0, 65535, 32768)
    key = tl.where((bits & 32767) > 32640, 65535, bits ^ mask)
    # Valid keys are positive, including infinities. Zero is an invalid lane.
    return key + 1


@triton.jit
def _best_pair(key_a, id_a, key_b, id_b):
    take_a = (key_a > key_b) | ((key_a == key_b) & (id_a < id_b))
    return tl.where(take_a, key_a, key_b), tl.where(take_a, id_a, id_b)


@triton.jit
def top8_partials(Logits, PartialKeys, PartialIds,
                  V: tl.constexpr, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    token = part * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(Logits + row * V + token, token < V, other=0)
    key = tl.where(token < V, _bf16_key(value), 0)
    for rank in tl.static_range(8):
        best_key, best_id = tl.reduce((key, token), 0, _best_pair)
        off = (row * PARTS + part) * 8 + rank
        tl.store(PartialKeys + off, best_key)
        tl.store(PartialIds + off, best_id)
        key = tl.where(token == best_id, 0, key)


@triton.jit
def top8_merge(PartialKeys, PartialIds, Out,
               PARTS: tl.constexpr, MERGE_BLOCK: tl.constexpr):
    row = tl.program_id(0)
    slot = tl.arange(0, MERGE_BLOCK)
    offset = row * PARTS * 8 + slot
    key = tl.load(PartialKeys + offset, slot < PARTS * 8, other=0)
    token = tl.load(PartialIds + offset, slot < PARTS * 8, other=2147483647)
    for rank in tl.static_range(8):
        best_key, best_id = tl.reduce((key, token), 0, _best_pair)
        tl.store(Out + row * 8 + rank, best_id)
        key = tl.where(token == best_id, 0, key)


class _PickerDeadline(Exception):
    pass


class VerifierTop8:
    """Plan-owned scratch, outputs and temporary captured comparison graphs.

    Buffers are disjoint, contiguous and overwritten completely on every call.
    There are no counters or persistent proposal state to reset. The caller's
    plan already owns stream synchronization and lifetime during graph replay.
    """

    def __init__(self, vocab: int, rows: int, device):
        self.vocab, self.rows = vocab, rows
        self.keys = torch.empty((rows, 75, 8), dtype=torch.int32, device=device)
        self.ids = torch.empty_like(self.keys)
        self.out = torch.empty((rows, 8), dtype=torch.int32, device=device)
        self.chosen = False
        self.enabled = False
        self.failed = False
        self.trial_graphs = []
        self.trial_inputs = None

    def _top(self, logits):
        top8_partials[(self.rows, 75)](
            logits, self.keys, self.ids, 151936, 75, 2048,
            num_warps=4, num_stages=3, enable_fp_fusion=False)
        top8_merge[(self.rows,)](
            self.keys, self.ids, self.out, 75, 1024,
            num_warps=4, num_stages=3, enable_fp_fusion=False)
        return self.out

    @staticmethod
    def _original(table, tokens, logits, pair_top):
        top = torch.topk(logits, 8, dim=-1).indices.to(torch.int32)
        table.index_copy_(0, tokens, top)
        pair_top.copy_(top)

    def _candidate(self, table, tokens, logits, pair_top):
        top = self._top(logits)
        table.index_copy_(0, tokens, top)
        pair_top.copy_(top)

    def _check_deadline(self, deadline):
        if time.monotonic() >= deadline or budget.remaining() <= 30.0:
            raise _PickerDeadline()

    def _choose(self, table, tokens, logits, pair_top):
        self.chosen = True  # sticky choice, including rejection and low budget
        if budget.remaining() < 42.0 or torch.cuda.is_current_stream_capturing():
            return
        deadline = time.monotonic() + 12.0
        reason = "rejected"
        # Keep trial owners alive until synchronization succeeds, on every exit.
        self.trial_inputs = (table, tokens, logits, pair_top)
        try:
            self._check_deadline(deadline)
            snapshot = logits.clone()
            self.trial_inputs = (table, tokens, logits, pair_top, snapshot)
            reference = torch.topk(logits, 8, dim=-1).values
            top = self._top(logits)
            if not torch.equal(logits.view(torch.int16), snapshot.view(torch.int16)):
                raise RuntimeError("top8 changed authoritative greedy input")
            self._check_deadline(deadline)
            ordered_ids = top.sort(dim=-1).values
            valid = ((top >= 0) & (top < self.vocab)).all()
            unique = (ordered_ids[:, 1:] != ordered_ids[:, :-1]).all()
            # Validate addresses before gather, and reject unexpected nonfinite
            # model rows while preserving the original behavior on those rows.
            if not bool(valid.item()) or not bool(unique.item()):
                reason = "invalid IDs"
                return
            if not bool(torch.isfinite(logits).all().item()):
                reason = "nonfinite logits"
                return
            selected = logits.gather(1, top.long())
            if not torch.equal(selected, reference):
                reason = "rank mismatch"
                return
            self._check_deadline(deadline)
            functions = (self._original, self._candidate)
            for fn in functions:
                self._check_deadline(deadline)
                fn(table, tokens, logits, pair_top)
                torch.cuda.synchronize()
                self._check_deadline(deadline)
                graph = torch.cuda.CUDAGraph()
                self.trial_graphs.append(graph)
                with torch.cuda.graph(graph):
                    for _ in range(4):
                        fn(table, tokens, logits, pair_top)
                self._check_deadline(deadline)
                graph.replay()
                torch.cuda.synchronize()
                self._check_deadline(deadline)
            times = [[], []]
            # ABBA order, eight full updates per observation. Event timings
            # include top-k, index conversion, table scatter, and pair-ID copy.
            for which in (0, 1, 1, 0):
                self._check_deadline(deadline)
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(2):
                    self.trial_graphs[which].replay()
                end.record()
                end.synchronize()
                self._check_deadline(deadline)
                times[which].append(start.elapsed_time(end) / 8)
            baseline_ms, candidate_ms = max(times[0]), max(times[1])
            # Each candidate observation must beat each baseline observation.
            self.enabled = max(times[1]) < 0.95 * min(times[0])
            reason = (f"{'selected' if self.enabled else 'slower'} "
                      f"torch={baseline_ms * 1000:.1f}us top8={candidate_ms * 1000:.1f}us")
        except _PickerDeadline:
            reason = "warmup budget"
        except BaseException:
            self.failed = True
            self.enabled = False
            raise
        finally:
            # Device/capture/compile exceptions propagate after draining, rather
            # than treating an unknown CUDA failure as a safe kernel rejection.
            try:
                torch.cuda.synchronize()
            except BaseException:
                self.failed = True
                self.enabled = False
                _FAILED_TRIALS.append(self)
                raise
            self.trial_graphs.clear()
            self.trial_inputs = None
            print(f"[engine] verifier top8 N={self.rows}: {reason}", file=sys.stderr, flush=True)

    def update(self, table, tokens, logits, pair_top):
        if self.failed:
            raise RuntimeError("top8 picker cannot be reused after a failed trial")
        if not self.chosen:
            self._choose(table, tokens, logits, pair_top)
        if self.enabled:
            self._candidate(table, tokens, logits, pair_top)
        else:
            self._original(table, tokens, logits, pair_top)
