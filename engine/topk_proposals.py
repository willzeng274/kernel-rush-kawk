"""Standalone proposal-only kernels. No acceptance or model arithmetic changes.

Call *after* the existing torch.argmax(logits, dim=-1, out=greedy).
All tensors contiguous CUDA: logits BF16 [M,151936], greedy int64 [M],
partial_values FP32 [M,75,K-1], partial_ids int32 [M,75,K-1],
out int64 [M,K], M in 1..64, K in {2,4}. Caller owns all allocations;
inputs and outputs must not alias. Capture both launches in the verifier graph.
out[:,0] copies the authoritative original argmax, never recomputes it.
Remaining columns are distinct highest non-NaN logits excluding that argmax,
ties ordered by smaller token ID; missing alternatives are -1.
This defines proposal behavior even for infinities/NaNs. Model validation
continues to require finite full logits; this does not relax that condition.
"""
import triton
import triton.language as tl


@triton.jit
def proposal_partials(Logits, Greedy, PartialValues, PartialIds,
                      V: tl.constexpr, PARTS: tl.constexpr,
                      ALT: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.program_id(1)
    token = part * BLOCK + tl.arange(0, BLOCK)
    val = tl.load(Logits + row * V + token, token < V,
                  other=-float("inf")).to(tl.float32)
    greedy = tl.load(Greedy + row)
    live = (token < V) & (token != greedy) & (val == val)
    for rank in tl.static_range(ALT):
        best = tl.max(tl.where(live, val, -float("inf")), axis=0)
        idx = tl.min(tl.where(live & (val == best), token, 2147483647), axis=0)
        off = (row * PARTS + part) * ALT + rank
        tl.store(PartialValues + off, best)
        tl.store(PartialIds + off, tl.where(idx == 2147483647, -1, idx))
        live = live & (token != idx)


@triton.jit
def proposal_merge(Greedy, PartialValues, PartialIds, Out,
                   PARTS: tl.constexpr, ALT: tl.constexpr,
                   MERGE_BLOCK: tl.constexpr):
    row = tl.program_id(0)
    slot = tl.arange(0, MERGE_BLOCK)
    off = row * PARTS * ALT + slot
    val = tl.load(PartialValues + off, slot < PARTS * ALT,
                  other=-float("inf"))
    token = tl.load(PartialIds + off, slot < PARTS * ALT, other=-1)
    live = (slot < PARTS * ALT) & (token >= 0) & (val == val)
    greedy = tl.load(Greedy + row)
    tl.store(Out + row * (ALT + 1), greedy)
    for rank in tl.static_range(ALT):
        best = tl.max(tl.where(live, val, -float("inf")), axis=0)
        idx = tl.min(tl.where(live & (val == best), token, 2147483647), axis=0)
        tl.store(Out + row * (ALT + 1) + rank + 1,
                 tl.where(idx == 2147483647, -1, idx))
        live = live & (token != idx)


def proposals_out(logits, greedy, partial_values, partial_ids, out, k):
    """Allocation-free fixed-shape helper; validate buffer contracts at setup."""
    rows, vocab = logits.shape
    assert 1 <= rows <= 64 and vocab == 151936 and k in (2, 4)
    parts = triton.cdiv(vocab, 2048)
    alt = k - 1
    assert tuple(partial_values.shape) == (rows, parts, alt)
    assert tuple(partial_ids.shape) == (rows, parts, alt)
    assert tuple(greedy.shape) == (rows,) and tuple(out.shape) == (rows, k)
    proposal_partials[(rows, parts)](
        logits, greedy, partial_values, partial_ids, vocab, parts, alt, 2048,
        num_warps=4, enable_fp_fusion=False)
    proposal_merge[(rows,)](
        greedy, partial_values, partial_ids, out, parts, alt,
        triton.next_power_of_2(parts * alt), num_warps=4, enable_fp_fusion=False)
