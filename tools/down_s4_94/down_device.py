from __future__ import annotations

import triton
import triton.language as tl

@triton.jit
def _down_s4_kernel(
    a_ptr, w_ptr, part_ptr, stride_am, stride_wn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr, K_PER_SPLIT: tl.constexpr,
):
    tl.static_assert(BLOCK_M == 64 and BLOCK_N == 64 and BLOCK_K == 64)
    tl.static_assert(SPLIT_K == 4 and K_PER_SPLIT == 2432)
    tile = tl.program_id(0)
    split = tile % SPLIT_K
    rm = tl.arange(0, BLOCK_M)
    rn = (tile // SPLIT_K) * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    for k0 in range(0, K_PER_SPLIT, BLOCK_K):
        kk = split * K_PER_SPLIT + k0 + rk
        a = tl.load(a_ptr + rm[:, None] * stride_am + kk[None, :])
        w = tl.load(w_ptr + rn[:, None] * stride_wn + kk[None, :])
        acc += tl.dot(a, tl.trans(w))
    tl.store(part_ptr + (split * 64 + rm[:, None]) * 2560 + rn[None, :], acc)


@triton.jit
def _sum_kernel(part_ptr, c_ptr, MN, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < MN
    acc = tl.zeros([BLOCK], tl.float32)
    for s in range(SPLIT_K):
        acc += tl.load(part_ptr + s * MN + idx, mask=mask, other=0.0)
    tl.store(c_ptr + idx, acc.to(tl.bfloat16), mask=mask)

