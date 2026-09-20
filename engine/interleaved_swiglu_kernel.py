"""Isolated decode prototype; no engine integration or runtime selection.

X is BF16 [M, K], W is the unchanged BF16 [2*I, K] gate-over-up
matrix, and Y is caller-owned BF16 [M, I]. All inner strides are one;
outer strides are in elements. Inputs must not alias Y. Target: 2 <= M <=
16, K=2560, I=9728, MP=16, BLOCK_N=64, BLOCK_K=128, four warps,
three stages. Launch grid: (triton.cdiv(I, BLOCK_N // 2),), i.e. 304 CTAs.

This adapts #15's logical gate/up interleave and BF16 reshape/split to
the retained #73 narrow GEMV geometry and load/reduction expression.
It has not been compiled or run on CUDA. No speed or bitwise GPU parity
claim is implied by the source-level indexing and rounding checks.
"""

import triton
import triton.language as tl


@triton.jit
def interleaved_gemv_swiglu(
    X, W, Y, M, K,
    stride_xm, stride_wn, stride_ym,
    I: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    MP: tl.constexpr,
):
    tl.static_assert(BLOCK_N == 64)
    tl.static_assert(BLOCK_K == 128)
    tl.static_assert(MP == 16)
    pid = tl.program_id(0)
    logical_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    channels = logical_n // 2
    weight_rows = channels + (logical_n % 2) * I
    nmask = channels < I
    offs_m = tl.arange(0, MP)
    mmask = offs_m < M
    acc = tl.zeros([MP, BLOCK_N], tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        kmask = offs_k < K
        x = tl.load(
            X + offs_m[:, None] * stride_xm + offs_k[None, :],
            mask=mmask[:, None] & kmask[None, :], other=0.0,
        )
        w = tl.load(
            W + weight_rows[:, None] * stride_wn + offs_k[None, :],
            mask=nmask[:, None] & kmask[None, :], other=0.0,
        )
        acc += tl.dot(x, tl.trans(w))

    # Native/retained projection materializes BF16 before activation.
    pairs = tl.reshape(
        acc.to(tl.bfloat16), (MP, BLOCK_N // 2, 2), can_reorder=False,
    )
    gate, up = tl.split(pairs)
    gate = gate.to(tl.float32)
    # Keep the separate BF16 SiLU boundary before the BF16 product.
    act = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16)
    result = (act * up).to(tl.bfloat16)
    out_channels = pid * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)
    tl.store(
        Y + offs_m[:, None] * stride_ym + out_channels[None, :], result,
        mask=mmask[:, None] & (out_channels[None, :] < I),
    )
