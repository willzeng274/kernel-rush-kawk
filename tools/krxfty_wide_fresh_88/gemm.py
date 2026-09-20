from __future__ import annotations

import triton
import triton.language as tl

@triton.jit
def _norm_stats(x_ptr, y_ptr, xout_ptr, M, K, stride_am, eps, write_out,
                BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    """Residual add over the whole row, its bf16 sum written once, and the
    per-row rsqrt(mean(sum^2) + eps) every program needs for its A tile."""
    rm = tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_K)
    m_mask = rm < M
    sumsq = tl.zeros([BLOCK_M], tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + rk
        mask = m_mask[:, None] & (kk[None, :] < K)
        x = tl.load(x_ptr + rm[:, None] * stride_am + kk[None, :], mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + rm[:, None] * stride_am + kk[None, :], mask=mask, other=0.0).to(tl.float32)
        sb = (x + y).to(tl.bfloat16)
        if write_out:
            tl.store(xout_ptr + rm[:, None] * stride_am + kk[None, :], sb, mask=mask)
        sf = sb.to(tl.float32)
        sumsq += tl.sum(sf * sf, axis=1)
    return tl.math.rsqrt(sumsq / K + eps)


@triton.jit
def _normed_tile(x_ptr, y_ptr, wn_ptr, rstd, rm, kk, stride_am, mask):
    """bf16(w * bf16(bf16(x + y) * rstd)): the RMSNorm output tile, recomputed on the fly."""
    x = tl.load(x_ptr + rm[:, None] * stride_am + kk[None, :], mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + rm[:, None] * stride_am + kk[None, :], mask=mask, other=0.0).to(tl.float32)
    s = (x + y).to(tl.bfloat16).to(tl.float32)
    n = (s * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
    w = tl.load(wn_ptr + kk, mask=kk < 1_000_000_000, other=0.0).to(tl.float32)
    return (n * w[None, :]).to(tl.bfloat16)


@triton.jit
def _skinny_kernel(
    a_ptr, w_ptr, c_ptr, part_ptr,
    M, N, K, stride_am, stride_wn, k_per_split, num_tiles,
    y_ptr, wn_ptr, xout_ptr, eps,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr,
    NORM: tl.constexpr,
):
    """Tiles are (n-block, k-split) pairs walked by a persistent 1-D grid: with
    fewer programs than tiles each program loops, so the work always fills the
    SMs in whole waves instead of leaving a fractional last wave idle."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    rm = tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_K)
    m_mask = rm < M
    if NORM:
        rstd = _norm_stats(a_ptr, y_ptr, xout_ptr, M, K, stride_am, eps, pid == 0, BLOCK_M, BLOCK_K)
    for tile in range(pid, num_tiles, nprog):
        pid_n = tile // SPLIT_K
        pid_k = tile % SPLIT_K
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = rn < N
        k_start = pid_k * k_per_split
        k_end = tl.minimum(k_start + k_per_split, K)
        acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        for k0 in range(k_start, k_end, BLOCK_K):
            kk = k0 + rk
            k_mask = kk < k_end
            a_mask = m_mask[:, None] & k_mask[None, :]
            if NORM:
                a = _normed_tile(a_ptr, y_ptr, wn_ptr, rstd, rm, kk, stride_am, a_mask)
            else:
                a = tl.load(a_ptr + rm[:, None] * stride_am + kk[None, :], mask=a_mask, other=0.0)
            w = tl.load(w_ptr + rn[:, None] * stride_wn + kk[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0.0)
            acc += tl.dot(a, tl.trans(w))
        if SPLIT_K == 1:
            tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])
        else:
            tl.store(part_ptr + (pid_k * M + rm[:, None]) * N + rn[None, :], acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _sum_kernel(part_ptr, c_ptr, MN, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < MN
    acc = tl.zeros([BLOCK], tl.float32)
    for s in range(SPLIT_K):
        acc += tl.load(part_ptr + s * MN + idx, mask=mask, other=0.0)
    tl.store(c_ptr + idx, acc.to(tl.bfloat16), mask=mask)
