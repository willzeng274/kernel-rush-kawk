from __future__ import annotations

import triton
import triton.language as tl

@triton.jit
def _qkv_n32_kernel(
    a_ptr, w_ptr, c_ptr,
    M, N, K, stride_am, stride_wn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    tile = tl.program_id(0)
    rm = tl.arange(0, BLOCK_M)
    rn = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    m_mask = rm < M
    n_mask = rn < N
    acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + rk
        k_mask = kk < K
        a = tl.load(a_ptr + rm[:, None] * stride_am + kk[None, :],
                    mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(w_ptr + rn[:, None] * stride_wn + kk[None, :],
                    mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        acc += tl.dot(a, tl.trans(w))
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16),
             mask=m_mask[:, None] & n_mask[None, :])

