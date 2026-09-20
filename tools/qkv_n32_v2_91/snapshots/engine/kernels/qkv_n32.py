"""One QKV output tile per CTA; no persistent outer loop or fused norm."""

import torch
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


class QKVN32Matmul:
    def __init__(self, device):
        self.grid = (192,)

    def __call__(self, a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        c = torch.empty((64, 6144), dtype=torch.bfloat16, device=a.device)
        _qkv_n32_kernel[self.grid](
            a, w, c, 64, 6144, 2560, a.stride(0), w.stride(0),
            BLOCK_M=64, BLOCK_N=32, BLOCK_K=64,
            num_warps=4, num_stages=3,
        )
        return c
