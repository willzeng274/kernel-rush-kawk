"""One M64 down geometry: four disjoint FP32 K partials, then BF16 merge.

Exactly X[64,9728], W[2560,9728], original contiguous BF16. Each of 160
CTAs owns one 64-channel tile and one 2432-element K interval. No persistent
outer loop, atomics, residual/norm fusion, or intermediate BF16 rounding.
"""

import torch
import triton
import triton.language as tl

from kernels.gemm import _sum_kernel


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


class DownS4Matmul:
    """Callable-owned scratch; fresh nonaliasing BF16 output per invocation.

Scratch may be shared only by invocations ordered on the same CUDA stream.
The owning verifier and captured graph must retain this callable throughout
all queued producer/merge work. No allocation occurs on graph replay.
"""

    def __init__(self, device):
        self.device = device
        self.part = torch.empty((4, 64, 2560), dtype=torch.float32, device=device)

    def __call__(self, a, w, *, allocation_owners=None):
        for tensor, shape in ((a, (64, 9728)), (w, (2560, 9728))):
            if (tuple(tensor.shape) != shape or tensor.dtype != torch.bfloat16
                    or not tensor.is_cuda or tensor.device != self.device
                    or not tensor.is_contiguous() or tensor.data_ptr() % 16):
                raise ValueError("down S4 requires aligned contiguous BF16 inputs of the exact shape")
        c = torch.empty((64, 2560), dtype=torch.bfloat16, device=a.device)
        if allocation_owners is not None:
            allocation_owners.append(c)
        _down_s4_kernel[(160,)](
            a, w, self.part, a.stride(0), w.stride(0),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, SPLIT_K=4, K_PER_SPLIT=2432,
            num_warps=4, num_stages=3, num_ctas=1, enable_fp_fusion=True,
        )
        _sum_kernel[(160,)](
            self.part, c, 163840, SPLIT_K=4, BLOCK=1024,
            num_warps=4, num_stages=3, num_ctas=1, enable_fp_fusion=True,
        )
        return c
