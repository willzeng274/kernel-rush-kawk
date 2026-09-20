"""Isolated M64 gate/up + SwiGLU prototype; never imported by an engine.

Exact domain: BF16 X[64,2560], unchanged concatenated gate-then-up
W[19456,2560], caller-owned Y[64,9728]. All tensors are contiguous and
16-byte aligned; strides are elements, and Y must not alias either input. The kernel
does not mutate X/W, allocate, normalize, split K, or repack any weight.

One CTA owns every M row and 64 final output channels. Its single dot has
shape [64,128] with adjacent gate/up columns. Grid (152,), four warps;
the only proposed compile cases are BK128/stages2 and BK64/stages3.
These are source-level choices, not measured performance results.

Logical interleave/reshape/split follows the earlier isolated #76 prototype,
but the full 64-row tile newly meets the pinned compiler's MMA3 condition.
"""

import triton
import triton.language as tl


@triton.jit
def gateup64_kernel(
    X, W, Y, stride_xm, stride_wn, stride_ym,
    K: tl.constexpr, I: tl.constexpr, BLOCK_K: tl.constexpr,
):
    tl.static_assert(K == 2560)
    tl.static_assert(I == 9728)
    tl.static_assert((BLOCK_K == 64) | (BLOCK_K == 128))
    pid = tl.program_id(0)
    rows = tl.arange(0, 64)
    paired_cols = pid * 128 + tl.arange(0, 128)
    # Preserve all original BF16 weight bits; K remains contiguous.
    weight_rows = paired_cols // 2 + (paired_cols % 2) * I
    kk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((64, 128), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + kk
        x = tl.load(X + rows[:, None] * stride_xm + k[None, :])
        w = tl.load(
            W + weight_rows[:, None] * stride_wn + k[None, :],
            eviction_policy="evict_first",
        )
        acc = tl.dot(x, tl.trans(w), acc)

    # Materialize the same three BF16 boundaries as gemm.py/swiglu.py:
    # projection -> BF16, FP32 SiLU -> BF16, FP32 product -> BF16.
    paired_bf16 = tl.reshape(
        acc.to(tl.bfloat16), (64, 64, 2), can_reorder=False,
    )
    gate, up = tl.split(paired_bf16)
    gate_f = gate.to(tl.float32)
    up_f = up.to(tl.float32)
    silu_f = (gate_f / (1.0 + tl.exp(-gate_f))).to(tl.bfloat16).to(tl.float32)
    result = (silu_f * up_f).to(tl.bfloat16)
    out_cols = pid * 64 + tl.arange(0, 64)
    tl.store(Y + rows[:, None] * stride_ym + out_cols[None, :], result)


def launch_gateup64(x, wgu, out, *, block_k=128):
    """Allocation-free launcher sketch; no engine selector integration."""
    import torch

    if block_k not in (64, 128):
        raise ValueError("only the two bounded compile configurations are supported")
    for tensor, shape in ((x, (64, 2560)), (wgu, (19456, 2560)),
                          (out, (64, 9728))):
        if (tuple(tensor.shape) != shape or tensor.dtype != torch.bfloat16
                or not tensor.is_cuda or tensor.device != x.device
                or tensor.stride(1) != 1 or tensor.stride(0) != shape[1]
                or tensor.data_ptr() % 16):
            raise ValueError("expected aligned contiguous BF16 tensors of the exact shape")
    output_storage = out.untyped_storage().data_ptr()
    if output_storage in (x.untyped_storage().data_ptr(), wgu.untyped_storage().data_ptr()):
        raise ValueError("output may not share input storage")
    gateup64_kernel[(152,)](
        x, wgu, out, x.stride(0), wgu.stride(0), out.stride(0),
        K=2560, I=9728, BLOCK_K=block_k,
        num_warps=4, num_stages=2 if block_k == 128 else 3,
        enable_fp_fusion=True,
    )
    return out
