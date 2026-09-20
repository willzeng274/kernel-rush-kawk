"""Exact M64 gate/up geometry with 32 final channels per CTA.

X[64,2560], original gate-then-up W[19456,2560], Y[64,9728]:
contiguous aligned BF16 on one device; output storage cannot alias inputs.
The single 64x64 dot interleaves gate/up columns, then preserves all three
BF16 boundaries. There are 304 CTAs, no split-K, and no weight repacking.
Only BK128/stages2 and BK64/stages3 with four warps are eligible.
"""

import triton
import triton.language as tl


@triton.jit
def gateup_n32_kernel(
    X, W, Y, stride_xm, stride_wn, stride_ym,
    K: tl.constexpr, I: tl.constexpr, BLOCK_K: tl.constexpr,
):
    tl.static_assert(K == 2560)
    tl.static_assert(I == 9728)
    tl.static_assert((BLOCK_K == 64) | (BLOCK_K == 128))
    pid = tl.program_id(0)
    rows = tl.arange(0, 64)
    paired_cols = pid * 64 + tl.arange(0, 64)
    weight_rows = paired_cols // 2 + (paired_cols % 2) * I
    kk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((64, 64), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + kk
        x = tl.load(X + rows[:, None] * stride_xm + k[None, :])
        w = tl.load(
            W + weight_rows[:, None] * stride_wn + k[None, :],
            eviction_policy="evict_first",
        )
        acc = tl.dot(x, tl.trans(w), acc)

    paired_bf16 = tl.reshape(
        acc.to(tl.bfloat16), (64, 32, 2), can_reorder=False,
    )
    gate, up = tl.split(paired_bf16)
    gate_f = gate.to(tl.float32)
    up_f = up.to(tl.float32)
    silu_f = (gate_f / (1.0 + tl.exp(-gate_f))).to(tl.bfloat16).to(tl.float32)
    result = (silu_f * up_f).to(tl.bfloat16)
    out_cols = pid * 32 + tl.arange(0, 32)
    tl.store(Y + rows[:, None] * stride_ym + out_cols[None, :], result)


def launch_gateup_n32(x, wgu, out, *, block_k=128):
    """Allocation-free launch; returned output belongs to the caller."""
    import torch

    if block_k not in (64, 128):
        raise ValueError("only the two frozen N32 configurations are supported")
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
    gateup_n32_kernel[(304,)](
        x, wgu, out, x.stride(0), wgu.stride(0), out.stride(0),
        K=2560, I=9728, BLOCK_K=block_k,
        num_warps=4, num_stages=2 if block_k == 128 else 3,
        enable_fp_fusion=True,
    )
    return out
