"""Fused residual add + RMSNorm.

The reference does ``x = residual + branch`` in bf16 and then normalises the
bf16 sum. This kernel writes the bf16 sum back into ``x`` (the residual stream
for the next block) and emits the normalised row, reproducing the two
roundings: the add rounds to bf16, the normalised value rounds to bf16 before
the weight multiply.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _add_rms_norm_kernel(x_ptr, y_ptr, w_ptr, xout_ptr, out_ptr, row_stride, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    offsets = row * row_stride + cols
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    s = (x + y).to(tl.bfloat16)
    tl.store(xout_ptr + offsets, s, mask=mask)
    sf = s.to(tl.float32)
    variance = tl.sum(sf * sf, axis=0) / n_cols
    normed = sf * tl.math.rsqrt(variance + eps)
    weight = tl.load(w_ptr + cols, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, normed.to(tl.bfloat16) * weight, mask=mask)


def add_rms_norm(x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor, eps: float,
                 xout: torch.Tensor | None = None) -> torch.Tensor:
    """``xout = x + y`` (bf16; in place into ``x`` when ``xout`` is None), returns
    ``rms_norm(xout, weight)``. All [M, H] contiguous."""
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    block = triton.next_power_of_2(n_cols)
    _add_rms_norm_kernel[(n_rows,)](
        x, y, weight, x if xout is None else xout, out, x.stride(0), n_cols, eps,
        BLOCK=block, num_warps=max(4, min(16, block // 256)),
    )
    return out
