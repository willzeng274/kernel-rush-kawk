"""SwiGLU activation matching ``Qwen3MLP``: ``down(silu(gate(x)) * up(x))``.

The reference rounds twice: ``silu`` on a bf16 tensor computes in fp32 and
stores bf16, then the bf16 * bf16 product rounds again. Both roundings are
reproduced here. ``gu`` is the fused gate/up projection, ``[M, 2*I]`` with the
gate half first.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_kernel(gu_ptr, out_ptr, I, ROW, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    blk = tl.program_id(1)
    cols = blk * BLOCK + tl.arange(0, BLOCK)
    mask = cols < I
    base = gu_ptr + row * ROW
    g = tl.load(base + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(base + I + cols, mask=mask, other=0.0).to(tl.float32)
    s = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + row * I + cols, (s * u).to(tl.bfloat16), mask=mask)


def swiglu(gu: torch.Tensor) -> torch.Tensor:
    M, two_i = gu.shape
    I = two_i // 2
    out = torch.empty((M, I), dtype=gu.dtype, device=gu.device)
    BLOCK = 1024
    _swiglu_kernel[(M, triton.cdiv(I, BLOCK))](gu, out, I, gu.stride(0), BLOCK=BLOCK, num_warps=4)
    return out
