"""Numerically conservative Qwen3 pointwise kernels, compatible with Triton 3.1."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rms(X, W, Y, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + col, col < N, 0).to(tl.float32)
    scale = tl.rsqrt(tl.sum(x * x, 0) / N + EPS)
    # HF casts normalized values to BF16 before applying the learned gain.
    normed = (x * scale).to(Y.dtype.element_ty)
    weight = tl.load(W + col, col < N, 0)
    tl.store(Y + row * N + col, normed * weight, col < N)


def rms_norm(x, weight, eps):
    # Projection outputs and residual streams are contiguous in Qwen3. Retain
    # a correct fallback for alternate callers of this module.
    x = x.contiguous()
    width = x.shape[-1]
    y = torch.empty_like(x)
    _rms[(x.numel() // width,)](
        x, weight, y, width, eps, triton.next_power_of_2(width),
        num_warps=4 if width <= 1024 else 8,
        enable_fp_fusion=False,
    )
    return y


@triton.jit
def _swiglu(GATE, UP, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(GATE + offsets, offsets < N, 0).to(tl.float32)
    u = tl.load(UP + offsets, offsets < N, 0)
    # torch.nn.functional.silu writes a BF16 tensor, then PyTorch multiplies
    # that rounded tensor by up. Preserve both rounding boundaries.
    activated = (g / (1.0 + tl.exp(-g))).to(OUT.dtype.element_ty)
    tl.store(OUT + offsets, activated * u, offsets < N)


def swiglu(gate, up):
    out = torch.empty_like(gate)
    n = gate.numel()
    _swiglu[(triton.cdiv(n, 256),)](
        gate, up, out, n, 256, num_warps=4, enable_fp_fusion=False,
    )
    return out


class FusedRMSNorm(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        self.weight = original.weight
        self.variance_epsilon = original.variance_epsilon

    def forward(self, x):
        return rms_norm(x, self.weight, self.variance_epsilon)


class FusedMLP(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        self.gate_proj = original.gate_proj
        self.up_proj = original.up_proj
        self.down_proj = original.down_proj

    def forward(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))
