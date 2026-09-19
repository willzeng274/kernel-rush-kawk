"""Optional, allocation-free BF16 decode projections for Triton 3.1 / H100.

Inputs: contiguous BF16 CUDA X[B,K], W[N,K], OUT[B,N], 1 <= B <= 16.
All multiplication and reduction are dense. Products use BF16 inputs and FP32
accumulation, with one final BF16 cast. There are no intermediate BF16 partials,
atomics, quantization, or approximate matrix products. Reduction order can differ
from cuBLAS; this is not a bitwise-equivalence promise.

Construct LinearPlan outside CUDA graph capture, then call it with persistent
buffers. Construction allocates optional FP32 split-K storage. A plan is bound
to shape/device, not weight values, so one plan can serve all 36 model layers.
Do not use one plan concurrently on multiple CUDA streams: workspace is shared.

No automatic dispatch to unmeasured candidates is installed. Benchmark the
explicit configurations against torch.mm before choosing them for a workload.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit
def _gemv(X, W, OUT, PART, N: tl.constexpr, K: tl.constexpr,
          BN: tl.constexpr, BK: tl.constexpr, SPLITS: tl.constexpr,
          STEPS: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    split = tl.program_id(1)
    k = split * STEPS * BK + tl.arange(0, BK)
    # Each lane accumulates a strided subsequence in FP32, then the program
    # reduces across K. Never spill partial sums to BF16.
    acc = tl.full((BN, BK), 0, tl.float32)
    for _ in range(STEPS):
        x = tl.load(X + k, k < K, 0).to(tl.float32)
        w = tl.load(W + n[:, None] * K + k[None, :],
                    (n[:, None] < N) & (k[None, :] < K), 0).to(tl.float32)
        acc += w * x[None, :]
        k += BK
    total = tl.sum(acc, 1)
    if SPLITS == 1:
        tl.store(OUT + n, total, n < N)
    else:
        tl.store(PART + split * N + n, total, n < N)


@triton.jit
def _small_mma(X, W, OUT, PART, B: tl.constexpr, N: tl.constexpr,
               K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               SPLITS: tl.constexpr, STEPS: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    split = tl.program_id(1)
    m = tl.arange(0, 16)
    k = split * STEPS * BK + tl.arange(0, BK)
    acc = tl.full((16, BN), 0, tl.float32)
    for _ in range(STEPS):
        x = tl.load(X + m[:, None] * K + k[None, :],
                    (m[:, None] < B) & (k[None, :] < K), 0)
        w = tl.load(W + n[None, :] * K + k[:, None],
                    (n[None, :] < N) & (k[:, None] < K), 0)
        # BF16 operands select tensor-core BF16 arithmetic. FP32 input_precision
        # modes (TF32, IEEE, etc.) have no role here.
        acc = tl.dot(x, w, acc)
        k += BK
    mask = (m[:, None] < B) & (n[None, :] < N)
    if SPLITS == 1:
        tl.store(OUT + m[:, None] * N + n[None, :], acc, mask)
    else:
        tl.store(PART + (split * B + m[:, None]) * N + n[None, :],
                 acc, mask)


@triton.jit
def _merge(PART, OUT, TOTAL: tl.constexpr, SPLITS: tl.constexpr,
           BLOCK: tl.constexpr):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = tl.full((BLOCK,), 0, tl.float32)
    # Deliberately deterministic. Atomic split-K gives nondeterministic rounding
    # and would also need a separate zeroing kernel for every invocation.
    for split in tl.static_range(SPLITS):
        total += tl.load(PART + split * TOTAL + p, p < TOTAL, 0)
    tl.store(OUT + p, total, p < TOTAL)


@dataclass(frozen=True)
class Config:
    kind: str
    block_n: int
    block_k: int
    splits: int = 1
    warps: int = 4
    stages: int = 2

    @property
    def name(self):
        return (f"{self.kind}_n{self.block_n}_k{self.block_k}"
                f"_s{self.splits}_w{self.warps}_p{self.stages}")


def candidates(batch, n, k):
    """Small explicit search set, not a triton.autotune decorator.

    Benchmark outside model loading if possible. On-platform, prefer one
    previously measured config per shape to protect the 300-second budget.
    """
    result = []
    if batch == 1:
        result.extend((Config("gemv", 8, 256, stages=1),
                       Config("gemv", 16, 256, stages=1)))
    # N=2560 gives only 40/80 CTAs without split-K, insufficient to occupy
    # all 132 SMs. Splitting K improves parallelism at a second-launch cost.
    if n <= 2560:
        result.extend((Config("mma", 32, 64, 2),
                       Config("mma", 32, 128, 4),
                       Config("mma", 64, 128, 4)))
    elif n <= 6144:
        # QKV has 192 CTAs at BN=32 even without splitting. Include a direct
        # launch: avoiding the merge can beat the higher split-K occupancy.
        result.extend((Config("mma", 32, 128, 1),
                       Config("mma", 32, 128, 4),
                       Config("mma", 64, 128, 4)))
    else:
        result.extend((Config("mma", 32, 128, 1),
                       Config("mma", 64, 128, 1),
                       Config("mma", 64, 128, 2)))
    return result


class LinearPlan:
    """Validated, reusable projection launch with caller-owned output.

    ``plan = LinearPlan(x, weight, out, config)`` allocates storage once.
    ``plan(x, weight, out)`` launches without allocation or synchronization.
    The caller must preserve construction-time shapes, strides, and dtypes on
    subsequent calls. Launches only read X/W and overwrite OUT/plan.workspace.
    X, W, and OUT must not alias. Unknown batches/prefill should use torch.mm.
    """

    def __init__(self, x, weight, out, config):
        self.b, self.k = x.shape
        self.n, wk = weight.shape
        if not (1 <= self.b <= 16) or wk != self.k:
            raise ValueError("expected X[B,K], W[N,K], 1 <= B <= 16")
        if tuple(out.shape) != (self.b, self.n):
            raise ValueError("output shape must be [B,N]")
        for tensor in (x, weight, out):
            if (not tensor.is_cuda or tensor.dtype != torch.bfloat16
                    or not tensor.is_contiguous() or tensor.device != x.device):
                raise ValueError("expected contiguous BF16 tensors on one CUDA device")
        if len({t.data_ptr() for t in (x, weight, out)}) != 3:
            raise ValueError("X, W, and OUT cannot alias")
        if config.kind not in ("gemv", "mma"):
            raise ValueError("kind must be gemv or mma")
        if config.kind == "gemv" and self.b != 1:
            raise ValueError("SIMT GEMV is only for B=1")
        if config.block_n not in (8, 16, 32, 64, 128):
            raise ValueError("unsupported block_n")
        if config.kind == "mma" and config.block_n < 16:
            raise ValueError("tensor-core N tile must be at least 16")
        if config.block_k not in (32, 64, 128, 256):
            raise ValueError("unsupported block_k")
        if config.splits not in (1, 2, 4, 8):
            raise ValueError("unsupported split count")
        if config.warps not in (4, 8) or config.stages not in (1, 2, 3, 4):
            raise ValueError("unsupported launch configuration")
        self.config = config
        self.steps = triton.cdiv(self.k, config.splits * config.block_k)
        self.grid = (triton.cdiv(self.n, config.block_n), config.splits)
        self.workspace = (torch.empty((config.splits, self.b, self.n),
                                      device=x.device, dtype=torch.float32)
                          if config.splits > 1 else None)

    def __call__(self, x, weight, out):
        c = self.config
        # Direct kernels specialize away PART accesses; a valid OUT pointer
        # avoids passing None to Triton's pointer argument inference.
        part = self.workspace if self.workspace is not None else out
        kwargs = dict(BN=c.block_n, BK=c.block_k, SPLITS=c.splits,
                      STEPS=self.steps, num_warps=c.warps,
                      num_stages=c.stages)
        if c.kind == "gemv":
            _gemv[self.grid](x, weight, out, part, self.n, self.k, **kwargs)
        else:
            _small_mma[self.grid](x, weight, out, part,
                                  self.b, self.n, self.k, **kwargs)
        if c.splits > 1:
            _merge[(triton.cdiv(self.b * self.n, 1024),)](
                part, out, self.b * self.n, c.splits, 1024, num_warps=4)
        return out
