"""Unsplit, full-K FP32 accumulation for B9--32 dense BF16 projections.

One M tile shares each weight load across every active decode row. Narrow N16
creates 160 CTAs even for N2560. Two large K tiles reduce serial loop overhead
versus the earlier K64/128 experiment: K512 with two stages, or K256 with three.
No copies, split workspace, intermediate BF16 sums, or measured-time tuning.

These BM16/32 tiles use MMA v2 in Triton 3.1, not Hopper WGMMA. The larger tile
is an experiment: compiler resource use and actual speed must be measured.
"""

import sys
import time

import torch
import triton
import triton.language as tl
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources

from wide_gemv import WideGemvLayout


@triton.jit
def _large_k_dot(X, W, OUT, N, X_ROW, OUT_ROW,
                 B: tl.constexpr, K: tl.constexpr,
                 BM: tl.constexpr, BK: tl.constexpr):
    m = tl.arange(0, BM)
    n = tl.program_id(0) * 16 + tl.arange(0, 16)
    k = tl.arange(0, BK)
    acc = tl.full((BM, 16), 0, tl.float32)
    for block in range(tl.cdiv(K, BK)):
        kk = block * BK + k
        x = tl.load(X + m[:, None] * X_ROW + kk[None, :],
                    (m[:, None] < B) & (kk[None, :] < K), 0)
        w = tl.load(W + n[None, :] * K + kk[:, None],
                    (n[None, :] < N) & (kk[:, None] < K), 0)
        # Original BF16 operands, with one FP32 accumulator spanning all K.
        acc = tl.dot(x, w, acc)
    tl.store(OUT + m[:, None] * OUT_ROW + n[None, :], acc,
             (m[:, None] < B) & (n[None, :] < N))


class LargeKPlan:
    def __init__(self, x, weight, output, block_k):
        self.batch, self.k = x.shape
        self.n = weight.shape[0]
        if (not 9 <= self.batch <= 32 or self.k not in (2560, 4096, 9728)
                or tuple(weight.shape) != (self.n, self.k)
                or tuple(output.shape) != (self.batch, self.n)
                or block_k not in (256, 512)):
            raise ValueError("unsupported dense decode shape or tile")
        for tensor in (x, weight, output):
            if (not tensor.is_cuda or tensor.dtype != torch.bfloat16
                    or tensor.device != x.device or tensor.stride(1) != 1):
                raise ValueError("expected row-major BF16 tensors on one CUDA device")
        if not weight.is_contiguous():
            raise ValueError("weight must be original contiguous [N,K]")
        self.bm = 16 if self.batch <= 16 else 32
        self.block_k = block_k
        self.stages = 2 if block_k == 512 else 3
        self.name = f"dense_m{self.bm}_n16_k{block_k}_w4_p{self.stages}"

    def __call__(self, x, weight, output):
        _large_k_dot[(triton.cdiv(self.n, 16),)](
            x, weight, output, self.n, x.stride(0), output.stride(0),
            B=self.batch, K=self.k, BM=self.bm, BK=self.block_k,
            num_warps=4, num_stages=self.stages,
        )


class LargeKGemmLayout(WideGemvLayout):
    """Reuse the accepted numerical, graph and memory guards from #17."""

    def __init__(self, engine, native, deadline):
        self.native, self.batch = native, engine.batch
        self.plans = {}
        self.device = engine.normalized.device
        if not 9 <= self.batch <= 32:
            return
        # Three K values x two tiles: at most six specializations per workload.
        # Runtime N/strides let gate/up, QKV and head share compiled code.
        deadline = min(deadline, time.monotonic() + 50.0)
        groups = (
            ("gateup", engine.normalized, engine.gateup,
             [pair[1] for pair in engine.packed[:6]]),
            ("down", engine.intermediate, engine.branch,
             [layer.mlp.down_proj.weight for layer in engine.layers[:6]]),
            ("qkv", engine.normalized, engine.qkv,
             [pair[0] for pair in engine.packed[:6]]),
            ("output", engine.attention, engine.branch,
             [layer.self_attn.o_proj.weight for layer in engine.layers[:6]]),
            ("head", engine.normalized, engine.logits,
             [engine.model.lm_head.weight]),
        )
        generator = torch.Generator(device=self.device)
        generator.manual_seed(48591)
        for name, template, output, weights in groups:
            if time.monotonic() >= deadline:
                break
            winner = self._select(name, template, weights, output, generator, deadline)
            if winner is not None and time.monotonic() < deadline:
                self.plans[name] = winner
            chosen = self.plans.get(name)
            self._log(name, "selected " + (chosen.name if chosen else "existing layout"))
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()

    def _log(self, name, message):
        print(f"[large-k-gemm] B={self.batch} {name}: {message}",
              file=sys.stderr, flush=True)

    def _select(self, name, template, weights, output, generator, deadline):
        if time.monotonic() >= deadline:
            return None
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        if time.monotonic() >= deadline or not self._room(output):
            self._log(name, "existing: deadline or numerical memory guard")
            return None
        try:
            reference = torch.empty_like(output)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self._log(name, "existing: numerical reference allocation failed")
            return None
        # Decode buffers are scratch until real prefill overwrites them.
        x = template
        x.normal_(generator=generator)
        if time.monotonic() >= deadline:
            return None
        native_graph = self._graph(name, None, x, weights, output, deadline)
        if native_graph is None:
            return None
        winner, winner_ms = None, float("inf")
        for block_k in (512, 256):
            if time.monotonic() >= deadline:
                break
            plan = LargeKPlan(x, weights[0], output, block_k)
            try:
                if not self._check(name, plan, x, weights, output, reference,
                                   generator, deadline):
                    self._log(name, f"{plan.name}: numerical check rejected")
                    continue
                if time.monotonic() >= deadline:
                    break
                candidate_graph = self._graph(name, plan, x, weights, output, deadline)
                if candidate_graph is None:
                    break
            except (CompilationError, OutOfResources) as error:
                self._log(name, f"{plan.name}: {type(error).__name__}")
                continue
            native_times, candidate_times = [], []
            for graph, samples in ((native_graph, native_times),
                                   (candidate_graph, candidate_times),
                                   (candidate_graph, candidate_times),
                                   (native_graph, native_times)):
                if time.monotonic() >= deadline:
                    break
                samples.append(self._time(graph, len(weights)))
            if len(native_times) != 2 or len(candidate_times) != 2:
                del candidate_graph
                break
            native_ms, custom_ms = min(native_times), max(candidate_times)
            self._log(name, f"{plan.name}: existing {native_ms * 1000:.2f} us, "
                      f"custom {custom_ms * 1000:.2f} us")
            if custom_ms < native_ms * 0.95 and custom_ms < winner_ms:
                winner, winner_ms = plan, custom_ms
            del candidate_graph
        return winner if time.monotonic() < deadline else None

    def run(self, name, layer, x, weight, output):
        plan = self.plans.get(name) if x.shape[0] == self.batch else None
        if (plan is None or tuple(x.shape) != (plan.batch, plan.k)
                or tuple(weight.shape) != (plan.n, plan.k)
                or tuple(output.shape) != (plan.batch, plan.n)):
            self.native.run(name, layer, x, weight, output)
        else:
            plan(x, weight, output)
