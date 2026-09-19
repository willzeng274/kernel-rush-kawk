"""B1 residual + RMSNorm + full-K QKV, selected against the accepted path.

X/BRANCH/XOUT are contiguous BF16 [1,2560], GAIN is BF16 [2560],
W is original row-major BF16 [6144,2560], OUT is BF16 [1,6144].
XOUT must be independent of all inputs: every CTA reads the old residual,
and CTA zero alone writes the new residual. No cross-CTA dependency exists.
The caller alternates two hidden buffers across the 35 inter-layer boundaries.
The embedding and final-layer residual/norm retain their original kernels.
"""

import math
import sys
import time

import torch
import triton
import triton.language as tl
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources

from custom_kernels import residual_norm_kernel
from wide_gemv import WideGemvLayout


@triton.jit
def _residual_norm_qkv(X, BRANCH, GAIN, W, XOUT, OUT,
                       H: tl.constexpr, N: tl.constexpr, EPS: tl.constexpr,
                       ROWS: tl.constexpr, WIDTH: tl.constexpr):
    pid = tl.program_id(0)
    k = tl.arange(0, WIDTH)
    x = tl.load(X + k, k < H, 0).to(tl.float32)
    branch = tl.load(BRANCH + k, k < H, 0).to(tl.float32)
    # Preserve eager residual addition, normalized value and gain-product
    # BF16 casts before the FP32 dense projection reduction.
    x = (x + branch).to(tl.bfloat16).to(tl.float32)
    if pid == 0:
        tl.store(XOUT + k, x, k < H)
    inv = tl.rsqrt(tl.sum(x * x, 0) / H + EPS)
    norm = (x * inv).to(tl.bfloat16).to(tl.float32)
    gain = tl.load(GAIN + k, k < H, 0).to(tl.float32)
    norm = (norm * gain).to(tl.bfloat16).to(tl.float32)
    n = pid * ROWS + tl.arange(0, ROWS)
    weight = tl.load(W + n[:, None] * H + k[None, :],
                     (n[:, None] < N) & (k[None, :] < H), 0).to(tl.float32)
    result = tl.sum(weight * norm[None, :], axis=1)
    tl.store(OUT + n, result, n < N)


class ResidualQKVPlan:
    def __init__(self, eps, warps):
        self.eps, self.warps = eps, warps
        self.name = f"residual_norm_qkv_r4_k4096_w{warps}"

    def __call__(self, branch, hidden, gain, weight, next_hidden, output):
        _residual_norm_qkv[(1536,)](
            hidden, branch, gain, weight, next_hidden, output,
            H=2560, N=6144, EPS=self.eps, ROWS=4, WIDTH=4096,
            num_warps=self.warps, num_stages=1, enable_fp_fusion=False,
        )


class ResidualQKV:
    """One fixed B1 segment choice, made before real prefill and capture."""

    def __init__(self, engine, deadline):
        self.fallback = engine.native_layout
        self.device, self.eps = engine.hidden.device, engine.eps
        self.plan, self.hidden = None, None
        self.deadline = min(deadline, time.monotonic() + 25.0)
        if (engine.batch != 1 or engine.h != 2560
                or time.monotonic() >= self.deadline):
            return
        # Six distinct 30 MiB matrices exceed L2, including the layer-indexed
        # transposes owned by the actual accepted Native/Wide/Hopper fallback.
        weights = [pair[0] for pair in engine.packed[1:7]]
        gains = [layer.input_layernorm.weight for layer in engine.layers[1:7]]
        if (len(weights) != 6 or len(gains) != 6
                or any(tuple(w.shape) != (6144, 2560) for w in weights)):
            return
        if not self._room(engine.qkv):
            return
        try:
            self.hidden = torch.empty_like(engine.hidden)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return
        generator = torch.Generator(device=self.device)
        generator.manual_seed(39173)
        try:
            self.plan = self._select(engine, weights, gains, generator)
            self._log("selected " + (self.plan.name if self.plan else "accepted residual norm + QKV"))
        finally:
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        if self.plan is None:
            self.hidden = None

    @staticmethod
    def _log(message):
        print("[residual-qkv] B=1 " + message, file=sys.stderr, flush=True)

    def _room(self, output):
        free, total = torch.cuda.mem_get_info(self.device)
        # Reference QKV/residual and comparison temporaries, plus ping-pong.
        return free - (output.numel() * 32 + 3 * 2560 * 32 + 65536) >= total // 4

    def enabled(self, engine):
        return (self.plan is not None and self.hidden is not None
                and engine.batch == 1 and engine.h == 2560
                and tuple(engine.hidden.shape) == (1, 2560)
                and tuple(engine.qkv.shape) == (1, 6144)
                and engine.hidden.device == self.device)

    def _baseline(self, layer, branch, hidden, gain, weight, normalized, output):
        residual_norm_kernel[(hidden.shape[0],)](
            branch, hidden, gain, normalized, 2560, self.eps, 4096,
            num_warps=4, enable_fp_fusion=False,
        )
        self.fallback.run("qkv", layer, normalized, weight, output)

    def _check(self, plan, engine, weights, gains, reference, ref_hidden, generator):
        for index, scale in ((0, 1.0), (5, 0.1), (0, 10.0)):
            if time.monotonic() >= self.deadline or not self._room(engine.qkv):
                return False
            engine.hidden.normal_(generator=generator).mul_(scale)
            engine.branch.normal_(generator=generator).mul_(scale)
            ref_hidden.copy_(engine.hidden)
            self._baseline(index + 1, engine.branch, ref_hidden, gains[index],
                           weights[index], engine.normalized, reference)
            if time.monotonic() >= self.deadline:
                return False
            plan(engine.branch, engine.hidden, gains[index], weights[index],
                 self.hidden, engine.qkv)
            if time.monotonic() >= self.deadline or not self._room(engine.qkv):
                return False
            try:
                if (not torch.equal(self.hidden, ref_hidden)
                        or not torch.allclose(engine.qkv, reference, rtol=0.01, atol=0.01)):
                    return False
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                return False
        return True

    def _select(self, engine, weights, gains, generator):
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        if time.monotonic() >= self.deadline or not self._room(engine.qkv):
            return None
        try:
            reference = torch.empty_like(engine.qkv)
            ref_hidden = torch.empty_like(engine.hidden)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None
        engine.hidden.normal_(generator=generator)
        engine.branch.normal_(generator=generator)
        baseline_graph = self._graph(None, engine, weights, gains)
        if baseline_graph is None:
            return None
        winner, winner_ms = None, float("inf")
        for warps in (4, 8):
            if time.monotonic() >= self.deadline:
                break
            plan = ResidualQKVPlan(self.eps, warps)
            try:
                if not self._check(plan, engine, weights, gains, reference, ref_hidden, generator):
                    self._log(plan.name + ": numerical or resource check rejected")
                    continue
                candidate_graph = self._graph(plan, engine, weights, gains)
                if candidate_graph is None:
                    break
            except (CompilationError, OutOfResources, torch.cuda.OutOfMemoryError) as error:
                self._log(plan.name + ": " + type(error).__name__)
                if isinstance(error, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                continue
            baseline_times, candidate_times = [], []
            for graph, samples in ((baseline_graph, baseline_times),
                                   (candidate_graph, candidate_times),
                                   (candidate_graph, candidate_times),
                                   (baseline_graph, baseline_times)):
                if time.monotonic() >= self.deadline:
                    break
                samples.append(WideGemvLayout._time(graph, len(weights)))
            if len(baseline_times) != 2 or len(candidate_times) != 2:
                del candidate_graph
                break
            baseline_ms, candidate_ms = min(baseline_times), max(candidate_times)
            self._log(f"{plan.name}: accepted {baseline_ms * 1000:.2f} us, "
                      f"fused {candidate_ms * 1000:.2f} us")
            if (all(math.isfinite(t) and t > 0 for t in baseline_times + candidate_times)
                    and candidate_ms < baseline_ms * 0.95 and candidate_ms < winner_ms):
                winner, winner_ms = plan, candidate_ms
            del candidate_graph
        return winner if time.monotonic() < self.deadline else None

    def _graph(self, plan, engine, weights, gains):
        if time.monotonic() >= self.deadline:
            return None
        current = torch.cuda.current_stream(self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(current)

        def launch():
            hidden, other = engine.hidden, self.hidden
            for index, (weight, gain) in enumerate(zip(weights, gains)):
                if plan is None:
                    self._baseline(index + 1, engine.branch, hidden, gain,
                                   weight, engine.normalized, engine.qkv)
                else:
                    plan(engine.branch, hidden, gain, weight, other, engine.qkv)
                    hidden, other = other, hidden
            # An even-sized pool returns the candidate's final hidden to the
            # original address, just as the baseline's in-place residual does.

        try:
            with torch.cuda.stream(stream):
                launch()
        finally:
            current.wait_stream(stream)
        torch.cuda.synchronize(self.device)
        if time.monotonic() >= self.deadline:
            return None
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=stream):
                launch()
        finally:
            current.wait_stream(stream)
        if time.monotonic() >= self.deadline:
            return None
        graph.replay()
        torch.cuda.synchronize(self.device)
        return graph

    def run(self, layer, branch, hidden, gain, weight, next_hidden, normalized, output):
        if (self.plan is not None and tuple(hidden.shape) == (1, 2560)
                and tuple(branch.shape) == (1, 2560)
                and tuple(next_hidden.shape) == (1, 2560)
                and tuple(weight.shape) == (6144, 2560)
                and tuple(output.shape) == (1, 6144)):
            self.plan(branch, hidden, gain, weight, next_hidden, output)
        else:
            # Defensive fallback preserves the out-of-place wrapper contract.
            # Normal disabled dispatch keeps the accepted in-place path and
            # never pays this copy; shape changes re-run constructor selection.
            next_hidden.copy_(hidden)
            self._baseline(layer, branch, next_hidden, gain, weight, normalized, output)
