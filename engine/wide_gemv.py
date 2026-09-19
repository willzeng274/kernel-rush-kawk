"""A bounded B=1 full-K reduction experiment with the native layout fallback.

Unlike the earlier 256-wide loop GEMV, each CTA sees an entire dot product.
There is no serial K loop, split-K workspace, extra weight layout, or second
kernel. BF16 operands are converted losslessly to FP32 before multiplication;
the FP32 reduction is cast only when storing the final BF16 projection.

The decoder calls only configurations accepted during its untimed warmup.
Prefill and batches above one retain the existing native implementation.
"""

import statistics
import sys
import time

import torch
import triton
import triton.language as tl
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources


@triton.jit
def _wide_dot(X, W, OUT, N, K: tl.constexpr,
              ROWS: tl.constexpr, WIDTH: tl.constexpr):
    n = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    k = tl.arange(0, WIDTH)
    x = tl.load(X + k, k < K, 0).to(tl.float32)
    w = tl.load(W + n[:, None] * K + k[None, :],
                (n[:, None] < N) & (k[None, :] < K), 0).to(tl.float32)
    # Both operands and the reduction are FP32. BF16 products fit in FP32
    # exactly for ordinary finite model values; no BF16 arithmetic is used.
    result = tl.sum(w * x[None, :], axis=1)
    tl.store(OUT + n, result, n < N)


class WidePlan:
    def __init__(self, k, warps):
        self.k = k
        self.width = triton.next_power_of_2(k)
        # Bound the full-K tile to 16384 elements, including padded zeros.
        # Two launch choices trade warp occupancy against per-thread registers.
        self.rows = max(1, 16384 // self.width)
        self.warps = warps
        self.name = f"fullK_r{self.rows}_k{self.width}_w{warps}"

    def __call__(self, x, weight, output):
        _wide_dot[(triton.cdiv(weight.shape[0], self.rows),)](
            x, weight, output, weight.shape[0], self.k,
            ROWS=self.rows, WIDTH=self.width,
            num_warps=self.warps, num_stages=1, enable_fp_fusion=False,
        )


class WideGemvLayout:
    """Same run interface as NativeLayout; no allocations during graph replay."""

    def __init__(self, engine, native, deadline):
        self.native = native
        self.batch = engine.batch
        self.plans = {}
        self.device = engine.normalized.device
        if self.batch != 1:
            return
        # Six specializations in total: three K widths and two warp counts.
        # N remains a runtime argument, allowing QKV/GU/head to share code.
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
            # The tied head alone is 742 MiB, far larger than the H100 L2.
            ("head", engine.normalized, engine.logits,
             [engine.model.lm_head.weight]),
        )
        generator = torch.Generator(device=engine.normalized.device)
        generator.manual_seed(38571)
        for name, template, output, weights in groups:
            if time.monotonic() >= deadline:
                break
            winner = self._select(name, template, weights, output, generator, deadline)
            if winner is not None and time.monotonic() < deadline:
                self.plans[name] = winner
            chosen = self.plans.get(name)
            self._log(name, "selected " + (chosen.name if chosen else "native layout"))
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()

    @property
    def weights(self):
        """Expose native transpose ownership for compatible prefill tuners."""
        return self.native.weights

    @property
    def extra_bytes(self):
        return self.native.extra_bytes

    @staticmethod
    def _log(name, message):
        print(f"[wide-gemv] B=1 {name}: {message}", file=sys.stderr, flush=True)

    def _room(self, output, reference_live=False):
        free, total = torch.cuda.mem_get_info(self.device)
        # Reserve one BF16 reference and conservative allclose temporaries.
        # Existing model/cache/layout/Lt workspace ownership is reflected in
        # physical free memory. No activation or weight copies are allocated.
        required = output.numel() * 32 + 65536
        if reference_live:
            required -= output.numel() * output.element_size()
        return free - required >= total // 4

    def _check(self, name, plan, x, weights, output, reference, generator, deadline):
        # Include both ends of the rotated pool, with independent values and
        # three scales. Reuse caller-owned scratch and the same reference.
        for index, scale in ((0, 1.0), (len(weights) - 1, 0.1), (0, 10.0)):
            if time.monotonic() >= deadline:
                return False
            x.normal_(generator=generator).mul_(scale)
            self.native.run(name, index, x, weights[index], reference)
            if time.monotonic() >= deadline:
                return False
            plan(x, weights[index], output)
            if time.monotonic() >= deadline:
                return False
            if not self._room(output, reference_live=True):
                self._log(name, "native: comparison memory guard")
                return False
            try:
                if not torch.allclose(output, reference, rtol=0.01, atol=0.01):
                    return False
            except torch.cuda.OutOfMemoryError:
                # A failed PyTorch temporary allocation is recoverable. Device
                # execution errors and unexpected RuntimeErrors still escape.
                torch.cuda.empty_cache()
                self._log(name, "native: comparison allocation failed")
                return False
        return True

    def _select(self, name, template, weights, output, generator, deadline):
        if time.monotonic() >= deadline:
            return None
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        if time.monotonic() >= deadline or not self._room(output):
            self._log(name, "native: deadline or numerical memory guard")
            return None
        try:
            reference = torch.empty_like(output)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self._log(name, "native: numerical reference allocation failed")
            return None
        # These persistent decode activations are scratch until real prefill.
        # Real prefill and decode overwrite them before consuming their values.
        x = template
        x.normal_(generator=generator)
        if time.monotonic() >= deadline:
            return None
        native_graph = self._graph(name, None, x, weights, output, deadline)
        if native_graph is None:
            return None
        winner, winner_ms = None, float("inf")
        for warps in (4, 8):
            if time.monotonic() >= deadline:
                break
            plan = WidePlan(x.shape[1], warps)
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
            # Rotate six actual layer weights beyond L2. Require the slower
            # custom median to beat the faster native median by at least 5%.
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
            self._log(name, f"{plan.name}: native {native_ms * 1000:.2f} us, "
                      f"custom {custom_ms * 1000:.2f} us")
            if custom_ms < native_ms * 0.95 and custom_ms < winner_ms:
                winner, winner_ms = plan, custom_ms
            del candidate_graph
        return winner if time.monotonic() < deadline else None

    def _graph(self, name, plan, x, weights, output, deadline):
        if time.monotonic() >= deadline:
            return None
        current = torch.cuda.current_stream(x.device)
        stream = torch.cuda.Stream(device=x.device)
        stream.wait_stream(current)

        def launch():
            for index, weight in enumerate(weights):
                if plan is None:
                    self.native.run(name, index, x, weight, output)
                else:
                    plan(x, weight, output)

        try:
            with torch.cuda.stream(stream):
                launch()
        finally:
            # A later launch may fail after earlier work reached this stream.
            # Order that work before the caller releases candidate buffers.
            current.wait_stream(stream)
        torch.cuda.synchronize(x.device)
        if time.monotonic() >= deadline:
            return None
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=stream):
                launch()
        finally:
            current.wait_stream(stream)
        if time.monotonic() >= deadline:
            return None
        graph.replay()
        torch.cuda.synchronize(x.device)
        return graph

    @staticmethod
    def _time(graph, count):
        samples = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(3):
            start.record()
            for _ in range(32):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / (32 * count))
        return statistics.median(samples)

    def run(self, name, layer, x, weight, output):
        plan = self.plans.get(name) if x.shape[0] == 1 else None
        if plan is None:
            self.native.run(name, layer, x, weight, output)
        else:
            plan(x, weight, output)
