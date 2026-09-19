"""Select dense BF16 projection implementations only during untimed warmup."""

import statistics
import sys
import time

import torch
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources

from small_batch_gemm import LinearPlan, candidates


class LinearTuner:
    def __init__(self, engine, deadline):
        self.plans = {}
        if engine.batch > 16 or time.monotonic() >= deadline:
            return
        # Rotate actual layer weights to exceed H100 L2. Benchmarking one hot
        # weight would misrepresent the full-model stream through 36 layers.
        layers = engine.layers[:6]
        # Spend a bounded search on the largest weight streams first.
        groups = (
            ("gateup", engine.normalized, engine.gateup,
             [pair[1] for pair in engine.packed[:6]]),
            ("down", engine.intermediate, engine.branch,
             [layer.mlp.down_proj.weight for layer in layers]),
            ("qkv", engine.normalized, engine.qkv,
             [pair[0] for pair in engine.packed[:6]]),
            ("output", engine.attention, engine.branch,
             [layer.self_attn.o_proj.weight for layer in layers]),
        )
        # Do not consume or reseed the caller's RNG state. These synthetic
        # activations are independent of every prompt and generated token.
        generator = torch.Generator(device=engine.normalized.device)
        generator.manual_seed(12423)
        for name, template, output, weights in groups:
            if time.monotonic() >= deadline:
                break
            x = torch.randn(template.shape, dtype=template.dtype,
                            device=template.device, generator=generator)
            reference = torch.mm(x, weights[0].t())
            base_ms = self._time(None, x, weights, output)
            best_ms, best = base_ms, None
            for config in candidates(engine.batch, weights[0].shape[0], x.shape[1]):
                # Compilation cannot be interrupted safely. This soft cutoff
                # is checked before each specialization, with 120 seconds of
                # the overall budget reserved by the engine for normal warmup.
                if time.monotonic() >= deadline:
                    break
                # Initialize and check the exact dense product outside capture.
                try:
                    plan = LinearPlan(x, weights[0], output, config)
                    plan(x, weights[0], output)
                except (CompilationError, OutOfResources) as error:
                    print(f"[linear] {name} {config.name}: {type(error).__name__}",
                          file=sys.stderr, flush=True)
                    continue
                torch.cuda.synchronize()
                if not torch.allclose(output, reference, rtol=0.01, atol=0.01):
                    continue
                timing = self._time(plan, x, weights, output)
                # Require headroom over measurement noise and native cuBLAS.
                if timing < best_ms * 0.97:
                    best_ms, best = timing, plan
            if best is not None and time.monotonic() < deadline:
                # Recheck a second layer and activation before accepting a
                # winner. This catches indexing mistakes, not model-level
                # numerical regressions; the full generation still needs its
                # ordinary end-to-end correctness check.
                probe = torch.randn(template.shape, dtype=template.dtype,
                                    device=template.device, generator=generator)
                expected = torch.mm(probe, weights[-1].t())
                best(probe, weights[-1], output)
                if torch.allclose(output, expected, rtol=0.01, atol=0.01):
                    # A cold initial baseline and winner-selection noise must
                    # not move a projection away from cuBLAS. Require repeated
                    # wins against a freshly measured, warmed native path.
                    native = [self._time(None, x, weights, output)]
                    custom = [self._time(best, x, weights, output)]
                    custom.append(self._time(best, x, weights, output))
                    native.append(self._time(None, x, weights, output))
                    base_ms, best_ms = min(native), max(custom)
                    if best_ms < base_ms * 0.95:
                        self.plans[name] = best
            selected = self.plans.get(name)
            selected_name = selected.config.name if selected is not None else "torch.mm"
            chosen_ms = best_ms if selected is not None else base_ms
            print(f"[linear] B={engine.batch} {name}: {selected_name} "
                  f"{chosen_ms * 1000:.2f} us; native {base_ms * 1000:.2f} us",
                  file=sys.stderr, flush=True)

    @staticmethod
    def _call(plan, x, weight, output):
        if plan is None:
            torch.mm(x, weight.t(), out=output)
        else:
            plan(x, weight, output)

    @classmethod
    def _time(cls, plan, x, weights, output):
        current = torch.cuda.current_stream()
        stream = torch.cuda.Stream()
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for weight in weights:
                cls._call(plan, x, weight, output)
        current.wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for weight in weights:
                cls._call(plan, x, weight, output)
        graph.replay()
        torch.cuda.synchronize()
        samples = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(3):
            start.record()
            for _ in range(32):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / (32 * len(weights)))
        return statistics.median(samples)

    def run(self, name, x, weight, output):
        self._call(self.plans.get(name), x, weight, output)
