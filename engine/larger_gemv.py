"""Warmup-selected larger full-K B1 tiles, with the accepted path fallback.

The exact accepted full-K kernel is reused. Its tile grows from 16,384 to
32,768 padded values with eight warps: four to eight output rows at K<=4096,
or one to two rows at K=9728. No serial or persistent loop, partial buffer,
weight copy, new arithmetic, or intermediate cast is introduced.
"""

import math
import sys
import time

import torch
import triton
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources

from wide_gemv import WideGemvLayout, _wide_dot


class LargerPlan:
    """B1 row-major BF16 CUDA projection to caller-owned BF16 output."""

    def __init__(self, x, weight, output):
        self.batch, self.k = x.shape
        self.n = weight.shape[0]
        if (self.batch != 1 or self.k not in (2560, 4096, 9728)
                or tuple(weight.shape) != (self.n, self.k)
                or tuple(output.shape) != (1, self.n)):
            raise ValueError("unsupported larger full-K projection shape")
        for tensor in (x, weight, output):
            if (not tensor.is_cuda or tensor.dtype != torch.bfloat16
                    or tensor.device != x.device or tensor.stride(1) != 1):
                raise ValueError("expected row-major BF16 tensors on one CUDA device")
        if not weight.is_contiguous():
            raise ValueError("weight must be original contiguous [N,K]")
        self.width = triton.next_power_of_2(self.k)
        self.rows = 32768 // self.width
        self.name = f"larger_fullK_r{self.rows}_k{self.width}_w8"

    def __call__(self, x, weight, output):
        _wide_dot[(triton.cdiv(self.n, self.rows),)](
            x, weight, output, self.n, self.k,
            ROWS=self.rows, WIDTH=self.width,
            num_warps=8, num_stages=1, enable_fp_fusion=False,
        )


class LargerGemvLayout(WideGemvLayout):
    """Select one additional tile per K against actual Native/Wide/Hopper."""

    def __init__(self, engine, native, deadline):
        self.native, self.batch = native, engine.batch
        self.plans = {}
        self.device = engine.normalized.device
        if self.batch != 1 or time.monotonic() >= deadline:
            return
        # At most three new specializations; N is a runtime kernel argument.
        # Retain the engine-wide deadline and bound this stage to 30 seconds.
        deadline = min(deadline, time.monotonic() + 30.0)
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
        generator.manual_seed(69317)
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

    @staticmethod
    def _log(name, message):
        print(f"[larger-gemv] B=1 {name}: {message}", file=sys.stderr, flush=True)

    def _check(self, name, plan, x, weights, output, reference, generator, deadline):
        # Every actual weight in the timing pool must match at every scale.
        # Existing persistent activations are scratch until real prefill.
        for scale in (1.0, 0.1, 10.0):
            for index, weight in enumerate(weights):
                if time.monotonic() >= deadline:
                    return False
                x.normal_(generator=generator).mul_(scale)
                self.native.run(name, index, x, weight, reference)
                if time.monotonic() >= deadline:
                    return False
                plan(x, weight, output)
                if time.monotonic() >= deadline:
                    return False
                if not self._room(output, reference_live=True):
                    self._log(name, "existing: comparison memory guard")
                    return False
                try:
                    if not torch.allclose(output, reference, rtol=0.01, atol=0.01):
                        return False
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    self._log(name, "existing: comparison allocation failed")
                    return False
        return True

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
        x = template
        x.normal_(generator=generator)
        if time.monotonic() >= deadline:
            return None
        existing_graph = self._graph(name, None, x, weights, output, deadline)
        if existing_graph is None or time.monotonic() >= deadline:
            return None
        plan = LargerPlan(x, weights[0], output)
        try:
            if not self._check(name, plan, x, weights, output, reference,
                               generator, deadline):
                self._log(name, f"{plan.name}: numerical check rejected")
                return None
            if time.monotonic() >= deadline:
                return None
            candidate_graph = self._graph(name, plan, x, weights, output, deadline)
            if candidate_graph is None:
                return None
        except (CompilationError, OutOfResources) as error:
            self._log(name, f"{plan.name}: {type(error).__name__}")
            return None
        # Inherited graphs include the complete six-real-weight rotation.
        # The 742 MiB head is one real weight and already much larger than L2.
        existing_times, candidate_times = [], []
        for graph, samples in ((existing_graph, existing_times),
                               (candidate_graph, candidate_times),
                               (candidate_graph, candidate_times),
                               (existing_graph, existing_times)):
            if time.monotonic() >= deadline:
                return None
            samples.append(self._time(graph, len(weights)))
        if time.monotonic() >= deadline:
            return None
        if not all(math.isfinite(value) and value > 0
                   for value in existing_times + candidate_times):
            return None
        existing_ms, custom_ms = min(existing_times), max(candidate_times)
        self._log(name, f"{plan.name}: existing {existing_ms * 1000:.2f} us, "
                  f"custom {custom_ms * 1000:.2f} us")
        return plan if custom_ms < existing_ms * 0.95 else None

    def run(self, name, layer, x, weight, output):
        plan = self.plans.get(name) if x.shape[0] == 1 else None
        if (plan is None or tuple(x.shape) != (1, plan.k)
                or tuple(weight.shape) != (plan.n, plan.k)
                or tuple(output.shape) != (1, plan.n)):
            self.native.run(name, layer, x, weight, output)
        else:
            plan(x, weight, output)
