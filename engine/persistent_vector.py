"""Bounded B1 full-K GEMV with persistent independent output tiles.

Each CTA retains X across a runtime loop over output tiles. No CTA waits for
another, and each output belongs to exactly one CTA/iteration. There is no
serial K loop, split reduction, extra weight layout, or activation reformulation.
"""
import math
import sys
import time

import torch
import triton
import triton.language as tl
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources

from wide_gemv import WideGemvLayout


@triton.jit
def _persistent_dot(X, W, OUT, N, K: tl.constexpr,
                    ROWS: tl.constexpr, WIDTH: tl.constexpr):
    k = tl.arange(0, WIDTH)
    r = tl.arange(0, ROWS)
    x = tl.load(X + k, k < K, 0).to(tl.float32)
    # N and the grid stride are runtime values, so output widths/grid caps do
    # not require distinct machine-code specializations.
    for tile in range(tl.program_id(0), tl.cdiv(N, ROWS), tl.num_programs(0)):
        n = tile * ROWS + r
        w = tl.load(W + n[:, None] * K + k[None, :],
                    (n[:, None] < N) & (k[None, :] < K), 0).to(tl.float32)
        result = tl.sum(w * x[None, :], axis=1)
        tl.store(OUT + n, result, n < N)


class PersistentPlan:
    def __init__(self, x, weight, output, sms, multiplier, warps):
        self.batch, self.k = x.shape
        self.n = weight.shape[0]
        if (self.batch != 1 or self.k not in (2560, 4096) or self.n <= 0
                or tuple(weight.shape) != (self.n, self.k)
                or tuple(output.shape) != (1, self.n)
                or not isinstance(sms, int) or isinstance(sms, bool) or sms <= 0
                or multiplier not in (2, 4) or warps not in (4, 8)):
            raise ValueError("unsupported persistent vector shape or launch")
        for tensor in (x, weight, output):
            if (not tensor.is_cuda or tensor.dtype != torch.bfloat16
                    or tensor.device != x.device or not tensor.is_contiguous()):
                raise ValueError("expected contiguous BF16 tensors on one CUDA device")
        self.width, self.rows = 4096, 4
        self.multiplier, self.warps = multiplier, warps
        self.grid = min((self.n + self.rows - 1) // self.rows, sms * multiplier)
        self.name = f"persistent_r4_k{self.k}_g{self.grid}_w{warps}"

    def __call__(self, x, weight, output):
        _persistent_dot[(self.grid,)](
            x, weight, output, self.n, self.k,
            ROWS=self.rows, WIDTH=self.width,
            num_warps=self.warps, num_stages=1, enable_fp_fusion=False,
        )


class PersistentVectorLayout(WideGemvLayout):
    """Actual Native/Wide/Hopper reference and fallback, fixed before capture."""

    def __init__(self, engine, native, deadline):
        self.native, self.batch = native, engine.batch
        self.plans = {}
        self.device = engine.normalized.device
        if self.batch != 1 or time.monotonic() >= deadline:
            return
        deadline = min(deadline, time.monotonic() + 50.0)
        self.sms = int(torch.cuda.get_device_properties(self.device).multi_processor_count)
        if self.sms <= 0:
            self._log("all", "existing: invalid SM count")
            return
        groups = (
            ("gateup", engine.normalized, engine.gateup,
             [pair[1] for pair in engine.packed[:6]]),
            ("qkv", engine.normalized, engine.qkv,
             [pair[0] for pair in engine.packed[:6]]),
            ("output", engine.attention, engine.branch,
             [layer.self_attn.o_proj.weight for layer in engine.layers[:6]]),
            ("head", engine.normalized, engine.logits,
             [engine.model.lm_head.weight]),
        )
        generator = torch.Generator(device=self.device)
        generator.manual_seed(58599)
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
        print(f"[persistent-vector] B={self.batch} {name}: {message}",
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
        x = template
        x.normal_(generator=generator)
        if time.monotonic() >= deadline:
            return None
        existing_graph = self._graph(name, None, x, weights, output, deadline)
        if existing_graph is None:
            return None
        winner, winner_ms = None, float("inf")
        # Four launch configurations share two K x two warp specializations;
        # both grid caps are runtime values consumed through num_programs.
        for multiplier, warps in ((4, 4), (2, 4), (4, 8), (2, 8)):
            if time.monotonic() >= deadline:
                break
            plan = PersistentPlan(x, weights[0], output, self.sms, multiplier, warps)
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
            existing_times, candidate_times = [], []
            for graph, samples in ((existing_graph, existing_times),
                                   (candidate_graph, candidate_times),
                                   (candidate_graph, candidate_times),
                                   (existing_graph, existing_times)):
                if time.monotonic() >= deadline:
                    break
                samples.append(self._time(graph, len(weights)))
            if len(existing_times) != 2 or len(candidate_times) != 2:
                del candidate_graph
                break
            existing_ms, custom_ms = min(existing_times), max(candidate_times)
            self._log(name, f"{plan.name}: existing {existing_ms * 1000:.2f} us, "
                      f"custom {custom_ms * 1000:.2f} us")
            if (all(math.isfinite(value) and value > 0
                    for value in existing_times + candidate_times)
                    and custom_ms < existing_ms * 0.95 and custom_ms < winner_ms):
                winner, winner_ms = plan, custom_ms
            del candidate_graph
        return winner if time.monotonic() < deadline else None

    def run(self, name, layer, x, weight, output):
        plan = self.plans.get(name) if x.shape[0] == 1 else None
        if (plan is None or tuple(x.shape) != (1, plan.k)
                or tuple(weight.shape) != (plan.n, plan.k)
                or tuple(output.shape) != (1, plan.n)):
            self.native.run(name, layer, x, weight, output)
        else:
            plan(x, weight, output)
