"""Bounded eight-warp refinements of frozen dense Hopper device kernels.

M128/K128/stages2 covers B2-32; M64/K128/stages3 covers only B17-32.
M64/BB16 is excluded because pinned Triton3.1 selects instruction N8,
which is outside its Hopper linear-layout converter's asserted domain.
"""
import math
import sys
import time

import torch
import triton
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources
from hopper_gemm import HopperPlan, _hopper_dot, _hopper_merge
from hopper_tiles import HopperTilesPlan, HopperTilesLayout, _hopper_tiles_dot


class HopperWarpsPlan(HopperTilesPlan, HopperPlan):
    def __init__(self, x, weight, output, block_m, splits):
        if block_m == 128:
            HopperTilesPlan.__init__(self, x, weight, output, splits)
        elif (block_m == 64 and 17 <= x.shape[0] <= 32
                and splits == HopperPlan.split_count(weight.shape[0], 128)):
            HopperPlan.__init__(self, x, weight, output, 128)
        else:
            raise ValueError("unsupported eight-warp dense tile")
        self.block_m = block_m
        self.name = "eight_warps_" + self.name

    @staticmethod
    def choices(n, k, batch, sm_count):
        if not 2 <= batch <= 32:
            return ()
        result = tuple((128, split) for split in
                       HopperTilesPlan.split_choices(n, k, batch, sm_count))
        if batch >= 17:
            result += ((64, HopperPlan.split_count(n, 128)),)
        return result

    def __call__(self, x, weight, output):
        part = output if self.workspace is None else self.workspace
        kernel = _hopper_tiles_dot if self.block_m == 128 else _hopper_dot
        kernel[(triton.cdiv(self.n, self.block_m), self.splits)](
            x, weight, output, part, self.n, x.stride(0), output.stride(0),
            B=self.batch, K=self.k, BB=self.bb, BK=128, SPLITS=self.splits,
            num_warps=8, num_stages=self.stages,
        )
        if self.splits > 1:
            _hopper_merge[(triton.cdiv(self.batch * self.n, 512),)](
                self.workspace, output, self.n, output.stride(0), self.batch * self.n,
                SPLITS=self.splits, BLOCK=512, num_warps=4,
            )


class HopperWarpsLayout(HopperTilesLayout):
    """Eight-warp complete operations against actual selected Tiles/Hopper."""

    def __init__(self, engine, native, deadline):
        self.native, self.batch = native, engine.batch
        self.plans = {}
        self.device = engine.normalized.device
        if not 2 <= self.batch <= 32:
            return
        if time.monotonic() >= deadline:
            return
        self.sm_count = torch.cuda.get_device_properties(self.device).multi_processor_count
        if self.sm_count <= 0:
            return
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
        generator.manual_seed(63179)
        try:
            for name, template, output, weights in groups:
                if time.monotonic() >= deadline:
                    break
                winner = self._select(name, template, weights, output, generator, deadline)
                if winner is not None and time.monotonic() < deadline:
                    self.plans[name] = winner
                chosen = self.plans.get(name)
                self._log(name, "selected " + (chosen.name if chosen else "existing layout"))
        finally:
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()

    @property
    def extra_bytes(self):
        return self.native.extra_bytes + sum(plan.extra_bytes for plan in self.plans.values())

    def _log(self, name, message):
        print(f"[hopper-warps] B={self.batch} {name}: {message}",
              file=sys.stderr, flush=True)

    def _workspace_room(self, output, workspace_bytes):
        free, total = torch.cuda.mem_get_info(self.device)
        # Reference already exists. Reserve comparison temporaries as well as
        # the new persistent split buffer before allocating it.
        required = output.numel() * (32 - output.element_size()) + 65536
        return free - required - workspace_bytes >= total // 4

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
        native_graph = self._graph(name, None, x, weights, output, deadline)
        if native_graph is None:
            return None
        winner, winner_ms = None, float("inf")
        for block_m, splits in HopperWarpsPlan.choices(
                weights[0].shape[0], x.shape[1], self.batch, self.sm_count):
            if time.monotonic() >= deadline:
                break
            workspace_bytes = output.numel() * splits * 4 if splits > 1 else 0
            if not self._workspace_room(output, workspace_bytes):
                self._log(name, "existing: split workspace memory guard")
                continue
            try:
                plan = HopperWarpsPlan(x, weights[0], output, block_m, splits)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self._log(name, "existing: split workspace allocation failed")
                continue
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
            except (CompilationError, OutOfResources, torch.cuda.OutOfMemoryError) as error:
                self._log(name, f"{plan.name}: {type(error).__name__}")
                torch.cuda.empty_cache()
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
            if (all(math.isfinite(value) and value > 0
                    for value in native_times + candidate_times)
                    and custom_ms < native_ms * 0.95 and custom_ms < winner_ms):
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

