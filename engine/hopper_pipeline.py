"""Two bounded K64 Hopper pipelines over the unchanged dense device kernels.

M128 uses three stages; M64 uses four. Original BF16 operands, FP32
accumulation/merge, final BF16 storage, and no retained weight copies.
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


class HopperPipelinePlan:
    def __init__(self, x, weight, output, tile_m, splits):
        self.batch, self.k = x.shape
        self.n = weight.shape[0]
        if (not 2 <= self.batch <= 32 or self.k not in (2560, 4096, 9728)
                or tuple(weight.shape) != (self.n, self.k)
                or tuple(output.shape) != (self.batch, self.n)
                or tile_m not in (64, 128) or splits not in (1, 2, 4, 8)):
            raise ValueError("unsupported dense decode shape or tile")
        for tensor in (x, weight, output):
            if (not tensor.is_cuda or tensor.dtype != torch.bfloat16
                    or tensor.device != x.device or tensor.stride(1) != 1):
                raise ValueError("expected row-major BF16 tensors on one CUDA device")
        if not weight.is_contiguous():
            raise ValueError("weight must be original contiguous [N,K]")
        self.bb = 16 if self.batch <= 16 else 32
        self.tile_m = tile_m
        self.block_k, self.stages, self.splits = 64, (3 if tile_m == 128 else 4), splits
        self.workspace = (torch.empty((self.splits, self.batch, self.n),
                                      dtype=torch.float32, device=x.device)
                          if self.splits > 1 else None)
        self.name = f"pipeline_m{tile_m}_n{self.bb}_k64_s{splits}_p{self.stages}"

    @staticmethod
    def configurations(n, k, batch, sm_count):
        # Exactly two configurations. Preserve each existing tile's split
        # policy; M128 takes only the first eligible choice, never a split grid.
        m128_splits = HopperTilesPlan.split_choices(n, k, batch, sm_count)[0]
        m64_splits = HopperPlan.split_count(n, 128)
        return ((128, m128_splits), (64, m64_splits))

    @property
    def extra_bytes(self):
        return 0 if self.workspace is None else self.workspace.numel() * 4

    def __call__(self, x, weight, output):
        part = output if self.workspace is None else self.workspace
        kernel = _hopper_tiles_dot if self.tile_m == 128 else _hopper_dot
        kernel[(triton.cdiv(self.n, self.tile_m), self.splits)](
            x, weight, output, part, self.n, x.stride(0), output.stride(0),
            B=self.batch, K=self.k, BB=self.bb, BK=self.block_k, SPLITS=self.splits,
            num_warps=4, num_stages=self.stages,
        )
        if self.splits > 1:
            _hopper_merge[(triton.cdiv(self.batch * self.n, 512),)](
                self.workspace, output, self.n, output.stride(0), self.batch * self.n,
                SPLITS=self.splits, BLOCK=512, num_warps=4,
            )


class HopperPipelineLayout(HopperTilesLayout):
    """Two K64 pipelines against the actual accepted Tiles/Hopper dispatcher."""

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
        generator.manual_seed(62983)
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
        print(f"[hopper-pipeline] B={self.batch} {name}: {message}",
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
        for tile_m, splits in HopperPipelinePlan.configurations(
                weights[0].shape[0], x.shape[1], self.batch, self.sm_count):
            if time.monotonic() >= deadline:
                break
            workspace_bytes = output.numel() * splits * 4 if splits > 1 else 0
            if not self._workspace_room(output, workspace_bytes):
                self._log(name, "existing: split workspace memory guard")
                continue
            try:
                plan = HopperPipelinePlan(x, weights[0], output, tile_m, splits)
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

    def run(self, name, layer, x, weight, output):
        plan = self.plans.get(name) if x.shape[0] == self.batch else None
        if (plan is None or tuple(x.shape) != (plan.batch, plan.k)
                or tuple(weight.shape) != (plan.n, plan.k)
                or tuple(output.shape) != (plan.batch, plan.n)):
            self.native.run(name, layer, x, weight, output)
        else:
            plan(x, weight, output)
