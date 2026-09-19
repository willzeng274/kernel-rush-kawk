"""Bounded M128/K128 refinement against the accepted Hopper dispatcher.

Original BF16 operands, FP32 accumulation/merge and final BF16 storage.
Four warps, two pipeline stages; no weight copies or replay allocations.
"""

import math
import sys
import time

import torch
import triton
import triton.language as tl
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources

from hopper_gemm import HopperGemmLayout, _hopper_merge


@triton.jit
def _hopper_tiles_dot(X, W, OUT, PART, N, X_ROW, OUT_ROW,
                B: tl.constexpr, K: tl.constexpr, BB: tl.constexpr,
                BK: tl.constexpr, SPLITS: tl.constexpr):
    n = tl.program_id(0) * 128 + tl.arange(0, 128)
    split = tl.program_id(1)
    b = tl.arange(0, BB)
    k = tl.arange(0, BK)
    steps = tl.cdiv(K, SPLITS * BK)
    acc = tl.full((128, BB), 0, tl.float32)
    for block in range(steps):
        kk = (split * steps + block) * BK + k
        w = tl.load(W + n[:, None] * K + kk[None, :],
                    (n[:, None] < N) & (kk[None, :] < K), 0)
        x = tl.load(X + b[None, :] * X_ROW + kk[:, None],
                    (b[None, :] < B) & (kk[:, None] < K), 0)
        acc = tl.dot(w, x, acc)
    if SPLITS == 1:
        tl.store(OUT + b[None, :] * OUT_ROW + n[:, None], acc,
                 (b[None, :] < B) & (n[:, None] < N))
    else:
        tl.store(PART + (split * B + b[None, :]) * N + n[:, None], acc,
                 (b[None, :] < B) & (n[:, None] < N))


class HopperTilesPlan:
    def __init__(self, x, weight, output, splits):
        self.batch, self.k = x.shape
        self.n = weight.shape[0]
        if (not 2 <= self.batch <= 32 or self.k not in (2560, 4096, 9728)
                or tuple(weight.shape) != (self.n, self.k)
                or tuple(output.shape) != (self.batch, self.n)
                or splits not in (1, 2, 4, 8)):
            raise ValueError("unsupported dense decode shape or tile")
        for tensor in (x, weight, output):
            if (not tensor.is_cuda or tensor.dtype != torch.bfloat16
                    or tensor.device != x.device or tensor.stride(1) != 1):
                raise ValueError("expected row-major BF16 tensors on one CUDA device")
        if not weight.is_contiguous():
            raise ValueError("weight must be original contiguous [N,K]")
        self.bb = 16 if self.batch <= 16 else 32
        self.block_k, self.stages, self.splits = 128, 2, splits
        self.workspace = (torch.empty((self.splits, self.batch, self.n),
                                      dtype=torch.float32, device=x.device)
                          if self.splits > 1 else None)
        self.name = f"transpose_m128_n{self.bb}_k128_s{splits}_p2"

    @staticmethod
    def split_choices(n, k, batch, sm_count):
        # Low-channel projections need at least one full SM wave. Keep K
        # partitions and workspace bounded even on an unexpected device.
        channel_tiles = triton.cdiv(n, 128)
        splits = 1
        if n < 8192:
            while channel_tiles * splits < sm_count and splits < 8:
                splits *= 2
            return (splits,)
        # Gate/up already fills one SM wave on H100. Test a second wave only
        # when added FP32 partial traffic is <=2.5% of original weight bytes.
        # Head has ample CTAs and keeps its single complete K reduction.
        if (n == 19456 and channel_tiles < 2 * sm_count
                and 8 * 2 * batch * n * 40 <= 2 * n * k):
            return (2, 1)
        return (1,)

    @property
    def extra_bytes(self):
        return 0 if self.workspace is None else self.workspace.numel() * 4

    def __call__(self, x, weight, output):
        part = output if self.workspace is None else self.workspace
        _hopper_tiles_dot[(triton.cdiv(self.n, 128), self.splits)](
            x, weight, output, part, self.n, x.stride(0), output.stride(0),
            B=self.batch, K=self.k, BB=self.bb, BK=self.block_k, SPLITS=self.splits,
            num_warps=4, num_stages=self.stages,
        )
        if self.splits > 1:
            _hopper_merge[(triton.cdiv(self.batch * self.n, 512),)](
                self.workspace, output, self.n, output.stride(0), self.batch * self.n,
                SPLITS=self.splits, BLOCK=512, num_warps=4,
            )


class HopperTilesLayout(HopperGemmLayout):
    """Whole-operation selection against the already selected Hopper path."""

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
        generator.manual_seed(59273)
        for name, template, output, weights in groups:
            if time.monotonic() >= deadline:
                break
            winner = self._select(name, template, weights, output, generator, deadline)
            # _select returns only fully checked, timed, and drained plans.
            if winner is not None:
                self.plans[name] = winner
            chosen = self.plans.get(name)
            self._log(name, "selected " + (chosen.name if chosen else "existing layout"))
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()

    @property
    def extra_bytes(self):
        return self.native.extra_bytes + sum(plan.extra_bytes for plan in self.plans.values())

    def _log(self, name, message):
        print(f"[hopper-tiles] B={self.batch} {name}: {message}",
              file=sys.stderr, flush=True)

    def _workspace_room(self, output, workspace_bytes):
        free, total = torch.cuda.mem_get_info(self.device)
        # Reference already exists. Reserve comparison temporaries as well as
        # the new persistent split buffer before allocating it.
        required = output.numel() * (32 - output.element_size()) + 65536
        return free - required - workspace_bytes >= total // 4

    def _select(self, name, template, weights, output, generator, deadline):
        try:
            if time.monotonic() >= deadline:
                return None
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            if time.monotonic() >= deadline or not self._room(output):
                self._log(name, "existing: deadline or numerical memory guard")
                return None
            if time.monotonic() >= deadline:
                return None
            try:
                reference = torch.empty_like(output)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self._log(name, "existing: numerical reference allocation failed")
                return None
            x = template
            if time.monotonic() >= deadline:
                return None
            x.normal_(generator=generator)
            if time.monotonic() >= deadline:
                return None
            native_graph = self._graph(name, None, x, weights, output, deadline)
            if native_graph is None:
                return None
            winner, winner_ms = None, float("inf")
            for splits in HopperTilesPlan.split_choices(
                    weights[0].shape[0], x.shape[1], self.batch, self.sm_count):
                if time.monotonic() >= deadline:
                    break
                workspace_bytes = output.numel() * splits * 4 if splits > 1 else 0
                if not self._workspace_room(output, workspace_bytes):
                    self._log(name, "existing: split workspace memory guard")
                    continue
                if time.monotonic() >= deadline:
                    break
                try:
                    plan = HopperTilesPlan(x, weights[0], output, splits)
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
                except (CompilationError, OutOfResources) as error:
                    self._log(name, f"{plan.name}: {type(error).__name__}")
                    torch.cuda.synchronize(self.device)
                    continue
                native_times, candidate_times = [], []
                for graph, samples in ((native_graph, native_times),
                                       (candidate_graph, candidate_times),
                                       (candidate_graph, candidate_times),
                                       (native_graph, native_times)):
                    if time.monotonic() >= deadline:
                        break
                    value = self._time(graph, len(weights), deadline)
                    if value is None:
                        break
                    samples.append(value)
                if len(native_times) != 2 or len(candidate_times) != 2:
                    del candidate_graph
                    break
                if time.monotonic() >= deadline:
                    break
                native_ms, custom_ms = min(native_times), max(candidate_times)
                self._log(name, f"{plan.name}: existing {native_ms * 1000:.2f} us, "
                          f"custom {custom_ms * 1000:.2f} us")
                if (all(math.isfinite(value) and value > 0
                        for value in native_times + candidate_times)
                        and custom_ms < native_ms * 0.95 and custom_ms < winner_ms):
                    winner, winner_ms = plan, custom_ms
                del candidate_graph
            # A later incomplete trial must not erase an earlier valid winner.
            return winner
        finally:
            # Drain before local references/workspaces leave scope.
            # CUDA execution or drain errors must still propagate.
            torch.cuda.synchronize(self.device)

    def run(self, name, layer, x, weight, output):
        plan = self.plans.get(name) if x.shape[0] == self.batch else None
        if (plan is None or tuple(x.shape) != (plan.batch, plan.k)
                or tuple(weight.shape) != (plan.n, plan.k)
                or tuple(output.shape) != (plan.batch, plan.n)):
            self.native.run(name, layer, x, weight, output)
        else:
            plan(x, weight, output)
