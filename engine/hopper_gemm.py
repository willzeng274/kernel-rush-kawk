"""Transpose the product so M64 covers channels, enabling Hopper WGMMA.

W[N,K] @ X.T[K,B] is stored as OUT[B,N]. Original BF16 operands and FP32
accumulators retain the projection boundary. Low-channel-count operations use
bounded split-K with deterministic FP32 merging; there are no weight copies.
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
def _hopper_dot(X, W, OUT, PART, N, X_ROW, OUT_ROW,
                B: tl.constexpr, K: tl.constexpr, BB: tl.constexpr,
                BK: tl.constexpr, SPLITS: tl.constexpr):
    n = tl.program_id(0) * 64 + tl.arange(0, 64)
    split = tl.program_id(1)
    b = tl.arange(0, BB)
    k = tl.arange(0, BK)
    steps = tl.cdiv(K, SPLITS * BK)
    acc = tl.full((64, BB), 0, tl.float32)
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


@triton.jit
def _hopper_merge(PART, OUT, N, OUT_ROW, TOTAL,
                  SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    result = tl.full((BLOCK,), 0, tl.float32)
    # No atomics or BF16 partials; one deterministic FP32 merge then BF16 store.
    for split in tl.static_range(SPLITS):
        result += tl.load(PART + split * TOTAL + p, p < TOTAL, 0)
    tl.store(OUT + (p // N) * OUT_ROW + p % N, result, p < TOTAL)


class HopperPlan:
    def __init__(self, x, weight, output, block_k):
        self.batch, self.k = x.shape
        self.n = weight.shape[0]
        if (not 1 <= self.batch <= 32 or self.k not in (2560, 4096, 9728)
                or tuple(weight.shape) != (self.n, self.k)
                or tuple(output.shape) != (self.batch, self.n)
                or block_k not in (128, 256)):
            raise ValueError("unsupported dense decode shape or tile")
        for tensor in (x, weight, output):
            if (not tensor.is_cuda or tensor.dtype != torch.bfloat16
                    or tensor.device != x.device or tensor.stride(1) != 1):
                raise ValueError("expected row-major BF16 tensors on one CUDA device")
        if not weight.is_contiguous():
            raise ValueError("weight must be original contiguous [N,K]")
        self.bb = 16 if self.batch <= 16 else 32
        self.block_k = block_k
        self.stages = 3 if block_k == 128 else 2
        self.splits = self.split_count(self.n, block_k)
        self.workspace = (torch.empty((self.splits, self.batch, self.n),
                                      dtype=torch.float32, device=x.device)
                          if self.splits > 1 else None)
        self.name = f"transpose_m64_n{self.bb}_k{block_k}_s{self.splits}_p{self.stages}"

    @staticmethod
    def split_count(n, block_k):
        if n < 4096:
            return 8 if block_k == 128 else 4
        if n < 8192:
            return 4 if block_k == 128 else 2
        return 1

    @property
    def extra_bytes(self):
        return 0 if self.workspace is None else self.workspace.numel() * 4

    def __call__(self, x, weight, output):
        part = output if self.workspace is None else self.workspace
        _hopper_dot[(triton.cdiv(self.n, 64), self.splits)](
            x, weight, output, part, self.n, x.stride(0), output.stride(0),
            B=self.batch, K=self.k, BB=self.bb, BK=self.block_k, SPLITS=self.splits,
            num_warps=4, num_stages=self.stages,
        )
        if self.splits > 1:
            _hopper_merge[(triton.cdiv(self.batch * self.n, 512),)](
                self.workspace, output, self.n, output.stride(0), self.batch * self.n,
                SPLITS=self.splits, BLOCK=512, num_warps=4,
            )


class HopperGemmLayout(WideGemvLayout):
    """Bounded whole-operation selection against the accepted Native/Wide path."""

    def __init__(self, engine, native, deadline):
        self.native, self.batch = native, engine.batch
        self.plans = {}
        self.device = engine.normalized.device
        if not 1 <= self.batch <= 32:
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
        generator.manual_seed(48599)
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

    @property
    def extra_bytes(self):
        return self.native.extra_bytes + sum(plan.extra_bytes for plan in self.plans.values())

    def _log(self, name, message):
        print(f"[hopper-gemm] B={self.batch} {name}: {message}",
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
        for block_k in (256, 128):
            if time.monotonic() >= deadline:
                break
            splits = HopperPlan.split_count(weights[0].shape[0], block_k)
            workspace_bytes = output.numel() * splits * 4 if splits > 1 else 0
            if not self._workspace_room(output, workspace_bytes):
                self._log(name, "existing: split workspace memory guard")
                continue
            try:
                plan = HopperPlan(x, weights[0], output, block_k)
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
