"""Fuse only an already-selected split projection's merge/residual/RMSNorm.

The accepted dispatcher and its FP32 partial workspace retain ownership. No
new inference scratch, projection tuning, K partition or BF16 boundary changes.
Private tuning states are reset equally in both complete-segment graphs.
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
from hopper_gemm import HopperPlan, HopperGemmLayout
from hopper_tiles import HopperTilesPlan, HopperTilesLayout
from native_layout import NativeLayout
from persistent_vector import PersistentVectorLayout
from wide_gemv import WideGemvLayout


@triton.jit
def _merge_residual_norm(PART, X, GAIN, NORMALIZED,
                         B: tl.constexpr, H: tl.constexpr, SPLITS: tl.constexpr,
                         EPS: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    branch = tl.full((BLOCK,), 0, tl.float32)
    # Identical per-element sequential FP32 order to _hopper_merge.
    for split in tl.static_range(SPLITS):
        branch += tl.load(PART + (split * B + b) * H + d, d < H, 0)
    branch = branch.to(tl.bfloat16).to(tl.float32)
    x = tl.load(X + b * H + d, d < H, 0).to(tl.float32)
    x = (x + branch).to(tl.bfloat16).to(tl.float32)
    tl.store(X + b * H + d, x, d < H)
    inv = tl.rsqrt(tl.sum(x * x, 0) / H + EPS)
    n = (x * inv).to(tl.bfloat16).to(tl.float32)
    gain = tl.load(GAIN + d, d < H, 0).to(tl.float32)
    tl.store(NORMALIZED + b * H + d, n * gain, d < H)


def effective_plan(layout, name, x, weight, output):
    """Mirror each known run guard; never pass a selected unsupported plan."""
    seen = set()
    while id(layout) not in seen:
        seen.add(id(layout))
        kind = type(layout)
        if kind is NativeLayout:
            return None
        if kind is PersistentVectorLayout:
            plan = layout.plans.get(name) if x.shape[0] == 1 else None
            selected = (plan is not None and tuple(x.shape) == (1, plan.k)
                        and tuple(weight.shape) == (plan.n, plan.k)
                        and tuple(output.shape) == (1, plan.n))
        elif kind in (HopperTilesLayout, HopperGemmLayout):
            plan = layout.plans.get(name) if x.shape[0] == layout.batch else None
            selected = (plan is not None
                        and tuple(x.shape) == (plan.batch, plan.k)
                        and tuple(weight.shape) == (plan.n, plan.k)
                        and tuple(output.shape) == (plan.batch, plan.n))
        elif kind is WideGemvLayout:
            plan = layout.plans.get(name) if x.shape[0] == 1 else None
            selected = plan is not None
        else:
            return None
        if selected:
            return layout, plan
        layout = layout.native
    return None


def _signature(plan):
    return (type(plan), plan.batch, plan.k, plan.n, plan.bb,
            plan.block_k, plan.stages, plan.splits)


class Boundary:
    """Fixed metadata and strong references for this allocation's boundary."""
    def __init__(self, owner, plan, x, branch, hidden, normalized, weights, gains):
        self.owner, self.plan, self.workspace = owner, plan, plan.workspace
        self.signature = _signature(plan)
        self.x, self.branch, self.hidden, self.normalized = x, branch, hidden, normalized
        self.weights, self.gains = tuple(weights), tuple(gains)

    def valid(self, fallback, name, layer, x, weight, hidden, gain, normalized):
        p, b = self.plan, self.plan.batch
        if (not isinstance(layer, int) or not 0 <= layer < len(self.weights)
                or x is not self.x or hidden is not self.hidden
                or normalized is not self.normalized
                or weight is not self.weights[layer] or gain is not self.gains[layer]
                or type(p) not in (HopperPlan, HopperTilesPlan)
                or _signature(p) != self.signature
                or p.workspace is not self.workspace
                or not 2 <= b <= 32 or p.n != 2560
                or p.k != (4096 if name == "output" else 9728)
                or p.splits not in (2, 4, 8)):
            return False
        current = effective_plan(fallback, name, x, weight, self.branch)
        if current is None or current[0] is not self.owner or current[1] is not p:
            return False
        tensors = (x, weight, self.branch, hidden, normalized, gain)
        if any(not t.is_cuda or t.dtype != torch.bfloat16
               or t.device != x.device for t in tensors):
            return False
        if (tuple(x.shape) != (b, p.k) or x.stride(1) != 1
                or x.stride(0) < p.k
                or tuple(weight.shape) != (2560, p.k) or not weight.is_contiguous()
                or tuple(gain.shape) != (2560,) or not gain.is_contiguous()
                or any(tuple(t.shape) != (b, 2560) or not t.is_contiguous()
                       for t in (self.branch, hidden, normalized))):
            return False
        w = self.workspace
        return (w is not None and w.is_cuda and w.device == x.device
                and w.dtype == torch.float32 and w.is_contiguous()
                and tuple(w.shape) == (p.splits, b, 2560))


class MergeNorm:
    """One fixed epilogue, independently measured for output and down."""
    def __init__(self, engine, deadline):
        self.fallback, self.batch = engine.native_layout, engine.batch
        self.device, self.eps = engine.hidden.device, engine.eps
        self.bindings = {}
        self.deadline = min(deadline, time.monotonic() + 15.0)
        if (not 2 <= self.batch <= 32 or engine.h != 2560
                or time.monotonic() >= self.deadline):
            self._log("all", "accepted path: batch, shape or deadline")
            return
        try:
            pending = []
            for name, x, weights, gains in (
                ("output", engine.attention,
                 [l.self_attn.o_proj.weight for l in engine.layers],
                 [l.post_attention_layernorm.weight for l in engine.layers]),
                ("down", engine.intermediate,
                 [l.mlp.down_proj.weight for l in engine.layers],
                 [l.input_layernorm.weight for l in engine.layers[1:]]
                 + [engine.base.norm.weight]),
            ):
                found = effective_plan(self.fallback, name, x, weights[0], engine.branch)
                if found is None or type(found[1]) not in (HopperPlan, HopperTilesPlan):
                    self._log(name, "accepted path: no eligible selected split plan")
                    continue
                binding = Boundary(*found, x, engine.branch, engine.hidden,
                                   engine.normalized, weights, gains)
                if (len(weights) < 6 or len(gains) != len(weights)
                        or not all(binding.valid(self.fallback, name, i, x, w,
                                                 engine.hidden, gains[i], engine.normalized)
                                   for i, w in enumerate(weights))):
                    self._log(name, "accepted path: current metadata rejected")
                    continue
                pending.append((name, binding))
            if not pending:
                return
            generator = torch.Generator(device=self.device)
            generator.manual_seed(62171)
            for name, binding in pending:
                if time.monotonic() >= self.deadline:
                    self._log(name, "accepted path: deadline")
                    continue
                accepted = self._select(name, binding, generator)
                # Recheck ownership after the last synchronized timing trial.
                if (accepted and time.monotonic() < self.deadline
                        and binding.valid(self.fallback, name, 0, binding.x,
                                          binding.weights[0], binding.hidden,
                                          binding.gains[0], binding.normalized)):
                    self.bindings[name] = binding
                self._log(name, "selected " + (binding.plan.name + "+merge_norm"
                          if name in self.bindings else "accepted path"))
        finally:
            # All temporary graphs and states must be drained before release.
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()

    def _log(self, name, message):
        print(f"[merge-norm] B={self.batch} {name}: {message}", file=sys.stderr, flush=True)

    def _room(self, binding):
        free, total = torch.cuda.mem_get_info(self.device)
        # Five private BF16 [B,H] tensors plus conservative finite/equality
        # temporaries. Model, cache and existing split buffers already live.
        required = binding.hidden.numel() * 64 + 65536
        return free - required >= total // 4

    def _baseline(self, name, index, binding, hidden, normalized):
        self.fallback.run(name, index, binding.x, binding.weights[index], binding.branch)
        residual_norm_kernel[(self.batch,)](
            binding.branch, hidden, binding.gains[index], normalized,
            2560, self.eps, 4096, num_warps=4, enable_fp_fusion=False,
        )

    def _fused(self, binding, x, weight, hidden, gain, normalized):
        binding.plan.produce_partials(x, weight, binding.branch)
        _merge_residual_norm[(self.batch,)](
            binding.workspace, hidden, gain, normalized,
            B=self.batch, H=2560, SPLITS=binding.plan.splits,
            EPS=self.eps, BLOCK=4096, num_warps=4, enable_fp_fusion=False,
        )

    def _check(self, name, binding, seed, reference, candidate, generator):
        for index in range(6):
            for scale in (0.1, 1.0, 10.0):
                if time.monotonic() >= self.deadline or not self._room(binding):
                    return False
                binding.x.normal_(generator=generator).mul_(scale)
                seed.normal_(generator=generator).mul_(scale)
                reference[0].copy_(seed)
                candidate[0].copy_(seed)
                self._baseline(name, index, binding, *reference)
                if time.monotonic() >= self.deadline:
                    return False
                self._fused(binding, binding.x, binding.weights[index],
                            candidate[0], binding.gains[index], candidate[1])
                if (time.monotonic() >= self.deadline
                        or not self._room(binding)
                        or not all(torch.equal(a.view(torch.int16), b.view(torch.int16))
                                   for a, b in zip(reference, candidate))
                        or not all(torch.isfinite(t).all().item() for t in reference)):
                    return False
        return True

    def _select(self, name, binding, generator):
        baseline_graph = candidate_graph = None
        try:
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            if time.monotonic() >= self.deadline or not self._room(binding):
                return False
            seed = torch.empty_like(binding.hidden)
            reference = (torch.empty_like(seed), torch.empty_like(seed))
            candidate = (torch.empty_like(seed), torch.empty_like(seed))
            if not self._check(name, binding, seed, reference, candidate, generator):
                self._log(name, "accepted path: exact numerical/resource check rejected")
                return False
            baseline_graph = self._graph(name, binding, seed, reference, False)
            if baseline_graph is None:
                return False
            candidate_graph = self._graph(name, binding, seed, candidate, True)
            if candidate_graph is None:
                return False
            baseline_times, candidate_times = [], []
            for graph, samples in ((baseline_graph, baseline_times),
                                   (candidate_graph, candidate_times),
                                   (candidate_graph, candidate_times),
                                   (baseline_graph, baseline_times)):
                if time.monotonic() >= self.deadline:
                    return False
                samples.append(WideGemvLayout._time(graph, 6))
            if not all(math.isfinite(t) and t > 0 for t in baseline_times + candidate_times):
                return False
            old, new = min(baseline_times), max(candidate_times)
            self._log(name, f"accepted {old * 1000:.2f} us, fused {new * 1000:.2f} us")
            return time.monotonic() < self.deadline and new < old * 0.95
        except (CompilationError, OutOfResources, torch.cuda.OutOfMemoryError) as error:
            self._log(name, "accepted path: " + type(error).__name__)
            return False
        finally:
            # These local references stay live through the drain on ALL exits.
            torch.cuda.synchronize(self.device)
            baseline_graph = candidate_graph = None
            torch.cuda.empty_cache()

    def _graph(self, name, binding, seed, state, fused):
        if time.monotonic() >= self.deadline:
            return None
        current = torch.cuda.current_stream(self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(current)

        def launch():
            for index in range(6):
                # This equal reset is part of BOTH timed segments; its time
                # is never subtracted and replay cannot accumulate residuals.
                state[0].copy_(seed)
                if fused:
                    self._fused(binding, binding.x, binding.weights[index],
                                state[0], binding.gains[index], state[1])
                else:
                    self._baseline(name, index, binding, *state)

        try:
            with torch.cuda.stream(stream):
                launch()
        finally:
            current.wait_stream(stream)
            stream.synchronize()
        if time.monotonic() >= self.deadline:
            return None
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=stream):
                launch()
        finally:
            current.wait_stream(stream)
            stream.synchronize()
        if time.monotonic() >= self.deadline:
            return None
        graph.replay()
        torch.cuda.synchronize(self.device)
        return graph

    def try_run(self, name, layer, x, weight, hidden, gain, normalized):
        binding = self.bindings.get(name)
        if binding is None or not binding.valid(
                self.fallback, name, layer, x, weight, hidden, gain, normalized):
            return False
        self._fused(binding, x, weight, hidden, gain, normalized)
        return True
