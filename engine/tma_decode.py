"""Load-only TMA transport for the actual selected Hopper decode producer.

Actual BF16 domains, original split loop, pointer stores and accepted merge.
The selector is allocation-local; it never adds a dispatcher or changes tiling.
"""
import math
import re
import statistics
import sys
import time
from contextlib import contextmanager

import numpy as np
import torch
import triton
import triton.language as tl
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources

from hopper_gemm import HopperPlan, HopperGemmLayout
from hopper_tiles import HopperTilesPlan, HopperTilesLayout
from native_layout import NativeLayout
from persistent_vector import PersistentVectorLayout
from wide_gemv import WideGemvLayout


@triton.jit
def _tma_decode_dot(X_DESC, W_DESC, OUT, PART, N, OUT_ROW,
                    B: tl.constexpr, K: tl.constexpr, TN: tl.constexpr,
                    BB: tl.constexpr, BK: tl.constexpr, SPLITS: tl.constexpr):
    tile_n = tl.program_id(0)
    n = tile_n * TN + tl.arange(0, TN)
    split = tl.program_id(1)
    b = tl.arange(0, BB)
    steps = tl.cdiv(K, SPLITS * BK)
    acc = tl.full((TN, BB), 0, tl.float32)
    for block in range(steps):
        k0 = (split * steps + block) * BK
        w = tl._experimental_descriptor_load(W_DESC, [tile_n * TN, k0],
                                              [TN, BK], tl.bfloat16)
        x = tl._experimental_descriptor_load(X_DESC, [0, k0],
                                              [BB, BK], tl.bfloat16)
        acc = tl.dot(w, tl.trans(x), acc)
    if SPLITS == 1:
        tl.store(OUT + b[None, :] * OUT_ROW + n[:, None], acc,
                 (b[None, :] < B) & (n[:, None] < N))
    else:
        tl.store(PART + (split * B + b[None, :]) * N + n[:, None], acc,
                 (b[None, :] < B) & (n[:, None] < N))


def effective_plan(layout, name, x, weight, output):
    """Match the real guards, stopping at any selected unsupported backend."""
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


def _matrix_ok(t):
    return (t.is_cuda and t.dtype == torch.bfloat16 and len(t.shape) == 2
            and t.is_contiguous() and tuple(t.stride()) == (t.shape[1], 1)
            and all(type(v) is int and 0 < v < 2**32 for v in t.shape)
            and t.data_ptr() > 0 and t.data_ptr() % 16 == 0
            and t.shape[1] * 2 % 16 == 0 and t.shape[1] * 2 < 2**40)


class TensorOwner:
    def __init__(self, tensor):
        self.tensor = tensor
        self.metadata = self.snapshot(tensor)

    @staticmethod
    def snapshot(t):
        return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device, t.is_cuda)

    def matches(self, tensor):
        return tensor is self.tensor and self.snapshot(tensor) == self.metadata


class DecodeDescriptor:
    """Separate from prefill: actual B rows, documented zero-filled load OOB."""
    def __init__(self, tensor, box):
        if (not _matrix_ok(tensor) or len(box) != 2
                or any(type(v) is not int or not 1 <= v <= 256 for v in box)
                or box[1] not in (128, 256)
                or box[0] not in (16, 32, 64, 128)
                or torch.cuda.is_current_stream_capturing()
                or torch.cuda.get_device_capability(tensor.device)[0] != 9):
            raise ValueError("unsupported decode descriptor tensor, box, device or capture")
        self.owner, self.box = TensorOwner(tensor), tuple(box)
        self.storage = np.zeros(255, dtype=np.int8)
        offset = (-self.storage.ctypes.data) % 128
        self.host = self.storage[offset:offset + 128]
        if (self.host.nbytes != 128 or self.host.ctypes.data % 128
                or not self.host.flags.c_contiguous or not self.host.flags.writeable):
            raise ValueError("unaligned descriptor host storage")
        # Prevalidation above covers the pinned C helper's asserted encoding.
        fill = triton.runtime.driver.active.utils.fill_2d_tma_descriptor
        fill(tensor.data_ptr(), *tensor.shape, *self.box, 2, self.host)
        try:
            self.gpu = torch.tensor(self.host, dtype=torch.int8, device=tensor.device)
            if self.gpu.data_ptr() % 128:
                raise ValueError("unaligned descriptor device storage")
            self.gpu_owner = TensorOwner(self.gpu)
        finally:
            # Keep self, host bytes and uploaded storage alive through failures.
            torch.cuda.synchronize(tensor.device)

    def matches(self, tensor):
        return (self.owner.matches(tensor) and _matrix_ok(tensor)
                and self.gpu_owner.matches(self.gpu))


class TmaRejected(Exception):
    """Only a known failure of this new producer, never the baseline."""


def _is_ptxas_failure(message):
    return (message.startswith("Internal Triton PTX codegen error: \n")
            or re.match(r"\APlease run `ptxas [^`\n]+` to confirm that this is a bug in `ptxas`\n", message) is not None
            or re.match(r"\A`ptxas` failed with error code -?\d+: \n", message) is not None)


class LoadTransport:
    SHAPES = {"qkv": (6144, 2560), "gateup": (19456, 2560),
              "output": (2560, 4096), "down": (2560, 9728),
              "head": (151936, 2560)}

    @staticmethod
    def supported(name, plan, x, weights, output):
        if (type(plan) not in (HopperPlan, HopperTilesPlan) or not weights
                or not 2 <= plan.batch <= 32
                or (plan.n, plan.k) != LoadTransport.SHAPES.get(name)
                or plan.bb != (16 if plan.batch <= 16 else 32)
                or plan.block_k not in (128, 256) or plan.splits not in (1, 2, 4, 8)
                or getattr(plan, "load_transport", None) is not None
                or tuple(x.shape) != (plan.batch, plan.k)
                or tuple(output.shape) != (plan.batch, plan.n)
                or any(not _matrix_ok(t) or t.device != x.device for t in (x, output, *weights))
                or any(tuple(w.shape) != (plan.n, plan.k) for w in weights)):
            return False
        if type(plan) is HopperTilesPlan:
            if plan.block_k != 128 or plan.stages != 2:
                return False
        elif plan.stages != (3 if plan.block_k == 128 else 2):
            return False
        workspace = plan.workspace
        if plan.splits == 1:
            return workspace is None
        return (workspace is not None and workspace.is_cuda
                and workspace.device == x.device and workspace.dtype == torch.float32
                and workspace.is_contiguous()
                and tuple(workspace.shape) == (plan.splits, plan.batch, plan.n)
                and tuple(workspace.stride()) == (plan.batch * plan.n, plan.n, 1)
                and workspace.data_ptr() > 0 and workspace.data_ptr() % 16 == 0)

    def __init__(self, name, plan, x, weights, output, deadline):
        if not self.supported(name, plan, x, weights, output):
            raise ValueError("unsupported effective decode plan or owners")
        self.plan, self.signature = plan, _signature(plan)
        self.tn = 64 if type(plan) is HopperPlan else 128
        self.output = TensorOwner(output)
        self.workspace = TensorOwner(plan.workspace) if plan.workspace is not None else None
        self.weights = {}
        self.compiled_verified = False
        try:
            if time.monotonic() >= deadline:
                raise TimeoutError("descriptor deadline")
            self.x = DecodeDescriptor(x, (plan.bb, plan.block_k))
            for weight in weights:
                if time.monotonic() >= deadline:
                    raise TimeoutError("descriptor family deadline")
                self.weights[id(weight)] = DecodeDescriptor(weight, (self.tn, plan.block_k))
        finally:
            # Incomplete family and GPU copies stay owned until fully drained.
            torch.cuda.synchronize(x.device)

    @property
    def extra_bytes(self):
        return (1 + len(self.weights)) * 128

    def eligible(self, plan, x, weight, output):
        descriptor = self.weights.get(id(weight))
        return (plan is self.plan and _signature(plan) == self.signature
                and descriptor is not None and descriptor.matches(weight)
                and self.x.matches(x) and self.output.matches(output)
                and ((self.workspace is None and plan.workspace is None)
                     or (self.workspace is not None and self.workspace.matches(plan.workspace))))

    def __call__(self, plan, x, weight, output):
        part = output if self.workspace is None else self.workspace.tensor
        try:
            compiled = _tma_decode_dot[(triton.cdiv(plan.n, self.tn), plan.splits)](
                self.x.gpu, self.weights[id(weight)].gpu, output, part, plan.n, output.stride(0),
                B=plan.batch, K=plan.k, TN=self.tn, BB=plan.bb,
                BK=plan.block_k, SPLITS=plan.splits, num_warps=4, num_stages=plan.stages)
        except (CompilationError, OutOfResources, torch.cuda.OutOfMemoryError) as error:
            raise TmaRejected(type(error).__name__) from error
        except RuntimeError as error:
            if _is_ptxas_failure(str(error)):
                raise TmaRejected(str(error)) from error
            raise
        if not self.compiled_verified:
            # Selection's first eager launch proves the pinned compiler actually
            # emitted TMA loads and Hopper MMA before any graph can be published.
            ptx = compiled.asm.get("ptx", "")
            if "cp.async.bulk.tensor.2d.shared::cluster.global" not in ptx or "wgmma.mma_async" not in ptx:
                raise TmaRejected("expected TMA loads and WGMMA absent from generated PTX")
            self.compiled_verified = True


class TmaDecode:
    """Install one transport per effective plan, after complete exact/ABBA checks."""
    def __init__(self, engine, deadline):
        self.engine, self.fallback = engine, engine.native_layout
        self.device, self.batch = engine.normalized.device, engine.batch
        self.deadline = min(deadline, time.monotonic() + 20.0)
        self.transports = {}
        if (not 2 <= self.batch <= 32 or time.monotonic() >= self.deadline
                or torch.cuda.is_current_stream_capturing()
                or torch.cuda.get_device_capability(self.device)[0] != 9
                or not callable(getattr(triton.runtime.driver.active.utils,
                                        "fill_2d_tma_descriptor", None))):
            return
        groups = (
            ("gateup", engine.normalized, engine.gateup, [p[1] for p in engine.packed]),
            ("down", engine.intermediate, engine.branch, [l.mlp.down_proj.weight for l in engine.layers]),
            ("qkv", engine.normalized, engine.qkv, [p[0] for p in engine.packed]),
            ("output", engine.attention, engine.branch, [l.self_attn.o_proj.weight for l in engine.layers]),
            ("head", engine.normalized, engine.logits, [engine.model.lm_head.weight]),
        )
        generator = torch.Generator(device=self.device)
        generator.manual_seed(94271)
        try:
            for name, x, output, weights in groups:
                if time.monotonic() >= self.deadline:
                    break
                found = effective_plan(self.fallback, name, x, weights[0], output)
                if found is None or not LoadTransport.supported(name, found[1], x, weights, output):
                    continue
                owner, plan = found
                if not all(effective_plan(self.fallback, name, x, w, output) == found for w in weights):
                    continue
                if not self._room(output, len(weights)):
                    continue
                transport = None
                try:
                    try:
                        transport = LoadTransport(name, plan, x, weights, output, self.deadline)
                    except (torch.cuda.OutOfMemoryError, TimeoutError):
                        continue
                    accepted = self._select(name, transport, x, weights, output, generator)
                    if (accepted and time.monotonic() < self.deadline
                            and effective_plan(self.fallback, name, x, weights[0], output) == (owner, plan)
                            and all(transport.eligible(plan, x, w, output) for w in weights)):
                        self.transports[name] = transport
                        plan.load_transport = transport
                    self._log(name, "selected TMA loads" if name in self.transports else "accepted pointer loads")
                finally:
                    torch.cuda.synchronize(self.device)
                    transport = None
        except BaseException:
            self.close()
            raise
        finally:
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()

    @property
    def extra_bytes(self):
        return sum(t.extra_bytes for t in self.transports.values())

    def close(self):
        torch.cuda.synchronize(self.device)
        for transport in self.transports.values():
            if getattr(transport.plan, "load_transport", None) is transport:
                transport.plan.load_transport = None
        self.transports.clear()

    def _log(self, name, message):
        print(f"[tma-decode] B={self.batch} {name}: {message}", file=sys.stderr, flush=True)

    def _room(self, output, count):
        free, total = torch.cuda.mem_get_info(self.device)
        # References, finite/bitwise temporaries and optional residual states.
        required = output.numel() * 64 + (count + 1) * 512 + 65536
        return free - required >= total // 4

    @contextmanager
    def _using(self, transport, enabled):
        plan = transport.plan
        previous = getattr(plan, "load_transport", None)
        plan.load_transport = transport if enabled else None
        try:
            yield
        finally:
            plan.load_transport = previous

    def _merge_binding(self, name, transport, weights):
        # Composition with retained #41 is explicit; no wrapper hides its plan.
        merge = getattr(self.engine, "merge_norm", None)
        binding = merge.bindings.get(name) if merge is not None else None
        if binding is None:
            return None
        if (binding.plan is not transport.plan
                or binding.x is not transport.x.owner.tensor
                or binding.branch is not transport.output.tensor
                or binding.hidden is not self.engine.hidden
                or binding.normalized is not self.engine.normalized
                or any(not binding.valid(self.fallback, name, i, binding.x, w,
                                          binding.hidden, binding.gains[i], binding.normalized)
                       for i, w in enumerate(weights))):
            raise ValueError("accepted merge/norm binding no longer matches decode")
        return merge, binding

    def _launch(self, name, transport, x, weights, output, index, fused, state):
        if fused is None:
            self.fallback.run(name, index, x, weights[index], output)
        else:
            merge, binding = fused
            merge._fused(binding, x, weights[index], state[0], binding.gains[index], state[1])

    def _select(self, name, transport, x, weights, output, generator):
        baseline_graph = candidate_graph = None
        seed = reference = candidate = None
        try:
            if time.monotonic() >= self.deadline or not self._room(output, len(weights)):
                return False
            fused = self._merge_binding(name, transport, weights)
            # Six original layer weights; the tied head has one original weight.
            indices = [round(i * (len(weights) - 1) / 5) for i in range(6)] if len(weights) >= 6 else list(range(len(weights)))
            try:
                reference = (torch.empty_like(output),) if fused is None else (
                    torch.empty_like(fused[1].hidden), torch.empty_like(fused[1].normalized))
                candidate = None if fused is None else tuple(torch.empty_like(t) for t in reference)
                seed = None if fused is None else torch.empty_like(reference[0])
            except torch.cuda.OutOfMemoryError:
                return False
            for index in indices:
                for scale in (0.1, 1.0, 10.0):
                    if time.monotonic() >= self.deadline or not self._room(output, len(weights)):
                        return False
                    x.normal_(generator=generator).mul_(scale)
                    if fused is not None:
                        seed.normal_(generator=generator).mul_(scale)
                        reference[0].copy_(seed)
                        candidate[0].copy_(seed)
                    # Baseline failures never enter an own-kernel catch.
                    with self._using(transport, False):
                        self._launch(name, transport, x, weights, output, index, fused, reference)
                    if fused is None:
                        reference[0].copy_(output)
                    if time.monotonic() >= self.deadline:
                        return False
                    with self._using(transport, True):
                        self._launch(name, transport, x, weights, output, index, fused, candidate)
                    observed = (output,) if fused is None else candidate
                    if (time.monotonic() >= self.deadline
                            or not all(torch.isfinite(t).all().item() for t in reference)
                            or not all(torch.equal(a.view(torch.int16), b.view(torch.int16))
                                       for a, b in zip(reference, observed))):
                        return False
            baseline_graph = self._graph(name, transport, x, weights, output, indices, fused, seed, reference, False)
            if baseline_graph is None:
                return False
            candidate_graph = self._graph(name, transport, x, weights, output, indices, fused, seed, candidate, True)
            if candidate_graph is None:
                return False
            old, new = [], []
            for graph, samples in ((baseline_graph, old), (candidate_graph, new),
                                   (candidate_graph, new), (baseline_graph, old)):
                value = self._time(graph, len(indices))
                if value is None or not math.isfinite(value) or value <= 0:
                    return False
                samples.append(value)
            self._log(name, f"accepted {min(old)*1000:.2f} us, TMA {max(new)*1000:.2f} us")
            return time.monotonic() < self.deadline and max(new) < min(old) * .95
        except TmaRejected as error:
            self._log(name, "accepted pointer loads: " + str(error))
            return False
        finally:
            # Graphs and all trial owners remain alive until every exit drains.
            torch.cuda.synchronize(self.device)
            baseline_graph = candidate_graph = seed = reference = candidate = None

    def _graph(self, name, transport, x, weights, output, indices, fused, seed, state, enabled):
        if time.monotonic() >= self.deadline:
            return None
        current = torch.cuda.current_stream(self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(current)
        def launch():
            with self._using(transport, enabled):
                for index in indices:
                    if fused is not None:
                        state[0].copy_(seed)
                    self._launch(name, transport, x, weights, output, index, fused, state)
        try:
            with torch.cuda.stream(stream):
                launch()
        finally:
            try:
                current.wait_stream(stream)
            finally:
                stream.synchronize()
        if time.monotonic() >= self.deadline:
            return None
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=stream):
                launch()
        finally:
            try:
                current.wait_stream(stream)
            finally:
                stream.synchronize()
        if time.monotonic() >= self.deadline:
            return None
        try:
            graph.replay()
        finally:
            torch.cuda.synchronize(self.device)
        return graph if time.monotonic() < self.deadline else None

    def _time(self, graph, count):
        samples = []
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(3):
            if time.monotonic() >= self.deadline:
                return None
            start.record()
            for _ in range(32):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / (32 * count))
        return statistics.median(samples) if time.monotonic() < self.deadline else None
