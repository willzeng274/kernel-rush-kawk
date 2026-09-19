"""Same-kernel, exact-BF16 Hopper inline compression experiment.

The accepted #32 dispatcher is the reference. Only exact known Triton plan
classes can use a foreign pointer. Native cuBLAS families remain native. All
selection, copying and validation happens before engine graph capture.
"""
import math
import statistics
import time

import torch
import triton
import triton.language as tl
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources

from ilc_memory import Allocator, ILCUnavailable, Pointer
from native_layout import NativeLayout
from wide_gemv import WideGemvLayout, WidePlan
from persistent_vector import PersistentVectorLayout, PersistentPlan
from hopper_gemm import HopperGemmLayout, HopperPlan
from hopper_tiles import HopperTilesLayout, HopperTilesPlan


class BaselineFailure(RuntimeError):
    """An inherited baseline failed; candidate fallback must not hide it."""


@triton.jit
def _copy_bits(SOURCE, DESTINATION, WORDS, BLOCK: tl.constexpr):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    source = SOURCE.to(tl.pointer_type(tl.uint32))
    destination = DESTINATION.to(tl.pointer_type(tl.uint32))
    value = tl.load(source + p, p < WORDS, 0)
    tl.store(destination + p, value, p < WORDS)


@triton.jit
def _check_bits(SOURCE, COPY, ERROR, WORDS, BLOCK: tl.constexpr):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    source = SOURCE.to(tl.pointer_type(tl.uint32))
    copy = COPY.to(tl.pointer_type(tl.uint32))
    original = tl.load(source + p, p < WORDS, 0)
    alternate = tl.load(copy + p, p < WORDS, 0)
    bad = tl.sum(((original != alternate) & (p < WORDS)).to(tl.int32), 0)
    tl.atomic_or(ERROR, bad != 0)


def effective_plan(layout, name, x, weight, output):
    """Mirror each reviewed #32 run guard, without guessing unknown dispatch."""
    while type(layout) is not NativeLayout:
        kind = type(layout)
        if kind not in (PersistentVectorLayout, WideGemvLayout,
                        HopperGemmLayout, HopperTilesLayout):
            return None
        plan = layout.plans.get(name)
        if kind in (PersistentVectorLayout, WideGemvLayout):
            eligible = x.shape[0] == 1
        else:
            eligible = x.shape[0] == layout.batch
        if eligible and plan is not None:
            if kind is WideGemvLayout:
                if type(plan) is not WidePlan:
                    return None
            else:
                expected = {PersistentVectorLayout: PersistentPlan,
                            HopperGemmLayout: HopperPlan,
                            HopperTilesLayout: HopperTilesPlan}[kind]
                if type(plan) is not expected:
                    return None
                if (tuple(x.shape) != (plan.batch, plan.k)
                        or tuple(weight.shape) != (plan.n, plan.k)
                        or tuple(output.shape) != (plan.batch, plan.n)):
                    layout = layout.native
                    continue
            if (weight.dtype != torch.bfloat16 or not weight.is_contiguous()
                    or weight.device != x.device or x.dtype != torch.bfloat16
                    or output.dtype != torch.bfloat16 or x.device != output.device
                    or not x.is_cuda or not weight.is_cuda or not output.is_cuda
                    or x.stride(1) != 1 or output.stride(1) != 1
                    or tuple(weight.shape) != (output.shape[1], x.shape[1])
                    or plan.k != x.shape[1]):
                return None
            return plan
        layout = layout.native
    return None


def material_win(times):
    """Slow ON must beat fast real baseline by 20% and matched OFF by 15%."""
    if (set(times) != {"original", "off", "on"}
            or any(len(values) != 2 for values in times.values())
            or not all(math.isfinite(v) and v > 0
                       for values in times.values() for v in values)):
        return False
    return (max(times["on"]) <= 0.80 * min(times["original"])
            and max(times["on"]) <= 0.85 * min(times["off"]))


class ILCWeights:
    # No __del__ or automatic freeing: accepted owners are pinned until process
    # exit, including graphs retained by old shape-specific engine objects.
    def __init__(self, engine, native, absolute_deadline):
        self.native, self.batch = native, engine.batch
        self.device = engine.normalized.device
        self.families, self.allocator = {}, None
        self._quarantine_graphs = []
        self._quarantine_streams = []
        self.started = time.monotonic()
        self.attempted = []
        self._previous = None
        self.supported_bytes = self.selected_bytes = self.total_bytes = 0
        self.deadline = min(absolute_deadline, time.monotonic() + 30.0)
        groups = (
            ("gateup", engine.normalized, engine.gateup,
             [pair[1] for pair in engine.packed]),
            ("down", engine.intermediate, engine.branch,
             [layer.mlp.down_proj.weight for layer in engine.layers]),
            ("qkv", engine.normalized, engine.qkv,
             [pair[0] for pair in engine.packed]),
            ("output", engine.attention, engine.branch,
             [layer.self_attn.o_proj.weight for layer in engine.layers]),
            ("head", engine.normalized, engine.logits,
             [engine.model.lm_head.weight]),
        )
        eligible = []
        for name, x, output, weights in groups:
            size = sum(w.numel() * w.element_size() for w in weights)
            self.total_bytes += size
            plan = effective_plan(native, name, x, weights[0], output)
            if plan is None:
                self._log(name, f"native/unsupported dispatch; logical bytes={size}")
            else:
                self.supported_bytes += size
                self._log(name, f"eligible accepted plan={plan.name}; logical bytes={size}")
                eligible.append((name, x, output, weights, plan, size))
        if not eligible or time.monotonic() >= self.deadline:
            self._report("ineligible or deadline")
            return
        properties = torch.cuda.get_device_properties(self.device)
        if (properties.major, properties.minor) != (9, 0):
            self._report("requires Hopper SM90")
            return
        # Release only cached Torch blocks, never live weights/KV/native plans.
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        try:
            self.allocator = Allocator(
                self.device.index if self.device.index is not None else torch.cuda.current_device(),
                lambda: torch.cuda.synchronize(self.device),
                lambda: torch.cuda.mem_get_info(self.device))
        except ILCUnavailable as error:
            self._report(str(error))
            return
        self._log("device", f"{properties.name}; SMs={properties.multi_processor_count}; "
                  f"physical={properties.total_memory}; support={self.allocator.support}; "
                  f"granularity={self.allocator.granularity}")
        generator = torch.Generator(device=self.device)
        generator.manual_seed(76913)
        for name, x, output, weights, plan, size in eligible:
            if time.monotonic() >= self.deadline:
                self._log(name, "skipped: independent ILC deadline")
                continue
            self.attempted.append(name)
            # One completed family remains installed even if a later family
            # exhausts the cooperative deadline. Never publish partial copies.
            try:
                selected = self._select(name, x, output, weights, plan, generator)
            except (ILCUnavailable, torch.cuda.OutOfMemoryError,
                    CompilationError, OutOfResources) as error:
                self._log(name, f"accepted baseline: {type(error).__name__}: {error}")
                selected = None
                torch.cuda.empty_cache()
            if selected is not None:
                self.families[name] = (plan, selected,
                                       tuple(w.data_ptr() for w in weights))
                self.selected_bytes += size
                self._log(name, f"selected exact BF16 ILC; full-family logical bytes={size}")
        self._report("selection complete")

    @property
    def weights(self):
        return self.native.weights

    @property
    def extra_bytes(self):
        custom = 0 if self.allocator is None else self.allocator.live_bytes
        previous = 0 if self._previous is None else self._previous._custom_bytes()
        return self.native.extra_bytes + custom + previous

    def _custom_bytes(self):
        own = 0 if self.allocator is None else self.allocator.live_bytes
        return own + (0 if self._previous is None else self._previous._custom_bytes())

    def _log(self, name, message):
        print(f"[ilc-weights] B={self.batch} {name}: {message}", flush=True)

    def _report(self, reason):
        total = max(1, self.total_bytes)
        native_share = 1 - self.supported_bytes / total
        selected_share = self.selected_bytes / total
        live = 0 if self.allocator is None else self.allocator.live_bytes
        peak = 0 if self.allocator is None else self.allocator.peak_bytes
        elapsed = time.monotonic() - self.started
        remaining = max(0.0, self.deadline - time.monotonic())
        self._log("coverage", f"{reason}; elapsed={elapsed:.3f}s; left={remaining:.3f}s; "
                  f"attempted={self.attempted}; unsupported/native weight share={native_share:.4%}; "
                  f"selected weight share={selected_share:.4%}; custom live={live}; "
                  f"custom peak={peak}; original weights retained")

    def _copy(self, weight, compressed, error_flag, scratch):
        if time.monotonic() >= self.deadline:
            raise ILCUnavailable("copy deadline")
        if weight.numel() % 2 or weight.dtype != torch.bfloat16 or not weight.is_contiguous():
            raise ILCUnavailable("copy requires even contiguous BF16 elements")
        allocation = self.allocator.allocate(weight.numel() * 2, compressed, scratch)
        pointer = Pointer(allocation, weight.shape, weight.dtype, weight.device)
        try:
            words = weight.numel() // 2
            grid = (triton.cdiv(words, 4096),)
            _copy_bits[grid](weight, pointer, words, BLOCK=4096, num_warps=4)
            error_flag.zero_()
            _check_bits[grid](weight, pointer, error_flag, words, BLOCK=4096, num_warps=4)
            # Covers every original bit, including signed zero/NaN payloads.
            if error_flag.item() != 0:
                raise ILCUnavailable("full weight bit verification failed")
        except BaseException:
            if not self.allocator.quarantined:
                self.allocator.retire([allocation])
            raise
        return pointer

    def _select(self, name, x, output, weights, plan, generator):
        control, compressed = [], []
        published = False
        # Two output tensors plus integer-comparison temporaries and bit flag.
        scratch = output.numel() * 16 + 65536
        pool = weights[:6] if name != "head" else weights
        required = sum(self.allocator.rounded_size(w.numel() * 2, mode)
                       for w in pool for mode in (False, True))
        if not self.allocator.has_room(required, scratch):
            raise ILCUnavailable("control/ON pool exceeds physical reserve")
        reference = torch.empty_like(output)
        flag = torch.zeros((), dtype=torch.int32, device=self.device)
        try:
            for weight in pool:
                control.append(self._copy(weight, False, flag, scratch))
                compressed.append(self._copy(weight, True, flag, scratch))
            self._log(name, "VMM grants: OFF=0, ON=1 verified for every pool allocation")
            # Same actual accepted dispatcher and selected plan. Test all six
            # real matrices, then both ends at additional activation scales.
            probes = [(i, 1.0) for i in range(len(pool))]
            probes += [(0, 0.1), (len(pool) - 1, 10.0)]
            for index, scale in probes:
                if time.monotonic() >= self.deadline:
                    raise ILCUnavailable("numerical deadline")
                x.normal_(generator=generator).mul_(scale)
                self._baseline_run(name, index, x, pool[index], reference)
                if not torch.isfinite(reference).all().item():
                    raise ILCUnavailable("nonfinite accepted baseline projection")
                for pointer in (control[index], compressed[index]):
                    plan(x, pointer, output)
                    if (not torch.isfinite(output).all().item()
                            or not torch.equal(output.view(torch.int16), reference.view(torch.int16))):
                        raise ILCUnavailable("projection bits differ from accepted baseline")
            x.normal_(generator=generator)
            times = self._measure(name, plan, x, output, pool, control, compressed)
            self._log(name, "same selected kernel " + plan.name + "; " +
                      "; ".join(f"{arm}={','.join(f'{v * 1000:.2f}' for v in values)} us"
                                for arm, values in times.items()))
            if not material_win(times):
                return None
            # Benchmark graphs are destroyed on return from _measure. OFF is
            # retired before full-family extension to keep the live peak small.
            self.allocator.retire([p.owner for p in control])
            control.clear()
            for weight in weights[len(compressed):]:
                compressed.append(self._copy(weight, True, flag, scratch))
            self.allocator.synchronize()
            if not self.allocator.has_room(0, scratch):
                raise ILCUnavailable("full family physical reserve")
            published = True
            return compressed
        finally:
            retire = [p.owner for p in control]
            if not published:
                retire += [p.owner for p in compressed]
            if retire and not self.allocator.quarantined:
                self.allocator.retire(retire)

    def _baseline_run(self, name, index, x, weight, output):
        try:
            self.native.run(name, index, x, weight, output)
        except (torch.cuda.OutOfMemoryError, CompilationError, OutOfResources) as error:
            raise BaselineFailure("accepted baseline projection failed") from error

    def _graph(self, launch):
        if time.monotonic() >= self.deadline:
            raise ILCUnavailable("graph deadline")
        current = torch.cuda.current_stream(self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(current)
        graph = None
        try:
            # Join the side stream even when its first warmup launch fails.
            try:
                with torch.cuda.stream(stream):
                    launch()
            finally:
                current.wait_stream(stream)
            self.allocator.synchronize()
            if time.monotonic() >= self.deadline:
                raise ILCUnavailable("graph warmup deadline")
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph, stream=stream):
                    launch()
            finally:
                current.wait_stream(stream)
            graph.replay()
            self.allocator.synchronize()
        except BaseException:
            try:
                self.allocator.synchronize()
            except BaseException:
                self._quarantine_streams.append(stream)
                if graph is not None:
                    self._quarantine_graphs.append(graph)
                raise
            if graph is not None:
                try:
                    graph.reset()
                except BaseException:
                    self.allocator.quarantined = True
                    self._quarantine_graphs.append(graph)
                    raise
            raise
        return graph

    def _measure(self, name, plan, x, output, pool, control, compressed):
        graphs = {}
        times = {"original": [], "off": [], "on": []}
        try:
            def original():
                for index, weight in enumerate(pool):
                    self._baseline_run(name, index, x, weight, output)
            try:
                graphs["original"] = self._graph(original)
            except (torch.cuda.OutOfMemoryError, CompilationError, OutOfResources) as error:
                raise BaselineFailure("accepted baseline graph failed") from error
            for arm, pointers in (("off", control), ("on", compressed)):
                def alternate(pointers=pointers):
                    for pointer in pointers:
                        plan(x, pointer, output)
                graphs[arm] = self._graph(alternate)
            # Symmetric order balances drift. Whole operation includes any
            # inherited split-K merge. Real six-weight pools exceed L2; head is
            # always the full 151936 x 2560 matrix, never a narrow slice.
            for arm in ("original", "off", "on", "on", "off", "original"):
                if time.monotonic() >= self.deadline:
                    return times  # incomplete measurements cannot win
                samples = []
                for _ in range(3):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(8):
                        graphs[arm].replay()
                    end.record()
                    try:
                        end.synchronize()
                    except BaseException:
                        self.allocator.quarantined = True
                        raise
                    samples.append(start.elapsed_time(end) / (8 * len(pool)))
                times[arm].append(statistics.median(samples))
            return times
        finally:
            try:
                self.allocator.synchronize()
            except BaseException:
                self._quarantine_graphs.extend(graphs.values())
                raise
            try:
                for graph in graphs.values():
                    graph.reset()
            except BaseException:
                self.allocator.quarantined = True
                self._quarantine_graphs.extend(graphs.values())
                raise
            graphs.clear()

    def run(self, name, layer, x, weight, output):
        selected = self.families.get(name) if x.shape[0] == self.batch else None
        if selected is not None:
            plan, pointers, original_addresses = selected
            if (0 <= layer < len(pointers) and weight.data_ptr() == original_addresses[layer]
                    and tuple(weight.shape) == pointers[layer].shape
                    and tuple(x.shape) == (self.batch, plan.k)
                    and tuple(output.shape) == (self.batch, weight.shape[0])
                    and x.device == self.device and output.device == self.device
                    and x.dtype == torch.bfloat16 and output.dtype == torch.bfloat16
                    and x.stride(1) == 1 and output.stride(1) == 1):
                plan(x, pointers[layer], output)
                return
        self.native.run(name, layer, x, weight, output)
