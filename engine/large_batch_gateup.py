"""Optional full-K BF16 gate/up decode plan for actual batches 33 through 256.

The persistent dense device function and original prefill policy are unchanged.
Only complete numerical and actual-parent timing winners replace decode dispatch.
"""
import math
import statistics
import time
from types import SimpleNamespace

import torch
import triton
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources
from dense_prefill import _persistent_dense
from wide_gemv import WideGemvLayout


class Expired(Exception):
    pass


def _live(deadline):
    if time.monotonic() >= deadline:
        raise Expired()


def guard_gateup(engine):
    if getattr(engine, '_gateup_pending', None) is not None:
        raise RuntimeError('large-batch gate/up owner has not drained')


def _eligible(rows, device, x, weight, output):
    return (33 <= rows <= 256 and tuple(x.shape) == (rows, 2560) and
            tuple(weight.shape) == (19456, 2560) and
            tuple(output.shape) == (rows, 19456) and
            all(t.is_cuda and t.dtype == torch.bfloat16 and
                t.device == device and t.is_contiguous()
                for t in (x, weight, output)) and
            len({t.untyped_storage()._cdata for t in (x, weight, output)}) == 3)


class DecodeGateupPlan:
    """Narrow host plan; never widens the original DensePlan constructor."""
    def __init__(self, x, weight, output, sms):
        if (len(x.shape) != 2 or type(sms) is not int or sms <= 0):
            raise ValueError('unsupported gate/up shape or SM count')
        self.rows, self.k = x.shape
        self.n, self.device = 19456, x.device
        if (not 33 <= self.rows <= 256 or self.k != 2560 or
                not self.eligible(x, weight, output)):
            raise ValueError('unsupported gate/up shape, dtype, layout or alias')
        self.sms = sms
        self.bm, self.bn, self.bk, self.stages = (64 if self.rows <= 64 else 128), 128, 64, 4

    def eligible(self, x, weight, output):
        return _eligible(self.rows, self.device, x, weight, output)

    def __call__(self, x, weight, output):
        count = triton.cdiv(self.rows, self.bm) * triton.cdiv(self.n, self.bn)
        _persistent_dense[(min(self.sms, count),)](
            x, weight, output, M=self.rows, N=self.n, K=self.k,
            SMS=self.sms, BM=self.bm, BN=self.bn, BK=self.bk, GROUP=8,
            num_warps=4, num_stages=self.stages, enable_fp_fusion=True,
        )


class GateupLayout:
    def __init__(self, engine, native, deadline):
        self.native = native
        self.device = engine.normalized.device
        self.deadline = deadline - 3.0
        self.x, self.output = engine.normalized, engine.gateup
        self.weights = [(i, engine.packed[i][1]) for i in (0, 7, 14, 21, 28, 35)]
        self.trials, self.plan = [], None
        self.drain_error = None

    def _room(self, output, reference_live=False):
        return WideGemvLayout._room(self, output, reference_live)

    def _budget(self, pool_ms):
        _live(self.deadline)
        return (math.isfinite(pool_ms) and pool_ms > 0 and
                time.monotonic() + 64.0 * pool_ms / 1000.0 < self.deadline)

    def _drain(self):
        error = None
        for trial in self.trials:
            stream = getattr(trial, 'stream', None)
            if stream is not None:
                try:
                    torch.cuda.current_stream(self.device).wait_stream(stream)
                except BaseException as exc:
                    if error is None:
                        error = exc
        try:
            torch.cuda.synchronize(self.device)
        except BaseException as exc:
            if error is None:
                error = exc
        if error is not None and self.drain_error is None:
            self.drain_error = error
        # A later nominally successful sync must not clear an earlier failure.
        if self.drain_error is not None:
            raise self.drain_error

    def _launch(self, plan, output, checked=True):
        for layer, weight in self.weights:
            if checked:
                _live(self.deadline)
            if plan is None:
                self.native.run('gateup', layer, self.x, weight, output)
            else:
                plan(self.x, weight, output)

    def _events(self):
        _live(self.deadline)
        trial = SimpleNamespace(start=None, end=None)
        self.trials.append(trial)
        trial.start = torch.cuda.Event(enable_timing=True)
        trial.end = torch.cuda.Event(enable_timing=True)
        return trial

    def _sample(self, launch, events, repeats=1):
        _live(self.deadline)
        events.start.record()
        for _ in range(repeats):
            _live(self.deadline)
            launch()
        events.end.record()
        events.end.synchronize()
        _live(self.deadline)
        elapsed = events.start.elapsed_time(events.end)
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise Expired()
        return elapsed

    def _pilot(self, launch):
        return self._sample(launch, self._events())

    def _graph(self, plan):
        _live(self.deadline)
        trial = SimpleNamespace(stream=None, graph=None)
        self.trials.append(trial)
        trial.stream = torch.cuda.Stream(device=self.device)
        current = torch.cuda.current_stream(self.device)
        trial.stream.wait_stream(current)
        try:
            with torch.cuda.stream(trial.stream):
                self._launch(plan, self.output)
        finally:
            current.wait_stream(trial.stream)
        torch.cuda.synchronize(self.device)
        _live(self.deadline)
        trial.graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(trial.graph, stream=trial.stream):
                # The exact six launches were warmed. Do not throw a deadline
                # exception halfway through recording the fixed graph.
                self._launch(plan, self.output, checked=False)
        finally:
            current.wait_stream(trial.stream)
        _live(self.deadline)
        trial.graph.replay()
        torch.cuda.synchronize(self.device)
        _live(self.deadline)
        return trial.graph

    def _time(self, graph):
        events = self._events()
        elapsed = self._sample(graph.replay, events)
        repeats = max(1, min(4, int(8.0 / max(elapsed, 0.001))))
        samples = [self._sample(graph.replay, events, repeats) / (repeats * 6)
                   for _ in range(3)]
        return statistics.median(samples)

    def _check(self, plan, reference, generator):
        for scale in (1.0, 0.1, 10.0):
            for layer, weight in self.weights:
                _live(self.deadline)
                self.x.normal_(generator=generator).mul_(scale)
                _live(self.deadline)
                self.native.run('gateup', layer, self.x, weight, reference)
                _live(self.deadline)
                plan(self.x, weight, self.output)
                _live(self.deadline)
                if not self._room(self.output, reference_live=True):
                    return False
                _live(self.deadline)
                if not torch.isfinite(reference).all().item():
                    return False
                _live(self.deadline)
                if not torch.isfinite(self.output).all().item():
                    return False
                _live(self.deadline)
                if not torch.allclose(self.output, reference, rtol=0.01, atol=0.01):
                    return False
                _live(self.deadline)
        return True

    def select(self):
        _live(self.deadline)
        if torch.cuda.get_device_capability(self.device)[0] != 9:
            return
        _live(self.deadline)
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        _live(self.deadline)
        if type(sms) is not int or sms <= 0:
            return
        plan = DecodeGateupPlan(self.x, self.weights[0][1], self.output, sms)
        trial = SimpleNamespace(reference=None, plan=plan, generator=None)
        self.trials.append(trial)
        if not all(plan.eligible(self.x, weight, self.output) for _, weight in self.weights):
            return
        if not self._room(self.output):
            return
        _live(self.deadline)
        trial.reference = torch.empty_like(self.output)
        _live(self.deadline)
        trial.generator = torch.Generator(device=self.device).manual_seed(94271)
        self.x.normal_(generator=trial.generator)
        self._launch(None, self.output)
        torch.cuda.synchronize(self.device)
        _live(self.deadline)
        native_ms = self._pilot(lambda: self._launch(None, self.output))
        if not self._budget(native_ms):
            return
        # Complete compilation/real warmup before events measure a candidate.
        self._launch(plan, self.output)
        torch.cuda.synchronize(self.device)
        _live(self.deadline)
        custom_ms = self._pilot(lambda: self._launch(plan, self.output))
        if not self._budget(max(native_ms, custom_ms)):
            return
        if not self._check(plan, trial.reference, trial.generator):
            return
        native_graph, graph = self._graph(None), self._graph(plan)
        natives, customs = [], []
        for timed, values in ((native_graph, natives), (graph, customs),
                              (graph, customs), (native_graph, natives)):
            values.append(self._time(timed))
        _live(self.deadline)
        if (all(math.isfinite(value) and value > 0 for value in natives + customs) and
                max(customs) < 0.95 * min(natives)):
            self.plan = plan

    def run(self, name, layer, x, weight, output):
        if name == 'gateup' and self.plan is not None and self.plan.eligible(x, weight, output):
            self.plan(x, weight, output)
        else:
            self.native.run(name, layer, x, weight, output)


def extend_gateup(engine, native, tuner_deadline):
    guard_gateup(engine)
    now = time.monotonic()
    deadline = min(tuner_deadline, now + 12.0)
    if (not 33 <= engine.batch <= 256 or engine.capacity - engine.prompt <= 1 or
            now >= deadline - 3.0):
        return native
    if len(engine.layers) != 36 or len(engine.packed) != 36:
        return native
    if not all(_eligible(engine.batch, engine.normalized.device,
                         engine.normalized, engine.packed[i][1], engine.gateup)
               for i in (0, 7, 14, 21, 28, 35)):
        return native
    owner = GateupLayout(engine, native, deadline)
    engine._gateup_pending = owner
    try:
        try:
            owner.select()
        except (Expired, CompilationError, OutOfResources, torch.cuda.OutOfMemoryError):
            pass
    finally:
        # A failed drain leaves every object and the engine registry intact.
        owner._drain()
        owner.trials.clear()
        owner.x, owner.output, owner.weights = None, None, []
        engine._gateup_pending = None
    return owner if owner.plan is not None else native


def retire_gateup(engine):
    """Before any old graph/buffer replacement, retire only our own wrapper."""
    guard_gateup(engine)
    owner = getattr(engine, 'native_layout', None)
    if isinstance(owner, GateupLayout):
        engine._gateup_pending = owner
        owner._drain()
        engine.native_layout = owner.native
        engine._gateup_pending = None
