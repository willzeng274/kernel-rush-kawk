"""Optional large-row admission after the unchanged DensePrefill search.

No new device math. Only families excluded by the original pool FLOP limit
may try the original DensePlan within the original search's remaining time.
"""
import math
import statistics
import time
from types import SimpleNamespace

import torch
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources

from dense_prefill import DensePlan, DensePrefill
from wide_gemv import WideGemvLayout


class Expired(Exception):
    pass


def guard_postpass(engine):
    if getattr(engine, "_prefill_postpass_pending", None) is not None:
        raise RuntimeError("prefill post-pass owner has not drained")


def _live(deadline):
    if time.monotonic() >= deadline:
        raise Expired()


class _Postpass:
    def __init__(self, engine, native, groups, deadline):
        self.native, self.plans = native, {}
        self.device = engine.prefill_normalized.device
        self.groups, self.trials = groups, []
        self.drain_error = None
        # This is optional tuning time, not the original model-capture budget.
        self.deadline = deadline - 3.0
        self.sms = None

    def _room(self, output, reference_live=False):
        return WideGemvLayout._room(self, output, reference_live)

    def _budget(self, pool_ms):
        _live(self.deadline)
        return (math.isfinite(pool_ms) and pool_ms > 0
                and time.monotonic() + 64.0 * pool_ms / 1000.0 < self.deadline)

    def _drain(self):
        # Keep every owner live even if a join fails; still attempt device drain.
        error = None
        for trial in self.trials:
            stream = getattr(trial, "stream", None)
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
        if self.drain_error is not None:
            raise self.drain_error

    def _launch(self, name, plan, x, weights, output, checked=True):
        for index, weight in weights:
            if checked:
                _live(self.deadline)
            if plan is None:
                self.native.run(name, index, x, weight, output)
            else:
                plan(x, weight, output)

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

    def _graph(self, name, plan, x, weights, output):
        _live(self.deadline)
        trial = SimpleNamespace(stream=None, graph=None)
        self.trials.append(trial)
        trial.stream = torch.cuda.Stream(device=self.device)
        current = torch.cuda.current_stream(self.device)
        trial.stream.wait_stream(current)
        try:
            with torch.cuda.stream(trial.stream):
                self._launch(name, plan, x, weights, output)
        finally:
            current.wait_stream(trial.stream)
        torch.cuda.synchronize(self.device)
        _live(self.deadline)
        trial.graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(trial.graph, stream=trial.stream):
                # All six exact launches were warmed above. Avoid an exception
                # halfway through recording a fixed six-operation graph.
                self._launch(name, plan, x, weights, output, checked=False)
        finally:
            current.wait_stream(trial.stream)
        _live(self.deadline)
        trial.graph.replay()
        torch.cuda.synchronize(self.device)
        _live(self.deadline)
        return trial.graph

    def _time(self, graph, count):
        events = self._events()
        elapsed = self._sample(graph.replay, events)
        repeats = max(1, min(4, int(8.0 / max(elapsed, 0.001))))
        values = [self._sample(graph.replay, events, repeats) / (repeats * count)
                  for _ in range(3)]
        return statistics.median(values)

    def _check(self, name, plan, x, weights, output, reference, generator):
        for scale in (1.0, 0.1, 10.0):
            for index, weight in weights:
                _live(self.deadline)
                x.normal_(generator=generator).mul_(scale)
                _live(self.deadline)
                self.native.run(name, index, x, weight, reference)
                _live(self.deadline)
                plan(x, weight, output)
                _live(self.deadline)
                if not self._room(output, reference_live=True):
                    return False
                _live(self.deadline)
                if not torch.isfinite(reference).all().item():
                    return False
                _live(self.deadline)
                if not torch.isfinite(output).all().item():
                    return False
                _live(self.deadline)
                if not torch.allclose(output, reference, rtol=0.01, atol=0.01):
                    return False
                _live(self.deadline)
        return True

    def _family(self, name, x, output, weights, generator):
        _live(self.deadline)
        torch.cuda.empty_cache()
        _live(self.deadline)
        if not self._room(output):
            return
        _live(self.deadline)
        trial = SimpleNamespace(x=x, output=output, weights=weights,
                                reference=None, plans=[])
        self.trials.append(trial)
        trial.reference = torch.empty_like(output)
        _live(self.deadline)
        x.normal_(generator=generator)
        self._launch(name, None, x, weights, output)
        torch.cuda.synchronize(self.device)
        _live(self.deadline)
        native_ms = self._pilot(lambda: self._launch(name, None, x, weights, output))
        if not self._budget(native_ms):
            return
        native_graph = self._graph(name, None, x, weights, output)
        winner_ms = float("inf")
        for config in DensePlan.CONFIGS:
            _live(self.deadline)
            if not self._budget(native_ms):
                break
            try:
                plan = DensePlan(x, weights[0][1], output, config, self.sms)
                trial.plans.append(plan)
                if not all(plan.eligible(x, weight, output) for _, weight in weights):
                    continue
                # Compile before recording the pool pilot: an event spanning a
                # cold compiler wait would mislabel startup as steady GPU cost.
                self._launch(name, plan, x, weights, output)
                torch.cuda.synchronize(self.device)
                _live(self.deadline)
                custom_ms = self._pilot(lambda: self._launch(name, plan, x, weights, output))
                # The warm compile above is still charged to the wall deadline.
                if not self._budget(max(native_ms, custom_ms)):
                    continue
                if not self._check(name, plan, x, weights, output, trial.reference, generator):
                    continue
                graph = self._graph(name, plan, x, weights, output)
                natives, customs = [], []
                for timed, values in ((native_graph, natives), (graph, customs),
                                      (graph, customs), (native_graph, natives)):
                    values.append(self._time(timed, len(weights)))
                _live(self.deadline)
                if max(customs) < min(natives) * 0.95 and max(customs) < winner_ms:
                    self.plans[name], winner_ms = plan, max(customs)
            except (CompilationError, OutOfResources, torch.cuda.OutOfMemoryError):
                self._drain()
                _live(self.deadline)
                continue

    def select(self):
        _live(self.deadline)
        if torch.cuda.get_device_capability(self.device)[0] != 9:
            return
        _live(self.deadline)
        self.sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        _live(self.deadline)
        generator = torch.Generator(device=self.device).manual_seed(93617)
        for group in self.groups:
            _live(self.deadline)
            try:
                self._family(*group, generator)
            except torch.cuda.OutOfMemoryError:
                pass
            finally:
                self._drain()
                self.trials.clear()

    def run(self, name, layer, x, weight, output):
        plan = self.plans.get(name)
        if plan is not None and plan.eligible(x, weight, output):
            plan(x, weight, output)
        else:
            self.native.run(name, layer, x, weight, output)


def extend_prefill(engine, native, tuner_deadline, dense_started):
    """Return exact original dispatch unless optional complete winners exist."""
    guard_postpass(engine)
    rows = engine.prefill_rows
    deadline = min(tuner_deadline, dense_started + 45.0)
    if rows < 256 or rows > 65536 or time.monotonic() >= deadline - 3.0:
        return native
    if len(engine.layers) < 6:
        return native
    indices = [round(i * (len(engine.layers) - 1) / 5) for i in range(6)]
    groups = (
        ("gateup", engine.prefill_normalized, engine.prefill_gateup,
         [(i, engine.packed[i][1]) for i in indices]),
        ("down", engine.prefill_intermediate, engine.prefill_branch,
         [(i, engine.layers[i].mlp.down_proj.weight) for i in indices]),
        ("qkv", engine.prefill_normalized, engine.prefill_qkv,
         [(i, engine.packed[i][0]) for i in indices]),
        ("output", engine.prefill_query, engine.prefill_branch,
         [(i, engine.layers[i].self_attn.o_proj.weight) for i in indices]),
    )
    groups = [group for group in groups if group[0] not in native.plans
              and 2 * rows * group[3][0][1].numel() * 6 > DensePrefill.MAX_POOL_FLOPS]
    if not groups:
        return native
    owner = _Postpass(engine, native, groups, deadline)
    engine._prefill_postpass_pending = owner
    try:
        try:
            owner.select()
        except (Expired, torch.cuda.OutOfMemoryError):
            pass  # Earlier complete winners and all original winners survive.
    finally:
        # Do not release anything, including engine quarantine, if drain fails.
        owner._drain()
        owner.trials.clear()
        owner.groups = []
        engine._prefill_postpass_pending = None
    return owner if owner.plans else native
