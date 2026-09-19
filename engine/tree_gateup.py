"""Bounded optional gate/up selection for exact W8 tree graphs.

Called once inside TreeGraph's already-quarantined constructor try block, before
_verify/capture. All staging owners remain pinned on tree.projection_trials.
After successful constructor drain they may be released; chosen plan remains.
"""
import math
import statistics
import time
from types import SimpleNamespace

import torch
from hopper_gemm import HopperPlan
from hopper_tiles import HopperTilesPlan
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources


class Expired(Exception):
    pass


def _live(deadline):
    if time.monotonic() >= deadline:
        raise Expired()


def _graph(owner, launch, deadline):
    _live(deadline)
    trial = SimpleNamespace(stream=torch.cuda.Stream(device=owner.normalized.device), graph=None)
    owner.projection_trials.append(trial)
    current = torch.cuda.current_stream(owner.normalized.device)
    trial.stream.wait_stream(current)
    with torch.cuda.stream(trial.stream):
        launch(deadline)
    current.wait_stream(trial.stream)
    torch.cuda.synchronize(owner.normalized.device)
    _live(deadline)
    trial.graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(trial.graph, stream=trial.stream):
        launch(deadline)
    current.wait_stream(trial.stream)
    torch.cuda.synchronize(owner.normalized.device)
    _live(deadline)
    return trial.graph


def _time(graph, count, deadline, device):
    values = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(3):
        _live(deadline)
        start.record()
        for _ in range(8):
            _live(deadline)
            graph.replay()
        end.record()
        end.synchronize()
        _live(deadline)
        value = start.elapsed_time(end) / (8 * count)
        if not math.isfinite(value) or value <= 0:
            raise Expired()
        values.append(value)
    return statistics.median(values)


def select_gateup(tree, deadline):
    """At most one family and 2–3 exact existing plans; no M>32 expansion.

    Caller MUST already quarantine tree, and MUST preserve quarantine if the
    drain below fails. Unknown runtime errors propagate; no broad fallback.
    """
    tree.gateup_plan = None
    tree.projection_trials = []
    if tree.rows not in (8, 16, 24, 32):
        return
    deadline = min(deadline - 35.0, time.monotonic() + 12.0)
    if time.monotonic() >= deadline:
        return
    x, out = tree.normalized, tree.gateup
    weights = [pair[1] for pair in tree.engine.packed[:6]]
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    choices = [('hopper', 256)] + [('tiles', splits) for splits in
               HopperTilesPlan.split_choices(19456, 2560, tree.rows, sm_count)]
    # All allocations/graphs/plans referenced by queued work are also reachable
    # from the quarantined tree, including incomplete candidate trials.
    trial = SimpleNamespace(reference=None, plans=[])
    tree.projection_trials.append(trial)
    winner_ms = float('inf')
    try:
        _live(deadline)
        free, total = torch.cuda.mem_get_info(x.device)
        if free - out.numel() * 64 - 65536 < total // 4:
            return
        trial.reference = torch.empty_like(out)
        generator = torch.Generator(device=x.device).manual_seed(59317)
        x.normal_(generator=generator)
        def baseline(until):
            for weight in weights:
                _live(until)
                torch.mm(x, weight.t(), out=out)
        baseline_graph = _graph(tree, baseline, deadline)
        for kind, parameter in choices:
            _live(deadline)
            try:
                plan = (HopperPlan(x, weights[0], out, parameter) if kind == 'hopper'
                        else HopperTilesPlan(x, weights[0], out, parameter))
                trial.plans.append(plan)
                good = True
                for scale in (1., .1, 10.):
                    for weight in weights:
                        _live(deadline)
                        x.normal_(generator=generator).mul_(scale)
                        torch.mm(x, weight.t(), out=trial.reference)
                        plan(x, weight, out)
                        _live(deadline)
                        if (not torch.isfinite(out).all()
                                or not torch.isfinite(trial.reference).all()
                                or not torch.allclose(out, trial.reference, rtol=.01, atol=.01)):
                            good = False
                            break
                    if not good:
                        break
                if not good:
                    continue
                def candidate(until):
                    for weight in weights:
                        _live(until)
                        plan(x, weight, out)
                candidate_graph = _graph(tree, candidate, deadline)
                native, custom = [], []
                for graph, times in ((baseline_graph, native), (candidate_graph, custom),
                                     (candidate_graph, custom), (baseline_graph, native)):
                    times.append(_time(graph, len(weights), deadline, x.device))
                if max(custom) < min(native) * .95 and max(custom) < winner_ms:
                    tree.gateup_plan, winner_ms = plan, max(custom)
            except (CompilationError, OutOfResources, torch.cuda.OutOfMemoryError):
                torch.cuda.synchronize(x.device)
                continue
    except (Expired, torch.cuda.OutOfMemoryError):
        pass  # Retain any earlier fully validated/timed winner.
    finally:
        # If this fails, caller must retain tree in engine quarantine. Do not
        # clear projection_trials during exception unwinding or before drain.
        torch.cuda.synchronize(x.device)


def linear(tree, family, index, x, weight, out):
    plan = tree.gateup_plan if family == 'gateup' else None
    if plan is None:
        torch.mm(x, weight.t(), out=out)
    else:
        plan(x, weight, out)
