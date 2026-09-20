"""Warmup-only selection among the existing complete projection paths."""

from contextlib import contextmanager
import os
import statistics
import time

import torch


class _UnsafeCaptureState(BaseException):
    """Do not enter the engine's eager fallback with uncertain CUDA state."""


class _BudgetExpired(Exception):
    pass


@contextmanager
def _projection_flags(model, rows, arm):
    missing = object()
    previous = model._gemv_choice.get(rows, missing)
    split = model.split_k
    model._gemv_choice[rows] = arm != "C"
    model.split_k = arm == "P"
    try:
        yield
    finally:
        model.split_k = split
        if previous is missing:
            model._gemv_choice.pop(rows, None)
        else:
            model._gemv_choice[rows] = previous


def _healthy(stream):
    try:
        # The eager warmup context may have restored a different current stream.
        with torch.cuda.stream(stream):
            if torch.cuda.is_current_stream_capturing():
                return False
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


def _check_budget(deadline):
    if time.perf_counter() >= deadline:
        raise _BudgetExpired()


def _capture_arm(engine, b, bucket, q, first, positions, pad, length, kv, deadline):
    """Each arm owns its inputs, attention workspace, output and private pool."""
    dev = engine.device
    tokens = (first[:, None].expand(b, q).clone() if q else first.clone())
    pos = positions.clone()
    lens = torch.full((b if q else 1,), length, device=dev, dtype=torch.int64)
    slots = torch.empty(b, device=dev, dtype=torch.int64)
    start = torch.tensor(pad, device=dev, dtype=torch.int32)
    ws = engine._make_ws_verify(b, bucket, q) if q else engine._make_ws(b, bucket)

    def body():
        # decode increments this tensor; every replay must overwrite the same
        # suffix slots and preserve the real, initialized prefix [0, length).
        lens.fill_(length)
        if q:
            return engine.model.verify(tokens, pos, *kv, lens, start, ws)
        hidden = engine.model.decode(tokens, pos, *kv, slots, lens, start, ws)
        return engine.model.argmax_token(hidden)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    capture_started = False
    try:
        with torch.cuda.stream(stream):
            for _ in range(3):
                _check_budget(deadline)
                body()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        _check_budget(deadline)
        graph = torch.cuda.CUDAGraph()
        # No production pool or another candidate arm's pool is shared here.
        capture_started = True
        with torch.cuda.graph(graph, stream=stream):
            output = body()
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
    except Exception as exc:
        # Even a stream that now reports "not capturing" cannot prove that a
        # failed context exit restored allocator, generator and stream state.
        if capture_started:
            raise _UnsafeCaptureState("whole-path selector capture failed") from exc
        if not _healthy(stream):
            raise _UnsafeCaptureState("whole-path selector CUDA state is uncertain") from exc
        raise
    keep = (tokens, pos, lens, slots, start, ws, kv, output, stream)
    return graph, keep


def _select(engine, b, bucket, q, inputs, incumbent, deadline):
    ids, positions, pad, length, bias = inputs
    _check_budget(deadline)
    kv = ([t[:b] for t in engine.k_cache], [t[:b] for t in engine.v_cache])
    # This extra prefill is untimed warmup only. Production capture still comes
    # next, followed by the original prefill which repairs all capture writes.
    first = engine.model.argmax_token(engine._prefill(ids, positions, *kv, bias))
    pos = positions[:, -1].contiguous() + 1
    torch.cuda.synchronize()
    _check_budget(deadline)
    rows = b * max(1, q)
    arms = ("C", "U", "P")
    entries = {}
    for arm in arms:
        _check_budget(deadline)
        with _projection_flags(engine.model, rows, arm):
            entries[arm] = _capture_arm(
                engine, b, bucket, q, first, pos, pad, length, kv, deadline)
    measurements = {arm: [] for arm in arms}
    for r in range(3):
        for arm in arms[r:] + arms[:r]:
            _check_budget(deadline)
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(8):
                entries[arm][0].replay()
            end.record()
            end.synchronize()
            measurements[arm].append(begin.elapsed_time(end) / 8)
    _check_budget(deadline)
    medians = {arm: statistics.median(values) for arm, values in measurements.items()}
    winner = min(arms, key=medians.get)
    return winner if medians[winner] < medians[incumbent] * 0.98 else incumbent


def capture_projection_path(engine, b, bucket, q, inputs, capture, deadline):
    """Select once per graph shape, then bake only its flags into production."""
    model = engine.model
    rows = b * max(1, q)
    forced = any(k.startswith(("ENGINE_GEMV", "ENGINE_SPLIT", "ENGINE_SK_"))
                 for k in os.environ)
    if (inputs is None or not engine.triton or rows > 32 or model.fused or forced
            or model.layers[0]["qkv"] is None):
        return capture(b, bucket)
    tri = model.use_gemv_for(rows)
    incumbent = "P" if tri and model.split_k else "U" if tri else "C"
    key = ("verify" if q else "decode", b, q, bucket)
    choices = engine.__dict__.setdefault("_projection_paths", {})
    if key not in choices:
        choices[key] = incumbent
        # Leave substantial time for the existing production warmup/capture.
        local_deadline = min(deadline, time.perf_counter() + 6.0)
        try:
            choices[key] = _select(engine, b, bucket, q, inputs, incumbent, local_deadline)
        except Exception as exc:
            if not _healthy(torch.cuda.current_stream()):
                raise _UnsafeCaptureState("whole-path selector CUDA state is uncertain") from exc
            # Budget, allocation, or healthy eager-warmup failures keep the exact
            # incumbent. The original real prefill still resets the KV cache.
    with _projection_flags(model, rows, choices[key]):
        return capture(b, bucket)
