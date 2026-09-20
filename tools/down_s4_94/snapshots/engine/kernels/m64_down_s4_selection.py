"""One optional down geometry after all M64 incumbent selections.

The chosen gate/up is fixed. Admission compares gu+down over all 36 layer
records in both orders, using median-of-five graph timings. One eight-second
cooperative window retains twelve seconds of the shared budget; no active
CUDA call can be preempted by the deadline.
"""

import math
import os
import time

import torch

import budget
from kernels.add_rmsnorm import add_rms_norm
from kernels.down_s4 import DownS4Matmul


_QUARANTINED = []


class DownS4DrainFailed(BaseException):
    """Unconfirmed CUDA completion must bypass ordinary fallback handlers."""


def _invoke_down(fn, act, weight, allocations):
    if isinstance(fn, DownS4Matmul):
        return fn(act, weight, allocation_owners=allocations)
    out = fn(act, weight)
    allocations.append(out)
    return out


def _stage(gu, down, record, x, y, residual, allocations):
    wn, wgu, wd, _ = record
    act = gu(x, y, wn, residual, wgu)
    allocations.append(act)
    return _invoke_down(down, act, wd, allocations)


def _owned_time(fn, records, owners):
    """Private graph/pool and all exposed allocations survive every drain."""
    owner = {"fn": fn, "records": records, "allocations": [], "events": []}
    owners.append(owner)
    stream = torch.cuda.Stream()
    owner["stream"] = stream
    stream.wait_stream(torch.cuda.current_stream())

    def call(index):
        return fn(records[index % len(records)], owner["allocations"])

    with torch.cuda.stream(stream):
        for i in range(3):
            call(i)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    owner["graph"] = graph
    capture = torch.cuda.graph(graph)
    owner["capture"] = capture
    with capture:
        for i in range(36):
            call(i)
    graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        owner["events"].append(start)
        end = torch.cuda.Event(enable_timing=True)
        owner["events"].append(end)
        start.record()
        for _ in range(3):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / 108)
    samples.sort()
    ms = samples[2]
    owners.remove(owner)
    return ms


def _finite_close(ref, out, owner):
    ref_f, out_f = ref.float(), out.float()
    owner["allocations"].extend((ref_f, out_f))
    error = (out_f - ref_f).abs()
    owner["allocations"].append(error)
    tolerance = 0.02 * ref_f.abs() + 1e-3
    owner["allocations"].append(tolerance)
    valid = (torch.isfinite(ref_f).all().item()
             and torch.isfinite(out_f).all().item()
             and (error <= tolerance).all().item())
    return valid, error.max().item()


def _validate(gu, incumbent, candidate, record, x, y, residual, eps, owners):
    owner = {"functions": (gu, incumbent, candidate), "record": record,
             "inputs": (x, y, residual), "allocations": []}
    owners.append(owner)
    allocations = owner["allocations"]
    wn, wgu, wd, next_wn = record
    act = gu(x, y, wn, residual, wgu)
    allocations.append(act)
    act_snapshot = act.clone()
    allocations.append(act_snapshot)
    residual_snapshot = residual.clone()
    allocations.append(residual_snapshot)
    ref = _invoke_down(incumbent, act, wd, allocations)
    out = _invoke_down(candidate, act, wd, allocations)
    valid, max_error = _finite_close(ref, out, owner)
    ref_next = torch.empty_like(residual)
    allocations.append(ref_next)
    out_next = torch.empty_like(residual)
    allocations.append(out_next)
    ref_norm = add_rms_norm(residual, ref, next_wn, eps, ref_next)
    allocations.append(ref_norm)
    out_norm = add_rms_norm(residual, out, next_wn, eps, out_next)
    allocations.append(out_norm)
    for ref_value, out_value in ((ref_next, out_next), (ref_norm, out_norm)):
        close, error = _finite_close(ref_value, out_value, owner)
        valid = close and valid
        max_error = max(max_error, error)
    valid = (torch.equal(act, act_snapshot)
             and torch.equal(residual, residual_snapshot) and valid)
    torch.cuda.synchronize()
    owners.remove(owner)
    return valid, max_error


def select_m64_down_s4(original, model, log=None):
    if _QUARANTINED:
        raise DownS4DrainFailed("down S4 selector cannot reuse quarantined CUDA state")
    if torch.cuda.is_current_stream_capturing():
        return original
    owners = [{"original": original, "model": model}]
    try:
        result = _select_m64_down_s4(original, model, owners, log)
    except BaseException as exc:
        try:
            torch.cuda.synchronize()
        except BaseException as drain_error:
            _QUARANTINED.append((owners, exc, drain_error))
            raise DownS4DrainFailed("down S4 CUDA drain failed; owners quarantined") from exc
        if isinstance(exc, Exception):
            if log:
                log("M64 down S4: incumbent retained after drained optional error: "
                    + type(exc).__name__)
            return original
        raise
    try:
        torch.cuda.synchronize()
    except BaseException as exc:
        _QUARANTINED.append((owners, result, exc))
        raise DownS4DrainFailed("down S4 final drain failed; owners quarantined") from exc
    return result


def _select_m64_down_s4(original, model, owners, log=None):
    def emit(message):
        if log:
            log("M64 down S4: " + message)

    if (os.environ.get("ENGINE_FORCE_CUBLAS") == "1"
            or os.environ.get("ENGINE_NORM_FUSED", "1") != "1"):
        emit("incumbent retained: forced projection configuration")
        return original
    remaining = budget.remaining()
    if not math.isfinite(remaining) or remaining < 20.0:
        emit("incumbent retained: fewer than 20s of shared budget remain")
        return original
    if (len(model.layers) != 36 or "gu" not in original or "d" not in original
            or torch.cuda.get_device_capability(model.device) != (9, 0)):
        emit("incumbent retained: unsupported model, incumbent or device")
        return original

    def aligned(tensor, shape):
        return (tuple(tensor.shape) == shape and tensor.dtype == torch.bfloat16
                and tensor.is_cuda and tensor.device == model.device
                and tensor.is_contiguous() and tensor.data_ptr() % 16 == 0)

    records = [(layer.post_norm, layer.wgu, layer.wd,
                model.layers[i + 1].in_norm if i + 1 < 36 else model.final_norm)
               for i, layer in enumerate(model.layers)]
    owners.append(records)
    if not all(aligned(wn, (2560,)) and aligned(wgu, (19456, 2560))
               and aligned(wd, (2560, 9728)) and aligned(next_wn, (2560,))
               for wn, wgu, wd, next_wn in records):
        emit("incumbent retained: unsupported norm/weight layout")
        return original
    deadline = time.monotonic() + min(8.0, remaining - 12.0)

    def available():
        left = budget.remaining()
        return (math.isfinite(left) and left >= 12.0
                and time.monotonic() < deadline)

    def incomplete():
        emit("incumbent retained: bounded comparison expired or incomplete")
        return original

    private = {}
    owners.append(private)
    x = torch.randn((64, 2560), dtype=torch.bfloat16, device=model.device)
    private["x"] = x
    y = torch.randn_like(x)
    private["y"] = y
    residual = torch.empty_like(x)
    private["residual"] = residual
    private["x_snapshot"] = x.clone()
    private["y_snapshot"] = y.clone()
    gu, incumbent = original["gu"], original["d"]
    candidate = DownS4Matmul(model.device)
    owners.append(candidate)
    max_error = 0.0
    for layer_index, record in enumerate(records):
        if not available():
            return incomplete()
        valid, error = _validate(gu, incumbent, candidate, record, x, y,
                                 residual, model.cfg.eps, owners)
        max_error = max(max_error, error)
        if not valid:
            emit(f"incumbent retained: rejected at layer {layer_index}; max error {max_error:.4g}")
            return original
    if not available():
        return incomplete()
    if not (torch.equal(x, private["x_snapshot"])
            and torch.equal(y, private["y_snapshot"])):
        emit("incumbent retained: validation input mutation")
        return original

    def measured(down):
        return _owned_time(lambda record, allocations:
                           _stage(gu, down, record, x, y, residual, allocations),
                           records, owners)

    ratios = []
    for order in (("incumbent", "candidate"), ("candidate", "incumbent")):
        timings = {}
        for which in order:
            if not available():
                return incomplete()
            ms = measured(incumbent if which == "incumbent" else candidate)
            if not math.isfinite(ms) or ms <= 0:
                emit("incumbent retained: invalid timing")
                return original
            timings[which] = ms
        ratios.append(timings["candidate"] / timings["incumbent"])
    if not available():
        return incomplete()
    if not (torch.equal(x, private["x_snapshot"])
            and torch.equal(y, private["y_snapshot"])):
        emit("incumbent retained: timed input mutation")
        return original
    emit(f"ratios {ratios[0]:.4f}/{ratios[1]:.4f}; max error {max_error:.4g}")
    if max(ratios) >= 0.97:
        emit("incumbent retained: both orders must improve by more than 3%")
        return original
    selected = dict(original)
    selected["d"] = candidate
    owners.append(selected)
    emit("selected n64-k64-split4-w4-s3; chosen gate/up and all other incumbents retained")
    return selected
