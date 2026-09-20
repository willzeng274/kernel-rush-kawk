"""One optional M64 QKV configuration, after every incumbent selector.

No device code changes: logical BM64/BN32/BK64, W4/S3, split1, full grid192.
The complete add/norm/projection call competes across all 36 actual layers.
The six-second local window is cooperative; an active call is not preempted.
"""

import math
import os
import time

import torch

import budget
from kernels.add_rmsnorm import add_rms_norm
from kernels.gemm import SkinnyMatmul


_QUARANTINED = []


class QKVN32DrainFailed(BaseException):
    """Fatal: an unconfirmed CUDA drain cannot enter an ordinary fallback."""


class _QKVN32(SkinnyMatmul):
    def __init__(self, eps, device):
        super().__init__(64, 6144, 2560, device, block_n=32, block_k=64,
                         split_k=1, num_warps=4, num_stages=3, persist=0)
        self.BLOCK_M = 64
        self.eps = eps

    def __call__(self, x, y, wn, xout, w, *, allocation_owners=None):
        normalized = add_rms_norm(x, y, wn, self.eps, xout)
        if allocation_owners is not None:
            allocation_owners.append(normalized)
        out = super().__call__(normalized, w)
        if allocation_owners is not None:
            allocation_owners.append(out)
        return out


def _invoke(fn, record, x, y, residual, allocations):
    wn, w = record
    if isinstance(fn, _QKVN32):
        return fn(x, y, wn, residual, w, allocation_owners=allocations)
    out = fn(x, y, wn, residual, w)
    allocations.append(out)
    return out


def _owned_time(fn, records, owners):
    """Retain streams, graph/pool, events and every exposed allocation.

The graph owns its captured allocation pool, including temporary allocations
inside the incumbent callable. Explicit candidate normalized tensors and all
returned outputs additionally stay in allocations through the final drain.
"""
    owner = {"fn": fn, "records": records, "allocations": []}
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
    start = torch.cuda.Event(enable_timing=True)
    owner["start"] = start
    end = torch.cuda.Event(enable_timing=True)
    owner["end"] = end
    start.record()
    for _ in range(3):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / 108
    owners.remove(owner)
    return ms


def _validate(incumbent, candidate, record, x, y, ref_residual,
              candidate_residual, owners):
    owner = {"functions": (incumbent, candidate), "record": record,
             "inputs": (x, y, ref_residual, candidate_residual),
             "allocations": []}
    owners.append(owner)
    ref = _invoke(incumbent, record, x, y, ref_residual, owner["allocations"])
    out = _invoke(candidate, record, x, y, candidate_residual, owner["allocations"])
    ref_f, out_f = ref.float(), out.float()
    owner.update(ref_float=ref_f, out_float=out_f)
    error = (out_f - ref_f).abs()
    owner["error"] = error
    tolerance = 0.02 * ref_f.abs() + 1e-3
    owner["tolerance"] = tolerance
    valid = (torch.isfinite(ref_f).all().item()
             and torch.isfinite(out_f).all().item()
             and (error <= tolerance).all().item()
             and torch.equal(ref_residual, candidate_residual))
    max_error = error.max().item()
    torch.cuda.synchronize()
    owners.remove(owner)
    return valid, max_error


def select_m64_qkv_n32(original, model, log=None):
    if _QUARANTINED:
        raise QKVN32DrainFailed("QKV N32 selector cannot reuse quarantined CUDA state")
    # A capture-time no-op does not allocate or synchronize.
    if torch.cuda.is_current_stream_capturing():
        return original
    owners = [{"original": original, "model": model}]
    try:
        result = _select_m64_qkv_n32(original, model, owners, log)
    except BaseException as exc:
        try:
            torch.cuda.synchronize()
        except BaseException as drain_error:
            # The exception traceback also retains interrupted-call locals.
            _QUARANTINED.append((owners, exc, drain_error))
            raise QKVN32DrainFailed("QKV N32 CUDA drain failed; owners quarantined") from exc
        if isinstance(exc, Exception):
            if log:
                log("M64 QKV N32: incumbent retained after drained optional error: "
                    + type(exc).__name__)
            return original
        raise
    try:
        torch.cuda.synchronize()
    except BaseException as exc:
        _QUARANTINED.append((owners, result, exc))
        raise QKVN32DrainFailed("QKV N32 final drain failed; owners quarantined") from exc
    return result


def _select_m64_qkv_n32(original, model, owners, log=None):
    def emit(message):
        if log:
            log("M64 QKV N32: " + message)

    if (os.environ.get("ENGINE_FORCE_CUBLAS") == "1"
            or os.environ.get("ENGINE_NORM_FUSED", "1") != "1"):
        emit("incumbent retained: forced projection configuration")
        return original
    remaining = budget.remaining()
    if not math.isfinite(remaining) or remaining < 12.0:
        emit("incumbent retained: fewer than 12s of shared budget remain")
        return original
    if (len(model.layers) != 36 or "qkv" not in original
            or torch.cuda.get_device_capability(model.device) != (9, 0)):
        emit("incumbent retained: unsupported model, incumbent or device")
        return original

    def aligned(tensor, shape):
        return (tuple(tensor.shape) == shape
                and tensor.dtype == torch.bfloat16 and tensor.is_cuda
                and tensor.device == model.device and tensor.is_contiguous()
                and tensor.data_ptr() % 16 == 0)

    records = [(layer.in_norm, layer.wqkv) for layer in model.layers]
    owners.append(records)
    if not all(aligned(wn, (2560,)) and aligned(w, (6144, 2560))
               for wn, w in records):
        emit("incumbent retained: unsupported norm/weight layout")
        return original
    deadline = time.monotonic() + min(6.0, remaining - 6.0)

    def available():
        remaining_now = budget.remaining()
        return (math.isfinite(remaining_now) and remaining_now >= 6.0
                and time.monotonic() < deadline)

    def incomplete():
        emit("incumbent retained: bounded comparison expired or incomplete")
        return original

    # Private immutable test inputs never alias model/plan activations. Keep
    # each allocation owned immediately, including during an allocation error.
    private = {}
    owners.append(private)
    x = torch.randn((64, 2560), dtype=torch.bfloat16, device=model.device)
    private["x"] = x
    y = torch.randn_like(x)
    private["y"] = y
    ref_residual = torch.empty_like(x)
    private["ref_residual"] = ref_residual
    candidate_residual = torch.empty_like(x)
    private["candidate_residual"] = candidate_residual
    private["x_snapshot"] = x.clone()
    private["y_snapshot"] = y.clone()
    incumbent = original["qkv"]
    candidate = _QKVN32(model.cfg.eps, model.device)
    owners.append(candidate)
    max_error = 0.0
    for layer_index, record in enumerate(records):
        if not available():
            return incomplete()
        valid, error = _validate(incumbent, candidate, record, x, y,
                                 ref_residual, candidate_residual, owners)
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

    def measured(fn):
        residual = candidate_residual if fn is candidate else ref_residual
        return _owned_time(lambda record, allocations:
                           _invoke(fn, record, x, y, residual, allocations),
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
    worst_ratio = max(ratios)
    emit(f"ratios {ratios[0]:.4f}/{ratios[1]:.4f}; max error {max_error:.4g}")
    if worst_ratio >= 0.97:
        emit("incumbent retained: both orders must improve by more than 3%")
        return original
    selected = dict(original)
    selected["qkv"] = candidate
    owners.append(selected)
    emit("selected n32-k64-w4-s3; every other incumbent retained")
    return selected
