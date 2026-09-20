"""Optional N32 gate/up selection after every incumbent projection is known.

Called only by the M64 recycling verifier, after decode_matmuls returns.
The incoming gate/up and down choices are authoritative incumbents. This
work cannot take budget from their earlier selectors. Every provisional
choice is abandoned if the bounded comparison suite is incomplete.
"""

import math
import os
import time

import torch

import budget
from kernels.add_rmsnorm import add_rms_norm
from kernels.gateup_n32 import launch_gateup_n32


_QUARANTINED = []


class N32SelectionDrainFailed(BaseException):
    """Fatal: unconfirmed CUDA completion must bypass ordinary fallbacks."""


def _candidate(eps, block_k):
    def full_call(x, y, w_norm, xout, w):
        normalized = add_rms_norm(x, y, w_norm, eps, xout)
        out = torch.empty((64, 9728), dtype=torch.bfloat16, device=x.device)
        return launch_gateup_n32(normalized, w, out, block_k=block_k)

    return full_call


def _owned_time(fn, records, owners):
    """Time 36 complete calls; hold every asynchronous owner until drained."""
    owner = {"fn": fn, "records": records, "outputs": []}
    owners.append(owner)
    stream = torch.cuda.Stream()
    owner["stream"] = stream
    stream.wait_stream(torch.cuda.current_stream())

    def call(index):
        owner["outputs"].append(fn(records[index % len(records)]))

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
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    owner["events"] = (start, end)
    start.record()
    for _ in range(3):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / 108
    owners.remove(owner)
    return ms


def _validate_pair(incumbent, candidate, down, record, x, y, xout, owners):
    """Compare complete calls and their consumer, retaining intermediate owners."""
    owner = {"record": record, "inputs": (x, y, xout),
             "functions": (incumbent, candidate, down)}
    owners.append(owner)
    wn, wgu, wd = record
    ref_gu = incumbent(x, y, wn, xout, wgu)
    owner["ref_gu"] = ref_gu
    ref_residual = xout.clone()
    owner["ref_residual"] = ref_residual
    ref_down = down(ref_gu, wd)
    owner["ref_down"] = ref_down
    ref_gu_f, ref_down_f = ref_gu.float(), ref_down.float()
    owner["ref_float"] = (ref_gu_f, ref_down_f)
    out_gu = candidate(x, y, wn, xout, wgu)
    owner["out_gu"] = out_gu
    out_down = down(out_gu, wd)
    owner["out_down"] = out_down
    out_gu_f, out_down_f = out_gu.float(), out_down.float()
    owner["out_float"] = (out_gu_f, out_down_f)
    valid = torch.equal(xout, ref_residual)
    max_error = 0.0
    for ref, out in ((ref_gu_f, out_gu_f), (ref_down_f, out_down_f)):
        error = (out - ref).abs()
        owner["error"] = error
        max_error = max(max_error, error.max().item())
        valid = (torch.isfinite(ref).all().item()
                 and torch.isfinite(out).all().item()
                 and (error <= 0.02 * ref.abs() + 1e-3).all().item()
                 and valid)
    torch.cuda.synchronize()
    owners.remove(owner)
    return valid, max_error


def select_m64_gateup_n32(original, model, log=None):
    if _QUARANTINED:
        raise N32SelectionDrainFailed("N32 selector cannot be reused after a failed drain")
    # No synchronization, allocations or optional work inside capture.
    if torch.cuda.is_current_stream_capturing():
        return original
    owners = [{"original": original, "model": model}]
    try:
        result = _select_m64_gateup_n32(original, model, owners, log)
    except BaseException as exc:
        try:
            torch.cuda.synchronize()
        except BaseException as drain_error:
            _QUARANTINED.append((owners, exc, drain_error))
            raise N32SelectionDrainFailed("N32 CUDA drain failed; owners quarantined") from exc
        raise
    try:
        torch.cuda.synchronize()
    except BaseException as exc:
        _QUARANTINED.append((owners, result, exc))
        raise N32SelectionDrainFailed("N32 final drain failed; owners quarantined") from exc
    return result


def _select_m64_gateup_n32(original, model, owners, log=None):
    def emit(message):
        if log:
            log("M64 N32 gate/up: " + message)

    if (os.environ.get("ENGINE_FORCE_CUBLAS") == "1"
            or os.environ.get("ENGINE_NORM_FUSED", "1") != "1"):
        emit("incumbent retained: forced projection configuration")
        return original
    remaining = budget.remaining()
    if not math.isfinite(remaining) or remaining < 24.0:
        emit("incumbent retained: fewer than 24s of shared budget remain")
        return original
    if (len(model.layers) != 36 or "gu" not in original or "d" not in original
            or torch.cuda.get_device_capability(model.device) != (9, 0)):
        emit("incumbent retained: unsupported model, incumbent or device")
        return original

    def aligned(tensor, shape):
        return (tuple(tensor.shape) == shape
                and tensor.dtype == torch.bfloat16 and tensor.is_cuda
                and tensor.device == model.device and tensor.is_contiguous()
                and tensor.data_ptr() % 16 == 0)

    records = [(layer.post_norm, layer.wgu, layer.wd) for layer in model.layers]
    owners.append(records)
    if not all(aligned(wn, (2560,)) and aligned(wgu, (19456, 2560))
               and aligned(wd, (2560, 9728)) for wn, wgu, wd in records):
        emit("incumbent retained: unsupported norm/weight layout")
        return original
    deadline = time.monotonic() + min(12.0, remaining - 12.0)

    def available():
        return budget.remaining() >= 12.0 and time.monotonic() < deadline

    def incomplete():
        emit("incumbent retained: bounded comparison suite incomplete")
        return original

    x = torch.randn((64, 2560), dtype=torch.bfloat16, device=model.device)
    y = torch.randn_like(x)
    xout = torch.empty_like(x)
    owners.append((x, y, xout))
    incumbent, down = original["gu"], original["d"]
    winner, winning_ratio, winning_name = incumbent, 1.0, "incumbent"

    def invoke(fn, record, with_down):
        wn, wgu, wd = record
        gu = fn(x, y, wn, xout, wgu)
        # The timer retains both the intermediate and its dependent output.
        return (gu, down(gu, wd)) if with_down else gu

    def measured(fn, with_down=True):
        return _owned_time(lambda record: invoke(fn, record, with_down), records, owners)

    for block_k in (128, 64):
        if not available():
            return incomplete()
        candidate = _candidate(model.cfg.eps, block_k)
        owners.append(candidate)
        valid, max_error = True, 0.0
        for layer_index, record in enumerate(records):
            if not available():
                return incomplete()
            valid, error = _validate_pair(incumbent, candidate, down, record, x, y, xout, owners)
            max_error = max(max_error, error)
            if not valid:
                emit(f"BK{block_k} rejected at layer {layer_index}; max error {max_error:.4g}")
                break
        if not valid:
            continue
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
        # Attribution only: selection depends on complete gu+down timings.
        gu_times = {}
        for which in ("incumbent", "candidate"):
            if not available():
                return incomplete()
            ms = measured(incumbent if which == "incumbent" else candidate, with_down=False)
            if not math.isfinite(ms) or ms <= 0:
                emit("incumbent retained: invalid gate/up timing")
                return original
            gu_times[which] = ms
        if not available():
            return incomplete()
        worst_ratio = max(ratios)
        emit(f"BK{block_k}: gu+down ratios {ratios[0]:.4f}/{ratios[1]:.4f}; "
             f"gu ratio {gu_times['candidate'] / gu_times['incumbent']:.4f}; "
             f"max error {max_error:.4g}")
        if worst_ratio < 0.97 and worst_ratio < winning_ratio:
            winner, winning_ratio, winning_name = candidate, worst_ratio, f"BK{block_k}"
    if not available():
        return incomplete()
    emit(f"selected {winning_name}; worst gu+down ratio {winning_ratio:.4f}")
    if winner is incumbent:
        return original
    selected = dict(original)
    selected["gu"] = winner
    owners.append(selected)
    return selected
