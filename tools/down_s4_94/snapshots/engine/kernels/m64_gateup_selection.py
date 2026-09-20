"""Bounded whole-call selection for the two compiled M64 Hopper variants.

Selection is optional: the original add/RMSNorm + cuBLAS + SwiGLU callable
remains the fallback. No weights or captured graph buffers are replaced here.
"""

import math
import time

import torch

import budget
from kernels.add_rmsnorm import add_rms_norm
from kernels.gateup64 import launch_gateup64


def _candidate(eps, block_k):
    def full_call(x, y, w_norm, xout, w):
        normalized = add_rms_norm(x, y, w_norm, eps, xout)
        # A fresh output follows the original callable's ownership semantics.
        # During capture the graph owns this allocation; no layer-shared
        # scratch is overwritten before down-projection consumes it.
        out = torch.empty((64, 9728), dtype=torch.bfloat16, device=x.device)
        return launch_gateup64(normalized, w, out, block_k=block_k)

    return full_call


def pick_m64_gateup(x, y, w_norm, xout, eps, original, weights, timer, log=None):
    """Return original or one fully checked/timed callable; never a partial trial.

    ``original`` and candidates share the (x,y,norm,xout,w) interface. Every
    measured graph rotates all 36 original layer matrices. The 12-second
    local admission window is capped by the existing shared warmup deadline;
    it does not preempt an already-running compilation or graph timing call.
    """
    def emit(message):
        if log:
            log("M64 WGMMA: " + message)

    def aligned(tensor, shape):
        return (tuple(tensor.shape) == shape and tensor.dtype == torch.bfloat16
                and tensor.is_cuda and tensor.device == x.device
                and tensor.is_contiguous() and tensor.data_ptr() % 16 == 0)

    if (len(weights) != 36 or tuple(x.shape) != (64, 2560)
            or not all(aligned(t, (64, 2560)) for t in (x, y, xout))
            or not aligned(w_norm, (2560,))
            or not all(aligned(w, (19456, 2560)) for w in weights)
            or torch.cuda.get_device_capability(x.device) != (9, 0)):
        emit("baseline retained: unsupported shape, layout, layer count or device")
        return original
    if budget.remaining() < 10.0:
        emit("baseline retained: fewer than 10s remain in shared picker budget")
        return original
    deadline = time.monotonic() + min(12.0, budget.remaining() - 2.0)

    def available():
        return not budget.expired() and time.monotonic() < deadline

    def measured(fn):
        return timer(lambda w: fn(x, y, w_norm, xout, w), iters=36, rotate=weights)

    winner, winning_ratio, winning_name = original, 1.0, "baseline"
    for block_k in (128, 64):
        if not available():
            emit("baseline retained: incomplete bounded selection")
            return original
        candidate = _candidate(eps, block_k)
        valid, max_error = True, 0.0
        for layer_index, w in enumerate(weights):
            if not available():
                emit("baseline retained: validation budget exhausted")
                return original
            # Baseline execution errors must propagate, not be hidden as
            # candidate failures. FP32 copies survive candidate allocations.
            ref = original(x, y, w_norm, xout, w).float()
            ref_xout = xout.clone()
            try:
                out = candidate(x, y, w_norm, xout, w).float()
                error = (out - ref).abs()
                max_error = max(max_error, error.max().item())
                # Same rtol/atol constants as the retained picker, applied
                # pointwise instead of to the single largest reference value.
                # This is strictly no weaker than its global error envelope.
                matches = (torch.isfinite(out).all().item()
                           and torch.isfinite(ref).all().item()
                           and (error <= 0.02 * ref.abs() + 1e-3).all().item()
                           and torch.equal(xout, ref_xout))
            except Exception as exc:
                emit(f"BK{block_k} rejected during validation: {exc}")
                valid = False
                break
            if not matches:
                emit(f"BK{block_k} rejected at layer {layer_index}: max error {max_error:.4g}")
                valid = False
                break
        if not valid:
            continue
        ratios = []
        for candidate_first in (False, True):
            times = {}
            order = ("candidate", "baseline") if candidate_first else ("baseline", "candidate")
            for name in order:
                if not available():
                    emit("baseline retained: timing budget exhausted")
                    return original
                if name == "baseline":
                    ms = measured(original)
                else:
                    try:
                        ms = measured(candidate)
                    except Exception as exc:
                        emit(f"BK{block_k} rejected during timing: {exc}")
                        valid = False
                        break
                if not math.isfinite(ms) or ms <= 0:
                    emit("baseline retained: invalid timing sample")
                    return original
                times[name] = ms
            if not valid:
                break
            ratios.append(times["candidate"] / times["baseline"])
        if valid:
            worst_ratio = max(ratios)
            emit(f"BK{block_k} whole-call ratios {ratios[0]:.4f}/{ratios[1]:.4f}; max error {max_error:.4g}")
            # Each of two opposite-order comparisons must clear the margin.
            if worst_ratio < 0.97 and worst_ratio < winning_ratio:
                winner, winning_ratio, winning_name = candidate, worst_ratio, f"BK{block_k}"
    if not available():
        emit("baseline retained: selection deadline reached before commit")
        return original
    emit(f"selected {winning_name}; worst whole-call ratio {winning_ratio:.4f}")
    return winner
