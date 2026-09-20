"""Warmup time budget shared by the kernel pickers.

The platform allows 300 s for load plus one warmup generation. Candidate
kernels are only tried while the budget has room; once it is spent the pickers
keep the best choice measured so far (cuBLAS is always among the candidates).
"""

from __future__ import annotations

import time

_deadline: float | None = None


def start(seconds: float) -> None:
    global _deadline
    _deadline = time.monotonic() + seconds


def expired() -> bool:
    return _deadline is not None and time.monotonic() > _deadline


def remaining() -> float:
    return float("inf") if _deadline is None else _deadline - time.monotonic()
