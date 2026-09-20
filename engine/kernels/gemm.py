"""Skinny GEMM for decode: ``C[M, N] = A[M, K] @ W[N, K]^T`` with M <= 32.

At decode the projections are pure weight streaming, so the kernel is built
to keep every SM reading: each program owns ``BLOCK_N`` rows of ``W`` and one
``K`` slice (split-K), accumulates in fp32 with tensor-core ``tl.dot`` over a
zero-padded 16-row ``A`` tile, and writes fp32 partials that a second kernel
sums and rounds to bf16. With ``SPLIT_K == 1`` the first kernel rounds and
stores directly. Whether this beats cuBLAS is measured per shape at warmup
(``pick_matmul``), never assumed.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

import budget


@triton.jit
def _norm_stats(x_ptr, y_ptr, xout_ptr, M, K, stride_am, eps, write_out,
                BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    """Residual add over the whole row, its bf16 sum written once, and the
    per-row rsqrt(mean(sum^2) + eps) every program needs for its A tile."""
    rm = tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_K)
    m_mask = rm < M
    sumsq = tl.zeros([BLOCK_M], tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + rk
        mask = m_mask[:, None] & (kk[None, :] < K)
        x = tl.load(x_ptr + rm[:, None] * stride_am + kk[None, :], mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + rm[:, None] * stride_am + kk[None, :], mask=mask, other=0.0).to(tl.float32)
        sb = (x + y).to(tl.bfloat16)
        if write_out:
            tl.store(xout_ptr + rm[:, None] * stride_am + kk[None, :], sb, mask=mask)
        sf = sb.to(tl.float32)
        sumsq += tl.sum(sf * sf, axis=1)
    return tl.math.rsqrt(sumsq / K + eps)


@triton.jit
def _normed_tile(x_ptr, y_ptr, wn_ptr, rstd, rm, kk, stride_am, mask):
    """bf16(w * bf16(bf16(x + y) * rstd)): the RMSNorm output tile, recomputed on the fly."""
    x = tl.load(x_ptr + rm[:, None] * stride_am + kk[None, :], mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + rm[:, None] * stride_am + kk[None, :], mask=mask, other=0.0).to(tl.float32)
    s = (x + y).to(tl.bfloat16).to(tl.float32)
    n = (s * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
    w = tl.load(wn_ptr + kk, mask=kk < 1_000_000_000, other=0.0).to(tl.float32)
    return (n * w[None, :]).to(tl.bfloat16)


@triton.jit
def _skinny_kernel(
    a_ptr, w_ptr, c_ptr, part_ptr,
    M, N, K, stride_am, stride_wn, k_per_split, num_tiles,
    y_ptr, wn_ptr, xout_ptr, eps,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr,
    NORM: tl.constexpr,
):
    """Tiles are (n-block, k-split) pairs walked by a persistent 1-D grid: with
    fewer programs than tiles each program loops, so the work always fills the
    SMs in whole waves instead of leaving a fractional last wave idle."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    rm = tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_K)
    m_mask = rm < M
    if NORM:
        rstd = _norm_stats(a_ptr, y_ptr, xout_ptr, M, K, stride_am, eps, pid == 0, BLOCK_M, BLOCK_K)
    for tile in range(pid, num_tiles, nprog):
        pid_n = tile // SPLIT_K
        pid_k = tile % SPLIT_K
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = rn < N
        k_start = pid_k * k_per_split
        k_end = tl.minimum(k_start + k_per_split, K)
        acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        for k0 in range(k_start, k_end, BLOCK_K):
            kk = k0 + rk
            k_mask = kk < k_end
            a_mask = m_mask[:, None] & k_mask[None, :]
            if NORM:
                a = _normed_tile(a_ptr, y_ptr, wn_ptr, rstd, rm, kk, stride_am, a_mask)
            else:
                a = tl.load(a_ptr + rm[:, None] * stride_am + kk[None, :], mask=a_mask, other=0.0)
            w = tl.load(w_ptr + rn[:, None] * stride_wn + kk[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0.0)
            acc += tl.dot(a, tl.trans(w))
        if SPLIT_K == 1:
            tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])
        else:
            tl.store(part_ptr + (pid_k * M + rm[:, None]) * N + rn[None, :], acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _sum_kernel(part_ptr, c_ptr, MN, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < MN
    acc = tl.zeros([BLOCK], tl.float32)
    for s in range(SPLIT_K):
        acc += tl.load(part_ptr + s * MN + idx, mask=mask, other=0.0)
    tl.store(c_ptr + idx, acc.to(tl.bfloat16), mask=mask)


@triton.jit
def _swiglu_epilogue(g, u):
    """bf16(gate), bf16(up), silu in fp32 rounded to bf16, product rounded to bf16."""
    gb = g.to(tl.bfloat16).to(tl.float32)
    ub = u.to(tl.bfloat16).to(tl.float32)
    s = (gb / (1.0 + tl.exp(-gb))).to(tl.bfloat16).to(tl.float32)
    return (s * ub).to(tl.bfloat16)


@triton.jit
def _gateup_kernel(
    a_ptr, wg_ptr, wu_ptr, c_ptr, part_ptr,
    M, N, K, stride_am, stride_wn, k_per_split, num_tiles,
    y_ptr, wn_ptr, xout_ptr, eps,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr,
    NORM: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    rm = tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_K)
    m_mask = rm < M
    if NORM:
        rstd = _norm_stats(a_ptr, y_ptr, xout_ptr, M, K, stride_am, eps, pid == 0, BLOCK_M, BLOCK_K)
    for tile in range(pid, num_tiles, nprog):
        pid_n = tile // SPLIT_K
        pid_k = tile % SPLIT_K
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = rn < N
        k_start = pid_k * k_per_split
        k_end = tl.minimum(k_start + k_per_split, K)
        acc_g = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        acc_u = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        for k0 in range(k_start, k_end, BLOCK_K):
            kk = k0 + rk
            k_mask = kk < k_end
            a_mask = m_mask[:, None] & k_mask[None, :]
            if NORM:
                a = _normed_tile(a_ptr, y_ptr, wn_ptr, rstd, rm, kk, stride_am, a_mask)
            else:
                a = tl.load(a_ptr + rm[:, None] * stride_am + kk[None, :], mask=a_mask, other=0.0)
            wg = tl.load(wg_ptr + rn[:, None] * stride_wn + kk[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0.0)
            wu = tl.load(wu_ptr + rn[:, None] * stride_wn + kk[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0.0)
            acc_g += tl.dot(a, tl.trans(wg))
            acc_u += tl.dot(a, tl.trans(wu))
        out_mask = m_mask[:, None] & n_mask[None, :]
        if SPLIT_K == 1:
            tl.store(c_ptr + rm[:, None] * N + rn[None, :], _swiglu_epilogue(acc_g, acc_u), mask=out_mask)
        else:
            tl.store(part_ptr + (pid_k * M + rm[:, None]) * N + rn[None, :], acc_g, mask=out_mask)
            tl.store(part_ptr + ((SPLIT_K + pid_k) * M + rm[:, None]) * N + rn[None, :], acc_u, mask=out_mask)


@triton.jit
def _sum_gateup_kernel(part_ptr, c_ptr, MN, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < MN
    g = tl.zeros([BLOCK], tl.float32)
    u = tl.zeros([BLOCK], tl.float32)
    for s in range(SPLIT_K):
        g += tl.load(part_ptr + s * MN + idx, mask=mask, other=0.0)
        u += tl.load(part_ptr + (SPLIT_K + s) * MN + idx, mask=mask, other=0.0)
    tl.store(c_ptr + idx, _swiglu_epilogue(g, u), mask=mask)


@triton.jit
def _row_rstd(x_ptr, y_ptr, xout_ptr, K, eps, write_out, BLOCK_K: tl.constexpr):
    rk = tl.arange(0, BLOCK_K)
    sumsq = tl.zeros([BLOCK_K], tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + rk
        mask = kk < K
        x = tl.load(x_ptr + kk, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + kk, mask=mask, other=0.0).to(tl.float32)
        sb = (x + y).to(tl.bfloat16)
        if write_out:
            tl.store(xout_ptr + kk, sb, mask=mask)
        sf = sb.to(tl.float32)
        sumsq += sf * sf
    return tl.math.rsqrt(tl.sum(sumsq, axis=0) / K + eps)


@triton.jit
def _normed_row(x_ptr, y_ptr, wn_ptr, rstd, kk, mask):
    x = tl.load(x_ptr + kk, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + kk, mask=mask, other=0.0).to(tl.float32)
    s = (x + y).to(tl.bfloat16).to(tl.float32)
    n = (s * rstd).to(tl.bfloat16).to(tl.float32)
    w = tl.load(wn_ptr + kk, mask=mask, other=0.0).to(tl.float32)
    return (n * w).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _gemv_kernel(
    a_ptr, w_ptr, w2_ptr, c_ptr, part_ptr,
    M, N, K, stride_am, stride_wn, k_per_split,
    y_ptr, wn_ptr, xout_ptr, eps,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr,
    NORM: tl.constexpr, GATEUP: tl.constexpr,
):
    """CUDA-core GEMV: every program streams BLOCK_N weight rows over its K slice
    in long contiguous segments and reduces in fp32 registers. Rows are unrolled
    statically and each re-reads the weights, so it is only offered for M == 1."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    n_mask = rn < N
    k_start = pid_k * k_per_split
    k_end = tl.minimum(k_start + k_per_split, K)
    first = (pid_n == 0) & (pid_k == 0)
    for m in tl.static_range(BLOCK_M):
        if NORM:
            rstd = _row_rstd(a_ptr + m * stride_am, y_ptr + m * stride_am, xout_ptr + m * stride_am, K, eps, first, BLOCK_K)
        acc = tl.zeros([BLOCK_N], tl.float32)
        if GATEUP:
            acc2 = tl.zeros([BLOCK_N], tl.float32)
        for k0 in range(k_start, k_end, BLOCK_K):
            kk = k0 + rk
            k_mask = kk < k_end
            if NORM:
                a = _normed_row(a_ptr + m * stride_am, y_ptr + m * stride_am, wn_ptr, rstd, kk, k_mask)
            else:
                a = tl.load(a_ptr + m * stride_am + kk, mask=k_mask, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + rn[:, None] * stride_wn + kk[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0.0)
            acc += tl.sum(w.to(tl.float32) * a[None, :], axis=1)
            if GATEUP:
                w2 = tl.load(w2_ptr + rn[:, None] * stride_wn + kk[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0.0)
                acc2 += tl.sum(w2.to(tl.float32) * a[None, :], axis=1)
        if SPLIT_K == 1:
            if GATEUP:
                tl.store(c_ptr + m * N + rn, _swiglu_epilogue(acc, acc2), mask=n_mask)
            else:
                tl.store(c_ptr + m * N + rn, acc.to(tl.bfloat16), mask=n_mask)
        else:
            tl.store(part_ptr + (pid_k * M + m) * N + rn, acc, mask=n_mask)
            if GATEUP:
                tl.store(part_ptr + ((SPLIT_K + pid_k) * M + m) * N + rn, acc2, mask=n_mask)


class Gemv:
    """Row-unrolled GEMV, optionally with the SwiGLU pair and the norm prologue."""

    def __init__(self, M: int, N: int, K: int, device, block_n: int, block_k: int, split_k: int,
                 num_warps: int, num_stages: int, gateup: bool = False):
        if M > 4:
            raise ValueError("Gemv handles at most 4 rows")
        self.M, self.N, self.K, self.gateup = M, N, K, gateup
        self.BLOCK_N, self.BLOCK_K = block_n, block_k
        self.num_warps, self.num_stages = num_warps, num_stages
        self.k_per_split = triton.cdiv(triton.cdiv(K, split_k), block_k) * block_k
        self.SPLIT_K = triton.cdiv(K, self.k_per_split)
        parts = (2 if gateup else 1) * self.SPLIT_K
        self.part = torch.empty((parts, M, N), dtype=torch.float32, device=device) if self.SPLIT_K > 1 else None
        self.grid = (triton.cdiv(N, block_n), self.SPLIT_K)

    def __call__(self, a: torch.Tensor, w: torch.Tensor, w2: torch.Tensor | None = None, norm=None) -> torch.Tensor:
        c = torch.empty((self.M, self.N), dtype=torch.bfloat16, device=a.device)
        y, wn, xout, eps = norm if norm is not None else (a, w, a, 0.0)
        _gemv_kernel[self.grid](
            a, w, w2 if w2 is not None else w, c, self.part if self.part is not None else c,
            self.M, self.N, self.K, a.stride(0), w.stride(0), self.k_per_split,
            y, wn, xout, eps,
            BLOCK_M=self.M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K, SPLIT_K=self.SPLIT_K,
            NORM=norm is not None, GATEUP=self.gateup,
            num_warps=self.num_warps, num_stages=self.num_stages,
        )
        if self.SPLIT_K > 1:
            MN = self.M * self.N
            if self.gateup:
                _sum_gateup_kernel[(triton.cdiv(MN, 1024),)](self.part, c, MN, SPLIT_K=self.SPLIT_K, BLOCK=1024, num_warps=4)
            else:
                _sum_kernel[(triton.cdiv(MN, 1024),)](self.part, c, MN, SPLIT_K=self.SPLIT_K, BLOCK=1024, num_warps=4)
        return c


GEMV_CONFIGS = [
    dict(block_n=32, block_k=256, split_k=1, num_warps=4, num_stages=3),
    dict(block_n=32, block_k=256, split_k=4, num_warps=4, num_stages=3),
    dict(block_n=16, block_k=256, split_k=8, num_warps=2, num_stages=3),
]


def _sm_count(device) -> int:
    try:
        return torch.cuda.get_device_properties(device).multi_processor_count
    except Exception:
        return 132


class SkinnyGateUp:
    """``swiglu(a @ wg.T, a @ wu.T)`` in one pass over both weight halves."""

    def __init__(self, M: int, N: int, K: int, device, block_n: int, block_k: int, split_k: int, num_warps: int,
                 num_stages: int, persist: int = 0):
        self.M, self.N, self.K = M, N, K
        self.BLOCK_M = 16 if M <= 16 else 32
        self.BLOCK_N, self.BLOCK_K = block_n, block_k
        self.num_warps, self.num_stages = num_warps, num_stages
        self.k_per_split = triton.cdiv(triton.cdiv(K, split_k), block_k) * block_k
        self.SPLIT_K = triton.cdiv(K, self.k_per_split)
        self.part = torch.empty((2 * self.SPLIT_K, M, N), dtype=torch.float32, device=device) if self.SPLIT_K > 1 else None
        self.num_tiles = triton.cdiv(N, block_n) * self.SPLIT_K
        programs = min(self.num_tiles, persist * _sm_count(device)) if persist else self.num_tiles
        self.grid = (programs,)

    def __call__(self, a: torch.Tensor, wg: torch.Tensor, wu: torch.Tensor, norm=None) -> torch.Tensor:
        """``norm=(y, w_norm, xout, eps)`` fuses ``xout = a + y; a = rms_norm(xout, w_norm)`` in front."""
        c = torch.empty((self.M, self.N), dtype=torch.bfloat16, device=a.device)
        y, wn, xout, eps = norm if norm is not None else (a, wg, a, 0.0)
        _gateup_kernel[self.grid](
            a, wg, wu, c, self.part if self.part is not None else c,
            self.M, self.N, self.K, a.stride(0), wg.stride(0), self.k_per_split, self.num_tiles,
            y, wn, xout, eps,
            BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K, SPLIT_K=self.SPLIT_K,
            NORM=norm is not None,
            num_warps=self.num_warps, num_stages=self.num_stages,
        )
        if self.SPLIT_K > 1:
            MN = self.M * self.N
            _sum_gateup_kernel[(triton.cdiv(MN, 1024),)](self.part, c, MN, SPLIT_K=self.SPLIT_K, BLOCK=1024, num_warps=4)
        return c


class SkinnyMatmul:
    def __init__(self, M: int, N: int, K: int, device, block_n: int, block_k: int, split_k: int, num_warps: int,
                 num_stages: int, persist: int = 0):
        self.M, self.N, self.K = M, N, K
        self.BLOCK_M = 16 if M <= 16 else 32
        self.BLOCK_N, self.BLOCK_K, self.SPLIT_K = block_n, block_k, split_k
        self.num_warps, self.num_stages = num_warps, num_stages
        self.k_per_split = triton.cdiv(triton.cdiv(K, split_k), block_k) * block_k
        self.SPLIT_K = triton.cdiv(K, self.k_per_split)
        self.part = torch.empty((self.SPLIT_K, M, N), dtype=torch.float32, device=device) if self.SPLIT_K > 1 else None
        self.num_tiles = triton.cdiv(N, block_n) * self.SPLIT_K
        programs = min(self.num_tiles, persist * _sm_count(device)) if persist else self.num_tiles
        self.grid = (programs,)

    def __call__(self, a: torch.Tensor, w: torch.Tensor, norm=None) -> torch.Tensor:
        """``norm=(y, w_norm, xout, eps)`` fuses ``xout = a + y; a = rms_norm(xout, w_norm)`` in front."""
        c = torch.empty((self.M, self.N), dtype=torch.bfloat16, device=a.device)
        y, wn, xout, eps = norm if norm is not None else (a, w, a, 0.0)
        _skinny_kernel[self.grid](
            a, w, c, self.part if self.part is not None else c,
            self.M, self.N, self.K, a.stride(0), w.stride(0), self.k_per_split, self.num_tiles,
            y, wn, xout, eps,
            BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K, SPLIT_K=self.SPLIT_K,
            NORM=norm is not None,
            num_warps=self.num_warps, num_stages=self.num_stages,
        )
        if self.SPLIT_K > 1:
            MN = self.M * self.N
            _sum_kernel[(triton.cdiv(MN, 1024),)](self.part, c, MN, SPLIT_K=self.SPLIT_K, BLOCK=1024, num_warps=4)
        return c


CONFIGS = [
    dict(block_n=64, block_k=128, split_k=1, num_warps=4, num_stages=3),
    dict(block_n=32, block_k=128, split_k=1, num_warps=4, num_stages=3),
    dict(block_n=64, block_k=64, split_k=4, num_warps=4, num_stages=4),
    dict(block_n=32, block_k=128, split_k=4, num_warps=4, num_stages=3),
    dict(block_n=64, block_k=128, split_k=8, num_warps=4, num_stages=3),
    dict(block_n=32, block_k=64, split_k=8, num_warps=4, num_stages=4),
    dict(block_n=64, block_k=128, split_k=1, num_warps=4, num_stages=3, persist=1),
    dict(block_n=32, block_k=128, split_k=2, num_warps=4, num_stages=3, persist=2),
    dict(block_n=64, block_k=128, split_k=4, num_warps=4, num_stages=4, persist=2),
    dict(block_n=32, block_k=128, split_k=4, num_warps=4, num_stages=3, persist=2),
]


def _time(fn, iters: int = 30, rotate: list | None = None) -> float:
    """Average milliseconds per call, measured as a CUDA graph replay.

    Host launch overhead exceeds the kernel time for many of these skinny
    shapes, so a host-driven loop would report launch cost and favour whatever
    launches fewest kernels. Capturing ``iters`` calls into one graph and
    replaying it measures device time, which is how the deployed step runs.
    With ``rotate``, call ``fn(w)`` over a cycle of distinct weight tensors so
    consecutive iterations cannot be served from L2.
    """
    ws = rotate or [None]
    call = (lambda i: fn(ws[i % len(ws)])) if rotate else (lambda i: fn())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for i in range(3):
            call(i)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(iters):
            call(i)
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(3):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / (3 * iters)


def pick_matmul(a: torch.Tensor, w: torch.Tensor, log=None, ws: list | None = None):
    """Return the fastest callable ``f(a, w) -> a @ w.T`` for this exact shape.

    Candidates: cuBLAS via ``torch.matmul`` and each Triton config whose
    output matches cuBLAS to bf16 rounding. Measured on the device the run
    will use, at warmup, rotating over ``ws`` (every layer's copy of this
    weight) so the timing reflects HBM streaming rather than L2 hits.
    """
    M, K = a.shape
    N = w.shape[0]
    rot = ws or [w]
    cublas = lambda a, w: a @ w.t()
    best_name, best_ms, best = "cublas", _time(lambda w: cublas(a, w), rotate=rot), cublas
    ref = cublas(a, w).float()
    candidates = []
    if M <= 32:
        candidates += [("skinny", cfg, lambda cfg=cfg: SkinnyMatmul(M, N, K, a.device, **cfg)) for cfg in CONFIGS]
    if M == 1:
        candidates += [("gemv", cfg, lambda cfg=cfg: Gemv(M, N, K, a.device, **cfg)) for cfg in GEMV_CONFIGS]
    for name, cfg, build in candidates:
        if budget.expired():
            break
        try:
            mm = build()
            out = mm(a, w).float()
            err = (out - ref).abs().max().item()
            tol = 0.02 * ref.abs().max().item() + 1e-3
            if err > tol:
                if log:
                    log(f"{name} {cfg} rejected: err {err:.4g} > {tol:.4g}")
                continue
            ms = _time(lambda w: mm(a, w), rotate=rot)
        except Exception as exc:  # a config that will not compile on this device is simply not used
            if log:
                log(f"{name} {cfg} failed: {exc}")
            continue
        if ms < best_ms:
            best_name, best_ms, best = f"{name}{cfg}", ms, mm
    if log:
        log(f"matmul M={M} N={N} K={K}: {best_name} {best_ms * 1000:.1f}us "
            f"({2 * M * N * K / best_ms / 1e6:.0f} GFLOP/s, {N * K * 2 / best_ms / 1e6:.0f} GB/s)")
    return best


def pick_gateup(a: torch.Tensor, wgu: torch.Tensor, log=None, ws: list | None = None):
    """Fastest ``f(a, wgu) -> swiglu(a @ wgu.T)``: cuBLAS + SwiGLU kernel, or the fused Triton GEMM."""
    from kernels.swiglu import swiglu

    M, K = a.shape
    I = wgu.shape[0] // 2
    wg, wu = wgu[:I], wgu[I:]
    rot = ws or [wgu]
    cublas = lambda a, wgu: swiglu(a @ wgu.t())
    best_name, best_ms, best = "cublas+swiglu", _time(lambda w: cublas(a, w), rotate=rot), cublas
    ref = cublas(a, wgu).float()
    candidates = []
    if M <= 32:
        candidates += [("gateup", cfg, lambda cfg=cfg: SkinnyGateUp(M, I, K, a.device, **cfg)) for cfg in CONFIGS]
    if M == 1:
        candidates += [("gemv-gateup", cfg, lambda cfg=cfg: Gemv(M, I, K, a.device, gateup=True, **cfg)) for cfg in GEMV_CONFIGS]
    for name, cfg, build in candidates:
        if budget.expired():
            break
        try:
            mm = build()
            out = mm(a, wg, wu).float()
            err = (out - ref).abs().max().item()
            tol = 0.02 * ref.abs().max().item() + 1e-3
            if err > tol:
                if log:
                    log(f"{name} {cfg} rejected: err {err:.4g} > {tol:.4g}")
                continue
            ms = _time(lambda w: mm(a, w[:I], w[I:]), rotate=rot)
        except Exception as exc:
            if log:
                log(f"{name} {cfg} failed: {exc}")
            continue
        if ms < best_ms:
            best_name, best_ms, best = f"{name}{cfg}", ms, (lambda a, wgu, mm=mm: mm(a, wgu[:I], wgu[I:]))
    if log:
        log(f"gateup M={M} I={I} K={K}: {best_name} {best_ms * 1000:.1f}us ({2 * I * K * 2 / best_ms / 1e6:.0f} GB/s)")
    return best


def pick_normed(kind: str, x: torch.Tensor, y: torch.Tensor, w_norm: torch.Tensor, w: torch.Tensor, eps: float, log=None, ws: list | None = None):
    """Fastest ``f(x, y, w_norm, xout, w) -> proj(rms_norm(x + y))`` that also writes ``xout = x + y``.

    ``kind`` is "matmul" or "gateup". The unfused pipeline (add_rms_norm
    kernel, then the best plain choice) competes against Triton configs with
    the norm folded into their prologue; both are timed as whole pipelines.
    """
    from kernels.add_rmsnorm import add_rms_norm

    M, K = x.shape
    xout = torch.empty_like(x)
    rot = ws or [w]
    picker = pick_matmul if kind == "matmul" else pick_gateup
    plain = picker(x, w, None, ws=rot)
    unfused = lambda x, y, w_norm, xout, w: plain(add_rms_norm(x, y, w_norm, eps, xout), w)
    best_name, best_ms, best = "add_norm+" + kind, _time(lambda w: unfused(x, y, w_norm, xout, w), rotate=rot), unfused
    ref = unfused(x, y, w_norm, xout, w).float()
    ref_xout = xout.clone()
    I = w.shape[0] // 2
    candidates = []
    if M <= 32:
        for cfg in CONFIGS:
            if kind == "matmul":
                candidates.append(("skinny", cfg, lambda cfg=cfg: SkinnyMatmul(M, w.shape[0], K, x.device, **cfg),
                                   lambda mm: (lambda x, y, w_norm, xout, w: mm(x, w, norm=(y, w_norm, xout, eps)))))
            else:
                candidates.append(("gateup", cfg, lambda cfg=cfg: SkinnyGateUp(M, I, K, x.device, **cfg),
                                   lambda mm: (lambda x, y, w_norm, xout, w: mm(x, w[:I], w[I:], norm=(y, w_norm, xout, eps)))))
    if M == 1:
        for cfg in GEMV_CONFIGS:
            if kind == "matmul":
                candidates.append(("gemv", cfg, lambda cfg=cfg: Gemv(M, w.shape[0], K, x.device, **cfg),
                                   lambda mm: (lambda x, y, w_norm, xout, w: mm(x, w, norm=(y, w_norm, xout, eps)))))
            else:
                candidates.append(("gemv-gateup", cfg, lambda cfg=cfg: Gemv(M, I, K, x.device, gateup=True, **cfg),
                                   lambda mm: (lambda x, y, w_norm, xout, w: mm(x, w[:I], w[I:], norm=(y, w_norm, xout, eps)))))
    for name, cfg, build, wrap in candidates:
            if budget.expired():
                break
            try:
                fused = wrap(build())
                out = fused(x, y, w_norm, xout, w).float()
                err = (out - ref).abs().max().item()
                if err > 0.02 * ref.abs().max().item() + 1e-3 or not torch.equal(xout, ref_xout):
                    if log:
                        log(f"normed {name} {cfg} rejected: err {err:.4g}")
                    continue
                ms = _time(lambda w: fused(x, y, w_norm, xout, w), rotate=rot)
            except Exception as exc:
                if log:
                    log(f"normed {name} {cfg} failed: {exc}")
                continue
            if ms < best_ms:
                best_name, best_ms, best = f"fused-norm {name}{cfg}", ms, fused
    if log:
        log(f"normed {kind} M={M} N={w.shape[0]} K={K}: {best_name} {best_ms * 1000:.1f}us")
    return best
