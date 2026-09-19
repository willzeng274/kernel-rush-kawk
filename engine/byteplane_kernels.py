"""Experimental exact byte-plane weight loads for retained #32 kernels.

Integrated device source. Not locally CUDA-compiled, benchmarked or submitted.
LO/HI are uint8 contiguous [N,K], each exactly N*K logical bytes. The caller
owns their allocation/compression/lifetime. Arithmetic and launch settings
must match the already-selected retained plan. No output/workspace allocation.
"""
import triton
import triton.language as tl


@triton.jit
def _load_bf16_planes(LO, HI, offset, mask):
    lo = tl.load(LO + offset, mask, 0).to(tl.uint16)
    hi = tl.load(HI + offset, mask, 0).to(tl.uint16)
    bits = (lo | (hi << 8)).to(tl.uint16)
    return bits.to(tl.bfloat16, bitcast=True)


@triton.jit
def pack_byteplanes(W, LO, HI, TOTAL, BLOCK: tl.constexpr):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    bits = tl.load(W + p, p < TOTAL, 0).to(tl.uint16, bitcast=True)
    tl.store(LO + p, (bits & 255).to(tl.uint8), p < TOTAL)
    tl.store(HI + p, (bits >> 8).to(tl.uint8), p < TOTAL)


@triton.jit
def check_byteplanes(W, LO, HI, ERROR, TOTAL, BLOCK: tl.constexpr):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    original = tl.load(W + p, p < TOTAL, 0).to(tl.uint16, bitcast=True)
    lo = tl.load(LO + p, p < TOTAL, 0).to(tl.uint16)
    hi = tl.load(HI + p, p < TOTAL, 0).to(tl.uint16)
    restored = (lo | (hi << 8)).to(tl.uint16)
    bad = tl.sum(((original != restored) & (p < TOTAL)).to(tl.int32), 0)
    tl.atomic_or(ERROR, bad != 0)


@triton.jit
def _wide_dot(X, LO, HI, OUT, N, K: tl.constexpr, ROWS: tl.constexpr, WIDTH: tl.constexpr):
    n = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    k = tl.arange(0, WIDTH)
    x = tl.load(X + k, k < K, 0).to(tl.float32)
    w = _load_bf16_planes(LO, HI, n[:, None] * K + k[None, :], (n[:, None] < N) & (k[None, :] < K)).to(tl.float32)
    result = tl.sum(w * x[None, :], axis=1)
    tl.store(OUT + n, result, n < N)


@triton.jit
def _persistent_dot(X, LO, HI, OUT, N, K: tl.constexpr, ROWS: tl.constexpr, WIDTH: tl.constexpr):
    k = tl.arange(0, WIDTH)
    r = tl.arange(0, ROWS)
    x = tl.load(X + k, k < K, 0).to(tl.float32)
    for tile in range(tl.program_id(0), tl.cdiv(N, ROWS), tl.num_programs(0)):
        n = tile * ROWS + r
        w = _load_bf16_planes(LO, HI, n[:, None] * K + k[None, :], (n[:, None] < N) & (k[None, :] < K)).to(tl.float32)
        result = tl.sum(w * x[None, :], axis=1)
        tl.store(OUT + n, result, n < N)


@triton.jit
def _load_tma_bf16_planes(LO_DESC, HI_DESC, n0, k0, TN: tl.constexpr, BK: tl.constexpr):
    # Existing UINT8 tensor-map ABI. Inline asm packs each corresponding group
    # of four uint8 elements and unpacks its four uint16 results automatically.
    lo = tl._experimental_descriptor_load(LO_DESC, [n0, k0], [TN, BK], tl.uint8)
    hi = tl._experimental_descriptor_load(HI_DESC, [n0, k0], [TN, BK], tl.uint8)
    bits = tl.inline_asm_elementwise(
        "prmt.b32 $0, $2, $3, 0x5140; prmt.b32 $1, $2, $3, 0x7362;",
        constraints="=r,=r,r,r", args=[lo, hi], dtype=tl.uint16, is_pure=True, pack=4)
    return bits.to(tl.bfloat16, bitcast=True)

@triton.jit
def _hopper_dot(X, LO, HI, OUT, PART, N, X_ROW, OUT_ROW, B: tl.constexpr, K: tl.constexpr, BB: tl.constexpr, BK: tl.constexpr, SPLITS: tl.constexpr):
    n = tl.program_id(0) * 64 + tl.arange(0, 64)
    split = tl.program_id(1)
    b = tl.arange(0, BB)
    k = tl.arange(0, BK)
    steps = tl.cdiv(K, SPLITS * BK)
    acc = tl.full((64, BB), 0, tl.float32)
    for block in range(steps):
        kk = (split * steps + block) * BK + k
        w = _load_tma_bf16_planes(LO, HI, tl.program_id(0) * 64, (split * steps + block) * BK, 64, BK)
        x = tl.load(X + b[None, :] * X_ROW + kk[:, None], (b[None, :] < B) & (kk[:, None] < K), 0)
        acc = tl.dot(w, x, acc)
    if SPLITS == 1:
        tl.store(OUT + b[None, :] * OUT_ROW + n[:, None], acc, (b[None, :] < B) & (n[:, None] < N))
    else:
        tl.store(PART + (split * B + b[None, :]) * N + n[:, None], acc, (b[None, :] < B) & (n[:, None] < N))

@triton.jit
def _hopper_tiles_dot(X, LO, HI, OUT, PART, N, X_ROW, OUT_ROW, B: tl.constexpr, K: tl.constexpr, BB: tl.constexpr, BK: tl.constexpr, SPLITS: tl.constexpr):
    n = tl.program_id(0) * 128 + tl.arange(0, 128)
    split = tl.program_id(1)
    b = tl.arange(0, BB)
    k = tl.arange(0, BK)
    steps = tl.cdiv(K, SPLITS * BK)
    acc = tl.full((128, BB), 0, tl.float32)
    for block in range(steps):
        kk = (split * steps + block) * BK + k
        w = _load_tma_bf16_planes(LO, HI, tl.program_id(0) * 128, (split * steps + block) * BK, 128, BK)
        x = tl.load(X + b[None, :] * X_ROW + kk[:, None], (b[None, :] < B) & (kk[:, None] < K), 0)
        acc = tl.dot(w, x, acc)
    if SPLITS == 1:
        tl.store(OUT + b[None, :] * OUT_ROW + n[:, None], acc, (b[None, :] < B) & (n[:, None] < N))
    else:
        tl.store(PART + (split * B + b[None, :]) * N + n[:, None], acc, (b[None, :] < B) & (n[:, None] < N))



@triton.jit
def _hopper_merge(PART, OUT, N, OUT_ROW, TOTAL, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    result = tl.full((BLOCK,), 0, tl.float32)
    for split in tl.static_range(SPLITS):
        result += tl.load(PART + split * TOTAL + p, p < TOTAL, 0)
    tl.store(OUT + p // N * OUT_ROW + p % N, result, p < TOTAL)


def launch_selected(kind, plan, x, lo, hi, output, descriptors=None):
    """Allocation-free selected-plan adapter; caller validates planes and chosen plan.

    kind must describe the retained selected plan exactly. Never retune shape,
    K partition, warps, stages, persistent grid or fusion in this adapter.
    """
    if kind == "wide":
        _wide_dot[(triton.cdiv(lo.shape[0], plan.rows),)](
            x, lo, hi, output, lo.shape[0], plan.k,
            ROWS=plan.rows, WIDTH=plan.width, num_warps=plan.warps,
            num_stages=1, enable_fp_fusion=False)
    elif kind == "persistent":
        _persistent_dot[(plan.grid,)](
            x, lo, hi, output, plan.n, plan.k,
            ROWS=plan.rows, WIDTH=plan.width, num_warps=plan.warps,
            num_stages=1, enable_fp_fusion=False)
    elif kind in ("hopper", "tiles"):
        if descriptors is None or len(descriptors) != 2:
            raise ValueError("batched byte-plane launch requires owned tensor maps")
        lo_descriptor, hi_descriptor = descriptors
        rows = 64 if kind == "hopper" else 128
        if (tuple(lo.shape) != (plan.n, plan.k) or tuple(hi.shape) != (plan.n, plan.k)
                or lo_descriptor.box != (rows, plan.block_k)
                or hi_descriptor.box != (rows, plan.block_k)
                or not lo_descriptor.matches(lo) or not hi_descriptor.matches(hi)):
            raise ValueError("byte-plane tensor map owner/shape/tile changed")
        kernel = _hopper_dot if kind == "hopper" else _hopper_tiles_dot
        part = output if plan.workspace is None else plan.workspace
        compiled = kernel[(triton.cdiv(plan.n, rows), plan.splits)](
            x, lo_descriptor.gpu, hi_descriptor.gpu, output, part, plan.n, x.stride(0), output.stride(0),
            B=plan.batch, K=plan.k, BB=plan.bb, BK=plan.block_k,
            SPLITS=plan.splits, num_warps=4, num_stages=plan.stages)
        if plan.splits > 1:
            _hopper_merge[(triton.cdiv(plan.batch * plan.n, 512),)](
                plan.workspace, output, plan.n, output.stride(0),
                plan.batch * plan.n, SPLITS=plan.splits, BLOCK=512,
                num_warps=4)
        return compiled
    else:
        raise ValueError("unsupported retained plan")
