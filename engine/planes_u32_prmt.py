"""Research only: offline compiler experiment, no GPU validation."""

import triton

import triton.language as tl

@triton.jit
def _load_tma_bf16_planes(LO_DESC, HI_DESC, n0, k0, TN: tl.constexpr, BK: tl.constexpr):
    lo = tl._experimental_descriptor_load(LO_DESC, [n0, k0 // 4], [TN, BK // 4], tl.uint32)
    hi = tl._experimental_descriptor_load(HI_DESC, [n0, k0 // 4], [TN, BK // 4], tl.uint32)
    w0, w1, w2, w3 = tl.inline_asm_elementwise(
        "{ .reg .b32 p0, p1; prmt.b32 p0, $4, $5, 0x5140; "
        "prmt.b32 p1, $4, $5, 0x7362; mov.b32 {$0, $1}, p0; mov.b32 {$2, $3}, p1; }",
        constraints="=h,=h,=h,=h,r,r", args=[lo, hi],
        dtype=(tl.uint16, tl.uint16, tl.uint16, tl.uint16), is_pure=True, pack=1)
    # join(a,b)[...,i,0]=a[...,i]; join(a,b)[...,i,1]=b[...,i].
    # The nested order here is [w0,w1,w2,w3], not [w0,w2,w1,w3].
    bits = tl.join(tl.join(w0, w2), tl.join(w1, w3))
    return tl.reshape(bits, (TN, BK), can_reorder=False).to(tl.bfloat16, bitcast=True)


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

