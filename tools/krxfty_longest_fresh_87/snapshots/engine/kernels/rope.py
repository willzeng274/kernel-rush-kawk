"""Per-head RMSNorm + rotary embedding + KV-cache write, fused, matching
Transformers 4.51.3 ``Qwen3Attention`` bit-for-bit in its cast placement.

Reference (per head of width D):
    n  = bf16(x_f32 * rsqrt(mean(x_f32^2) + eps))     # Qwen3RMSNorm, fp32 reduce
    y  = bf16(w * n)                                    # weight multiply in bf16
    q  = bf16(bf16(y * cos) + bf16(rotate_half(y) * sin))   # apply_rotary_pos_emb

cos/sin are the bf16 tables the reference builds (see ``model.rope_tables``);
the kernel only gathers rows from them by absolute position.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _norm_rope_row(x1, x2, w1, w2, cos1, cos2, sin1, sin2, eps, D: tl.constexpr):
    var = (tl.sum(x1 * x1, axis=0) + tl.sum(x2 * x2, axis=0)) / D
    r = tl.math.rsqrt(var + eps)
    n1 = (x1 * r).to(tl.bfloat16).to(tl.float32)
    n2 = (x2 * r).to(tl.bfloat16).to(tl.float32)
    y1 = (w1 * n1).to(tl.bfloat16).to(tl.float32)
    y2 = (w2 * n2).to(tl.bfloat16).to(tl.float32)
    a1 = (y1 * cos1).to(tl.bfloat16).to(tl.float32)
    b1 = ((-y2) * sin1).to(tl.bfloat16).to(tl.float32)
    a2 = (y2 * cos2).to(tl.bfloat16).to(tl.float32)
    b2 = (y1 * sin2).to(tl.bfloat16).to(tl.float32)
    return (a1 + b1).to(tl.bfloat16), (a2 + b2).to(tl.bfloat16)


@triton.jit
def _q_kernel(
    qkv_ptr, w_ptr, cos_ptr, sin_ptr, pos_ptr, q_out_ptr,
    T, eps,
    ROW: tl.constexpr, HQ: tl.constexpr, D: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    b = row // T
    t = row % T
    pos = tl.load(pos_ptr + b) + t
    HALF: tl.constexpr = D // 2
    d = tl.arange(0, HALF)
    src = qkv_ptr + row * ROW + head * D
    x1 = tl.load(src + d).to(tl.float32)
    x2 = tl.load(src + HALF + d).to(tl.float32)
    w1 = tl.load(w_ptr + d).to(tl.float32)
    w2 = tl.load(w_ptr + HALF + d).to(tl.float32)
    tab = pos * D
    cos1 = tl.load(cos_ptr + tab + d).to(tl.float32)
    cos2 = tl.load(cos_ptr + tab + HALF + d).to(tl.float32)
    sin1 = tl.load(sin_ptr + tab + d).to(tl.float32)
    sin2 = tl.load(sin_ptr + tab + HALF + d).to(tl.float32)
    o1, o2 = _norm_rope_row(x1, x2, w1, w2, cos1, cos2, sin1, sin2, eps, D)
    dst = q_out_ptr + ((b * HQ + head) * T + t) * D
    tl.store(dst + d, o1)
    tl.store(dst + HALF + d, o2)


@triton.jit
def _kv_kernel(
    qkv_ptr, w_ptr, cos_ptr, sin_ptr, pos_ptr, k_cache_ptr, v_cache_ptr,
    T, CAP, eps,
    ROW: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
):
    row = tl.program_id(0)
    kh = tl.program_id(1)
    b = row // T
    t = row % T
    pos = tl.load(pos_ptr + b) + t
    HALF: tl.constexpr = D // 2
    d = tl.arange(0, HALF)
    src = qkv_ptr + row * ROW + (HQ + kh) * D
    x1 = tl.load(src + d).to(tl.float32)
    x2 = tl.load(src + HALF + d).to(tl.float32)
    w1 = tl.load(w_ptr + d).to(tl.float32)
    w2 = tl.load(w_ptr + HALF + d).to(tl.float32)
    tab = pos * D
    cos1 = tl.load(cos_ptr + tab + d).to(tl.float32)
    cos2 = tl.load(cos_ptr + tab + HALF + d).to(tl.float32)
    sin1 = tl.load(sin_ptr + tab + d).to(tl.float32)
    sin2 = tl.load(sin_ptr + tab + HALF + d).to(tl.float32)
    o1, o2 = _norm_rope_row(x1, x2, w1, w2, cos1, cos2, sin1, sin2, eps, D)
    slot = ((b * HKV + kh) * CAP + pos) * D
    tl.store(k_cache_ptr + slot + d, o1)
    tl.store(k_cache_ptr + slot + HALF + d, o2)
    vsrc = qkv_ptr + row * ROW + (HQ + HKV + kh) * D
    tl.store(v_cache_ptr + slot + d, tl.load(vsrc + d))
    tl.store(v_cache_ptr + slot + HALF + d, tl.load(vsrc + HALF + d))


@triton.jit
def _qkv_kernel(
    qkv_ptr, qw_ptr, kw_ptr, cos_ptr, sin_ptr, pos_ptr, depth_ptr, q_out_ptr, k_cache_ptr, v_cache_ptr,
    T, CAP, eps,
    ROW: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, DEPTH: tl.constexpr,
):
    """One launch for every head: programs below HQ rotate a query head, the
    rest normalise/rotate one key head and copy its value head into the cache.
    Row t of a sequence lives in cache slot pos + t; its rotary angle is that of
    position pos + t for a chain, or pos + depth[t] for a draft tree, where
    siblings legitimately share one absolute position."""
    row = tl.program_id(0)
    head = tl.program_id(1)
    b = row // T
    t = row % T
    slot = tl.load(pos_ptr + b) + t
    if DEPTH:
        pos = tl.load(pos_ptr + b) + tl.load(depth_ptr + t)
    else:
        pos = slot
    HALF: tl.constexpr = D // 2
    d = tl.arange(0, HALF)
    tab = pos * D
    cos1 = tl.load(cos_ptr + tab + d).to(tl.float32)
    cos2 = tl.load(cos_ptr + tab + HALF + d).to(tl.float32)
    sin1 = tl.load(sin_ptr + tab + d).to(tl.float32)
    sin2 = tl.load(sin_ptr + tab + HALF + d).to(tl.float32)
    if head < HQ:
        src = qkv_ptr + row * ROW + head * D
        x1 = tl.load(src + d).to(tl.float32)
        x2 = tl.load(src + HALF + d).to(tl.float32)
        w1 = tl.load(qw_ptr + d).to(tl.float32)
        w2 = tl.load(qw_ptr + HALF + d).to(tl.float32)
        o1, o2 = _norm_rope_row(x1, x2, w1, w2, cos1, cos2, sin1, sin2, eps, D)
        dst = q_out_ptr + ((b * HQ + head) * T + t) * D
        tl.store(dst + d, o1)
        tl.store(dst + HALF + d, o2)
    else:
        kh = head - HQ
        src = qkv_ptr + row * ROW + (HQ + kh) * D
        x1 = tl.load(src + d).to(tl.float32)
        x2 = tl.load(src + HALF + d).to(tl.float32)
        w1 = tl.load(kw_ptr + d).to(tl.float32)
        w2 = tl.load(kw_ptr + HALF + d).to(tl.float32)
        o1, o2 = _norm_rope_row(x1, x2, w1, w2, cos1, cos2, sin1, sin2, eps, D)
        cslot = ((b * HKV + kh) * CAP + slot) * D
        tl.store(k_cache_ptr + cslot + d, o1)
        tl.store(k_cache_ptr + cslot + HALF + d, o2)
        vsrc = qkv_ptr + row * ROW + (HQ + HKV + kh) * D
        tl.store(v_cache_ptr + cslot + d, tl.load(vsrc + d))
        tl.store(v_cache_ptr + cslot + HALF + d, tl.load(vsrc + HALF + d))


def qk_norm_rope_cache(
    qkv: torch.Tensor,
    q_norm_w: torch.Tensor,
    k_norm_w: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    pos: torch.Tensor,
    q_out: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    T: int,
    eps: float,
    fused: bool = True,
    depth: torch.Tensor | None = None,
) -> None:
    """Normalise, rotate and scatter one projection buffer.

    qkv      [B*T, (HQ + 2*HKV) * D] bf16, rows ordered (b, t)
    pos      [B] int32, absolute position of row (b, 0); row (b, t) is pos[b] + t
    q_out    [B, HQ, T, D] bf16, written contiguous for SDPA / decode attention
    k_cache  [B, HKV, CAP, D] bf16 (one layer), written at pos[b] + t
    v_cache  same layout, written with the raw (un-normalised) value rows
    """
    B, HQ, _, D = q_out.shape
    HKV, CAP = k_cache.shape[1], k_cache.shape[2]
    M = B * T
    ROW = qkv.stride(0)
    if fused or depth is not None:
        _qkv_kernel[(M, HQ + HKV)](
            qkv, q_norm_w, k_norm_w, cos, sin, pos, depth if depth is not None else pos, q_out, k_cache, v_cache,
            T, CAP, eps,
            ROW=ROW, HQ=HQ, HKV=HKV, D=D, DEPTH=depth is not None, num_warps=1,
        )
        return
    _q_kernel[(M, HQ)](
        qkv, q_norm_w, cos, sin, pos, q_out, T, eps,
        ROW=ROW, HQ=HQ, D=D, num_warps=1,
    )
    _kv_kernel[(M, HKV)](
        qkv, k_norm_w, cos, sin, pos, k_cache, v_cache, T, CAP, eps,
        ROW=ROW, HQ=HQ, HKV=HKV, D=D, num_warps=1,
    )
