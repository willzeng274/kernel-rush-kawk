"""SOURCE-ONLY proposal for exact91's selected, two-split small-tree attention.

No host selector, retuning, compiler results, or GPU validation is included.
Requires the immutable completed QKV projection and serialized same-stream use
of the incumbent partial buffers. Current-tree cache values are never read.
"""

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
def _prefix_tile(q, k_ptr, v_ptr, kv_base, n0, end, pos, m, l, acc, scale,
                 D: tl.constexpr, BLOCK_N: tl.constexpr):
    # The incumbent's straight-line prefix recurrence. n < pos is explicit on
    # every cache load, including this already-prefix-only loop.
    d = tl.arange(0, D)
    n = n0 + tl.arange(0, BLOCK_N)
    kmask = (n < end) & (n < pos)
    k = tl.load(k_ptr + (kv_base + n[:, None]) * D + d[None, :], mask=kmask[:, None], other=0.0)
    sc = tl.dot(q, tl.trans(k)) * scale
    m_new = tl.maximum(m, tl.max(sc, axis=1))
    m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
    alpha = tl.exp(m - m_safe)
    p = tl.exp(sc - m_safe[:, None])
    l = l * alpha + tl.sum(p, axis=1)
    v = tl.load(v_ptr + (kv_base + n[:, None]) * D + d[None, :], mask=kmask[:, None], other=0.0)
    acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
    return m_new, l, acc


@triton.jit
def _current_tile(
    q, qkv_ptr, kw_ptr, cos_ptr, sin_ptr, depth_ptr, k_ptr, v_ptr,
    b, kh, kv_base, n0, end, pos, L, tree, m, l, acc, scale, eps,
    ROW: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr,
    R: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr,
):
    d = tl.arange(0, D)
    dh = tl.arange(0, D // 2)
    n = n0 + tl.arange(0, BLOCK_N)
    kmask = (n < end) & (n < pos)
    k = tl.load(k_ptr + (kv_base + n[:, None]) * D + d[None, :], mask=kmask[:, None], other=0.0)

    # At most eight scalar guards. Only the split/tile containing pos+j
    # computes/publishes node j; row_blocks==1 makes that owner unique.
    # Each norm materializes a single 128-value head, not BLOCK_N FP32 heads.
    for j in tl.static_range(R):
        slot = pos + j
        if (slot >= n0) & (slot < n0 + BLOCK_N) & (slot < end):
            rotary = pos + tl.load(depth_ptr + j)
            src = qkv_ptr + (b * R + j) * ROW + (HQ + kh) * D
            x1 = tl.load(src + dh).to(tl.float32)
            x2 = tl.load(src + D // 2 + dh).to(tl.float32)
            w1 = tl.load(kw_ptr + dh).to(tl.float32)
            w2 = tl.load(kw_ptr + D // 2 + dh).to(tl.float32)
            tab = rotary * D
            cos1 = tl.load(cos_ptr + tab + dh).to(tl.float32)
            cos2 = tl.load(cos_ptr + tab + D // 2 + dh).to(tl.float32)
            sin1 = tl.load(sin_ptr + tab + dh).to(tl.float32)
            sin2 = tl.load(sin_ptr + tab + D // 2 + dh).to(tl.float32)
            k1, k2 = _norm_rope_row(x1, x2, w1, w2, cos1, cos2, sin1, sin2, eps, D)
            # join is interleaved [64,2]; transpose restores half-major [2,64].
            current_k = tl.reshape(tl.trans(tl.join(k1, k2)), (D,), can_reorder=False)
            k = tl.where(n[:, None] == slot, current_k[None, :], k)
            dst = (kv_base + slot) * D
            tl.store(k_ptr + dst + dh, k1)
            tl.store(k_ptr + dst + D // 2 + dh, k2)

    # Exactly the incumbent masked-tail tile and online-softmax recurrence.
    # Do not peel current nodes into an extra attention update.
    sc = tl.dot(q, tl.trans(k)) * scale
    rel = n[None, :] - pos
    sh = tl.minimum(tl.maximum(rel, 0), 63).to(tl.int64)
    allowed = (rel < 0) | (((tree[:, None] >> sh) & 1) == 1)
    sc = tl.where(allowed & (n[None, :] < L), sc, float("-inf"))
    m_new = tl.maximum(m, tl.max(sc, axis=1))
    m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
    alpha = tl.exp(m - m_safe)
    p = tl.exp(sc - m_safe[:, None])
    l = l * alpha + tl.sum(p, axis=1)

    v = tl.load(v_ptr + (kv_base + n[:, None]) * D + d[None, :], mask=kmask[:, None], other=0.0)
    current_mask = (n >= pos) & (n < end) & (n < L)
    # Pointer arithmetic gathers raw V directly; no tl.gather API is needed.
    vsrc = qkv_ptr + (b * R + n[:, None] - pos) * ROW + (HQ + HKV + kh) * D + d[None, :]
    current_v = tl.load(vsrc, mask=current_mask[:, None], other=0.0)
    v = tl.where(current_mask[:, None], current_v, v)
    tl.store(v_ptr + (kv_base + n[:, None]) * D + d[None, :], current_v, mask=current_mask[:, None])
    acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
    return m_new, l, acc


@triton.jit
def _fused_tree_split_kernel(
    qkv_ptr, qw_ptr, kw_ptr, cos_ptr, sin_ptr, pos_ptr, depth_ptr, tree_ptr,
    k_ptr, v_ptr, o_part_ptr, m_part_ptr, l_part_ptr,
    CAP, scale, NSPLIT, SPLIT_LEN, eps,
    ROW: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr,
    G: tl.constexpr, GP: tl.constexpr, R: tl.constexpr,
    D: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Separate assertions avoid Triton 3.1 chained-boolean AST limitations.
    tl.static_assert(HQ == 32)
    tl.static_assert(HKV == 8)
    tl.static_assert(G == 4)
    tl.static_assert(D == 128)
    tl.static_assert(R >= 4)
    tl.static_assert(R <= 8)
    tl.static_assert((R & (R - 1)) == 0)
    tl.static_assert(GP == G * R)
    b = tl.program_id(0)
    kh = tl.program_id(1)
    s = tl.program_id(2)
    pos = tl.load(pos_ptr + b)
    L = pos + R
    start = s * SPLIT_LEN
    end = tl.minimum(start + SPLIT_LEN, L)
    rows = tl.arange(0, GP)
    t = rows // G
    g = rows % G
    head = kh * G + g
    d = tl.arange(0, D)
    dh = tl.arange(0, D // 2)
    row_mask = rows < G * R

    # [64,GP] lets the exact scalar-head helper reduce axis 0, separately for
    # both halves. Each column is one (tree-row, query-head) pair.
    rotary = pos + tl.load(depth_ptr + t, mask=row_mask, other=0)
    src = qkv_ptr + (b * R + t[None, :]) * ROW + head[None, :] * D
    x1 = tl.load(src + dh[:, None], mask=row_mask[None, :], other=0.0).to(tl.float32)
    x2 = tl.load(src + D // 2 + dh[:, None], mask=row_mask[None, :], other=0.0).to(tl.float32)
    w1 = tl.load(qw_ptr + dh[:, None]).to(tl.float32)
    w2 = tl.load(qw_ptr + D // 2 + dh[:, None]).to(tl.float32)
    tab = rotary[None, :] * D
    cos1 = tl.load(cos_ptr + tab + dh[:, None], mask=row_mask[None, :], other=0.0).to(tl.float32)
    cos2 = tl.load(cos_ptr + tab + D // 2 + dh[:, None], mask=row_mask[None, :], other=0.0).to(tl.float32)
    sin1 = tl.load(sin_ptr + tab + dh[:, None], mask=row_mask[None, :], other=0.0).to(tl.float32)
    sin2 = tl.load(sin_ptr + tab + D // 2 + dh[:, None], mask=row_mask[None, :], other=0.0).to(tl.float32)
    q1, q2 = _norm_rope_row(x1, x2, w1, w2, cos1, cos2, sin1, sin2, eps, D)
    q = tl.reshape(tl.permute(tl.join(q1, q2), (1, 2, 0)), (GP, D), can_reorder=False)
    kv_base = (b * HKV + kh) * CAP
    tree = tl.load(tree_ptr + t, mask=row_mask, other=0)
    m = tl.full([GP], float("-inf"), tl.float32)
    l = tl.zeros([GP], tl.float32)
    acc = tl.zeros([GP, D], tl.float32)

    # Exact91 S2 loop bounds and tile order, including the original mixed tile.
    prefix_end = tl.minimum(end, (pos // BLOCK_N) * BLOCK_N)
    for n0 in range(start, prefix_end, BLOCK_N):
        m, l, acc = _prefix_tile(q, k_ptr, v_ptr, kv_base, n0, end, pos, m, l, acc, scale, D, BLOCK_N)
    tail_start = tl.maximum(start, prefix_end)
    for n0 in range(tail_start, end, BLOCK_N):
        m, l, acc = _current_tile(
            q, qkv_ptr, kw_ptr, cos_ptr, sin_ptr, depth_ptr, k_ptr, v_ptr,
            b, kh, kv_base, n0, end, pos, L, tree, m, l, acc, scale, eps,
            ROW, HQ, HKV, R, D, BLOCK_N,
        )
    part = ((b * HQ + head) * R + t) * NSPLIT + s
    tl.store(o_part_ptr + part[:, None] * D + d[None, :], acc, mask=row_mask[:, None])
    tl.store(m_part_ptr + part, m, mask=row_mask)
    tl.store(l_part_ptr + part, l, mask=row_mask)
