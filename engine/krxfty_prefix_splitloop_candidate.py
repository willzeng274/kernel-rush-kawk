import triton
import triton.language as tl

@triton.jit
def _tree_tile(q, k_ptr, v_ptr, kv_base, n0, end, pos, L, tree, m, l, acc, scale,
               D: tl.constexpr, BLOCK_N: tl.constexpr, MASK_TREE: tl.constexpr):
    """One unchanged attention recurrence; the tree mask is compile-time optional."""
    d = tl.arange(0, D)
    n = n0 + tl.arange(0, BLOCK_N)
    kmask = n < end
    k = tl.load(k_ptr + (kv_base + n[:, None]) * D + d[None, :], mask=kmask[:, None], other=0.0)
    sc = tl.dot(q, tl.trans(k)) * scale
    if MASK_TREE:
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
    acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
    return m_new, l, acc


@triton.jit
def _split_kernel(
    q_ptr, k_ptr, v_ptr, pos_ptr, tree_ptr, o_part_ptr, m_part_ptr, l_part_ptr, o_ptr,
    CAP, scale, NSPLIT, SPLIT_LEN,
    HQ: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, GP: tl.constexpr, R: tl.constexpr,
    D: tl.constexpr, BLOCK_N: tl.constexpr,
    FINAL: tl.constexpr, TREE: tl.constexpr,
):
    """Query tile rows are (t, g): query row t of the sequence, head kh*G + g.
    Without TREE, row t may see keys 0 .. pos[b] + t (causal inside the block
    of R new tokens). With TREE, keys before pos are always visible and key
    pos + j is visible to row t iff bit j of tree[t] is set: an ancestor mask,
    so a whole draft tree is verified in one pass. The running max is guarded
    so rows with no valid key in a split stay at (m=-inf, l=0, acc=0)."""
    b = tl.program_id(0)
    kh = tl.program_id(1)
    s = tl.program_id(2) % NSPLIT
    rb = tl.program_id(2) // NSPLIT
    pos = tl.load(pos_ptr + b)
    L = pos + R
    start = s * SPLIT_LEN
    end = tl.minimum(start + SPLIT_LEN, L)

    rows = rb * GP + tl.arange(0, GP)
    t = rows // G
    g = rows % G
    head = kh * G + g
    d = tl.arange(0, D)
    row_mask = rows < G * R
    q = tl.load(
        q_ptr + (((b * HQ + head[:, None]) * R + t[:, None]) * D + d[None, :]),
        mask=row_mask[:, None], other=0.0,
    )
    kv_base = (b * HKV + kh) * CAP
    row_limit = pos + t
    if TREE:
        tree = tl.load(tree_ptr + t, mask=row_mask, other=0)

    m = tl.full([GP], float("-inf"), tl.float32)
    l = tl.zeros([GP], tl.float32)
    acc = tl.zeros([GP, D], tl.float32)
    if TREE:
        # Keep each loop straight-line so the compiler can pipeline its K/V
        # loads. Both bounds retain the original BLOCK_N-aligned tile order.
        prefix_end = tl.minimum(end, (pos // BLOCK_N) * BLOCK_N)
        for n0 in range(start, prefix_end, BLOCK_N):
            m, l, acc = _tree_tile(q, k_ptr, v_ptr, kv_base, n0, end, pos, L, tree, m, l, acc, scale,
                                    D, BLOCK_N, MASK_TREE=False)
        tail_start = tl.maximum(start, prefix_end)
        for n0 in range(tail_start, end, BLOCK_N):
            m, l, acc = _tree_tile(q, k_ptr, v_ptr, kv_base, n0, end, pos, L, tree, m, l, acc, scale,
                                    D, BLOCK_N, MASK_TREE=True)
    else:
        for n0 in range(start, end, BLOCK_N):
            n = n0 + tl.arange(0, BLOCK_N)
            kmask = n < end
            k = tl.load(k_ptr + (kv_base + n[:, None]) * D + d[None, :], mask=kmask[:, None], other=0.0)
            sc = tl.dot(q, tl.trans(k)) * scale
            if TREE:
                rel = n[None, :] - pos
                sh = tl.minimum(tl.maximum(rel, 0), 63).to(tl.int64)
                allowed = (rel < 0) | (((tree[:, None] >> sh) & 1) == 1)
                sc = tl.where(allowed & (n[None, :] < L), sc, float("-inf"))
            else:
                sc = tl.where(n[None, :] <= row_limit[:, None], sc, float("-inf"))
            m_new = tl.maximum(m, tl.max(sc, axis=1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.exp(m - m_safe)
            p = tl.exp(sc - m_safe[:, None])
            l = l * alpha + tl.sum(p, axis=1)
            v = tl.load(v_ptr + (kv_base + n[:, None]) * D + d[None, :], mask=kmask[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
            m = m_new

    if FINAL:
        out = acc / l[:, None]
        tl.store(o_ptr + ((b * R + t[:, None]) * HQ + head[:, None]) * D + d[None, :], out.to(tl.bfloat16), mask=row_mask[:, None])
    else:
        part = ((b * HQ + head) * R + t) * NSPLIT + s
        tl.store(o_part_ptr + part[:, None] * D + d[None, :], acc, mask=row_mask[:, None])
        tl.store(m_part_ptr + part, m, mask=row_mask)
        tl.store(l_part_ptr + part, l, mask=row_mask)
