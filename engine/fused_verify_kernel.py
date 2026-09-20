import triton
import triton.language as tl

@triton.jit
def _rope_attn_verify_kernel(
    QKV, QN, KN, COS, SIN, K, V, LenB, Start,
    Acc, Lsum, Mmax, Out,
    sm_scale,
    stride_qkv_r, stride_cos_r,
    stride_ob, stride_oh,
    stride_kb, stride_kh, stride_ks,
    stride_ab, stride_ah, stride_as, stride_ag,
    stride_lb, stride_lh, stride_ls,
    N_Q: tl.constexpr, N_KV: tl.constexpr, G: tl.constexpr, NQ: tl.constexpr,
    GP: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr, EPS: tl.constexpr,
    BLOCK_N: tl.constexpr, CHUNK, SPLITS_ONE: tl.constexpr,
):
    """Speculative-verify attention that does its own QK-norm, rotary and
    cache writes: _flash_verify_split_kernel starting from the raw fused QKV
    rows, as _rope_attn_decode_kernel does for plain decode. Row b*NQ+j of
    QKV is query j of sequence b; its key/value go to slot len_b+j, stored by
    whichever split owns that slot before it attends. Arithmetic and
    rounding points match _qk_norm_rope_kv_kernel."""
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    b = pid_bh // N_KV
    h = pid_bh % N_KV
    len_b = tl.load(LenB + b)
    start = tl.load(Start + b)
    lo0 = pid_s * CHUNK
    hi = tl.minimum(lo0 + CHUNK, len_b + NQ)
    lo = tl.maximum(lo0, start)

    cols = tl.arange(0, D)
    idx = tl.where(cols < HALF, cols + HALF, cols - HALF)
    kbase = K + b * stride_kb + h * stride_kh
    vbase = V + b * stride_kb + h * stride_kh
    wk = tl.load(KN + cols)
    wkp = tl.load(KN + idx)
    for jj in tl.static_range(NQ):
        slot = len_b + jj
        if (slot >= lo0) & (slot < lo0 + CHUNK):
            r = QKV + (b * NQ + jj) * stride_qkv_r
            c_ = tl.load(COS + (b * NQ + jj) * stride_cos_r + cols)
            s_ = tl.load(SIN + (b * NQ + jj) * stride_cos_r + cols)
            xk = tl.load(r + (N_Q + h) * D + cols).to(tl.float32)
            xkp = tl.load(r + (N_Q + h) * D + idx).to(tl.float32)
            rk = tl.rsqrt(tl.sum(xk * xk, axis=0) / D + EPS)
            kn_ = (xk * rk).to(tl.bfloat16) * wk
            kpn = (xkp * rk).to(tl.bfloat16) * wkp
            krot = tl.where(cols < HALF, -kpn, kpn)
            tl.store(kbase + slot * stride_ks + cols, (kn_ * c_) + (krot * s_))
            tl.store(vbase + slot * stride_ks + cols, tl.load(r + (N_Q + N_KV + h) * D + cols))

    offs_qi = tl.arange(0, GP)
    qmask = offs_qi < NQ * G
    j = offs_qi // G
    g = offs_qi % G
    qrow = b * NQ + j
    qlen = len_b + j + 1
    qb = QKV + qrow[:, None] * stride_qkv_r + ((h * G + g) * D)[:, None]
    xq = tl.load(qb + cols[None, :], mask=qmask[:, None], other=0.0).to(tl.float32)
    xqp = tl.load(qb + idx[None, :], mask=qmask[:, None], other=0.0).to(tl.float32)
    rq = tl.rsqrt(tl.sum(xq * xq, axis=1) / D + EPS)
    cq = tl.load(COS + qrow[:, None] * stride_cos_r + cols[None, :], mask=qmask[:, None], other=0.0)
    sq = tl.load(SIN + qrow[:, None] * stride_cos_r + cols[None, :], mask=qmask[:, None], other=0.0)
    qn_ = (xq * rq[:, None]).to(tl.bfloat16) * tl.load(QN + cols)[None, :]
    qpn = (xqp * rq[:, None]).to(tl.bfloat16) * tl.load(QN + idx)[None, :]
    qrot = tl.where(cols[None, :] < HALF, -qpn, qpn)
    q = tl.where(qmask[:, None], (qn_ * cq) + (qrot * sq), 0.0).to(tl.bfloat16)
    # the loop below streams the slots just written: every warp must see the
    # stores before any warp issues those loads
    tl.debug_barrier()

    m_i = tl.full([GP], -1e30, tl.float32)
    l_i = tl.zeros([GP], tl.float32)
    acc = tl.zeros([GP, D], tl.float32)
    for n0 in range(lo, hi, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        nmask = offs_n < hi
        k = tl.load(kbase + offs_n[:, None] * stride_ks + cols[None, :], mask=nmask[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        valid = nmask[None, :] & (offs_n[None, :] < qlen[:, None])
        qk = tl.where(valid, qk, -1e30)
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        v = tl.load(vbase + offs_n[:, None] * stride_ks + cols[None, :], mask=nmask[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p, v.to(tl.float32))
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new

    if SPLITS_ONE:
        tl.store(Out + qrow[:, None] * stride_ob + (h * G + g)[:, None] * stride_oh + cols[None, :],
                 (acc / l_i[:, None]).to(tl.bfloat16), mask=qmask[:, None])
    else:
        aptr = Acc + b * stride_ab + h * stride_ah + pid_s * stride_as
        tl.store(aptr + offs_qi[:, None] * stride_ag + cols[None, :], acc, mask=qmask[:, None])
        lptr = Lsum + b * stride_lb + h * stride_lh + pid_s * stride_ls
        mptr = Mmax + b * stride_lb + h * stride_lh + pid_s * stride_ls
        tl.store(lptr + offs_qi, l_i, mask=qmask)
        tl.store(mptr + offs_qi, m_i, mask=qmask)
