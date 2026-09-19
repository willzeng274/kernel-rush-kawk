"""Vector-length exact W1 kernels derived from the retained decode formula.

Each active root is necessarily consumed, so it writes its own main KV slot.
Fused attention excludes that slot from memory loads and supplies current K/V
from registers. Inactive members write no main cache and yield zero attention.
"""
import triton
import triton.language as tl


@triton.jit
def single_qkv_cache_kernel(QKV, QW, KW, COS, SIN, META, Q, SK, SV, K, V,
                     CAP: tl.constexpr, EPS: tl.constexpr,
                     D: tl.constexpr = 128):
    W: tl.constexpr = 1
    flat = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    b, row = flat // W, flat % W
    length = tl.load(META + b * (W + 2) + W).to(tl.int64)
    active = tl.load(META + b * (W + 2) + W + 1)
    valid = row < active
    # Never read past cos/sin even for inactive final rows of a finished member.
    position = tl.where(valid, length + row, 0)
    d = tl.arange(0, D)
    rd = (d + D // 2) % D
    if head < 32:
        base = flat * 6144 + head * D
        weight = tl.load(QW + d).to(tl.float32)
        rweight = tl.load(QW + rd).to(tl.float32)
    else:
        base = flat * 6144 + 4096 + (head - 32) * D
        weight = tl.load(KW + d).to(tl.float32)
        rweight = tl.load(KW + rd).to(tl.float32)
    x = tl.load(QKV + base + d).to(tl.float32)
    rx = tl.load(QKV + base + rd).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    x = ((x * inv).to(tl.bfloat16).to(tl.float32) * weight).to(tl.bfloat16).to(tl.float32)
    rx = ((rx * inv).to(tl.bfloat16).to(tl.float32) * rweight).to(tl.bfloat16).to(tl.float32)
    rx = tl.where(d < D // 2, -rx, rx)
    c = tl.load(COS + position * D + d).to(tl.float32)
    s = tl.load(SIN + position * D + d).to(tl.float32)
    a = (x * c).to(tl.bfloat16).to(tl.float32)
    z = (rx * s).to(tl.bfloat16).to(tl.float32)
    if head < 32:
        tl.store(Q + (flat * 32 + head) * D + d, tl.where(valid, a + z, 0))
    else:
        kh = head - 32
        offset = ((b * 8 + kh) * W + row) * D + d
        tl.store(SK + offset, tl.where(valid, a + z, 0))
        value = tl.load(QKV + flat * 6144 + 5120 + kh * D + d)
        tl.store(SV + offset, tl.where(valid, value, 0))
        main_offset = ((b * 8 + kh) * CAP + length) * D + d
        tl.store(K + main_offset, a + z, valid)
        tl.store(V + main_offset, value, valid)



@triton.jit
def single_fused_attention_kernel(QKV, QW, KW, COS, SIN, META, K, V, SK, SV, O,
                                  CAP: tl.constexpr, EPS: tl.constexpr,
                                  SCALE: tl.constexpr,
                                  BLOCK_N: tl.constexpr = 256,
                                  D: tl.constexpr = 128):
    b = tl.program_id(0).to(tl.int64)
    kh = tl.program_id(1).to(tl.int64)
    qh = tl.arange(0, 16)
    d = tl.arange(0, D)
    rd = (d + D // 2) % D
    active = tl.load(META + b * 3 + 2) > 0
    pos = tl.load(META + b * 3 + 1).to(tl.int64)
    safe_pos = tl.where(active, pos, 0)
    c = tl.load(COS + safe_pos * D + d).to(tl.float32)
    s = tl.load(SIN + safe_pos * D + d).to(tl.float32)

    # Every eager BF16 boundary in the accepted qkv_rope_cache_kernel remains.
    qbase = b * 6144 + (kh * 4 + qh[:, None]) * D
    qx = tl.load(QKV + qbase + d[None, :], qh[:, None] < 4, 0).to(tl.float32)
    qr = tl.load(QKV + qbase + rd[None, :], qh[:, None] < 4, 0).to(tl.float32)
    qw = tl.load(QW + d).to(tl.float32)
    qrw = tl.load(QW + rd).to(tl.float32)
    qi = tl.rsqrt(tl.sum(qx * qx, 1) / D + EPS)
    qx = ((qx * qi[:, None]).to(tl.bfloat16).to(tl.float32) * qw[None, :]).to(tl.bfloat16).to(tl.float32)
    qr = ((qr * qi[:, None]).to(tl.bfloat16).to(tl.float32) * qrw[None, :]).to(tl.bfloat16).to(tl.float32)
    qr = tl.where(d[None, :] < D // 2, -qr, qr)
    qa = (qx * c[None, :]).to(tl.bfloat16).to(tl.float32)
    qz = (qr * s[None, :]).to(tl.bfloat16).to(tl.float32)
    q = (qa + qz).to(tl.bfloat16)

    kbase = b * 6144 + 4096 + kh * D
    kx = tl.load(QKV + kbase + d).to(tl.float32)
    kr = tl.load(QKV + kbase + rd).to(tl.float32)
    kw = tl.load(KW + d).to(tl.float32)
    krw = tl.load(KW + rd).to(tl.float32)
    ki = tl.rsqrt(tl.sum(kx * kx, 0) / D + EPS)
    kx = ((kx * ki).to(tl.bfloat16).to(tl.float32) * kw).to(tl.bfloat16).to(tl.float32)
    kr = ((kr * ki).to(tl.bfloat16).to(tl.float32) * krw).to(tl.bfloat16).to(tl.float32)
    kr = tl.where(d < D // 2, -kr, kr)
    ka = (kx * c).to(tl.bfloat16).to(tl.float32)
    kz = (kr * s).to(tl.bfloat16).to(tl.float32)
    current_k = (ka + kz).to(tl.bfloat16)
    current_v = tl.load(QKV + b * 6144 + 5120 + kh * D + d)
    current_offset = ((b * 8 + kh) * CAP + pos) * D + d
    # The W1 root is always consumed; finished members never mutate main KV.
    tl.store(K + current_offset, current_k, active)
    tl.store(V + current_offset, current_v, active)
    # Keep a scratch copy for same-prefix all-layer validation only.
    scratch_offset = (b * 8 + kh) * D + d
    tl.store(SK + scratch_offset, tl.where(active, current_k, 0))
    tl.store(SV + scratch_offset, tl.where(active, current_v, 0))

    offsets = tl.arange(0, BLOCK_N)
    maximum = tl.full((16,), -1.0e30, tl.float32)
    denom = tl.zeros((16,), tl.float32)
    numerator = tl.zeros((16, D), tl.float32)
    for start in range(0, tl.where(active, tl.minimum(pos + 1, CAP), 0), BLOCK_N):
        t = start + offsets
        history = (t < CAP) & (t < pos)
        valid = (t < CAP) & (t <= pos)
        past_k = tl.load(K + ((b * 8 + kh) * CAP + t[None, :]) * D + d[:, None],
                         history[None, :], 0)
        k = tl.where(t[None, :] == pos, current_k[:, None], past_k)
        score = tl.dot(q, k).to(tl.float32) * SCALE
        score = tl.where(valid[None, :], score, float('-inf'))
        local_max = tl.maximum(tl.max(score, 1), -1.0e30)
        p = tl.exp(score - local_max[:, None])
        local_sum = tl.sum(p, 1)
        past_v = tl.load(V + ((b * 8 + kh) * CAP + t[:, None]) * D + d[None, :],
                         history[:, None], 0)
        v = tl.where(t[:, None] == pos, current_v[None, :], past_v)
        # Match each accepted 256-token split's probability rounding exactly.
        local_value = tl.dot(p.to(tl.bfloat16), v).to(tl.float32)
        next_max = tl.maximum(maximum, local_max)
        alpha = tl.exp(maximum - next_max)
        beta = tl.exp(local_max - next_max)
        numerator = numerator * alpha[:, None] + local_value * beta[:, None]
        denom = denom * alpha + local_sum * beta
        maximum = next_max
    h = b * 32 + kh * 4 + qh
    tl.store(O + h[:, None] * D + d[None, :],
             numerator / tl.maximum(denom[:, None], 1.0e-20), qh[:, None] < 4)



@triton.jit
def single_attention_split_kernel(Q, K, V, META, PART, PMAX, PSUM,
                           CAP: tl.constexpr, SPLITS: tl.constexpr,
                           SCALE: tl.constexpr,
                           BLOCK_N: tl.constexpr = 256,
                           D: tl.constexpr = 128):
    # Each program shares one K/V tile across the four grouped query heads.
    b = tl.program_id(0).to(tl.int64)
    kh = tl.program_id(1).to(tl.int64)
    split = tl.program_id(2).to(tl.int64)
    qh = tl.arange(0, 16)
    d = tl.arange(0, D)
    t = split * BLOCK_N + tl.arange(0, BLOCK_N)
    pos = tl.load(META + b * 3 + 1).to(tl.int64)
    active = tl.load(META + b * 3 + 2) > 0
    valid = (t < CAP) & (t <= pos) & active
    q = tl.load(Q + ((b * 32 + kh * 4 + qh[:, None]) * D + d[None, :]),
                qh[:, None] < 4, 0)
    k = tl.load(K + ((b * 8 + kh) * CAP + t[None, :]) * D + d[:, None],
                valid[None, :], 0)
    score = tl.dot(q, k).to(tl.float32) * SCALE
    score = tl.where(valid[None, :], score, float('-inf'))
    m = tl.maximum(tl.max(score, 1), -1.0e30)
    p = tl.exp(score - m[:, None])
    l = tl.sum(p, 1)
    v = tl.load(V + ((b * 8 + kh) * CAP + t[:, None]) * D + d[None, :],
                valid[:, None], 0)
    acc = tl.dot(p.to(tl.bfloat16), v).to(tl.float32)
    h = b * 32 + kh * 4 + qh
    tl.store(PMAX + h * SPLITS + split, m, qh < 4)
    tl.store(PSUM + h * SPLITS + split, l, qh < 4)
    tl.store(PART + (h[:, None] * SPLITS + split) * D + d[None, :],
             acc, qh[:, None] < 4)

