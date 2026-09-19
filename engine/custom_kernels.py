"""Qwen3 BF16 fused operations and exact dense grouped decode attention.

All caller-owned buffers are contiguous CUDA tensors. Hidden states and weights
are BF16; attention split buffers are FP32; token IDs/position are int64.
Rounding boundaries intentionally match eager Transformers 4.51.3.
"""
import triton
import triton.language as tl


@triton.jit
def rms_kernel(X, W, Y, H: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    x = tl.load(X + row * H + d, d < H, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / H + EPS)
    n = (x * inv).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + d, d < H, 0).to(tl.float32)
    tl.store(Y + row * H + d, n * w, d < H)


@triton.jit
def embedding_norm_kernel(IDS, EMB, W, X, N, H: tl.constexpr,
                          EPS: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    token = tl.load(IDS + b)
    d = tl.arange(0, BLOCK)
    x = tl.load(EMB + token * H + d, d < H, 0).to(tl.float32)
    tl.store(X + b * H + d, x, d < H)
    inv = tl.rsqrt(tl.sum(x * x, 0) / H + EPS)
    n = (x * inv).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + d, d < H, 0).to(tl.float32)
    tl.store(N + b * H + d, n * w, d < H)


@triton.jit
def residual_norm_kernel(BRANCH, X, W, N, H: tl.constexpr,
                         EPS: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    x = tl.load(X + b * H + d, d < H, 0).to(tl.float32)
    branch = tl.load(BRANCH + b * H + d, d < H, 0).to(tl.float32)
    # The residual addition itself produces BF16, before RMS normalization.
    x = (x + branch).to(tl.bfloat16).to(tl.float32)
    tl.store(X + b * H + d, x, d < H)
    inv = tl.rsqrt(tl.sum(x * x, 0) / H + EPS)
    n = (x * inv).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + d, d < H, 0).to(tl.float32)
    tl.store(N + b * H + d, n * w, d < H)


@triton.jit
def qkv_rope_cache_kernel(QKV, QW, KW, COS, SIN, POS, Q, KC, VC,
                          CAP: tl.constexpr, EPS: tl.constexpr,
                          NQ: tl.constexpr = 32, NKV: tl.constexpr = 8,
                          D: tl.constexpr = 128):
    b = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, D)
    rd = (d + D // 2) % D
    pos = tl.load(POS)
    is_q = head < NQ
    if is_q:
        base = b * (NQ + 2 * NKV) * D + head * D
        w = tl.load(QW + d).to(tl.float32)
        rw = tl.load(QW + rd).to(tl.float32)
    else:
        base = b * (NQ + 2 * NKV) * D + NQ * D + (head - NQ) * D
        w = tl.load(KW + d).to(tl.float32)
        rw = tl.load(KW + rd).to(tl.float32)
    x = tl.load(QKV + base + d).to(tl.float32)
    rx = tl.load(QKV + base + rd).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    x = ((x * inv).to(tl.bfloat16).to(tl.float32) * w).to(tl.bfloat16).to(tl.float32)
    rx = ((rx * inv).to(tl.bfloat16).to(tl.float32) * rw).to(tl.bfloat16).to(tl.float32)
    rx = tl.where(d < D // 2, -rx, rx)
    c = tl.load(COS + pos * D + d).to(tl.float32)
    s = tl.load(SIN + pos * D + d).to(tl.float32)
    # q*cos and rotate_half(q)*sin are distinct BF16 eager operations.
    a = (x * c).to(tl.bfloat16).to(tl.float32)
    z = (rx * s).to(tl.bfloat16).to(tl.float32)
    out = a + z
    if is_q:
        tl.store(Q + (b * NQ + head) * D + d, out)
    else:
        kv_head = head - NQ
        idx = ((b * NKV + kv_head) * CAP + pos) * D + d
        tl.store(KC + idx, out)
        v = tl.load(QKV + b * (NQ + 2 * NKV) * D + (NQ + NKV + kv_head) * D + d)
        tl.store(VC + idx, v)


@triton.jit
def attention_split_kernel(Q, K, V, POS, PART, PMAX, PSUM,
                           CAP: tl.constexpr, SPLITS: tl.constexpr,
                           SCALE: tl.constexpr,
                           BLOCK_N: tl.constexpr = 256,
                           D: tl.constexpr = 128):
    # Each program shares one K/V tile across the four grouped query heads.
    b = tl.program_id(0)
    kh = tl.program_id(1)
    split = tl.program_id(2)
    qh = tl.arange(0, 16)
    d = tl.arange(0, D)
    t = split * BLOCK_N + tl.arange(0, BLOCK_N)
    pos = tl.load(POS)
    valid = (t < CAP) & (t <= pos)
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


@triton.jit
def attention_merge_kernel(PART, PMAX, PSUM, O, SPLITS: tl.constexpr,
                           BLOCK_S: tl.constexpr, D: tl.constexpr = 128):
    h = tl.program_id(0)
    s = tl.arange(0, BLOCK_S)
    d = tl.arange(0, D)
    m = tl.load(PMAX + h * SPLITS + s, s < SPLITS, float('-inf'))
    l = tl.load(PSUM + h * SPLITS + s, s < SPLITS, 0)
    maximum = tl.max(m, 0)
    factor = tl.exp(m - maximum)
    denom = tl.sum(l * factor, 0)
    part = tl.load(PART + (h * SPLITS + s[:, None]) * D + d[None, :],
                   s[:, None] < SPLITS, 0)
    result = tl.sum(part * factor[:, None], 0) / denom
    tl.store(O + h * D + d, result)


@triton.jit
def swiglu_kernel(GU, OUT, I: tl.constexpr, TOTAL: tl.constexpr,
                   BLOCK: tl.constexpr = 1024):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    b = x // I
    d = x % I
    gate = tl.load(GU + b * (2 * I) + d, x < TOTAL, 0).to(tl.float32)
    up = tl.load(GU + b * (2 * I) + I + d, x < TOTAL, 0).to(tl.float32)
    act = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + x, act * up, x < TOTAL)
