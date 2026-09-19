"""Four real query positions sharing one B1 dense causal KV prefix.

The base position is a device scalar. Verification writes four physical KV
slots but never advances the logical position; host acceptance does that.
"""
import triton
import triton.language as tl


@triton.jit
def verify_qkv_rope_cache_kernel(
    QKV, QW, KW, COS, SIN, POS, Q, KC, VC,
    CAP: tl.constexpr, EPS: tl.constexpr,
    NQ: tl.constexpr = 32, NKV: tl.constexpr = 8,
    D: tl.constexpr = 128,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, D)
    rd = (d + D // 2) % D
    position = tl.load(POS) + row
    if head < NQ:
        base = row * (NQ + 2 * NKV) * D + head * D
        w = tl.load(QW + d).to(tl.float32)
        rw = tl.load(QW + rd).to(tl.float32)
    else:
        base = row * (NQ + 2 * NKV) * D + NQ * D + (head - NQ) * D
        w = tl.load(KW + d).to(tl.float32)
        rw = tl.load(KW + rd).to(tl.float32)
    x = tl.load(QKV + base + d).to(tl.float32)
    rx = tl.load(QKV + base + rd).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    x = ((x * inv).to(tl.bfloat16).to(tl.float32) * w).to(tl.bfloat16).to(tl.float32)
    rx = ((rx * inv).to(tl.bfloat16).to(tl.float32) * rw).to(tl.bfloat16).to(tl.float32)
    rx = tl.where(d < D // 2, -rx, rx)
    c = tl.load(COS + position * D + d).to(tl.float32)
    s = tl.load(SIN + position * D + d).to(tl.float32)
    # Preserve each eager BF16 product boundary before the final BF16 sum.
    a = (x * c).to(tl.bfloat16).to(tl.float32)
    z = (rx * s).to(tl.bfloat16).to(tl.float32)
    out = a + z
    if head < NQ:
        tl.store(Q + (row * NQ + head) * D + d, out)
    else:
        kv_head = head - NQ
        # All four time rows share sequence zero, NOT four separate batches.
        index = (kv_head * CAP + position) * D + d
        tl.store(KC + index, out)
        v = tl.load(QKV + row * (NQ + 2 * NKV) * D
                    + (NQ + NKV + kv_head) * D + d)
        tl.store(VC + index, v)


@triton.jit
def verify_attention_split_kernel(
    Q, K, V, POS, PART, PMAX, PSUM,
    CAP: tl.constexpr, SPLITS: tl.constexpr, SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr = 256, D: tl.constexpr = 128,
):
    kh = tl.program_id(0)
    split = tl.program_id(1)
    rows = tl.arange(0, 16)
    time = rows // 4
    head = kh * 4 + rows % 4
    d = tl.arange(0, D)
    t = split * BLOCK_N + tl.arange(0, BLOCK_N)
    position = tl.load(POS)
    valid = (t < CAP) & (t <= position + 3)
    q = tl.load(Q + ((time[:, None] * 32 + head[:, None]) * D
                    + d[None, :]))
    k = tl.load(K + (kh * CAP + t[None, :]) * D + d[:, None],
                valid[None, :], 0)
    score = tl.dot(q, k).to(tl.float32) * SCALE
    causal = valid[None, :] & (t[None, :] <= position + time[:, None])
    score = tl.where(causal, score, float('-inf'))
    maximum = tl.maximum(tl.max(score, 1), -1.0e30)
    probability = tl.exp(score - maximum[:, None])
    denom = tl.sum(probability, 1)
    v = tl.load(V + (kh * CAP + t[:, None]) * D + d[None, :],
                valid[:, None], 0)
    acc = tl.dot(probability.to(tl.bfloat16), v).to(tl.float32)
    # Existing merge sees [time, query_head], with one independent softmax each.
    output_head = time * 32 + head
    tl.store(PMAX + output_head * SPLITS + split, maximum)
    tl.store(PSUM + output_head * SPLITS + split, denom)
    tl.store(PART + (output_head[:, None] * SPLITS + split) * D
             + d[None, :], acc)
