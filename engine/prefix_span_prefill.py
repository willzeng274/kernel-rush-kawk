"""Span Q/K norm, exact-boundary RoPE, and static-cache KV writes.

QKV [B*SPAN,6144] and query output [B*SPAN,4096] are contiguous BF16.
K/V are BF16 [B,8,CAP,128]; cos/sin are BF16 [CAP,128].
Each program processes four rows for a single Q or K head. START is the
absolute position of the span; the existing full-prefill call defaults to 0.
"""
import triton
import triton.language as tl


@triton.jit
def prefill_qkv_rope_cache_kernel(
    QKV, QW, KW, COS, SIN, Q, KC, VC,
    ROWS: tl.constexpr, SPAN: tl.constexpr, CAP: tl.constexpr,
    EPS: tl.constexpr, R: tl.constexpr = 4,
    NQ: tl.constexpr = 32, NKV: tl.constexpr = 8,
    D: tl.constexpr = 128, START: tl.constexpr = 0,
):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    head = tl.program_id(1)
    d = tl.arange(0, D)
    rd = (d + D // 2) % D
    valid = rows < ROWS
    batch = rows // SPAN
    position = START + rows % SPAN
    if head < NQ:
        base = rows * ((NQ + 2 * NKV) * D) + head * D
        w = tl.load(QW + d).to(tl.float32)
        rw = tl.load(QW + rd).to(tl.float32)
    else:
        base = rows * ((NQ + 2 * NKV) * D) + NQ * D + (head - NQ) * D
        w = tl.load(KW + d).to(tl.float32)
        rw = tl.load(KW + rd).to(tl.float32)
    x = tl.load(QKV + base[:, None] + d[None, :],
                valid[:, None], 0).to(tl.float32)
    rx = tl.load(QKV + base[:, None] + rd[None, :],
                 valid[:, None], 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 1) / D + EPS)
    # Match native RMSNorm's cast before gain multiplication, then round the
    # gain product to BF16 before entering the separate rotary operations.
    x = ((x * inv[:, None]).to(tl.bfloat16).to(tl.float32)
         * w[None, :]).to(tl.bfloat16).to(tl.float32)
    rx = ((rx * inv[:, None]).to(tl.bfloat16).to(tl.float32)
          * rw[None, :]).to(tl.bfloat16).to(tl.float32)
    rx = tl.where(d[None, :] < D // 2, -rx, rx)
    c = tl.load(COS + position[:, None] * D + d[None, :],
                valid[:, None], 0).to(tl.float32)
    s = tl.load(SIN + position[:, None] * D + d[None, :],
                valid[:, None], 0).to(tl.float32)
    a = (x * c).to(tl.bfloat16).to(tl.float32)
    z = (rx * s).to(tl.bfloat16).to(tl.float32)
    out = a + z
    if head < NQ:
        tl.store(Q + (rows[:, None] * NQ + head) * D + d[None, :],
                 out, valid[:, None])
    else:
        kv_head = head - NQ
        index = ((batch[:, None] * NKV + kv_head) * CAP
                 + position[:, None]) * D + d[None, :]
        tl.store(KC + index, out, valid[:, None])
        vbase = (rows * ((NQ + 2 * NKV) * D)
                 + (NQ + NKV + kv_head) * D)
        v = tl.load(QKV + vbase[:, None] + d[None, :], valid[:, None], 0)
        tl.store(VC + index, v, valid[:, None])
