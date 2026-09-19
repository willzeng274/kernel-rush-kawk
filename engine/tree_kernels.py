"""Exact eight-node two-continuation tree with independent sequence lengths.

K/V main [B,8,CAP,128], scratch [B,8,W,128]; flattened activations [B*W,...].
META rows contain [W tokens, consumed length, active node bitmask]. All pointer
products for cache/position use int64. W=8; each attention CTA owns four GQA heads times eight tree nodes,
32 query rows, independent of batch. No M64 MMA tile is used.
"""
import triton
import triton.language as tl


@triton.jit
def tree_qkv_kernel(QKV, QW, KW, COS, SIN, META, Q, SK, SV,
                     CAP: tl.constexpr, W: tl.constexpr, EPS: tl.constexpr,
                     D: tl.constexpr = 128):
    flat = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    b, row = flat // W, flat % W
    length = tl.load(META + b * (W + 2) + W).to(tl.int64)
    active = tl.load(META + b * (W + 2) + W + 1)
    valid = (active & (1 << row)) != 0
    depth = tl.where(row < 5, row, row - 4)
    # Never read past cos/sin even for inactive final rows of a finished member.
    position = tl.where(valid, length + depth, 0)
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


@triton.jit
def tree_attention_kernel(Q, K, V, SK, SV, META, PART, PMAX, PSUM,
                           CAP: tl.constexpr, W: tl.constexpr,
                           SPLITS: tl.constexpr, SCALE: tl.constexpr,
                           BLOCK_N: tl.constexpr = 256, D: tl.constexpr = 128):
    b = tl.program_id(0).to(tl.int64)
    kh = tl.program_id(1).to(tl.int64)
    split = tl.program_id(2).to(tl.int64)
    length = tl.load(META + b * (W + 2) + W).to(tl.int64)
    active = tl.load(META + b * (W + 2) + W + 1)
    r = tl.arange(0, 32)
    row, head = r // 4, kh * 4 + r % 4
    ancestry = tl.where(row < 5, (1 << (row + 1)) - 1,
                        tl.where(row == 5, 33, tl.where(row == 6, 97, 225)))
    d = tl.arange(0, D)
    t = split * BLOCK_N + tl.arange(0, BLOCK_N)
    scratch_row = t - length
    committed = (t < length) & (t < CAP) & (active != 0)
    safe_node = tl.maximum(0, tl.minimum(scratch_row, W - 1))
    scratch = ((scratch_row >= 0) & (scratch_row < W)
               & ((active & (1 << safe_node)) != 0))
    q = tl.load(Q + (((b * W + row[:, None]) * 32 + head[:, None]) * D + d[None, :]),
                row[:, None] < W, 0)
    # Select an address before loading so only one K tile is live. Loading
    # both prefix and scratch tiles then selecting values spills on sm90.
    kbase = tl.where(committed,
                     K + ((b * 8 + kh) * CAP + t) * D,
                     SK + ((b * 8 + kh) * W + scratch_row) * D)
    k = tl.load(kbase[None, :] + d[:, None], (committed | scratch)[None, :], 0)
    score = tl.dot(q, k).to(tl.float32) * SCALE
    visible = (((active & (1 << row[:, None])) != 0)
               & (committed[None, :] | (scratch[None, :]
                  & ((ancestry[:, None] & (1 << safe_node[None, :])) != 0))))
    score = tl.where(visible, score, float('-inf'))
    maximum = tl.maximum(tl.max(score, 1), -1.0e30)
    p = tl.exp(score - maximum[:, None])
    denominator = tl.sum(p, 1)
    vbase = tl.where(committed,
                     V + ((b * 8 + kh) * CAP + t) * D,
                     SV + ((b * 8 + kh) * W + scratch_row) * D)
    v = tl.load(vbase[:, None] + d[None, :], (committed | scratch)[:, None], 0)
    numerator = tl.dot(p.to(tl.bfloat16), v).to(tl.float32)
    outhead = (b * W + row) * 32 + head
    tl.store(PMAX + outhead * SPLITS + split, maximum, row < W)
    tl.store(PSUM + outhead * SPLITS + split, denominator, row < W)
    tl.store(PART + (outhead[:, None] * SPLITS + split) * D + d[None, :],
             numerator, row[:, None] < W)


@triton.jit
def tree_compact_kernel(SK, SV, K, V, META, PATHS,
                         CAP: tl.constexpr, W: tl.constexpr, D: tl.constexpr = 128):
    b = tl.program_id(0).to(tl.int64)
    kh = tl.program_id(1).to(tl.int64)
    row = tl.program_id(2).to(tl.int64)
    length = tl.load(META + b * (W + 2) + W).to(tl.int64)
    node = tl.load(PATHS + b * W + row).to(tl.int64)
    d = tl.arange(0, D)
    valid = (node >= 0) & (node < W) & (length + row < CAP)
    source = ((b * 8 + kh) * W + node) * D + d
    target = ((b * 8 + kh) * CAP + length + row) * D + d
    key = tl.load(SK + source, valid, 0)
    value = tl.load(SV + source, valid, 0)
    tl.store(K + target, key, valid)
    tl.store(V + target, value, valid)


