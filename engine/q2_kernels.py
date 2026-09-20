"""Full-prefix Q2 attention and request-local device control; no host timing."""
import triton
import triton.language as tl


@triton.jit
def q2_draft(HIST, LENGTH, IDS, INPUT, CAP: tl.constexpr, BLOCK: tl.constexpr):
    ends = tl.arange(0, BLOCK)
    length = tl.load(LENGTH)
    pending = tl.load(IDS)
    selected = tl.full((), -1, tl.int32)
    # Preserve the retained 8/4/2 rule AND three observed successors. Only
    # the first proposed successor is used; no rejected value is indexed.
    for size in tl.static_range(3):
        n = 8 if size == 0 else (4 if size == 1 else 2)
        match = (ends >= n) & (ends + 3 <= length) & (ends < CAP)
        for j in range(n):
            left = tl.load(HIST + ends - n + j, match, other=-1)
            right = tl.load(HIST + length - n + j, length >= n, other=-2)
            match = match & (left == right)
        found = tl.max(tl.where(match, ends, -1), 0)
        selected = tl.where(selected >= 0, selected, found)
    draft = tl.load(HIST + selected, selected >= 0, other=0)
    draft = tl.where(selected >= 0, draft, pending)
    tl.store(INPUT, pending)
    tl.store(INPUT + 1, draft)


@triton.jit
def q2_accept(INPUT, PRED, HIST, LENGTH, POS, IDS, OUT, COUNT, STEP):
    step = tl.load(STEP)
    length = tl.load(LENGTH)
    p0 = tl.load(PRED)
    p1 = tl.load(PRED + 1)
    accepted = tl.load(INPUT + 1) == p0
    advance = 1 + accepted.to(tl.int32)
    tl.store(HIST + length, p0)
    tl.store(HIST + length + 1, p1, accepted)
    tl.store(OUT + step * 6, p0)
    tl.store(OUT + step * 6 + 1, p1, accepted)
    tl.store(COUNT + step, advance + 4)
    tl.store(LENGTH, length + advance)
    tl.store(POS, tl.load(POS) + advance)
    tl.store(IDS, tl.where(accepted, p1, p0))


@triton.jit
def q2_append(HIST, LENGTH, IDS, OUT, COUNT, STEP, INDEX: tl.constexpr):
    step = tl.load(STEP)
    length = tl.load(LENGTH)
    token = tl.load(IDS)
    advance = tl.load(COUNT + step) - 4
    tl.store(HIST + length, token)
    tl.store(OUT + step * 6 + advance + INDEX, token)
    tl.store(LENGTH, length + 1)
    if INDEX == 3:
        tl.store(STEP, step + 1)


@triton.jit
def q2_attention_split(
    Q, K, V, POS, PART, PMAX, PSUM,
    CAP: tl.constexpr, SPLITS: tl.constexpr, SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr = 256, D: tl.constexpr = 128,
):
    kh = tl.program_id(0)
    split = tl.program_id(1)
    rows = tl.arange(0, 16)
    time = rows // 4
    head = kh * 4 + rows % 4
    active = time < 2
    d = tl.arange(0, D)
    t = split * BLOCK_N + tl.arange(0, BLOCK_N)
    position = tl.load(POS)
    valid = (t < CAP) & (t <= position + 1)
    q = tl.load(Q + ((time[:, None] * 32 + head[:, None]) * D
                    + d[None, :]), active[:, None], other=0)
    k = tl.load(K + (kh * CAP + t[None, :]) * D + d[:, None],
                valid[None, :], other=0)
    score = tl.dot(q, k).to(tl.float32) * SCALE
    causal = active[:, None] & valid[None, :] & (t[None, :] <= position + time[:, None])
    score = tl.where(causal, score, float('-inf'))
    maximum = tl.maximum(tl.max(score, 1), -1.0e30)
    probability = tl.exp(score - maximum[:, None])
    denom = tl.sum(probability, 1)
    v = tl.load(V + (kh * CAP + t[:, None]) * D + d[None, :],
                valid[:, None], other=0)
    acc = tl.dot(probability.to(tl.bfloat16), v).to(tl.float32)
    output_head = time * 32 + head
    tl.store(PMAX + output_head * SPLITS + split, maximum, active)
    tl.store(PSUM + output_head * SPLITS + split, denom, active)
    tl.store(PART + (output_head[:, None] * SPLITS + split) * D
             + d[None, :], acc, active[:, None])
