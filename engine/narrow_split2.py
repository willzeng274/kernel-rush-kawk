"""Research-only two-split output/down producer derived from our #63 MMA.

New: tile-aligned split axis and FP32 [2, B, 2560] output; no BF16 partials.
Unchanged configuration: #63 narrow_config output/down tile/warp/stage choices.
The consumer below is exact function text from our submitted #41, not new work.
No production adapter, allocation, model state, selector or GPU result is here.
"""
import triton
import triton.language as tl


@triton.jit
def narrow_split2_projection(X, WEIGHT, PART,
                             ROWS: tl.constexpr, CHANNELS: tl.constexpr,
                             INNER: tl.constexpr, TILE_ROWS: tl.constexpr,
                             TILE_CHANNELS: tl.constexpr, TILE_INNER: tl.constexpr):
    rows = tl.arange(0, TILE_ROWS)
    channels = tl.program_id(0) * TILE_CHANNELS + tl.arange(0, TILE_CHANNELS)
    reduction = tl.arange(0, TILE_INNER)
    split = tl.program_id(1)
    steps = triton.cdiv(INNER, 2 * TILE_INNER)
    product = tl.full((TILE_ROWS, TILE_CHANNELS), 0, tl.float32)
    for offset in range(steps):
        inner = (split * steps + offset) * TILE_INNER + reduction
        left = tl.load(X + rows[:, None] * INNER + inner[None, :],
                       (rows[:, None] < ROWS) & (inner[None, :] < INNER), 0)
        right = tl.load(WEIGHT + channels[None, :] * INNER + inner[:, None],
                        (channels[None, :] < CHANNELS) & (inner[:, None] < INNER), 0)
        product = tl.dot(left, right, product)
    tl.store(PART + (split * ROWS + rows[:, None]) * CHANNELS + channels[None, :],
             product, (rows[:, None] < ROWS) & (channels[None, :] < CHANNELS))


@triton.jit
def _merge_residual_norm(PART, X, GAIN, NORMALIZED,
                         B: tl.constexpr, H: tl.constexpr, SPLITS: tl.constexpr,
                         EPS: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    branch = tl.full((BLOCK,), 0, tl.float32)
    # Identical per-element sequential FP32 order to _hopper_merge.
    for split in tl.static_range(SPLITS):
        branch += tl.load(PART + (split * B + b) * H + d, d < H, 0)
    branch = branch.to(tl.bfloat16).to(tl.float32)
    x = tl.load(X + b * H + d, d < H, 0).to(tl.float32)
    x = (x + branch).to(tl.bfloat16).to(tl.float32)
    tl.store(X + b * H + d, x, d < H)
    inv = tl.rsqrt(tl.sum(x * x, 0) / H + EPS)
    n = (x * inv).to(tl.bfloat16).to(tl.float32)
    gain = tl.load(GAIN + d, d < H, 0).to(tl.float32)
    tl.store(NORMALIZED + b * H + d, n * gain, d < H)


def producer_config(rows, family):
    """One fixed #63 configuration per actual B and either target family."""
    if type(rows) is not int or not 1 <= rows <= 32:
        raise ValueError('split2 rows must be an integer in1..32')
    if family == 'output':
        inner, kc, stages = 4096, 256, 5
    elif family == 'down':
        inner = 9728
        kc, stages = (512, 4) if rows == 1 else (512, 5) if rows <= 16 else (256, 5)
    else:
        raise ValueError('split2 supports only output/down')
    return dict(constants=dict(ROWS=rows, CHANNELS=2560, INNER=inner,
                TILE_ROWS=max(16, 1 << (rows-1).bit_length()),
                TILE_CHANNELS=32, TILE_INNER=kc),
                warps=4, stages=stages, fusion=True, grid=(80,2))


def compilation_cases():
    """64 exact producers +32 reused consumers; no parameter alternatives."""
    result=[]
    for rows in range(1,33):
        for family in ('output','down'):
            cfg=producer_config(rows,family)
            result.append(dict(id=f'split2_{family}_b{rows}',
                function='narrow_split2_projection',
                signature={'X':'*bf16','WEIGHT':'*bf16','PART':'*fp32'},**cfg))
        result.append(dict(id=f'split2_consumer_b{rows}',
            function='_merge_residual_norm',
            signature={'PART':'*fp32','X':'*bf16','GAIN':'*bf16','NORMALIZED':'*bf16'},
            constants=dict(B=rows,H=2560,SPLITS=2,EPS=1e-6,BLOCK=4096),
            warps=4,stages=3,fusion=False,grid=(rows,)))
    return result
