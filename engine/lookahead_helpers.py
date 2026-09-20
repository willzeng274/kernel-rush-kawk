"""Exact frozen metadata and split-softmax helpers; no chain tuner."""
import triton
import triton.language as tl

@triton.jit
def chain_merge_kernel(PART, PMAX, PSUM, OUT, SPLITS: tl.constexpr,
                       BLOCK_S: tl.constexpr, D: tl.constexpr = 128):
    h = tl.program_id(0).to(tl.int64)
    s = tl.arange(0, BLOCK_S)
    d = tl.arange(0, D)
    m = tl.load(PMAX + h * SPLITS + s, s < SPLITS, float('-inf'))
    den = tl.load(PSUM + h * SPLITS + s, s < SPLITS, 0)
    factor = tl.exp(m - tl.max(m, 0))
    part = tl.load(PART + (h * SPLITS + s[:, None]) * D + d[None, :],
                   s[:, None] < SPLITS, 0)
    total = tl.sum(den * factor, 0)
    result = tl.sum(part * factor[:, None], 0) / tl.maximum(total, 1.0e-20)
    tl.store(OUT + h * D + d, result)

@triton.jit
def chain_ids_kernel(META, IDS, W: tl.constexpr, ROWS: tl.constexpr,
                     BLOCK: tl.constexpr = 128):
    flat = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    b, row = flat // W, flat % W
    token = tl.load(META + b * (W + 2) + row, flat < ROWS, 0)
    tl.store(IDS + flat, token, flat < ROWS)

