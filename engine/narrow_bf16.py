"""Research-only, independently written BF16 projection compiler probe.

The tile/warp/stage choices are experimental parameters reported in public
jeojdi1/EngineKernel commit 3fec731d0dadc8a122afe93b21ae7e5a00a2dc7f.
This module is a new implementation of ordinary blocked matrix multiplication,
not a copy of that repository's implementation. It has no production launcher,
selector, weights, model state, or claimed device performance.
"""
import triton
import triton.language as tl


@triton.jit
def narrow_projection(X, WEIGHT, RESULT,
                      ROWS: tl.constexpr, CHANNELS: tl.constexpr,
                      INNER: tl.constexpr, TILE_ROWS: tl.constexpr,
                      TILE_CHANNELS: tl.constexpr, TILE_INNER: tl.constexpr):
    rows = tl.arange(0, TILE_ROWS)
    channels = tl.program_id(0) * TILE_CHANNELS + tl.arange(0, TILE_CHANNELS)
    reduction = tl.arange(0, TILE_INNER)
    product = tl.full((TILE_ROWS, TILE_CHANNELS), 0, tl.float32)
    for offset in range(triton.cdiv(INNER, TILE_INNER)):
        inner = offset * TILE_INNER + reduction
        left = tl.load(X + rows[:, None] * INNER + inner[None, :],
                       (rows[:, None] < ROWS) & (inner[None, :] < INNER), 0)
        right = tl.load(WEIGHT + channels[None, :] * INNER + inner[:, None],
                        (channels[None, :] < CHANNELS) & (inner[:, None] < INNER), 0)
        product = tl.dot(left, right, product)
    tl.store(RESULT + rows[:, None] * CHANNELS + channels[None, :], product,
             (rows[:, None] < ROWS) & (channels[None, :] < CHANNELS))


def configurations():
    """Fixed reported geometries and matched <=3-stage compiler controls."""
    shapes = {"qkv": (6144, 2560), "output": (2560, 4096),
              "gateup": (19456, 2560), "down": (2560, 9728)}
    result = []
    for rows in (1, 4, 16, 32):
        for family, (channels, inner) in shapes.items():
            if family == "qkv":
                nc, kc, warps, stages = ((16, 128, 2, 5) if rows == 1 else
                    (32, 64, 2, 8) if rows <= 16 else (32, 128, 2, 5))
            elif family == "output":
                nc, kc, warps, stages = 32, 256, 4, 5
            elif family == "gateup":
                nc, kc, warps, stages = ((16, 128, 1, 5) if rows == 1 else
                    (64, 128, 4, 3) if rows <= 16 else (32, 128, 2, 3))
            else:
                nc, kc, warps, stages = ((32, 512, 4, 4) if rows == 1 else
                    (32, 512, 4, 5) if rows <= 16 else (32, 256, 4, 5))
            for depth in sorted({stages, min(stages, 3)}):
                result.append(dict(id=f"narrow_{family}_m{rows}_s{depth}",
                    family=family, warps=warps, stages=depth, fusion=True,
                    grid=(triton.cdiv(channels, nc),),
                    constants=dict(ROWS=rows, CHANNELS=channels, INNER=inner,
                        TILE_ROWS=max(16, triton.next_power_of_2(rows)),
                        TILE_CHANNELS=nc, TILE_INNER=kc)))
    return result
