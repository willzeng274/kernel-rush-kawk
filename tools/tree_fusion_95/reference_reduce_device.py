import triton
import triton.language as tl


@triton.jit
def _reduce_kernel(
    o_part_ptr, m_part_ptr, l_part_ptr, o_ptr, NSPLIT,
    HQ: tl.constexpr, R: tl.constexpr, D: tl.constexpr, NSP: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    t = tl.program_id(2)
    s = tl.arange(0, NSP)
    smask = s < NSPLIT
    base = ((b * HQ + h) * R + t) * NSPLIT
    m = tl.load(m_part_ptr + base + s, mask=smask, other=float("-inf"))
    l = tl.load(l_part_ptr + base + s, mask=smask, other=0.0)
    M = tl.max(m, axis=0)
    w = tl.exp(m - M)
    L = tl.sum(w * l, axis=0)
    d = tl.arange(0, D)
    o = tl.load(o_part_ptr + (base + s[:, None]) * D + d[None, :], mask=smask[:, None], other=0.0)
    out = tl.sum(o * w[:, None], axis=0) / L
    tl.store(o_ptr + ((b * R + t) * HQ + h) * D + d, out.to(tl.bfloat16))
