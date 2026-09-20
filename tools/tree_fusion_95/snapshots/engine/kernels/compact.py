"""Move an accepted draft path's K/V rows into contiguous cache slots.

After a tree verify pass the draft nodes occupy slots pos .. pos+R-1 in tree
order. Once the host accepts a root-to-leaf path of ``a`` nodes with block
indices ``idx[b, 0..a-1]``, their K/V must sit at pos+1 .. pos+a so the next
pass sees a plain prefix. One program per (layer, sequence, kv head) loads all
``a`` source rows before storing any, so overlapping source/destination slots
cannot race.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _compact_kernel(k_ptr, v_ptr, pos_ptr, idx_ptr, len_ptr, CAP, B, HKV,
                    MAXA: tl.constexpr, D: tl.constexpr):
    layer = tl.program_id(0)
    b = tl.program_id(1)
    kh = tl.program_id(2)
    pos = tl.load(pos_ptr + b)
    a = tl.load(len_ptr + b)
    d = tl.arange(0, D)
    base = ((layer * B + b) * HKV + kh) * CAP
    for j in tl.static_range(MAXA):
        live = j < a  # a == -1 (frozen sequence) copies nothing
        src = tl.load(idx_ptr + b * MAXA + j, mask=live, other=0)
        k = tl.load(k_ptr + (base + pos + src) * D + d, mask=live, other=0.0)
        v = tl.load(v_ptr + (base + pos + src) * D + d, mask=live, other=0.0)
        # A later source can equal an earlier destination only if src >= j+1
        # and dst = j+1, i.e. src == dst, which is a no-op copy; every other
        # pair is disjoint because path indices strictly increase with depth.
        tl.store(k_ptr + (base + pos + 1 + j) * D + d, k, mask=live)
        tl.store(v_ptr + (base + pos + 1 + j) * D + d, v, mask=live)


def compact_paths(k_cache: torch.Tensor, v_cache: torch.Tensor, pos: torch.Tensor,
                  idx: torch.Tensor, lens: torch.Tensor) -> None:
    """k/v_cache [layers, B, HKV, cap, D]; pos [B] int32 (root slot); idx [B, MAXA]
    int32 block indices of accepted nodes in path order; lens [B] int32."""
    layers, B, HKV, cap, D = k_cache.shape
    MAXA = idx.shape[1]
    _compact_kernel[(layers, B, HKV)](k_cache, v_cache, pos, idx, lens, cap, B, HKV, MAXA=MAXA, D=D, num_warps=1)
