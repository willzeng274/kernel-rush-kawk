import triton
import triton.language as tl
from pair_cache import _pair_slot

@triton.jit
def _draft_kernel(root_ptr, table_ptr, parent_ptr, rank_ptr, spine_slot_ptr, spine_ptr,
                  nseen_ptr, anchor_ptr, blk_ptr, root_prev_ptr, node_keys_ptr,
                  pair_keys_ptr, pair_values_ptr,
                  K: tl.constexpr, R: tl.constexpr, S: tl.constexpr, SP: tl.constexpr):
    """Nodes are filled in index order (parents first). A node on the spine takes
    the host's n-gram token when one is present (>= 0); otherwise it tries the
    exact pair ending at its parent, then table[token(parent)][rank].

    The host writes the n-gram continuation a round late — it launches round
    t + 1 before it has read round t's tokens — so the pool it wrote is anchored
    ``off = nseen - anchor`` tokens behind this draft's root, and the spine reads
    ``pool[off + slot]``. That shift is only the right continuation if the model
    actually followed the pool over those tokens, which ``pool[off - 1] == root``
    checks for the last of them; a miss falls the whole spine back to the table.
    """
    b = tl.program_id(0)
    tok = tl.load(root_ptr + b)
    tl.store(blk_ptr + b * R, tok)
    previous = tl.load(root_prev_ptr + b)
    root_key = (previous.to(tl.uint64) << 32) | tok.to(tl.uint64)
    tl.store(node_keys_ptr + b * R, root_key)
    off = tl.load(nseen_ptr + b) - tl.load(anchor_ptr + b)
    prev = tl.load(spine_ptr + b * SP + tl.minimum(tl.maximum(off - 1, 0), SP - 1))
    fresh = (off >= 0) & (off + S <= SP) & ((off == 0) | (prev == tok))
    for i in tl.static_range(1, R):
        p = tl.load(parent_ptr + i)
        r = tl.load(rank_ptr + i)
        ptok = tl.load(blk_ptr + b * R + p)
        cand = tl.maximum(tl.load(table_ptr + ptok * K + r), 0)
        key = tl.load(node_keys_ptr + b * R + p)
        pair_slot = b * 4096 + _pair_slot(key)
        stored_key = tl.load(pair_keys_ptr + pair_slot)
        pair_cand = tl.load(pair_values_ptr + pair_slot * K + r,
                            mask=stored_key == key, other=-1)
        cand = tl.where(pair_cand >= 0, pair_cand, cand)
        slot = tl.load(spine_slot_ptr + i)
        sp = tl.load(spine_ptr + b * SP + tl.minimum(tl.maximum(off + slot, 0), SP - 1))
        use_spine = (slot >= 0) & (sp >= 0) & fresh
        node = tl.where(use_spine, sp, cand)
        tl.store(blk_ptr + b * R + i, node)
        node_key = (ptok.to(tl.uint64) << 32) | node.to(tl.uint64)
        tl.store(node_keys_ptr + b * R + i, node_key)
