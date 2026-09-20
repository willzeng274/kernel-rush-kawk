import triton
import triton.language as tl


@triton.jit
def _pair_slot(key):
    key = key.to(tl.uint64)
    mixed = (key ^ (key >> 23)) * 6364136223846793005
    return ((mixed ^ (mixed >> 32)) & 4095).to(tl.int32)


@triton.jit
def _triple_slot(key, oldest):
    salt = (oldest.to(tl.uint64) + 1) * 11400714819323198485
    return _pair_slot(key.to(tl.uint64) ^ salt)


@triton.jit
def _draft_kernel(root_ptr, table_ptr, parent_ptr, rank_ptr, spine_slot_ptr, spine_ptr,
                  nseen_ptr, anchor_ptr, blk_ptr, root_prev_ptr, node_keys_ptr,
                  pair_keys_ptr, pair_values_ptr,
                  root_prevprev_ptr, node_oldest_ptr, triple_keys_ptr,
                  triple_oldest_ptr, triple_values_ptr,
                  K: tl.constexpr, R: tl.constexpr, S: tl.constexpr, SP: tl.constexpr):
    """Nodes are filled in index order (parents first). A node on the spine takes
    the host's n-gram token when one is present (>= 0); otherwise it tries the
    exact triple ending at its parent, then the pair, then the unigram table.

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
    oldest = tl.load(root_prevprev_ptr + b)
    tl.store(node_oldest_ptr + b * R, oldest)
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
        oldest = tl.load(node_oldest_ptr + b * R + p)
        triple_slot = b * 4096 + _triple_slot(key, oldest)
        stored_oldest = tl.load(triple_oldest_ptr + triple_slot)
        valid_oldest = (oldest >= 0) & (stored_oldest == oldest)
        stored_tail = tl.load(triple_keys_ptr + triple_slot, mask=valid_oldest, other=-1)
        triple_cand = tl.load(triple_values_ptr + triple_slot * K + r,
                              mask=valid_oldest & (stored_tail == key), other=-1)
        cand = tl.where(triple_cand >= 0, triple_cand, cand)
        slot = tl.load(spine_slot_ptr + i)
        sp = tl.load(spine_ptr + b * SP + tl.minimum(tl.maximum(off + slot, 0), SP - 1))
        use_spine = (slot >= 0) & (sp >= 0) & fresh
        node = tl.where(use_spine, sp, cand)
        tl.store(blk_ptr + b * R + i, node)
        node_key = (ptok.to(tl.uint64) << 32) | node.to(tl.uint64)
        tl.store(node_keys_ptr + b * R + i, node_key)
        tl.store(node_oldest_ptr + b * R + i, (key.to(tl.uint64) >> 32).to(tl.int32))


@triton.jit
def _publish_pairs_kernel(pair_keys_ptr, pair_values_ptr, node_keys_ptr, top_ptr,
                          blk_ptr, root_prev_ptr, path_idx_ptr, path_len_ptr,
                          triple_keys_ptr, triple_oldest_ptr, triple_values_ptr,
                          node_oldest_ptr, root_prevprev_ptr,
                          K: tl.constexpr, R: tl.constexpr, MAXA: tl.constexpr,
                          P: tl.constexpr):
    """One warp owns a sequence's table; later consumed rows win collisions.

    Publication follows acceptance on the same stream, before any next draft.
    Saved node keys refer to the old root, even though accept changed root.
    No reader overlaps publication, and no other program writes these slots.
    """
    b = tl.program_id(0)
    plen = tl.load(path_len_ptr + b)
    ranks = tl.arange(0, P)
    for step in range(MAXA + 1):
        active = (plen >= 0) & (step <= plen)
        row = tl.load(path_idx_ptr + b * MAXA + tl.maximum(step - 1, 0),
                      mask=active & (step > 0), other=0)
        key = tl.load(node_keys_ptr + b * R + row, mask=active, other=0)
        slot = b * 4096 + _pair_slot(key)
        values = tl.load(top_ptr + (b * R + row) * K + ranks,
                         mask=active & (ranks < K), other=-1)
        tl.store(pair_values_ptr + slot * K + ranks, values, mask=active & (ranks < K))
        tl.store(pair_keys_ptr + slot, key, mask=active)
        oldest = tl.load(node_oldest_ptr + b * R + row, mask=active, other=-1)
        triple_active = active & (oldest >= 0)
        triple_slot = b * 4096 + _triple_slot(key, oldest)
        tl.store(triple_values_ptr + triple_slot * K + ranks, values,
                 mask=triple_active & (ranks < K))
        tl.store(triple_keys_ptr + triple_slot, key, mask=triple_active)
        # The oldest tag is the validity marker; the next reader follows this
        # whole kernel on the same stream, never an in-progress publication.
        tl.store(triple_oldest_ptr + triple_slot, oldest, mask=triple_active)
    last = tl.load(path_idx_ptr + b * MAXA + tl.maximum(plen - 1, 0),
                   mask=plen > 0, other=0)
    previous = tl.load(blk_ptr + b * R + last, mask=plen >= 0, other=0)
    last_key = tl.load(node_keys_ptr + b * R + last, mask=plen >= 0, other=0).to(tl.uint64)
    # New root is accept's final prediction. Its predecessor is the consumed
    # input that produced it, including old root when no draft matched.
    tl.store(root_prev_ptr + b, previous, mask=plen >= 0)
    tl.store(root_prevprev_ptr + b, (last_key >> 32).to(tl.int32), mask=plen >= 0)
