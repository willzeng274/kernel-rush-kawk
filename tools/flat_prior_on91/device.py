import triton
import triton.language as tl

@triton.jit
def _pair_slot(key):
    key = key.to(tl.uint64)
    mixed = (key ^ (key >> 23)) * 6364136223846793005
    return ((mixed ^ (mixed >> 32)) & 4095).to(tl.int32)


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


@triton.jit
def _publish_pairs_kernel(pair_keys_ptr, pair_values_ptr, node_keys_ptr, top_ptr,
                          blk_ptr, root_prev_ptr, path_idx_ptr, path_len_ptr,
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
    last = tl.load(path_idx_ptr + b * MAXA + tl.maximum(plen - 1, 0),
                   mask=plen > 0, other=0)
    previous = tl.load(blk_ptr + b * R + last, mask=plen >= 0, other=0)
    # New root is accept's final prediction. Its predecessor is the consumed
    # input that produced it, including old root when no draft matched.
    tl.store(root_prev_ptr + b, previous, mask=plen >= 0)


@triton.jit
def accept_kernel(blk_ptr, cand_ptr, child_start_ptr, child_list_ptr, child_par_ptr,
                  masks_ptr, depth_ptr,
                  done_ptr, nseen_ptr, pos_ptr, limit_ptr,
                  root_ptr, path_idx_ptr, path_len_ptr, acc_tok_ptr, acc_cnt_ptr,
                  CAP, R: tl.constexpr, C: tl.constexpr, P: tl.constexpr,
                  MAXA: tl.constexpr, GUARD: tl.constexpr):
    """One program per sequence.

    Frozen sequences (``done``) write ``path_len = -1``, which makes the compact
    copy nothing and ``pos += path_len + 1`` a no-op, and leave ``root`` alone so
    the next draft re-verifies the same block in place. Live sequences write the
    accepted path, its length, the new root, and the accepted tokens in order.

    ``done`` for the *next* round is recomputed here from the counters this
    kernel just advanced, which is the same test the host used to make from its
    queue lengths: a sequence is finished once it holds ``limit`` tokens, or
    once its next block would run past the end of the KV cache.
    """
    b = tl.program_id(0)
    live = tl.load(done_ptr + b) == 0
    n_prev = tl.load(nseen_ptr + b)
    pos = tl.load(pos_ptr + b)
    limit = tl.load(limit_ptr)

    j = tl.arange(0, P)
    cs = tl.load(child_start_ptr + j, mask=j < R + 1, other=0)
    cl = tl.load(child_list_ptr + j, mask=j < C, other=0)
    cp = tl.load(child_par_ptr + j, mask=j < C, other=0)
    cv = tl.load(cand_ptr + b * R + j, mask=j < R, other=0)
    # Per slot: does the child's drafted token match what the model predicted
    # after its parent? The two sentinels differ so masked-off slots never hit.
    drafted = tl.load(blk_ptr + b * R + cl, mask=j < C, other=-1)
    wanted = tl.load(cand_ptr + b * R + cp, mask=j < C, other=-2)
    hit = (j < C) & (drafted == wanted)

    tok = tl.sum(tl.where(j == 0, cv, 0), axis=0)
    tl.store(acc_tok_ptr + b * (MAXA + 1), tok)
    cur = 0
    alen = 0
    alive = live
    for _ in range(MAXA):
        start = tl.sum(tl.where(j == cur, cs, 0), axis=0)
        end = tl.sum(tl.where(j == cur + 1, cs, 0), axis=0)
        jsel = tl.min(tl.where(hit & (j >= start) & (j < end), j, P), axis=0)
        found = alive & (jsel < P)
        c = tl.where(found, tl.sum(tl.where(j == jsel, cl, 0), axis=0), 0)
        ntok = tl.sum(tl.where(j == c, cv, 0), axis=0)
        tl.store(path_idx_ptr + b * MAXA + tl.minimum(alen, MAXA - 1), c, mask=found)
        alen = tl.where(found, alen + 1, alen)
        tl.store(acc_tok_ptr + b * (MAXA + 1) + tl.minimum(alen, MAXA), ntok, mask=found)
        tok = tl.where(found, ntok, tok)
        cur = tl.where(found, c, cur)
        alive = found

    if R <= 64 and MAXA < R - 1:
        # A valid R-node tree with depth R-1 is a chain: its first-match path
        # is already longest. Compile the extra selection away in that case.
        # Each nonroot child occurs once in CSR. Summing distinct unsigned
        # child bits is OR, including bit 63; padded slots never contribute.
        bits = tl.full((P,), 1, tl.uint64) << cl.to(tl.uint64)
        bad = tl.sum(tl.where((j < C) & (cl > 0) & ~hit, bits, 0), axis=0)
        masks = tl.load(masks_ptr + j, mask=j < R, other=0).to(tl.uint64)
        depth = tl.load(depth_ptr + j, mask=j < R, other=-1)
        verified = (j < R) & ((masks & bad) == 0)
        best_depth = tl.max(tl.where(verified, depth, -1), axis=0)
        remaining = limit - n_prev
        improve = live & (tl.minimum(best_depth + 1, remaining) >
                          tl.minimum(alen + 1, remaining))
        if improve:
            endpoint = tl.min(tl.where(verified & (depth == best_depth), j, P), axis=0)
            tok = tl.sum(tl.where(j == endpoint, cv, 0), axis=0)
            tl.store(acc_tok_ptr + b * (MAXA + 1) + best_depth, tok)
            # Reconstruct backwards entirely from the already loaded CSR and
            # draft registers. Slot d-1 emits blk[node], then the endpoint's
            # prediction above supplies the bonus token at slot best_depth.
            node = endpoint
            for step in range(MAXA):
                d = best_depth - step
                active = d > 0
                slot = tl.min(tl.where((j < C) & (cl == node), j, P), axis=0)
                parent = tl.sum(tl.where(j == slot, cp, 0), axis=0)
                token = tl.sum(tl.where(j == slot, drafted, 0), axis=0)
                offset = tl.maximum(d - 1, 0)
                tl.store(path_idx_ptr + b * MAXA + offset, node, mask=active)
                tl.store(acc_tok_ptr + b * (MAXA + 1) + offset, token, mask=active)
                node = parent
            alen = best_depth

    plen = tl.where(live, alen, -1)
    cnt = tl.where(live, alen + 1, 0)
    n_new = n_prev + cnt
    tl.store(path_len_ptr + b, plen)
    tl.store(acc_cnt_ptr + b, cnt)
    tl.store(root_ptr + b, tok, mask=live)
    tl.store(nseen_ptr + b, n_new)
    # Sticky: neither counter moves once a sequence is frozen, so recomputing
    # alone would already hold, but the OR makes an externally set flag final.
    frozen = (n_new >= limit) | (pos + plen + 1 + GUARD >= CAP) | (live == 0)
    tl.store(done_ptr + b, frozen.to(tl.int32))


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


