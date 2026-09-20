import triton
import triton.language as tl

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
