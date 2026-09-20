"""Request-local two-token proposals; every proposal still needs verification."""

import triton
import triton.language as tl


PAIR_SLOTS = 4096


def pair_slot(key: int) -> int:
    """Host counterpart of the unsigned device hash; exact keys resolve aliases."""
    mixed = ((key ^ (key >> 23)) * 6364136223846793005) & ((1 << 64) - 1)
    return (mixed ^ (mixed >> 32)) & (PAIR_SLOTS - 1)


def prompt_pairs(input_ids: list[list[int]], k: int):
    """Sparse per-sequence slots, keys and newest distinct observed successors.

    The newest pair owns a colliding slot for this seed. Older occurrences of
    that exact pair may add distinct successors; older different pairs cannot
    displace it. No pair crosses a sequence boundary. Space is bounded by
    B * PAIR_SLOTS entries, each with at most min(k, 8) successors.
    """
    indices, keys, values = [], [], []
    width = min(k, 8)
    for b, ids in enumerate(input_ids):
        occupied = {}
        for i in range(len(ids) - 2, 0, -1):
            key = (ids[i - 1] << 32) | ids[i]
            slot = pair_slot(key)
            entry = occupied.get(slot)
            if entry is None:
                occupied[slot] = (key, [ids[i + 1]])
            elif entry[0] == key and len(entry[1]) < width and ids[i + 1] not in entry[1]:
                entry[1].append(ids[i + 1])
        for slot, (key, row) in occupied.items():
            indices.append(b * PAIR_SLOTS + slot)
            keys.append(key)
            values.append(row + [-1] * (k - len(row)))
    return indices, keys, values


@triton.jit
def _pair_slot(key):
    key = key.to(tl.uint64)
    mixed = (key ^ (key >> 23)) * 6364136223846793005
    return ((mixed ^ (mixed >> 32)) & 4095).to(tl.int32)


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
