"""Frozen exact device functions extracted from candidate; research only."""
import triton
import triton.language as tl


@triton.jit
def _bf16_key(value):
    bits = value.to(tl.int16, bitcast=True).to(tl.int32) & 65535
    mask = tl.where((bits & 32768) != 0, 65535, 32768)
    key = tl.where((bits & 32767) > 32640, 65535, bits ^ mask)
    # Valid keys are positive, including infinities. Zero is an invalid lane.
    return key + 1


@triton.jit
def _best_pair(key_a, id_a, key_b, id_b):
    take_a = (key_a > key_b) | ((key_a == key_b) & (id_a < id_b))
    return tl.where(take_a, key_a, key_b), tl.where(take_a, id_a, id_b)


@triton.jit
def top8_partials(Logits, PartialKeys, PartialIds,
                  V: tl.constexpr, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    token = part * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(Logits + row * V + token, token < V, other=0)
    key = tl.where(token < V, _bf16_key(value), 0)
    for rank in tl.static_range(8):
        best_key, best_id = tl.reduce((key, token), 0, _best_pair)
        off = (row * PARTS + part) * 8 + rank
        tl.store(PartialKeys + off, best_key)
        tl.store(PartialIds + off, best_id)
        key = tl.where(token == best_id, 0, key)


@triton.jit
def top8_merge(PartialKeys, PartialIds, Out,
               PARTS: tl.constexpr, MERGE_BLOCK: tl.constexpr):
    row = tl.program_id(0)
    slot = tl.arange(0, MERGE_BLOCK)
    offset = row * PARTS * 8 + slot
    key = tl.load(PartialKeys + offset, slot < PARTS * 8, other=0)
    token = tl.load(PartialIds + offset, slot < PARTS * 8, other=2147483647)
    for rank in tl.static_range(8):
        best_key, best_id = tl.reduce((key, token), 0, _best_pair)
        tl.store(Out + row * 8 + rank, best_id)
        key = tl.where(token == best_id, 0, key)
