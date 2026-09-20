# Source preparation: two-split small-tree fusion

Prepared isolated `tree_fused_attention_device.py`, pure-host `tree_fused_attention.py`, and standard-library `cpu_controls.py`. Nothing is installed, selected, submitted, or changed in an incumbent candidate. No Triton compilation, GPU execution, CI, or network access occurred. **GPU correctness and performance are unmeasured.**

## Integration interface

`TreeFusedAttention(incumbent, authorized=True)` accepts the already-selected `DecodeAttention` only for TREE, NSPLIT=2, NSP=2, row_blocks=1, HQ32/HKV8/D128/G4, R4/GP16 or R8/GP32. It preserves all eight incumbent `(BLOCK_N, warps, stages)` configurations, SPLIT_LEN, scale, and launch geometry. It retains the incumbent owner and borrows its partial buffers. Caller must serialize incumbent/fused calls on one stream and retain owners through graph drain. Authorization supports bounded selector evaluation as well as installation; this module does neither selection nor timing.

Call signature:

```python
fused(qkv, q_norm_w, k_norm_w, cos, sin, pos, depth,
      k_cache, v_cache, out, tree, eps)
```

The wrapper documents/checks static tensor metadata without reading device values. It imports the unchanged incumbent `_reduce_kernel`. No Q buffer is emitted. Original separate norm/cache plus attention remains the selector's comparison path.

## Device mechanism

Queries load as `[64,GP]` halves and use an AST-identical copy of incumbent `_norm_rope_row`: independent 64-element FP32 reductions, normalization cast, gain cast, separate rotary-product casts, final addition cast. Ordered join/permute/reshape assembles `[GP,128]`. Pinned Triton 3.1 `cat` requires reorder permission and rank one, so it is deliberately unused.

Original prefix and masked-tail bounds, BLOCK_N-aligned tile order, dot/online-softmax operations, BF16 probabilities, FP32 partial layout, and empty-split initialization are retained. Every K/V cache load explicitly requires `n<pos`. The tail unrolls at most R scalar guards, each normalizing one current K head directly from immutable QKV; the owning split/tile writes cache slot `pos+j`, using rotary position `pos+depth[j]`. V is gathered directly from QKV after the probability/sum calculation and published with the corresponding unique slot mask. No current cache value is read and no cross-CTA synchronization is needed.

Separate static assertions avoid the known Triton 3.1 chained-boolean parser failure reported by root.

## CPU evidence and next gates

Nine controls pass (0.740 seconds): 7,078 exhaustive position/split/tile geometries; equal unique K/V ownership; prefix-only reads and poisoned-current reconstruction; padded QKV, head, partial and output addressing; ancestor/sibling masks; BF16 ties; actual norm-source execution versus an independent scalar staged reference; wrong-interleaving/deferred-cast negative controls; empty-split merge; and all 16 supported R/config host dispatches. Narrow AST checks bind tests to cache masks, scalar norm source, recurrence expressions, ordered query assembly, original reducer import, and absence of unsupported boolean chains.

CPU balanced reduction and reciprocal-square-root models do not establish GPU reduction ordering or rsqrt/dot equivalence. Query layout conversion/coalescing and R guarded tail branches may raise registers, spills, or disrupt pipelining.

Manifest proposes three initial compile-only smoke fixtures covering GP16/32, BN32/64/128, W4/8, and stages2/3. They are not an exhaustive matrix: every actual selected tuple needs compilation and resource review before use. Then compare complete cache/output numerics and exact-greedy behavior, and measure serialized full-layer-rotation baseline versus fused calls. Preserve incumbent geometry; do not force two splits.
