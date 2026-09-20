# Exact longest verified path — frozen on #83

**Candidate ready for isolated compiler review; CPU semantics pass.** Source is `work/candidates/krxfty_longest_verified/engine`, based exactly on `krxfty_pair_cache_on82/engine` (#83). No compiler, CI, GPU, main-checkout edit, push or submission was performed. No top-8 or spine-swap change is included.

## Opportunity and rule

Duplicate sibling tokens can strand the old first-match walk on a shorter verified branch. The actual R8 template supplies a witness: root children 1/4 both match; child 1 stops, while child 4 continues to child 6. The old path `[1]` emits two tokens; `[4,6]` emits three.

Retain the original first-match walk in full. For R≤64, OR-reduce invalid edge bits indexed by child ID, using uint64 summation because each nonroot child occurs exactly once in CSR. An endpoint is eligible only if its ancestor mask intersects no invalid bits. Choose maximum depth, then lowest endpoint index, but replace the original only when `min(new_depth+1, limit−nseen) > min(old_depth+1, limit−nseen)`. Every selected edge satisfies its actual parent’s `blk[child] == cand[parent]`.

The kernel reuses existing register-loaded CSR, drafted tokens and candidates, plus stable `rec.masks/depth`. No new buffer, model operation or launch is added. Signed masks are reinterpreted as uint64; child bit 63 remains valid, root is excluded from invalid bits, and padded lanes are masked. R42 is covered. R>64 retains the old walk. Valid trees with MAXA=R−1 are chains, so the extra selector is statically excluded; this includes the existing R4 template.

## Preserved behavior

Only three files differ: acceptance kernel/wrapper; one metadata-argument line in `_round`; and its unused readable host reference, with an optional `remaining` argument. No engine caller invokes that host reference. The old graph chronology is byte-identical after removing that argument line. All model, attention, drafting, pair-cache, compaction, geometry, floor and controller sources are unchanged.

Backward reconstruction writes selected path indices and drafted tokens into forward path positions, followed by the selected endpoint’s prediction. The bonus becomes the pending root. `path_len`, accepted count and `nseen` keep full selected-path semantics, including counts exceeding the remaining limit; existing host output clipping is untouched. Frozen lanes retain path_len −1, count 0, root and nseen. Pair publication still consumes old root and selected path chronologically before the next draft; next-round compaction preserves exactly those rows.

## Evidence

`check_longest.py` executes actual extracted candidate AST with a NumPy pointer interpreter, checking against independent exhaustive DFS path enumeration that does not use candidate masks/depth. `CPU_RESULTS.json` records 993 cases: 896 live, 97 frozen, 24 strict gains and 41 deeper alternatives retained because clipping removes their gain. Coverage includes R4/8/16/32/42/64, R65 fallback, negative/zero/small remaining counts, equal-depth alternatives, ties preserving the original, bad ancestors, valid/invalid bit63, padding and capacity boundaries. Three additional mixed lanes check cross-sequence addressing.

There are 93 actual compaction/publication checks, including a write trace of every published row and deliberate hash collisions. Fourteen mutations are caught: omitted search, ignored ancestors, wrong parent predictions/bits, lost bit63, padded-root poisoning, missing clipping, tie replacement, reversed path stores, wrong bonus, clipped count, frozen-root overwrite, wrong compact source and omitted publication endpoint. All 18 source modules parse; preservation assertions restrict the diff to the stated edits.

## Limits and next step

The current [Docs](https://htn.dryft.ai/docs) require exact verification and prohibit approximate output; the live snapshot and full bundled contract were reviewed. This candidate never intentionally accepts a lower-logit token. Different verified physical rows can still produce different BF16 near-ties, so native judging on the emitted prefixes is authoritative. These synthetic CPU checks do not establish device memory behavior, compiler viability, numerical compliance or speed.

`COMPILE_PLAN.json` limits the next step to baseline/candidate acceptance specializations for R4/8/16/32/42/64, recording registers, spills and unsigned mask lowering. Additional reduction/register cost and unknown duplicate frequency may erase any gain. Compilation and any subsequent device or official run remain with the root agent.
