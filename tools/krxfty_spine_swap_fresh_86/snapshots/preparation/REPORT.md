# Exact draft sibling swap on retained #83

**Frozen, targeted CPU checks pass; compiler and GPU benefit remain unproved.** Base is exact retained #83 (`9a3e7ad5d4e763287a1e9b0de25a55281fab998d`, 1164.1 TPS). Only `work/candidates/krxfty_spine_swap/engine/recycle.py` changes. No floors, top8, tree geometry, row counts, scheduling, model/logits math, acceptance, pair publication or other file changes. No compiler, CI, GPU, main edit, push or submission occurred. No applicable ancestor AGENTS.md existed in the owned paths.

## Behavior

`Recycler.__init__` allocates one immutable int32 parent→global-spine-child vector, R entries (16–256 bytes for tested shapes). The existing captured draft receives this stable allocation and vocabulary size. A compile-time condition removes the swap body for chain-only trees, including R4.

For a later nonspine sibling of a global-spine parent, the draft checks that the existing n-gram override is fresh, present, and equals the sibling's original effective proposal. Only then may it restore the parent's raw preoverride rank0. On #83 that value is **exact-key pair0 when nonnegative, otherwise raw unigram0**. The restored ID must satisfy `0 <= ID < vocab` and differ from the override. Raw −1 is never converted to zero for restoration. Existing ordinary fallback still clamps unknown unigram proposals to zero.

The comparison uses the same pool value that the earlier spine child uses under these gates, avoiding an additional `blk` load/dependency. Parent metadata is restricted to the global rank0 spine. Stale, expired, missing, negative-offset and all-unknown behavior remains unchanged. Every changed child key is subsequently computed from its actual parent token and repaired token.

If the override is correct, unchanged first-match acceptance keeps the earlier spine child. If displaced rank0 is correct, its restored edge can now be accepted. Partial pair rows mix pair and unigram ranks and may already duplicate tokens; this change does **not** establish uniqueness or choose longest duplicate paths.

## CPU evidence

`check_spine_swap.py` extracts the actual candidate AST, executes its constructor/draft/unchanged accept with bounded NumPy pointers, and reuses only the existing pair fixture and independent full-prefix serial oracle. Prior broad suites were not rerun.

Passed:

- **117 directed cases** and **432 randomized lane cases**, covering R4/8/16/32/42/64, every eligible global-spine parent, off-spine branches, partial pairs, key misses, active/shifted/stale/expired pools, raw unknown IDs, zero and vocabulary boundaries including 151935/151936.
- **15,292 node ancestry checks**, including descendants of restored siblings.
- **20 acceptance comparisons**, proving recovery of lost pair-rank0 and preservation of the earlier spine first-match when override is greedy.
- **12 request runs / 248 rounds** matching independent serial outputs across all six shapes, including B3/R42; **156 metadata allocation/content reuse checks** across drafts and reused requests.
- **Three detected mutations:** restoring unigram0 instead of valid pair0; restoration without duplicate equality; clamping raw unknown rank0 into a restoreable zero.

AST/byte checks prove every other source file, TreeTemplate, update, publisher and acceptance remain unchanged. These tests establish the integer proposal/acceptance behavior in CPU emulation; they do not reproduce CUDA capture, lane ordering, BF16 model argmax or hardware timing.

## Frozen artifacts and next evidence

Source/diff/archive and before/after hashes are in `SOURCE_MANIFEST.json`, candidate `STATUS.json`, and `DIFF.patch`. Key SHA256 values:

- recycle.py: `c314d41e93af5bc392c471a95a9d4fb3dba26d38da5796d65feb0ebb0e9a4323`
- complete source tree: `58846308ed8fb9f2919ce4f4db8d60cb1dcdd50692dfe750cba46bb7d908b5e1`
- exact decorated draft: `06da897fa80c0f75aa5fc4c4ab0e96a023d0d751257c8e7fbcdfc67b38998600`
- proposed compile plan: `f5f24c0a6396dcf1abb2e19b77342ddec8c12776cdcb6e38ec937cb520bd30e3`

`COMPILE_PLAN.json` proposes exactly **six draft-only Triton3.1/SM90 builds**: R4/8/16/32/42/64, K8, vocab151936, native launch defaults and exact pointer signature. One process, 180-second total/40-second case limits, no retry/sweep. It specifies source verification, R4 elimination, masks/predicates/lane ownership and registers/spills/IR inspection. It is **not executed**.

GPU validation must still establish own-prefix exact acceptance, inherited scalar draft-load safety, capture behavior and resource/latency cost. Additional loads/comparisons may outweigh recovered edges; R4 has no coverage opportunity. No duplicate frequency, acceptance gain, saved-round or throughput claim follows from these CPU checks. Live rules prohibit intentional approximate outputs despite numeric tolerance2; acceptance remains strict and no margin policy is introduced.
