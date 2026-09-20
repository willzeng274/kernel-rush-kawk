**V2 is source-ready and uncompiled** at `work/candidates/krxfty_down_s4_v2_on91/engine`. It is a separate 23-file copy of the frozen down-S4 prototype on exact #91; the original candidate, frontier research and compiler inputs were not edited.

The only candidate-source change is in `kernels/down_s4.py`: two compound `tl.static_assert` conditions become five separate single-comparison assertions for BM64, BN64, BK64, split4 and K-per-split2432. This addresses the root-reported Triton 3.1 Boolean-expression failure in CI35506309022, before producer IR existed; that attempt never compiled the merge.

Removing assertion statements makes the entire old and new device-module ASTs identical. All other 22 source files are byte-identical. Therefore arithmetic, the single 38-step K loop, addresses, FP32 partials, merge, launch arguments/options, ownership, selector, budget and model hook are unchanged. No assertion was weakened or removed.

The existing actual-host CPU suite was adapted only for the new candidate path and a narrow check requiring exactly those five noncompound assertions. **All 42 controls passed in one run.** They retain the previous wrapper/launch, split-address coverage, negative-control, validation, timer, fallback, timeout and ownership checks. These are CPU stand-ins, not compiled device execution.

`PROTOTYPE_DELTA.diff` contains only the assertion correction; applying it to a disposable copy reproduces V2 byte-for-byte. `CANDIDATE.diff` records the complete difference from #91. Parent source and preparation manifest members were rechecked unchanged. The deterministic archive contains exactly the 23 frozen source files, and every member matches the source manifest.

Source set: `9e890ca3af849001ff621713a64a124c9f58db386e7a4abaeae9d62a7cd6fa47`.

Archive: `down-s4-v2-on91-source.tar.gz`, 45,130 bytes; SHA256 `1070df4860a996d1ba5fa38b73f5b2af26ae73414abb07edd8914a7ee8dffd82`.

Corrected producer-module SHA256: `4319be41e779f9d714de66e8d5e27d18039c9f54419b5c638ceab6ba2c1d87bc`.

`SOURCE_MANIFEST.json`, `SOURCE_AUDIT.json`, `HOST_CHECKS.json` and `PREPARATION_MANIFEST.json` retain the freeze and evidence. No compiler checkout/main edits, native compilation, CI, GPU execution or submission occurred. Separate compiler authorization remains necessary; unmasked-load tail bounds and final WGMMA/copy drains across 38 iterations remain unproved. Only an official GPU run can establish full-verifier correctness and performance.
