# Public EngineKernel with corrected prefill and fenced fused verification

The complete engine derives from public EngineKernel commit 51f8ee511c372edcfade2edd71f53cf79a0bf02f: https://github.com/jeojdi1/EngineKernel/tree/51f8ee511c372edcfade2edd71f53cf79a0bf02f/engine . Original authorship and comments are retained.

Accepted submission #70 scored1001.5tok/s with captured prefill in a private graph pool and retained position input. This revision preserves that engine's entrypoint byte-for-byte. The selector from #71 (982.5) and optional prefill attention from #72 (986.0) are not retained.

The fused speculative-verification kernel is byte-identical to public commit fed6eae986e8c7b2b03b87d47e980abad15212d0: https://github.com/jeojdi1/EngineKernel/commit/fed6eae986e8c7b2b03b87d47e980abad15212d0 . Compared with #70's already-present kernel, it adds an unconditional block barrier ordering cache writes before attention reads. The model's ENGINE_ROPE_VERIFY default changes from0to1. No ordinary-decode barrier or cuDNN initialization change is included. Drafting, exact argmax acceptance, pacing, model weights, prefill and other kernels are unchanged.

The existing speculative shape child now performs an independent comparison of unfused and fused attention on fresh raw QKV and separate caches. A prefix crosses a real split boundary when feasible; new cache values and outputs are checked, workspace poisoning prevents accidental reuse, and the reference output is cloned before workspace reuse. This is a bounded operation check, not full-model proof. Its failure follows the existing child-probe contract, which disables Triton for that graph path. Exact SM90 compilation and archive/source checks passed; official GPU correctness and performance remain to be evaluated.
