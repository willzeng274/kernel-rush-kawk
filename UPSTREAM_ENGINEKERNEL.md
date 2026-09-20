# Public EngineKernel with corrected captured prefill and optional Triton attention

The complete engine derives from public EngineKernel commit 51f8ee511c372edcfade2edd71f53cf79a0bf02f: https://github.com/jeojdi1/EngineKernel/tree/51f8ee511c372edcfade2edd71f53cf79a0bf02f/engine . Original authorship and kernel comments are retained.

Our accepted submission #70 scored 1001.5 tok/s by enabling captured prefill with a private graph pool and retaining the position input. This revision preserves that complete engine and its speculation/decode paths. The whole-model selector from #71 (982.5 tok/s) is removed.

The only new device function is the byte-identical causal GQA prefill kernel from public commit 7f18415792ec731a1b86518a745dd9bc36ef80bb: https://github.com/jeojdi1/EngineKernel/commit/7f18415792ec731a1b86518a745dd9bc36ef80bb . It writes projection-ready rows directly. Added host guards restrict supported layouts. The existing cuDNN route stays first choice; padded attention and unsupported inputs keep the previous path.

A separate, once-only child probe checks this optional kernel against PyTorch on full/partial tiles with actual strides and poisoned unused cache slots. It has a 45-second timeout and disables only optional prefill on failure. This probe runs during setup; it is not evidence of full-model correctness until the official run passes. No diagnostics about hidden workloads, unrelated upstream fusion changes, or runtime downloads are included.
