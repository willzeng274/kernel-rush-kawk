# Public EngineKernel with corrected prefill and whole-path selection

Derived from https://github.com/jeojdi1/EngineKernel/tree/51f8ee511c372edcfade2edd71f53cf79a0bf02f/engine . All original authorship and kernel comments remain. ek_kernels.py, ek_model.py and ek_probe.py are byte-identical to that public revision.

The corrected captured prefill uses a private graph pool and retains its position input. That complete engine passed official submission70 at1001.5tokens/second, following the unchanged public engine’s986.3result in69.

This revision adds ek_select.py and graph-construction hooks. During warmup it compares complete fixed-prefix decode/verification across the already present cuBLAS, unsplit Triton and split/fused Triton paths. It retains the incumbent unless a challenger is at least2%faster, uses independent temporary graph pools and restores flags after production capture. Device kernels, weight layouts, speculation and prefill arithmetic are unchanged. GPU correctness and speed are evaluated by the official run; local checks cover source and host policy only.
