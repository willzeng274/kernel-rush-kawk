# Public EngineKernel with bounded prefill pacing credit

The complete engine derives from public EngineKernel commit 51f8ee511c372edcfade2edd71f53cf79a0bf02f: https://github.com/jeojdi1/EngineKernel/tree/51f8ee511c372edcfade2edd71f53cf79a0bf02f/engine . Original authorship and comments are retained.

Retained submission #73 scored 1003.2 tok/s with captured prefill in a private graph pool, retained position input, and fenced fused verification from public fed6eae986e8c7b2b03b87d47e980abad15212d0. Its bounded verification probe is preserved. The slower last-layer prefill pruning from #74 is removed.

This experiment changes only the speculative host scheduler relative to #73. It measures ordinary work through the existing first-token CPU conversion, excluding earlier verifier setup and later consumer pauses. Cold prefill capture receives no usable credit. At unchanged R=1.2, the scheduler may remove at most min(F/6, existing tail waits). Before either final-yield path, it drains outstanding same-stream work. The tail retains absolute deadlines so required work can overlap the existing wait. Q3, B1 eligibility, model arithmetic, exact argmax acceptance, replay decisions, and all three other engine modules are byte-identical to #73.

Actual-method CPU scheduling checks and independent source review passed. The new drain can cost more than the pacing credit saves. No device arithmetic was changed; CUDA performance, spread, and official eligibility remain to be measured. This file is provenance outside the submitted engine archive.
