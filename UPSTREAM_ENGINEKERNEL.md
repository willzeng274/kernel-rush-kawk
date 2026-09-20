# Public EngineKernel with isolated captured prefill

Derived from https://github.com/jeojdi1/EngineKernel/tree/51f8ee511c372edcfade2edd71f53cf79a0bf02f/engine . Author attribution and kernel comments are preserved. The three files ek_kernels.py, ek_model.py and ek_probe.py remain byte-identical to upstream.

Only engine.py changes: enable captured prefill by default, omit pool=self.pool for that capture so PyTorch gives it a private pool, retain its external position tensor in the cached tuple, and update the related comments. The result clone, speculation, kernel settings, model operations and other interfaces remain unchanged.

The author identifies51f8ee5 as the last platform-verified revision in commit7d7bd106. This derivative has not been GPU-validated or submitted. Numerical correctness, memory and speed require the official evaluation. It is prepared pending exact baseline#69.
