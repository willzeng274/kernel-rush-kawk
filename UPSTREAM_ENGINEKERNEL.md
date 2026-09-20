# Complete last-verified public EngineKernel baseline

All four engine files are byte-identical to https://github.com/jeojdi1/EngineKernel/tree/51f8ee511c372edcfade2edd71f53cf79a0bf02f/engine . All upstream comments, attribution, defaults, and interfaces are retained.

Author commit 7d7bd106bec8cd730476ce33b354075289614e91 identifies this as the last state verified by the platform. Later commit 54f2e51ea23d7ff33345b2feebaa87b4f69a7ab3 reports batch-one divergence with captured prefill sharing the decode graph pool. Our earlier whole-engine baseline used the newer, subsequently failing 1ebd607 revision. The exact commit associated with LegoMan’s 1006.7 score remains independently unconfirmed.
