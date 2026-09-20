# Complete public EngineKernel baseline

All four engine files are copied byte-for-byte from https://github.com/jeojdi1/EngineKernel/tree/1ebd607be2336e3854880fcfa699dba799a3d9d2/engine . No algorithms, tuning, defaults, wrappers, or interfaces were modified. Author attribution and source comments are retained. This tests the complete public implementation; the exact commit associated with LegoMan’s1006.7score is not independently confirmed.

The next control changes only the default of ENGINE_SPEC from1to0, following a batch-one incorrect-output result for the unchanged baseline. All other source and defaults remain unchanged.
