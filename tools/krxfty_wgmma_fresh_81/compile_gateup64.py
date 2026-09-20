"""Two bounded source-exact Triton 3.1 SM90 compilations, pending approval.

This script is prepared only; do not run without root authorization. It
imports no Torch, needs no GPU/driver/model, and writes compiler evidence
beside itself. The source manifest is checked before each compile. To bound
CPU execution, launch this script under a 180-second process timeout.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


HERE = Path(__file__).resolve().parent
CASES = (("m64_n128_k128_s2", 128, 2), ("m64_n128_k64_s3", 64, 3))


def main():
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import _path_to_binary
    from triton.compiler import ASTSource
    from gateup64 import gateup64_kernel

    assert triton.__version__ == "3.1.0"
    assert os.environ.get("DISABLE_MMA_V3", "0").lower() in ("", "0", "false")
    manifest = json.loads((HERE / "SOURCE_MANIFEST.json").read_text())
    source = HERE / "gateup64.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["gateup64_sha256"]
    outdir = HERE / "compile-results"
    outdir.mkdir(exist_ok=True)
    kernel = gateup64_kernel
    names = kernel.arg_names
    assert names == ["X", "W", "Y", "stride_xm", "stride_wn", "stride_ym",
                     "K", "I", "BLOCK_K"]
    assert set(kernel.constexprs) == {6, 7, 8}

    class AlignedPointer:
        def data_ptr(self):
            return 0x100000

    results = []
    for name, bk, stages in CASES:
        constants = {"K": 2560, "I": 9728, "BLOCK_K": bk}
        values = dict(X=AlignedPointer(), W=AlignedPointer(), Y=AlignedPointer(),
                      stride_xm=2560, stride_wn=2560, stride_ym=9728, **constants)
        attrs = kernel._get_config(*(values[n] for n in names))
        assert set(attrs.divisible_by_16) == set(range(9))
        assert not attrs.equal_to_1
        signature = {i: "*bf16" if i < 3 else "i32" for i in range(6)}
        options = dict(num_warps=4, num_stages=stages, enable_fp_fusion=True)
        start = time.monotonic()
        compiled = triton.compile(
            ASTSource(kernel, signature, {names.index(k): v for k, v in constants.items()}, attrs),
            target=GPUTarget("cuda", 90, 32), options=options,
        )
        seconds = time.monotonic() - start
        stem = outdir / name
        for ext in ("ttir", "ttgir", "llir", "ptx"):
            stem.with_suffix("." + ext).write_text(compiled.asm[ext])
        ptxas, _ = _path_to_binary("ptxas")
        assembly = subprocess.run(
            [ptxas, "-v", "--gpu-name=sm_90a", str(stem.with_suffix(".ptx")),
             "-o", str(stem.with_suffix(".cubin"))],
            capture_output=True, text=True, timeout=45,
        )
        log = assembly.stdout + assembly.stderr
        stem.with_suffix(".ptxas.log").write_text(log)
        ptx = compiled.asm["ptx"]
        inventory = {}
        for key, pattern in {
            "wgmma": r"\bwgmma\.mma_async[^;]*;",
            "mma_sync": r"\bmma\.sync[^;]*;",
            "async_copy": r"\bcp\.async\.(?:ca|cg)\.shared\.global[^;]*;",
            "shared_load": r"\bld\.shared[^;]*;",
            "shared_store": r"\bst\.shared[^;]*;",
            "shuffle": r"\bshfl\.[^;]*;",
            "barrier": r"\bbar\.sync[^;]*;",
            "bf16_cast": r"\bcvt\.[^\s;]*bf16[^\s;]*",
            "exp2": r"\bex2\.[^\s;]*",
            "local_load": r"\bld\.local[^;]*;",
            "local_store": r"\bst\.local[^;]*;",
        }.items():
            found = re.findall(pattern, ptx)
            inventory[key] = dict(static_count=len(found), forms=sorted(set(found)))
        result = dict(
            case=name, source_sha256=manifest["gateup64_sha256"],
            triton=triton.__version__, target="cuda-sm90-warp32", cuda_execution=False,
            signature=signature, constants=constants, options=options, grid=[152],
            runtime_strides=[2560, 2560, 9728],
            attrs=dict(divisible_by_16=sorted(attrs.divisible_by_16), equal_to_1=[]),
            metadata=compiled.metadata._asdict(), ptxas_returncode=assembly.returncode,
            ptxas_resources=log, cpu_compile_seconds=seconds, inventory=inventory,
            hashes={ext: hashlib.sha256(compiled.asm[ext].encode()).hexdigest()
                    for ext in ("ttir", "ttgir", "llir", "ptx")},
        )
        stem.with_suffix(".json").write_text(json.dumps(result, indent=2, default=str) + "\n")
        results.append(result)
        (outdir / "summary.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
        print(json.dumps(result, default=str), flush=True)
        assert assembly.returncode == 0
        assert inventory["wgmma"]["static_count"] > 0, "expected Hopper WGMMA"
        assert inventory["mma_sync"]["static_count"] == 0


if __name__ == "__main__":
    main()
