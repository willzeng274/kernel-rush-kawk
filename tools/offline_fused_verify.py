"""Exact fenced Q3 verifier source compilation, two real launcher variants."""
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'compile-results'


def main():
    import triton
    from triton.compiler import ASTSource, AttrsDescriptor
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import _path_to_binary
    assert triton.__version__ == '3.1.0'
    OUT.mkdir(exist_ok=True)
    record = json.loads(Path(__file__).with_name('fused_verify_source.json').read_text())
    source = ROOT / 'engine/fused_verify_kernel.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == record['compiler_module_sha256']
    spec = importlib.util.spec_from_file_location('fused_verify_exact', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    kernel = module._rope_attn_verify_kernel
    names = kernel.arg_names
    assert names == record['arguments']
    constant_base = dict(N_Q=32, N_KV=8, G=4, NQ=3, GP=16, D=128, HALF=64, EPS=1e-6)
    options = dict(num_warps=4, num_stages=3, enable_fp_fusion=True)
    pointer_types = {name: '*bf16' for name in ('QKV', 'QN', 'KN', 'COS', 'SIN', 'K', 'V', 'Out')}
    pointer_types.update(LenB='*i64', Start='*i32', Acc='*fp32', Lsum='*fp32', Mmax='*fp32')

    class AlignedPointer:
        def data_ptr(self):
            return 0x100000

    results = []
    # Representative valid workspace strides. CHUNK is always a multiple of
    # BLOCK_N under plan_splits; neither launch folds it into a constexpr.
    for name, block, one, capacity, sp, chunk in [
            ('bn64_split', 64, False, 768, 16, 64),
            ('bn128_unsplit', 128, True, 512, 1, 512)]:
        begin = time.monotonic()
        constants = dict(constant_base, BLOCK_N=block, SPLITS_ONE=one)
        signature = {i: pointer_types[n] if n in pointer_types else 'fp32' if n == 'sm_scale' else 'i32'
                     for i, n in enumerate(names) if n not in constants}
        result = dict(id=name, source=record, signature={names[i]: t for i, t in signature.items()},
                      constants=constants, options=options, cuda_execution=False,
                      started_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        try:
            vals = {n: AlignedPointer() for n in pointer_types}
            vals.update(sm_scale=128 ** -.5, stride_qkv_r=6144, stride_cos_r=128,
                        stride_ob=4096, stride_oh=128, stride_kb=8 * capacity * 128,
                        stride_kh=capacity * 128, stride_ks=128,
                        stride_ab=8 * sp * 16 * 128, stride_ah=sp * 16 * 128,
                        stride_as=16 * 128, stride_ag=128, stride_lb=8 * sp * 16,
                        stride_lh=sp * 16, stride_ls=16, CHUNK=chunk, **constants)
            attrs = kernel._get_config(*(vals[n] for n in names))
            div = {i for i, n in enumerate(names) if n in pointer_types or
                   isinstance(vals[n], int) and vals[n] % 16 == 0}
            eq = {i for i, n in enumerate(names) if isinstance(vals[n], int)
                  and not isinstance(vals[n], bool) and vals[n] == 1}
            assert set(attrs.divisible_by_16) == div
            assert set(attrs.equal_to_1) == eq
            assert names.index('sm_scale') not in div and names.index('EPS') not in div
            assert names.index('CHUNK') in div and names.index('CHUNK') not in eq
            cfg = {names.index(n): v for n, v in constants.items()}
            for i in eq:
                cfg[i] = vals[names[i]]
            assert names.index('CHUNK') not in cfg
            result['attributes'] = dict(divisible_by_16=[names[i] for i in sorted(div)],
                                        equal_to_1=[names[i] for i in sorted(eq)])
            result['runtime_values'] = {n: v for n, v in vals.items() if n not in pointer_types and n not in constants}
            compiled = triton.compile(ASTSource(kernel, signature, cfg, AttrsDescriptor(tuple(sorted(div)), tuple(sorted(eq)))),
                                      target=GPUTarget('cuda', 90, 32), options=options)
            result['metadata'] = compiled.metadata._asdict()
            for ext in ('ttir', 'ttgir', 'ptx', 'llir'):
                (OUT / (name + '.' + ext)).write_text(compiled.asm[ext])
            ptxas, _ = _path_to_binary('ptxas')
            assembly = subprocess.run([ptxas, '-v', '--gpu-name=sm_90a', str(OUT / (name + '.ptx')),
                                       '-o', str(OUT / (name + '.cubin'))], capture_output=True, text=True, timeout=45)
            result.update(ptxas_returncode=assembly.returncode, ptxas_resources=assembly.stdout + assembly.stderr)
            (OUT / (name + '.ptxas.log')).write_text(result['ptxas_resources'])
            result['success'] = assembly.returncode == 0
        except Exception as error:
            import traceback
            result.update(success=False, error=str(error), traceback=traceback.format_exc())
        result['elapsed_seconds'] = time.monotonic() - begin
        result['completed_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        results.append(result)
        (OUT / (name + '.json')).write_text(json.dumps(result, indent=2, default=str) + '\n')
        (OUT / 'fused_verify_summary.json').write_text(json.dumps(results, indent=2, default=str) + '\n')
        print(json.dumps(result, default=str), flush=True)
    if len(results) != 2 or any(not r['success'] for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
