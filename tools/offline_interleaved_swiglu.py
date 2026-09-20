"""Seven exact-source Triton 3.1 SM90 compilations; no driver/GPU/model use."""
import datetime
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'compile-results'
MANIFEST = Path(__file__).with_name('interleaved_swiglu_source.json')


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def instruction_inventory(text):
    patterns = {
        'mma_sync': r'\bmma\.sync[^\s]+',
        'wgmma': r'\bwgmma\.mma_async[^\s]+',
        'async_copy': r'\bcp\.async\.(?:ca|cg)\.shared\.global[^;]*;',
        'global_load': r'\bld\.global[^\s]+',
        'shared_load': r'\bld\.shared[^\s]+',
        'shared_store': r'\bst\.shared[^\s]+',
        'global_store': r'\bst\.global[^\s]+',
        'shuffle': r'\bshfl\.[^\s]+',
        'barrier': r'\bbar\.sync[^;]*;',
        'bf16_convert': r'\bcvt\.[^\s]*bf16[^\s]*',
        'exp2': r'\bex2\.[^\s]+',
    }
    result = {}
    for name, pattern in patterns.items():
        found = re.findall(pattern, text)
        result[name] = {'static_count': len(found), 'forms': sorted(set(found))}
    return result


def child(case):
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import _path_to_binary
    assert triton.__version__ == '3.1.0'
    OUT.mkdir(exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    source_record = manifest['sources'][case['kind']]
    source = ROOT / 'engine' / source_record['compiler_module']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_record['compiler_module_sha256']
    spec = importlib.util.spec_from_file_location('exact_' + case['kind'], source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    kernel = getattr(module, case['kernel'])
    names = kernel.arg_names
    expected = ['X', 'W', 'Y', 'M', 'K', 'stride_xm', 'stride_wn', 'stride_ym',
                'N' if case['kind'] == 'ordinary' else 'I', 'BLOCK_N', 'BLOCK_K', 'MP']
    assert names == expected
    constants = {expected[8]: 19456 if case['kind'] == 'ordinary' else 9728,
                 'BLOCK_N': 64, 'BLOCK_K': 128, 'MP': 16}
    assert set(kernel.constexprs) == {names.index(n) for n in constants}
    signature = {i: '*bf16' if i < 3 else 'i32' for i in range(8)}
    options = dict(num_warps=4, num_stages=3, enable_fp_fusion=True)

    class AlignedPointer:
        # Mirrors the 16-byte-aligned BF16 allocations/views used by the
        # existing launcher. This object only answers JIT specialization queries.
        def data_ptr(self):
            return 0x100000

    values = dict(X=AlignedPointer(), W=AlignedPointer(), Y=AlignedPointer(),
                  M=case['m'], K=2560, stride_xm=2560, stride_wn=2560,
                  stride_ym=19456 if case['kind'] == 'ordinary' else 9728, **constants)
    attrs = kernel._get_config(*(values[n] for n in names))
    divisible = set(attrs.divisible_by_16)
    equal = set(attrs.equal_to_1)
    # Validate the actual installed launcher's result; no floating point
    # alignment assumptions or invented constexpr specialization is supplied.
    expected_divisible = {0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11}
    if case['m'] == 16:
        expected_divisible.add(3)
    assert divisible == expected_divisible and not equal
    cfg = {names.index(n): v for n, v in constants.items()}
    assert 3 not in cfg and 4 not in cfg  # M and K remain runtime i32.
    result = dict(case=case, source=source_record, triton=triton.__version__,
                  target=dict(backend='cuda', capability=90, warp_size=32),
                  signature={names[i]: t for i, t in signature.items()},
                  constexpr=constants, constants_by_index=cfg, options=options,
                  runtime_values={n: values[n] for n in names[3:8]},
                  attributes=dict(divisible_by_16=[names[i] for i in sorted(divisible)],
                                  equal_to_1=[names[i] for i in sorted(equal)]),
                  grid=[152 if case['kind'] == 'paired' else 304], k_tiles=20,
                  cuda_execution=False, started_at=utc_now())
    stem = OUT / case['id']
    t0 = time.monotonic()
    compiled = triton.compile(ASTSource(kernel, signature, cfg, attrs),
                              target=GPUTarget('cuda', 90, 32), options=options)
    result['compile_seconds'] = time.monotonic() - t0
    result['metadata'] = compiled.metadata._asdict()
    for ext in ('ttir', 'ttgir', 'ptx', 'llir'):
        stem.with_suffix('.' + ext).write_text(compiled.asm[ext])
    (OUT / 'triton31_get_config.py.txt').write_text(inspect.getsource(kernel._get_config))
    ptxas, _ = _path_to_binary('ptxas')
    t1 = time.monotonic()
    assembly = subprocess.run([ptxas, '-v', '--gpu-name=sm_90a', str(stem.with_suffix('.ptx')),
                               '-o', str(stem.with_suffix('.cubin'))],
                              capture_output=True, text=True, timeout=45)
    result['ptxas_seconds'] = time.monotonic() - t1
    result.update(ptxas_returncode=assembly.returncode,
                  ptxas_resources=assembly.stdout + assembly.stderr)
    stem.with_suffix('.ptxas.log').write_text(result['ptxas_resources'])
    ptx = compiled.asm['ptx']
    result['ptx_inventory'] = instruction_inventory(ptx)
    mma_lines = [i for i, line in enumerate(ptx.splitlines()) if 'mma.sync.' in line or 'wgmma.mma_async' in line]
    suffix = '\n'.join(ptx.splitlines()[mma_lines[-1] + 1:]) + '\n' if mma_lines else ptx
    # This suffix may contain the loop's backedge and is a textual inspection
    # aid, not a dynamic instruction count or GPU-time estimate.
    stem.with_suffix('.after_last_mma.ptx.txt').write_text(suffix)
    result['after_last_mma_inventory'] = instruction_inventory(suffix)
    result['assembly_sha256'] = {ext: hashlib.sha256(compiled.asm[ext].encode()).hexdigest()
                                 for ext in ('ttir', 'ttgir', 'ptx', 'llir')}
    result['success'] = assembly.returncode == 0
    result['completed_at'] = utc_now()
    stem.with_suffix('.json').write_text(json.dumps(result, indent=2, default=str) + '\n')
    print(json.dumps(dict(id=case['id'], success=result['success'], metadata=result['metadata'],
                         ptxas_resources=result['ptxas_resources']), default=str), flush=True)
    if not result['success']:
        raise SystemExit(1)


def main():
    OUT.mkdir(exist_ok=True)
    if len(sys.argv) > 1:
        child(json.loads(sys.argv[1]))
        return
    manifest = json.loads(MANIFEST.read_text())
    results = []
    for case in manifest['cases']:
        begin = time.monotonic()
        result = dict(case=case, started_at=utc_now())
        try:
            run = subprocess.run([sys.executable, __file__, json.dumps(case)],
                                 capture_output=True, text=True, timeout=90)
            (OUT / (case['id'] + '.compiler.log')).write_text(run.stdout + run.stderr)
            result['returncode'] = run.returncode
            path = OUT / (case['id'] + '.json')
            if path.exists():
                result.update(json.loads(path.read_text()))
            if run.returncode:
                result.update(success=False, error=run.stderr[-8000:])
        except Exception as error:
            result.update(success=False, error=str(error))
        result['total_seconds'] = time.monotonic() - begin
        result['completed_at'] = utc_now()
        results.append(result)
        (OUT / 'interleaved_swiglu_summary.json').write_text(json.dumps(results, indent=2, default=str) + '\n')
        print(json.dumps(result, default=str), flush=True)
    (OUT / 'interleaved_swiglu_source.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if len(results) != 7 or any(not r.get('success') for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
