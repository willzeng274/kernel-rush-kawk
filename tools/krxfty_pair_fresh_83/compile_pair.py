"""Exactly fourteen frozen pair-cache CPU-only Triton3.1 SM90 compiles."""
import ast
import datetime
import hashlib
import inspect
import json
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
OUT = HERE / 'compile-results'
MANIFEST = HERE / 'SOURCE_MANIFEST.json'

def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha(data):
    return hashlib.sha256(data).hexdigest()

def function_source(text, name):
    node = next(n for n in ast.parse(text).body if getattr(n, 'name', '') == name)
    first = min([node.lineno] + [n.lineno for n in node.decorator_list])
    return ''.join(text.splitlines(keepends=True)[first-1:node.end_lineno])

def verify_source(manifest):
    assert sha((HERE/'draft.py').read_bytes()) == manifest['compiler_module_sha256']
    for source in manifest['sources'].values():
        frozen = HERE / source['snapshot']
        assert sha(frozen.read_bytes()) == source['source_sha256']
        if 'function' in source:
            exact = function_source(frozen.read_text(), source['function'])
            assert exact == function_source((HERE/source['module']).read_text(), source['function'])
            assert sha(exact.encode()) == source['decorated_function_sha256']
        for name, item in source.get('functions', {}).items():
            assert sha(function_source(frozen.read_text(), name).encode()) == item['decorated_function_sha256']

def inventory(ptx):
    patterns = {
        'global_load': r'\bld\.global[^\s]+',
        'global_store': r'\bst\.global[^\s]+',
        'shared_load': r'\bld\.shared[^\s]+',
        'shared_store': r'\bst\.shared[^\s]+',
        'shuffle': r'\bshfl\.[^\s]+',
        'warp_reduce': r'\bred\.sync[^\s]+',
        'barrier': r'\bbar\.sync[^;]*;',
        'branch': r'\bbra(?:\.uni)?\s+[^;]+;',
        'mma': r'\b(?:mma|wgmma)\.[^\s]+',
    }
    return {name: {'static_count': len(found), 'forms': sorted(set(found))}
            for name, pattern in patterns.items() for found in [re.findall(pattern, ptx)]}


def ptxas_resources(log):
    regs = re.search(r'Used (\d+) registers', log)
    stack = re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads', log)
    shared = re.search(r'(\d+) bytes smem', log)
    assert regs and stack, log
    return dict(registers_per_thread=int(regs[1]), shared_bytes=int(shared[1]) if shared else 0,
                stack_bytes=int(stack[1]), spill_store_bytes=int(stack[2]), spill_load_bytes=int(stack[3]))



def child(case):
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import _path_to_binary
    from draft import _draft_kernel
    from pair_cache import _publish_pairs_kernel
    assert triton.__version__ == '3.1.0'
    manifest = json.loads(MANIFEST.read_text())
    verify_source(manifest)
    kernel = _draft_kernel if case['kind'] == 'draft' else _publish_pairs_kernel
    expected = (['root_ptr', 'table_ptr', 'parent_ptr', 'rank_ptr', 'spine_slot_ptr', 'spine_ptr',
                 'nseen_ptr', 'anchor_ptr', 'blk_ptr', 'root_prev_ptr', 'node_keys_ptr',
                 'pair_keys_ptr', 'pair_values_ptr', 'K', 'R', 'S', 'SP'] if case['kind'] == 'draft' else
                ['pair_keys_ptr', 'pair_values_ptr', 'node_keys_ptr', 'top_ptr', 'blk_ptr',
                 'root_prev_ptr', 'path_idx_ptr', 'path_len_ptr', 'K', 'R', 'MAXA', 'P'])
    names = kernel.arg_names
    assert names == expected
    constants = case['constexpr']
    cfg = {names.index(n): v for n, v in constants.items()}
    assert set(kernel.constexprs) == set(cfg)
    signature = {i:'*'+dtype for i, dtype in enumerate(case['pointer_dtypes'])}
    pointer_count = len(signature)
    class AlignedAllocation:
        def data_ptr(self):
            return 0x100000
    values = {n:AlignedAllocation() for n in names[:pointer_count]}
    values.update(constants)
    attrs = kernel._get_config(*(values[n] for n in names))
    divisible, equal = set(attrs.divisible_by_16), set(attrs.equal_to_1)
    assert divisible == set(range(pointer_count)) | {names.index(n) for n,v in constants.items() if v%16==0}
    assert equal == {names.index(n) for n,v in constants.items() if v==1}
    options = {} if case['kind'] == 'draft' else {'num_warps':1}
    result = dict(case=case, triton=triton.__version__, target={'backend':'cuda', 'capability':90, 'warp_size':32},
                  signature={names[i]:t for i,t in signature.items()}, constexpr=constants, constants_by_index=cfg,
                  requested_options=options, attributes={'divisible_by_16':[names[i] for i in sorted(divisible)],
                  'equal_to_1':[names[i] for i in sorted(equal)]}, cuda_execution=False, started_at=utc_now())
    stem = OUT / case['id']
    stem.with_suffix('.config.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'triton31_get_config.py.txt').write_text(inspect.getsource(kernel._get_config))
    begin = time.monotonic()
    compiled = triton.compile(ASTSource(kernel,signature,cfg,attrs),target=GPUTarget('cuda',90,32),options=options)
    result['compile_seconds'] = time.monotonic()-begin
    result['metadata'] = compiled.metadata._asdict()
    assert result['metadata']['num_warps'] == case['num_warps']
    assert result['metadata']['num_stages'] == 3 and result['metadata']['enable_fp_fusion'] is True
    for ext in ('ttir','ttgir','llir','ptx'):
        stem.with_suffix('.'+ext).write_text(compiled.asm[ext])
    ptxas, _ = _path_to_binary('ptxas')
    begin = time.monotonic()
    assembly = subprocess.run([ptxas,'-v','--gpu-name=sm_90a',str(stem.with_suffix('.ptx')),
                               '-o',str(stem.with_suffix('.cubin'))],capture_output=True,text=True,timeout=30)
    result['ptxas_seconds'] = time.monotonic()-begin
    result['ptxas_returncode'] = assembly.returncode
    log = assembly.stdout+assembly.stderr
    stem.with_suffix('.ptxas.log').write_text(log)
    result['ptxas_log'] = log
    result['ptxas_resources'] = ptxas_resources(log) if assembly.returncode==0 else None
    result['ptx_inventory'] = inventory(compiled.asm['ptx'])
    result['assembly_sha256'] = {ext:sha(compiled.asm[ext].encode()) for ext in ('ttir','ttgir','llir','ptx')}
    result['cubin_sha256'] = sha(stem.with_suffix('.cubin').read_bytes()) if assembly.returncode==0 else None
    result['success'] = assembly.returncode==0
    result['torch_imported'] = 'torch' in sys.modules
    assert not result['torch_imported']
    result['completed_at'] = utc_now()
    stem.with_suffix('.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
    print(json.dumps({'id':case['id'],'success':result['success'],'resources':result['ptxas_resources']}),flush=True)
    if not result['success']:
        raise SystemExit(1)

def main():
    OUT.mkdir(exist_ok=True)
    if len(sys.argv) > 1:
        child(json.loads(sys.argv[1]))
        return
    manifest = json.loads(MANIFEST.read_text())
    verify_source(manifest)
    (OUT / 'SOURCE_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    results = []
    for case in manifest['cases']:
        begin = time.monotonic()
        result = {'case': case, 'started_at': utc_now()}
        try:
            run = subprocess.run([sys.executable, __file__, json.dumps(case)],
                                 capture_output=True, text=True, timeout=40)
            (OUT / (case['id'] + '.compiler.log')).write_text(run.stdout + run.stderr)
            result['returncode'] = run.returncode
            path = OUT / (case['id'] + '.json')
            if path.exists():
                result.update(json.loads(path.read_text()))
            if run.returncode:
                result.update(success=False, error=run.stderr[-10000:])
        except Exception as error:
            result.update(success=False, error=str(error))
        result['total_seconds'] = time.monotonic() - begin
        result['completed_at'] = utc_now()
        results.append(result)
        (OUT / 'summary.json').write_text(json.dumps(results, indent=2, default=str) + '\n')
        print(json.dumps(result, default=str), flush=True)
    if len(results) != 14 or any(not r.get('success') for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
