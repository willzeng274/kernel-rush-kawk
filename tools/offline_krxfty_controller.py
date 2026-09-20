"""Bounded exact-source Triton 3.1 SM90 controller compilation without a GPU."""
import ast
import datetime
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import re
import subprocess
import sys
import textwrap
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'compile-results'
MANIFEST = Path(__file__).with_name('krxfty_controller_source.json')


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def function_source(text, name):
    node = next(n for n in ast.parse(text).body if getattr(n, 'name', '') == name)
    first = min([node.lineno] + [n.lineno for n in node.decorator_list])
    return textwrap.dedent(''.join(text.splitlines(keepends=True)[first - 1:node.end_lineno]))


def verify_source(manifest):
    module = ROOT / manifest['compiler_module']
    assert sha(module.read_bytes()) == manifest['compiler_module_sha256']
    for source in manifest['sources'].values():
        frozen = ROOT / source['snapshot']
        assert sha(frozen.read_bytes()) == source['source_sha256']
        original = function_source(frozen.read_text(), source['function'])
        compiled = function_source(module.read_text(), source['function'])
        assert original == compiled
        assert sha(compiled.encode()) == source['decorated_function_sha256']
    return module


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
    assert triton.__version__ == '3.1.0'
    manifest = json.loads(MANIFEST.read_text())
    source = verify_source(manifest)
    spec = importlib.util.spec_from_file_location('exact_controller', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    kernel = getattr(module, case['kernel'])
    names = kernel.arg_names
    constants = case['constexpr']
    if case['kind'] == 'accept':
        expected = ['blk_ptr', 'cand_ptr', 'child_start_ptr', 'child_list_ptr', 'child_par_ptr',
                    'done_ptr', 'nseen_ptr', 'pos_ptr', 'limit_ptr', 'root_ptr', 'path_idx_ptr',
                    'path_len_ptr', 'acc_tok_ptr', 'acc_cnt_ptr', 'CAP', 'R', 'C', 'P', 'MAXA', 'GUARD']
        pointer_count = 14
        wide_pointers = {0, 1, 9, 12}
        options = {'num_warps': 1}
    else:
        expected = ['root_ptr', 'table_ptr', 'parent_ptr', 'rank_ptr', 'spine_slot_ptr', 'spine_ptr',
                    'nseen_ptr', 'anchor_ptr', 'blk_ptr', 'K', 'R', 'S', 'SP']
        pointer_count = 9
        wide_pointers = {0, 5, 8}
        options = {}  # Actual draft launcher passes no backend options.
    assert names == expected
    assert set(kernel.constexprs) == {names.index(n) for n in constants}
    signature = {i: '*i64' if i in wide_pointers else '*i32' for i in range(pointer_count)}
    if case['kind'] == 'accept':
        signature[names.index('CAP')] = 'i32'
        assert 'CAP' not in constants

    class AlignedAllocation:
        def data_ptr(self):
            return 0x100000

    values = {n: AlignedAllocation() for n in names[:pointer_count]}
    values.update(case['runtime_values'])
    values.update(constants)
    attrs = kernel._get_config(*(values[n] for n in names))
    divisible = set(attrs.divisible_by_16)
    equal = set(attrs.equal_to_1)
    expected_divisible = set(range(pointer_count)) | {names.index(n) for n, v in values.items() if isinstance(v, int) and v % 16 == 0}
    expected_equal = {names.index(n) for n, v in constants.items() if v == 1}
    assert divisible == expected_divisible and equal == expected_equal
    cfg = {names.index(n): v for n, v in constants.items()}
    # Equal-to-one may include constexpr K=1, but no runtime value is folded.
    assert all(i in cfg for i in equal)
    result = dict(case=case, source=manifest['sources'][case['kind']], triton=triton.__version__,
                  target={'backend': 'cuda', 'capability': 90, 'warp_size': 32},
                  signature={names[i]: t for i, t in signature.items()},
                  constexpr=constants, constants_by_index=cfg, requested_options=options,
                  attributes={'divisible_by_16': [names[i] for i in sorted(divisible)],
                              'equal_to_1': [names[i] for i in sorted(equal)]},
                  cuda_execution=False, started_at=utc_now())
    stem = OUT / case['id']
    stem.with_suffix('.config.json').write_text(json.dumps(result, indent=2) + '\n')
    (OUT / 'triton31_get_config.py.txt').write_text(inspect.getsource(kernel._get_config))
    begin = time.monotonic()
    compiled = triton.compile(ASTSource(kernel, signature, cfg, attrs),
                              target=GPUTarget('cuda', 90, 32), options=options)
    result['compile_seconds'] = time.monotonic() - begin
    result['metadata'] = compiled.metadata._asdict()
    assert result['metadata']['num_warps'] == case['num_warps']
    assert result['metadata']['num_stages'] == 3
    assert result['metadata']['enable_fp_fusion'] is True
    for ext in ('ttir', 'ttgir', 'llir', 'ptx'):
        stem.with_suffix('.' + ext).write_text(compiled.asm[ext])
    ptxas, _ = _path_to_binary('ptxas')
    begin = time.monotonic()
    assembly = subprocess.run([ptxas, '-v', '--gpu-name=sm_90a', str(stem.with_suffix('.ptx')),
                               '-o', str(stem.with_suffix('.cubin'))],
                              capture_output=True, text=True, timeout=30)
    result['ptxas_seconds'] = time.monotonic() - begin
    result['ptxas_returncode'] = assembly.returncode
    log = assembly.stdout + assembly.stderr
    result['ptxas_log'] = log
    stem.with_suffix('.ptxas.log').write_text(log)
    result['ptxas_resources'] = ptxas_resources(log) if assembly.returncode == 0 else None
    result['ptx_inventory'] = inventory(compiled.asm['ptx'])
    result['ttgir_features'] = {
        'layout_definitions': [line for line in compiled.asm['ttgir'].splitlines() if line.startswith('#')],
        'reduce_sites': compiled.asm['ttgir'].count('"tt.reduce"'),
        'scf_for_sites': compiled.asm['ttgir'].count('scf.for'),
        'scf_while_sites': compiled.asm['ttgir'].count('scf.while'),
        'convert_layout_sites': compiled.asm['ttgir'].count('convert_layout'),
    }
    result['assembly_sha256'] = {ext: sha(compiled.asm[ext].encode()) for ext in ('ttir', 'ttgir', 'llir', 'ptx')}
    result['cubin_sha256'] = sha(stem.with_suffix('.cubin').read_bytes()) if assembly.returncode == 0 else None
    result['success'] = assembly.returncode == 0
    result['torch_imported'] = 'torch' in sys.modules
    assert not result['torch_imported']
    result['completed_at'] = utc_now()
    stem.with_suffix('.json').write_text(json.dumps(result, indent=2, default=str) + '\n')
    print(json.dumps({'id': case['id'], 'success': result['success'], 'resources': result['ptxas_resources']}), flush=True)
    if not result['success']:
        raise SystemExit(1)


def main():
    OUT.mkdir(exist_ok=True)
    if len(sys.argv) > 1:
        child(json.loads(sys.argv[1]))
        return
    manifest = json.loads(MANIFEST.read_text())
    verify_source(manifest)
    (OUT / 'krxfty_controller_source.json').write_text(json.dumps(manifest, indent=2) + '\n')
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
        (OUT / 'krxfty_controller_summary.json').write_text(json.dumps(results, indent=2, default=str) + '\n')
        print(json.dumps(result, default=str), flush=True)
    if len(results) != 14 or any(not r.get('success') for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
