"""Exactly six frozen sibling-swap CPU-only Triton3.1 SM90 compiles."""
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
import signal

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
    engine=HERE/'snapshots/engine'
    files={p.relative_to(engine).as_posix():sha(p.read_bytes()) for p in sorted(engine.rglob('*')) if p.is_file()}
    assert files==manifest['engine_files'] and len(files)==18
    lines=''.join(digest+'  '+name+'\n' for name,digest in files.items())
    assert sha(lines.encode())==manifest['source_tree_sha256']
    assert (HERE/'snapshots/SOURCE_SHA256.txt').read_text()==lines
    assert sha((HERE/'draft.py').read_bytes())==manifest['compiler_module_sha256']
    exact=function_source((engine/'recycle.py').read_text(),'_draft_kernel')
    assert exact==function_source((HERE/'draft.py').read_text(),'_draft_kernel')
    assert sha(exact.encode())==manifest['decorated_draft_sha256']
    assert (HERE/'pair_cache.py').read_bytes()==(engine/'pair_cache.py').read_bytes()
    assert sha(function_source((HERE/'pair_cache.py').read_text(),'_pair_slot').encode())==manifest['decorated_pair_slot_sha256']

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



def compile_case(case):
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import _path_to_binary
    from draft import _draft_kernel
    assert triton.__version__ == '3.1.0'
    manifest = json.loads(MANIFEST.read_text())
    verify_source(manifest)
    kernel = _draft_kernel
    expected = list(manifest['pointer_signature']) + ['K','R','S','SP','VOCAB']
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
    options = {}
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
        raise RuntimeError('PTXAS failed')
    return result

def main():
    OUT.mkdir(exist_ok=True)
    manifest=json.loads(MANIFEST.read_text())
    verify_source(manifest)
    (OUT/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    assert len(manifest['cases'])==6
    assert [c['constexpr']['R'] for c in manifest['cases']]==[4,8,16,32,42,64]
    def expired(signum,frame):raise TimeoutError('40-second case budget expired')
    signal.signal(signal.SIGALRM,expired)
    results=[]
    for case in manifest['cases']:
        begin=time.monotonic()
        signal.alarm(40)
        try:
            result=compile_case(case)
            result['returncode']=0
        except Exception as error:
            result=dict(case=case,success=False,error=repr(error))
            raise
        finally:
            signal.alarm(0)
            result['total_seconds']=time.monotonic()-begin
            results.append(result)
            (OUT/'summary.json').write_text(json.dumps(results,indent=2,default=str)+'\n')
    verify_source(manifest)
    assert len(results)==6 and all(r['success'] for r in results)

if __name__=='__main__':main()
