"""Ten frozen Triton 3.1 SM90 prefix-attention compiles; no driver/GPU use."""
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
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'compile-results'
MANIFEST=Path(__file__).with_name('krxfty_prefix_source.json')


def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(x): return hashlib.sha256(x).hexdigest()


def function_source(text,name):
    node=next(n for n in ast.parse(text).body if getattr(n,'name','')==name)
    first=min([node.lineno]+[n.lineno for n in node.decorator_list])
    return ''.join(text.splitlines(keepends=True)[first-1:node.end_lineno])


def verify_sources(manifest):
    for source in manifest['sources'].values():
        snapshot=(ROOT/source['snapshot']).read_bytes()
        module=(ROOT/source['compiler_module']).read_bytes()
        assert sha(snapshot)==source['source_sha256']
        assert sha(module)==source['compiler_module_sha256']
        exact=function_source(snapshot.decode(),source['kernel'])
        assert exact==function_source(module.decode(),source['kernel'])
        assert sha(exact.encode())==source['decorated_kernel_sha256']


def inventory(ptx):
    patterns={
        'mma_sync':r'\bmma\.sync[^\s]+',
        'wgmma':r'\bwgmma\.mma_async[^\s]+',
        'async_copy':r'\bcp\.async\.(?:ca|cg)\.shared\.global[^;]*;',
        'async_wait':r'\bcp\.async\.wait_group[^;]*;',
        'global_load':r'\bld\.global[^\s]+',
        'global_store':r'\bst\.global[^\s]+',
        'shared_load':r'\bld\.shared[^\s]+',
        'shared_store':r'\bst\.shared[^\s]+',
        'shuffle':r'\bshfl\.[^\s]+',
        'warp_reduce':r'\bredux\.sync[^\s]+',
        'barrier':r'\bbar\.sync[^;]*;',
        'branch':r'\bbra(?:\.uni)?\s+[^;]+;',
        'bf16_convert':r'\bcvt\.[^\s]*bf16[^\s]*',
        'exp2':r'\bex2\.[^\s]+',
        'shift64':r'\bshr\.[su]64',
        'shift_funnel':r'\bshf\.[^\s]+',
        'bit_and':r'\band\.b(?:32|64)',
        'lop3':r'\blop3\.[^\s]+',
    }
    return {name:{'static_count':len(found),'forms':sorted(set(found))}
            for name,pattern in patterns.items() for found in [re.findall(pattern,ptx)]}


def resources(log):
    regs=re.search(r'Used (\d+) registers',log)
    stack=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',log)
    shared=re.search(r'(\d+) bytes smem',log)
    assert regs and stack,log
    return dict(registers_per_thread=int(regs[1]),shared_bytes=int(shared[1]) if shared else 0,
                stack_bytes=int(stack[1]),spill_store_bytes=int(stack[2]),spill_load_bytes=int(stack[3]))


def child(case):
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import _path_to_binary
    assert triton.__version__=='3.1.0'
    manifest=json.loads(MANIFEST.read_text());verify_sources(manifest)
    source=manifest['sources'][case['kind']]
    spec=importlib.util.spec_from_file_location('prefix_'+case['kind'],ROOT/source['compiler_module'])
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    kernel=getattr(module,source['kernel']);names=kernel.arg_names
    expected=['q_ptr','k_ptr','v_ptr','pos_ptr','tree_ptr','o_part_ptr','m_part_ptr','l_part_ptr','o_ptr',
              'CAP','scale','NSPLIT','SPLIT_LEN','HQ','HKV','G','GP','R','D','BLOCK_N','FINAL','TREE']
    assert names==expected
    constants=case['constexpr']
    assert set(kernel.constexprs)=={names.index(n) for n in constants}
    signature={names.index(n):t for n,t in manifest['signature'].items()}

    class AlignedAllocation:
        def data_ptr(self):return 0x100000

    values={n:AlignedAllocation() for n in names[:9]}
    values.update(case['runtime']);values.update(constants)
    attrs=kernel._get_config(*(values[n] for n in names))
    divisible=set(attrs.divisible_by_16);equal=set(attrs.equal_to_1)
    # Match installed JIT rules, including bool constexpr divisibility handling.
    expected_divisible=set(range(9))|{names.index(n) for n,v in values.items() if isinstance(v,int) and v%16==0}
    expected_equal={names.index(n) for n,v in values.items() if isinstance(v,int) and not isinstance(v,bool) and v==1}
    assert divisible==expected_divisible and equal==expected_equal
    assert names.index('scale') not in divisible|equal
    cfg={names.index(n):v for n,v in constants.items()}
    # This is normal JITFunction.run specialization, not an invented constexpr.
    for i in equal:cfg[i]=values[names[i]]
    assert [names[i] for i in sorted(equal)]==case['natural_equal_to_one_folding']
    assert names.index('CAP') not in cfg and names.index('SPLIT_LEN') not in cfg
    options=case['options']
    result=dict(case=case,source=source,triton=triton.__version__,target=manifest['target'],
                signature={names[i]:t for i,t in signature.items()},constants_by_index=cfg,
                attributes={'divisible_by_16':[names[i] for i in sorted(divisible)],'equal_to_1':[names[i] for i in sorted(equal)]},
                cuda_execution=False,started_at=utc())
    stem=OUT/case['id'];stem.with_suffix('.config.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'triton31_get_config.py.txt').write_text(inspect.getsource(kernel._get_config))
    (OUT/'triton31_jit_run.py.txt').write_text(inspect.getsource(kernel.run))
    begin=time.monotonic()
    compiled=triton.compile(ASTSource(kernel,signature,cfg,attrs),target=GPUTarget('cuda',90,32),options=options)
    result['compile_seconds']=time.monotonic()-begin
    result['metadata']=compiled.metadata._asdict()
    for n,v in options.items():assert result['metadata'][n]==v
    assert result['metadata']['enable_fp_fusion'] is True
    for ext in ('ttir','ttgir','llir','ptx'):stem.with_suffix('.'+ext).write_text(compiled.asm[ext])
    ptxas,_=_path_to_binary('ptxas');begin=time.monotonic()
    assembly=subprocess.run([ptxas,'-v','--gpu-name=sm_90a',str(stem.with_suffix('.ptx')),'-o',str(stem.with_suffix('.cubin'))],capture_output=True,text=True,timeout=40)
    result['ptxas_seconds']=time.monotonic()-begin
    result['ptxas_returncode']=assembly.returncode
    log=assembly.stdout+assembly.stderr;stem.with_suffix('.ptxas.log').write_text(log)
    result['ptxas_log']=log
    result['ptxas_resources']=resources(log) if assembly.returncode==0 else None
    result['ptx_inventory']=inventory(compiled.asm['ptx'])
    ttgir=compiled.asm['ttgir']
    result['ttgir_features']=dict(layouts=[line for line in ttgir.splitlines() if re.match(r'#\w+ = #triton_gpu\.',line)],
                                  scf_for_sites=ttgir.count('scf.for'),scf_if_sites=ttgir.count('scf.if'),dot_sites=ttgir.count('tt.dot'),
                                  convert_layout_sites=ttgir.count('convert_layout'),async_copy_sites=ttgir.count('async_copy'),
                                  shift_right_sites=ttgir.count('arith.shrsi'),and_sites=ttgir.count('arith.andi'))
    result['assembly_sha256']={ext:sha(compiled.asm[ext].encode()) for ext in ('ttir','ttgir','llir','ptx')}
    result['cubin_sha256']=sha(stem.with_suffix('.cubin').read_bytes()) if assembly.returncode==0 else None
    result['success']=assembly.returncode==0
    result['torch_imported']='torch' in sys.modules;assert not result['torch_imported']
    result['completed_at']=utc()
    stem.with_suffix('.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
    print(json.dumps({'id':case['id'],'success':result['success'],'resources':result['ptxas_resources']}),flush=True)
    if not result['success']:raise SystemExit(1)


def main():
    OUT.mkdir(exist_ok=True)
    if len(sys.argv)>1:child(json.loads(sys.argv[1]));return
    manifest=json.loads(MANIFEST.read_text());verify_sources(manifest)
    assert len(manifest['cases'])==10
    (OUT/'krxfty_prefix_source.json').write_text(json.dumps(manifest,indent=2)+'\n')
    results=[]
    for case in manifest['cases']:
        begin=time.monotonic();result=dict(case=case,started_at=utc())
        try:
            run=subprocess.run([sys.executable,__file__,json.dumps(case)],capture_output=True,text=True,timeout=50)
            (OUT/(case['id']+'.compiler.log')).write_text(run.stdout+run.stderr)
            result['returncode']=run.returncode
            path=OUT/(case['id']+'.json')
            if path.exists():result.update(json.loads(path.read_text()))
            if run.returncode:result.update(success=False,error=run.stderr[-10000:])
        except Exception as error:result.update(success=False,error=str(error))
        result['total_seconds']=time.monotonic()-begin;result['completed_at']=utc();results.append(result)
        (OUT/'krxfty_prefix_summary.json').write_text(json.dumps(results,indent=2,default=str)+'\n')
        print(json.dumps(result,default=str),flush=True)
        if not result.get('success'):
            # An unexpected compiler failure stops this approved attempt; root reviews.
            raise SystemExit(1)
    assert len(results)==10


if __name__=='__main__':main()
