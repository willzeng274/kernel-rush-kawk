"""Exactly seven frozen Triton 3.1 SM90 builds; never creates a GPU context."""
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

HERE=Path(__file__).resolve().parent
OUT=HERE/'compile-results'
PREP=HERE/'snapshots/preparation'
REPRESENTATIVES=['A_qkv_main','A_o_main','B_qkv_main','B_o_main','C_qkv_main','C_o_main','C_qkv_sum']

def sha(data):return hashlib.sha256(data).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,default=str)+'\n')
def function_source(text,name):
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
    return ''.join(text.splitlines(keepends=True)[min([node.lineno]+[n.lineno for n in node.decorator_list])-1:node.end_lineno])

def verify_source():
    m=json.loads((HERE/'SOURCE_MANIFEST.json').read_text())
    for label,key in [('engine','engine_files'),('preparation','preparation_files')]:
        root=HERE/'snapshots'/label
        files={p.relative_to(root).as_posix():sha(p.read_bytes()) for p in root.rglob('*') if p.is_file()}
        assert files==m[key],label
    assert len(m['engine_files'])==19 and len(m['preparation_files'])==23
    assert sha(json.dumps(m['engine_files'],sort_keys=True).encode())==m['canonical_tree_sha256']
    frozen=json.loads((PREP/'SOURCE_MANIFEST.json').read_text())
    assert sha(''.join(r['path']+'\0'+r['sha256']+'\n' for r in frozen['files']).encode())==m['source_set_sha256']
    assert m['source_set_sha256']=='0d23da3fc2a5523b6ca207af5049f812344e326137fc454c81299a6362c806ad'
    assert sha((HERE/'gemm.py').read_bytes())==m['compiler_module_sha256']
    for name,digest in m['decorated_function_sha256'].items():
        source=function_source((HERE/'snapshots/engine/kernels/gemm.py').read_text(),name)
        assert source==function_source((HERE/'gemm.py').read_text(),name)
        assert sha(source.encode())==digest
    assert m['compile_representatives']==REPRESENTATIVES
    return m


def load_native():
    import triton
    import triton.runtime.jit as jit
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import make_backend
    assert triton.__version__=='3.1.0'
    expected=json.loads((PREP/'NATIVE_BINDER_KEYS.json').read_text())
    assert sha(Path(inspect.getsourcefile(jit)).read_bytes())==expected['jit_source_sha256']
    assert sha((HERE/'snapshots/engine/kernels/gemm.py').read_bytes())==expected['kernel_source_sha256']
    spec=importlib.util.spec_from_file_location('frozen_wide_gemm',HERE/'gemm.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return triton,module,make_backend(GPUTarget('cuda',90,32))


class Pointer:
    def __init__(self,dtype):self.dtype=dtype
    def data_ptr(self):return 0x100000


def native_matrix(module,backend):
    plan=json.loads((PREP/'COMPILE_PLAN.json').read_text())
    expected=json.loads((PREP/'NATIVE_BINDER_KEYS.json').read_text())
    assert plan['minimal_representatives']==REPRESENTATIVES
    assert len(plan['cases'])==12 and plan['distinct_native_keys']==7 and plan['reduction_launches']==4
    assert plan['bounds']==dict(single_worker=True,total_seconds=180,case_seconds=40,retry=False,expand_search=False)
    results=[];payloads={};mapping=[]
    for case in plan['cases']:
        m,n,k=case['shape_MNK'];cfg=case['constexpr'];r=case['runtime_scalars']
        split=cfg['SPLIT_K'];bn=cfg['BLOCK_N'];kp=((k+split-1)//split+63)//64*64
        assert m==64 and cfg['BLOCK_M']==64 and cfg['BLOCK_K']==64 and cfg['NORM'] is False
        assert r==dict(M=m,N=n,K=k,stride_am=k,stride_wn=k,k_per_split=kp,num_tiles=((n+bn-1)//bn)*split,eps=0.0)
        assert case['grid_assuming_132_SMs']==[min(r['num_tiles'],132*(2 if split==2 else 1))]
        assert case['part_bytes']==(split*m*n*4 if split>1 else 0)
        for phase in (['main','sum'] if case['reduction'] else ['main']):
            obj=case if phase=='main' else case['reduction']
            name='_skinny_kernel' if phase=='main' else '_sum_kernel'
            kernel=getattr(module,name)
            kernel.create_binder()
            if phase=='main':
                values={n:Pointer('torch.float32' if n=='part_ptr' and split>1 else 'torch.bfloat16') for n in case['tensor_signature']}
                options=dict(num_warps=case['launch_options']['num_warps'],num_stages=case['launch_options']['num_stages'],debug=kernel.debug)
            else:
                assert obj['runtime_scalars']=={'MN':m*n} and obj['constexpr']=={'SPLIT_K':2,'BLOCK':1024}
                assert obj['grid']==[(m*n+1023)//1024]
                values=dict(part_ptr=Pointer('torch.float32'),c_ptr=Pointer('torch.bfloat16'))
                options=dict(num_warps=4,debug=kernel.debug)
            values.update(obj['runtime_scalars']);values.update(obj['constexpr'])
            bound,sigspec,constvals,nonconst,extra=kernel.binder(**values,**options)
            attrs=kernel._get_config(*bound.values())
            signature={kernel.params[i].name:v for i,v in zip(kernel.non_constexpr_indices,sigspec)}
            constants={p.name:v for p,v in zip(kernel.params,bound.values()) if p.is_constexpr or p.num in attrs.equal_to_1 or v is None}
            assert constants==obj['constexpr'], 'runtime value unexpectedly folded'
            key=name+''.join(sigspec)+str((constvals,extra))
            launch=case['id']+'_'+phase
            observed=dict(launch=launch,kernel=name,native_key_sha256=sha(key.encode()),signature=signature,constants=constants,attrs=dict(divisible_by_16=sorted(attrs.divisible_by_16),equal_to_1=sorted(attrs.equal_to_1)),options_explicit=options)
            results.append(observed)
            parsed=backend.parse_options(options)
            assert parsed.num_ctas==1 and parsed.enable_fp_fusion is True
            assert parsed.num_stages==options.get('num_stages',3) and parsed.num_warps==options['num_warps']
            payloads[launch]=(kernel,signature,constants,attrs,parsed.__dict__,observed)
            mapping.append(dict(launch=launch,shape_MNK=case['shape_MNK'],runtime_values=obj['runtime_scalars'],grid=case['grid_assuming_132_SMs'] if phase=='main' else obj['grid'],part_bytes=case['part_bytes'],native_key_sha256=observed['native_key_sha256'],native_parsed_options=parsed.__dict__))
    groups={}
    for r in results:groups.setdefault(r['native_key_sha256'],[]).append(r['launch'])
    # Descriptive provenance fields are immutable snapshot data; all executable
    # key/signature/attribute/constant/option fields are reconstructed above.
    actual={**expected,'launch_count':len(results),'distinct_keys':len(groups),'groups':groups,'launches':results}
    write('NATIVE_BINDER_ACTUAL.json',actual)
    assert actual==expected,'STOP: complete frozen native binder JSON mismatch'
    assert len(results)==16 and len(groups)==7
    assert {payloads[n][5]['native_key_sha256'] for n in REPRESENTATIVES}==set(groups)
    for r in mapping:
        r['compiled_representative']=next(n for n in REPRESENTATIVES if payloads[n][5]['native_key_sha256']==r['native_key_sha256'])
    write('RUNTIME_MAPPING.json',mapping)
    assert 'torch' not in sys.modules
    return payloads


def inventory(ptx):
    patterns={'wgmma':r'\bwgmma\.[^\s;]+','mma':r'\bmma\.[^\s;]+','async_copy':r'\bcp\.async[^\s;]+','async_wait':r'\b(?:cp\.async\.wait_group|wgmma\.wait_group)[^;]*;', 'barrier':r'\b(?:bar\.sync|mbarrier\.[^\s;]+)[^;]*;', 'global_store':r'\bst\.global[^\s;]+','global_load':r'\bld\.global[^\s;]+','bf16_convert':r'\bcvt\.[^\s;]*bf16[^\s;]*','branch':r'\bbra(?:\.uni)?\s+[^;]+;'}
    return {n:dict(static_count=len(v),forms=sorted(set(v))) for n,p in patterns.items() for v in [re.findall(p,ptx)]}


def resources(log):
    regs=re.search(r'Used (\d+) registers',log)
    stack=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',log)
    shared=re.search(r'(\d+) bytes smem',log)
    assert regs and stack,log
    return dict(registers_per_thread=int(regs[1]),shared_bytes=int(shared[1]) if shared else 0,stack_bytes=int(stack[1]),spill_store_bytes=int(stack[2]),spill_load_bytes=int(stack[3]))


def child(launch):
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import _path_to_binary
    verify_source()
    triton,module,backend=load_native()
    payloads=native_matrix(module,backend)
    kernel,signature,constants,attrs,options,record=payloads[launch]
    result=dict(id=launch,native=record,parsed_options=options,triton=triton.__version__,target=dict(backend='cuda',arch=90,warp_size=32),cuda_execution=False,started_at=utc())
    stem=OUT/launch
    write(launch+'.config.json',result)
    begin=time.monotonic()
    compiled=triton.compile(ASTSource(kernel,signature,constants,attrs),target=GPUTarget('cuda',90,32),options=options)
    result['compile_seconds']=time.monotonic()-begin
    result['metadata']=compiled.metadata._asdict()
    for ext in ('ttir','ttgir','llir','ptx'):
        stem.with_suffix('.'+ext).write_text(compiled.asm[ext])
    stem.with_suffix('.cubin').write_bytes(compiled.asm['cubin'])
    ptxas,version=_path_to_binary('ptxas')
    result['ptxas_binary']=ptxas;result['ptxas_version']=version
    assembly=subprocess.run([ptxas,'-v','--gpu-name=sm_90a',str(stem.with_suffix('.ptx')),'-o',str(stem.with_suffix('.verbose.cubin'))],capture_output=True,text=True,timeout=30)
    log=assembly.stdout+assembly.stderr
    stem.with_suffix('.ptxas.log').write_text(log)
    result['ptxas_returncode']=assembly.returncode;result['ptxas_log']=log
    result['resources']=resources(log) if assembly.returncode==0 else None
    result['ptx_inventory']=inventory(compiled.asm['ptx'])
    result['ttgir_features']=dict(layout_definitions=[l for l in compiled.asm['ttgir'].splitlines() if l.startswith('#')],scf_for_sites=compiled.asm['ttgir'].count('scf.for'),async_copy_sites=compiled.asm['ttgir'].count('async_copy_global_to_local'),async_wait_sites=compiled.asm['ttgir'].count('async_wait'),wgmma_sites=compiled.asm['ttgir'].count('warp_group_dot'))
    result['artifact_sha256']={ext:sha(stem.with_suffix('.'+ext).read_bytes()) for ext in ('ttir','ttgir','llir','ptx','cubin','verbose.cubin','ptxas.log') if stem.with_suffix('.'+ext).exists()}
    assert result['metadata']['num_ctas']==1 and result['metadata']['enable_fp_fusion'] is True
    assert result['metadata']['num_stages']==options['num_stages'] and result['metadata']['num_warps']==options['num_warps']
    result['torch_imported']='torch' in sys.modules
    assert not result['torch_imported']
    verify_source()
    result['success']=assembly.returncode==0
    result['completed_at']=utc()
    write(launch+'.json',result)
    print(json.dumps(dict(id=launch,success=result['success'],resources=result['resources'])),flush=True)
    if not result['success']:raise SystemExit(1)


def main():
    OUT.mkdir(exist_ok=True)
    if len(sys.argv)>1:
        assert sys.argv[1] in REPRESENTATIVES
        child(sys.argv[1]);return
    started=time.monotonic()
    manifest=verify_source()
    triton,module,backend=load_native()
    native_matrix(module,backend) # All sixteen keys validated before first build.
    import triton.runtime.jit as jit
    (OUT/'native_triton310_jit.py.txt').write_bytes(Path(inspect.getsourcefile(jit)).read_bytes())
    write('NATIVE_PREFLIGHT.json',dict(status='PASS',launches=16,distinct_keys=7,representatives=REPRESENTATIVES,source_set_sha256=manifest['source_set_sha256'],expected_json_sha256=sha((PREP/'NATIVE_BINDER_KEYS.json').read_bytes()),cuda_execution=False))
    results=[]
    for launch in REPRESENTATIVES:
        remaining=180-(time.monotonic()-started)
        assert remaining>0,'Total deadline expired'
        begin=time.monotonic();result=dict(id=launch,started_at=utc())
        try:
            run=subprocess.run([sys.executable,__file__,launch],capture_output=True,text=True,timeout=min(40,remaining))
            (OUT/(launch+'.compiler.stdout')).write_text(run.stdout)
            (OUT/(launch+'.compiler.stderr')).write_text(run.stderr)
            result['returncode']=run.returncode
            path=OUT/(launch+'.json')
            if path.exists():result.update(json.loads(path.read_text()))
            if run.returncode:result.update(success=False,error=run.stderr[-10000:])
        except Exception as error:result.update(success=False,error=str(error))
        result['total_seconds']=time.monotonic()-begin
        results.append(result);write('summary.json',results)
        print(json.dumps(dict(id=launch,success=result.get('success',False),resources=result.get('resources'),total_seconds=result['total_seconds'],error=result.get('error'))),flush=True)
        if not result.get('success'):raise SystemExit(1) # Never retry or expand.
    verify_source()
    assert len(results)==7
    write('COMPLETION.json',dict(status='PASS',compile_calls=7,launches_covered=16,total_seconds=time.monotonic()-started,source_verified_before_after=True,cuda_execution=False,completed_at=utc()))

if __name__=='__main__':main()
