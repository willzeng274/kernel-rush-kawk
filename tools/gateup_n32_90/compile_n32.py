"""Two authorized source-exact SM90 builds; one CPU worker, no GPU context."""
import datetime,hashlib,importlib.util,inspect,json,os,re,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=HERE/'compile-results'
CASES=['m64_final32_dot64_k128_s2','m64_final32_dot64_k64_s3']
sha=lambda b:hashlib.sha256(b).hexdigest()
utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,default=str)+'\n')
def verify_source():
    m=json.loads((HERE/'SOURCE_MANIFEST.json').read_text())
    for label,key in [('engine','engine_files'),('preparation','preparation_files'),('primary','primary_files')]:
        folder=HERE/'snapshots'/label
        assert {p.relative_to(folder).as_posix():sha(p.read_bytes()) for p in folder.rglob('*') if p.is_file()}==m[key],label
    assert len(m['engine_files'])==20 and m['cases']==CASES
    assert m['bounds']==dict(single_worker=True,total_seconds=180,case_seconds=40,retry=False,expand_search=False)
    listing=(HERE/'snapshots/preparation/SOURCE_SHA256.txt').read_bytes()
    assert sha(listing)==m['source_set_sha256']=='912f2a1d09b38a98773613746175583f8a11ee336447792e35c0642d6a192773'
    expected={r['path']:r['sha256'] for r in json.loads((HERE/'snapshots/preparation/SOURCE_MANIFEST.json').read_text())['files']}
    assert expected==m['engine_files']
    assert (HERE/'gateup_n32.py').read_bytes()==(HERE/'snapshots/engine/kernels/gateup_n32.py').read_bytes()
    assert sha((HERE/'gateup_n32.py').read_bytes())==m['kernel_sha256']=='7059cb67f089f9b1654347524347b4f559a91e355e88dfc9023c5048b341fe87'
    return m
class Pointer:
    dtype='torch.bfloat16'
    def data_ptr(self):return 0x100000

def native():
    import triton
    import triton.runtime.jit as jit
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import make_backend
    expected=json.loads((HERE/'NATIVE_EXPECTED.json').read_text())
    assert triton.__version__=='3.1.0'
    assert os.environ.get('DISABLE_MMA_V3','0').lower() in ('','0','false')
    assert sha(Path(inspect.getsourcefile(jit)).read_bytes())==expected['jit_source_sha256']
    backend=make_backend(GPUTarget('cuda',90,32))
    backend_path=Path(backend.parse_options.__func__.__code__.co_filename)
    assert sha(backend_path.read_bytes())==expected['backend_source_sha256']
    spec=importlib.util.spec_from_file_location('frozen_gateup_n32',HERE/'gateup_n32.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    kernel=module.gateup_n32_kernel
    assert kernel.arg_names==['X','W','Y','stride_xm','stride_wn','stride_ym','K','I','BLOCK_K']
    kernel.create_binder()
    observed=[];payload={}
    for name,bk,stages in zip(CASES,(128,64),(2,3)):
        values=dict(X=Pointer(),W=Pointer(),Y=Pointer(),stride_xm=2560,stride_wn=2560,stride_ym=9728,K=2560,I=9728,BLOCK_K=bk)
        explicit=dict(num_warps=4,num_stages=stages,enable_fp_fusion=True,debug=kernel.debug)
        bound,sigspec,constvals,nonconst,extra=kernel.binder(**values,**explicit)
        attrs=kernel._get_config(*bound.values())
        signature={kernel.params[i].name:v for i,v in zip(kernel.non_constexpr_indices,sigspec)}
        constants={p.name:v for p,v in zip(kernel.params,bound.values()) if p.is_constexpr or p.num in attrs.equal_to_1 or v is None}
        assert signature==dict(X='*bf16',W='*bf16',Y='*bf16',stride_xm='i32',stride_wn='i32',stride_ym='i32')
        assert constants==dict(K=2560,I=9728,BLOCK_K=bk),'runtime stride folded into constant'
        assert set(attrs.divisible_by_16)==set(range(9)) and not attrs.equal_to_1
        key='gateup_n32_kernel'+''.join(sigspec)+str((constvals,extra))
        parsed=backend.parse_options(explicit)
        assert parsed.num_warps==4 and parsed.num_stages==stages and parsed.num_ctas==1 and parsed.enable_fp_fusion is True
        normalized=dict(parsed.__dict__)
        assert len(parsed.extern_libs)==1 and parsed.extern_libs[0][0]=='libdevice' and Path(parsed.extern_libs[0][1]).name=='libdevice.10.bc'
        normalized['extern_libs']=[['libdevice','<TRITON_LIBDEVICE>']]
        record=dict(case=name,kernel='gateup_n32_kernel',native_key_sha256=sha(key.encode()),signature=signature,constants=constants,attrs=dict(divisible_by_16=sorted(attrs.divisible_by_16),equal_to_1=sorted(attrs.equal_to_1)),options_explicit=explicit,parsed_options=normalized)
        observed.append(record)
        payload[name]=(kernel,signature,constants,attrs,parsed.__dict__,record)
    assert json.loads(json.dumps(observed))==expected['launches'],'native binder/parsed-options mismatch'
    write('NATIVE_BINDER_ACTUAL.json',dict(launches=observed,jit_source_sha256=expected['jit_source_sha256'],backend_source_sha256=expected['backend_source_sha256'],kernel_source_sha256=expected['kernel_source_sha256'],cuda_execution=False))
    assert 'torch' not in sys.modules
    return payload

def inventory(ptx):
    patterns={'wgmma':r'\bwgmma\.mma_async[^;]*;','mma_sync':r'\bmma\.sync[^;]*;','async_copy':r'\bcp\.async\.(?:ca|cg)\.shared\.global[^;]*;','async_wait':r'\b(?:cp\.async\.wait_group|wgmma\.wait_group)[^;]*;','shared_load':r'\bld\.shared[^;]*;','shared_store':r'\bst\.shared[^;]*;','stmatrix':r'\bstmatrix\.[^;]*;','barrier':r'\bbar\.sync[^;]*;','bf16_cast':r'\bcvt\.[^\s;]*bf16[^\s;]*','exp2':r'\bex2\.[^\s;]*','local_load':r'\bld\.local[^;]*;','local_store':r'\bst\.local[^;]*;'}
    return {k:dict(static_count=len(v),forms=sorted(set(v))) for k,p in patterns.items() for v in [re.findall(p,ptx)]}
def resources(log):
    regs=re.search(r'Used (\d+) registers',log)
    stack=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',log)
    assert regs and stack,log
    return dict(registers_per_thread=int(regs[1]),stack_bytes=int(stack[1]),spill_store_bytes=int(stack[2]),spill_load_bytes=int(stack[3]))

def child(name):
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import _path_to_binary
    verify_source()
    kernel,signature,constants,attrs,options,record=native()[name]
    stem=OUT/name
    result=dict(case=name,native=record,parsed_options=options,runtime_grid=[304],runtime_strides=[2560,2560,9728],cuda_execution=False,started_at=utc())
    write(name+'.config.json',result)
    begin=time.monotonic()
    compiled=triton.compile(ASTSource(kernel,signature,constants,attrs),target=GPUTarget('cuda',90,32),options=options)
    result['compile_seconds']=time.monotonic()-begin
    result['metadata']=compiled.metadata._asdict()
    for ext in ('ttir','ttgir','llir','ptx'):stem.with_suffix('.'+ext).write_text(compiled.asm[ext])
    stem.with_suffix('.cubin').write_bytes(compiled.asm['cubin'])
    ptxas,version=_path_to_binary('ptxas')
    assembly=subprocess.run([ptxas,'-v','--gpu-name=sm_90a',str(stem.with_suffix('.ptx')),'-o',str(stem.with_suffix('.verbose.cubin'))],capture_output=True,text=True,timeout=30)
    log=assembly.stdout+assembly.stderr;stem.with_suffix('.ptxas.log').write_text(log)
    result.update(ptxas_version=version,ptxas_returncode=assembly.returncode,ptxas_log=log,resources=resources(log) if assembly.returncode==0 else None,inventory=inventory(compiled.asm['ptx']))
    result['artifact_sha256']={ext:sha(stem.with_suffix('.'+ext).read_bytes()) for ext in ('ttir','ttgir','llir','ptx','cubin','verbose.cubin','ptxas.log') if stem.with_suffix('.'+ext).exists()}
    inv=result['inventory'];r=result['resources']
    result['success']=bool(assembly.returncode==0 and r and not any(r[k] for k in ('stack_bytes','spill_store_bytes','spill_load_bytes')) and inv['wgmma']['static_count']>0 and inv['mma_sync']['static_count']==0 and inv['local_load']['static_count']==0 and inv['local_store']['static_count']==0)
    assert all('m64n64k16' in x for x in inv['wgmma']['forms'])
    result['completed_at']=utc();result['torch_imported']='torch' in sys.modules
    assert not result['torch_imported'];verify_source()
    write(name+'.json',result)
    print(json.dumps(dict(case=name,success=result['success'],resources=r,shared_bytes=result['metadata']['shared'])),flush=True)
    if not result['success']:raise SystemExit(1)

def main():
    OUT.mkdir(exist_ok=True)
    if len(sys.argv)>1:
        assert sys.argv[1] in CASES
        child(sys.argv[1]);return
    started=time.monotonic();manifest=verify_source()
    native() # Both complete native launch records checked BEFORE first build.
    write('NATIVE_PREFLIGHT.json',dict(status='PASS',launches=2,compile_calls_before_preflight=0,source_set_sha256=manifest['source_set_sha256'],expected_sha256=sha((HERE/'NATIVE_EXPECTED.json').read_bytes()),cuda_execution=False))
    results=[]
    for name in CASES:
        remaining=180-(time.monotonic()-started)
        assert remaining>0
        begin=time.monotonic();result=dict(case=name,started_at=utc())
        try:
            run=subprocess.run([sys.executable,__file__,name],capture_output=True,text=True,timeout=min(40,remaining))
            (OUT/(name+'.compiler.stdout')).write_text(run.stdout)
            (OUT/(name+'.compiler.stderr')).write_text(run.stderr)
            result['returncode']=run.returncode
            if (OUT/(name+'.json')).exists():result.update(json.loads((OUT/(name+'.json')).read_text()))
            if run.returncode:result.update(success=False,error=run.stderr[-10000:])
        except Exception as exc:result.update(success=False,error=str(exc))
        result['total_seconds']=time.monotonic()-begin
        results.append(result);write('summary.json',results)
        print(json.dumps(dict(case=name,success=result.get('success',False),resources=result.get('resources'),total_seconds=result['total_seconds'],error=result.get('error'))),flush=True)
        if not result.get('success'):raise SystemExit(1)
    verify_source();assert len(results)==2
    write('COMPLETION.json',dict(status='PASS',compile_calls=2,total_seconds=time.monotonic()-started,source_verified_before_after=True,cuda_execution=False,completed_at=utc()))
if __name__=='__main__':main()
