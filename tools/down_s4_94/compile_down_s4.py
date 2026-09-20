"""Two proposed source-exact SM90 builds; requires separate root authorization."""
import ast,datetime,hashlib,importlib.util,inspect,json,os,re,signal,subprocess,sys,time
from pathlib import Path
from launch_inputs import CASES,capture_launches,launch_receipt,serialize
HERE=Path(__file__).resolve().parent
OUT=HERE/'compile-results'
sha=lambda b:hashlib.sha256(b).hexdigest()
utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,default=str)+'\n')

def verify_source():
    m=json.loads((HERE/'SOURCE_MANIFEST.json').read_text())
    for label in ('engine','preparation','primary'):
        folder=HERE/'snapshots'/label
        assert {p.relative_to(folder).as_posix():sha(p.read_bytes()) for p in folder.rglob('*') if p.is_file()}==m[label+'_files'],label
    assert len(m['engine_files'])==23 and m['cases']==CASES
    assert m['bounds']==dict(single_worker=True,total_seconds=90,case_seconds=40,retry=False,expand_search=False)
    assert sha((HERE/'snapshots/preparation/SOURCE_SHA256.txt').read_bytes())==m['source_set_sha256']=='09615652fb2ffd2d716bacd02a59d5736dde83d47050cddf910fb4ac15c7ae7d'
    extracted=HERE/'down_device.py'
    assert sha(extracted.read_bytes())==m['kernel_sha256']=='4486f4a358a57708d9a47a94a8b4356d5093668a4adb6891ee8e7105f6dfab11'
    def spans(path):
        source=path.read_text();lines=source.splitlines(keepends=True)
        return {n.name:''.join(lines[min(d.lineno for d in n.decorator_list)-1:n.end_lineno]) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in ('_down_s4_kernel','_sum_kernel')}
    extracted_spans=spans(extracted)
    assert set(extracted_spans)=={'_down_s4_kernel','_sum_kernel'}
    for record in m['extractions']:
        original=HERE/'snapshots/engine'/record['source']
        assert sha(original.read_bytes())==record['original_sha256']
        assert spans(original)[record['kernel']]==extracted_spans[record['kernel']]
        assert sha(extracted_spans[record['kernel']].encode())==record['span_sha256']
    expected=json.loads((HERE/'snapshots/preparation/SOURCE_MANIFEST.json').read_text())['files']
    assert expected==m['engine_files']
    for name,digest in json.loads((HERE/'AUDIT_MANIFEST.json').read_text())['files'].items():
        assert sha((HERE/name).read_bytes())==digest,name
    return m

def native():
    import triton
    import triton.runtime.jit as jit
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import make_backend
    expected=json.loads((HERE/'NATIVE_EXPECTED.json').read_text())
    assert triton.__version__=='3.1.0'
    assert os.environ.get('DISABLE_MMA_V3','0').lower() in ('','0','false')
    assert os.environ.get('TRITON_DEBUG','0')!='1'
    assert sha(Path(inspect.getsourcefile(jit)).read_bytes())==expected['jit_source_sha256']
    backend=make_backend(GPUTarget('cuda',90,32))
    assert sha(Path(backend.parse_options.__func__.__code__.co_filename).read_bytes())==expected['backend_source_sha256']
    spec=importlib.util.spec_from_file_location('frozen_down_s4',HERE/'down_device.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    calls=capture_launches()
    assert launch_receipt(calls)==expected['wrapper_launch_receipt']
    records=[];cases={}
    for call in calls:
        kernel=getattr(module,call['kernel']);kernel.create_binder()
        assert kernel.debug is None
        kwargs=dict(call['kwargs']);kwargs['debug']=kernel.debug
        bound,sigspec,constvals,nonconst,extra=kernel.binder(*call['args'],**kwargs)
        attrs=kernel._get_config(*bound.values())
        signature={kernel.params[i].name:v for i,v in zip(kernel.non_constexpr_indices,sigspec)}
        constants={p.name:v for p,v in zip(kernel.params,bound.values()) if p.is_constexpr or p.num in attrs.equal_to_1 or v is None}
        key=''.join(sigspec)+str((constvals,extra))
        parsed=backend.parse_options(kwargs)
        assert parsed.num_warps==4 and parsed.num_stages==3 and parsed.num_ctas==1 and parsed.enable_fp_fusion is True
        normalized=dict(parsed.__dict__)
        assert len(parsed.extern_libs)==1 and parsed.extern_libs[0][0]=='libdevice' and Path(parsed.extern_libs[0][1]).name=='libdevice.10.bc'
        normalized['extern_libs']=[['libdevice','<TRITON_LIBDEVICE>']]
        record=dict(case=call['case'],kernel=call['kernel'],grid=call['grid'],
          native_key_sha256=sha((call['kernel']+key).encode()),jit_cache_key_sha256=sha(key.encode()),jit_cache_key=key,
          bound_arguments={k:serialize(v) for k,v in bound.items()},signature_specialization=sigspec,constexpr_values=constvals,
          signature=signature,constants=constants,attrs=dict(divisible_by_16=sorted(attrs.divisible_by_16),equal_to_1=sorted(attrs.equal_to_1)),options_explicit=extra,parsed_options=normalized)
        records.append(record)
        cases[call['case']]=(kernel,signature,constants,attrs,parsed.__dict__,record)
    assert list(cases)==CASES and json.loads(json.dumps(records))==expected['launches'],'native binder/parsed-options mismatch'
    write('NATIVE_BINDER_ACTUAL.json',dict(launches=records,wrapper_launch_receipt=launch_receipt(calls),jit_source_sha256=expected['jit_source_sha256'],backend_source_sha256=expected['backend_source_sha256'],kernel_source_sha256=expected['kernel_source_sha256'],cuda_execution=False))
    assert 'torch' not in sys.modules
    return cases

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
    result=dict(case=name,native=record,parsed_options=options,runtime_grid=record['grid'],runtime_MNK=[64,2560,9728],runtime_partial_shape=[4,64,2560],cuda_execution=False,started_at=utc())
    write(name+'.config.json',result)
    assert not (OUT/(name+'.compile-start.json')).exists(),'No retries or resumed builds'
    write(name+'.compile-start.json',dict(case=name,started_at=utc(),no_retry=True))
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
    resources_ok=bool(assembly.returncode==0 and r and not any(r[k] for k in ('stack_bytes','spill_store_bytes','spill_load_bytes')) and inv['mma_sync']['static_count']==0 and inv['local_load']['static_count']==0 and inv['local_store']['static_count']==0)
    if name==CASES[0]:
        instruction_ok=inv['wgmma']['static_count']>0 and all('m64n64k16' in x for x in inv['wgmma']['forms']) and inv['bf16_cast']['static_count']==0
    else:
        instruction_ok=inv['wgmma']['static_count']==0 and inv['bf16_cast']['static_count']>0
    result['success']=bool(resources_ok and instruction_ok)
    result['runtime_release']='BLOCKED_PENDING_SEPARATE_38_STEP_ORDERING_BOUNDS_AND_STORE_AUDIT'
    result['completed_at']=utc();result['torch_imported']='torch' in sys.modules
    assert not result['torch_imported'];verify_source()
    write(name+'.json',result)
    print(json.dumps(dict(case=name,success=result['success'],resources=r,shared_bytes=result['metadata']['shared'])),flush=True)
    if not result['success']:raise SystemExit(1)

def run_bounded(argv, timeout):
    # Each case gets its own process group, including any ptxas descendant.
    # Timeout kills the complete group, so no abandoned compiler can overlap.
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return dict(returncode=process.returncode, stdout=stdout, stderr=stderr, timed_out=False)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        return dict(returncode=process.returncode, stdout=stdout, stderr=stderr, timed_out=True)
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise


def deadline_alarm(signum, frame):
    raise TimeoutError("88-second internal deadline;90-second absolute bound")

def main():
    OUT.mkdir(exist_ok=True)
    if len(sys.argv)>1:
        assert sys.argv[1] in CASES
        child(sys.argv[1]);return
    started=time.monotonic()
    signal.signal(signal.SIGALRM, deadline_alarm)
    signal.setitimer(signal.ITIMER_REAL, 88)
    manifest=verify_source()
    assert not list(OUT.glob('*.compile-start.json')), 'No retries or resumed builds'
    native() # BOTH complete launch records must match BEFORE the first of two builds.
    write('NATIVE_PREFLIGHT.json',dict(status='PASS',launches=2,compile_calls_before_preflight=0,source_set_sha256=manifest['source_set_sha256'],expected_sha256=sha((HERE/'NATIVE_EXPECTED.json').read_bytes()),cuda_execution=False))
    results=[]
    for name in CASES:
        remaining=88-(time.monotonic()-started)
        assert remaining>0
        begin=time.monotonic();result=dict(case=name,started_at=utc())
        try:
            run=run_bounded([sys.executable,'-B',__file__,name],min(40,remaining))
            (OUT/(name+'.compiler.stdout')).write_text(run['stdout'])
            (OUT/(name+'.compiler.stderr')).write_text(run['stderr'])
            result.update(returncode=run['returncode'],timed_out=run['timed_out'])
            if (OUT/(name+'.json')).exists():result.update(json.loads((OUT/(name+'.json')).read_text()))
            if run['returncode'] or run['timed_out']:result.update(success=False,error=run['stderr'][-10000:] or 'Child hard timeout')
        except Exception as exc:result.update(success=False,error=str(exc))
        result['total_seconds']=time.monotonic()-begin
        results.append(result);write('summary.json',results)
        print(json.dumps(dict(case=name,success=result.get('success',False),resources=result.get('resources'),total_seconds=result['total_seconds'],error=result.get('error'))),flush=True)
        if not result.get('success'):raise SystemExit(1)
    verify_source();assert len(results)==len(list(OUT.glob('*.compile-start.json')))==2
    write('COMPLETION.json',dict(status='PASS',compile_calls=2,total_seconds=time.monotonic()-started,source_verified_before_after=True,cuda_execution=False,completed_at=utc()))
    signal.setitimer(signal.ITIMER_REAL,0)
if __name__=='__main__':main()
