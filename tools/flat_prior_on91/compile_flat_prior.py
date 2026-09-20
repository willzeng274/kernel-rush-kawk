"""Exactly56 proposed source-exact SM90 builds; separate root authorization required."""
import ast,datetime,hashlib,importlib.util,inspect,io,json,os,re,signal,subprocess,sys,tarfile,time
from pathlib import Path
from launch_inputs import CASES,SCHEDULE,capture_launches,launch_receipt
from native_record import record_native,compare_all
HERE=Path(__file__).resolve().parent
OUT=HERE/'compile-results'
sha=lambda b:hashlib.sha256(b).hexdigest()
utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,default=str)+'\n')
def spans(path):
    source=path.read_text();lines=source.splitlines(keepends=True)
    return {n.name:''.join(lines[min(d.lineno for d in n.decorator_list)-1:n.end_lineno]) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.decorator_list}

def verify_source():
    m=json.loads((HERE/'SOURCE_MANIFEST.json').read_text())
    for label in ('base91','candidate','preparation','primary','maintained'):
        folder=HERE/'snapshots'/label
        assert {p.relative_to(folder).as_posix():sha(p.read_bytes()) for p in folder.rglob('*') if p.is_file()}==m[label+'_files'],label
    assert len(m['candidate_files'])==len(m['base91_files'])==21 and m['cases']==CASES and len(CASES)==56
    assert m['bounds']==dict(single_worker=True,total_seconds=180,internal_seconds=178,case_seconds=25,retry=False,expand_search=False)
    assert sha((HERE/'SOURCE_SHA256.txt').read_bytes())==m['source_set_sha256']=='96d42195c95d8778975023294cdc1fe437de669a98462d7b54653ad1176d1373'
    expected={name.removeprefix('engine/'):digest for name,digest in json.loads((HERE/'snapshots/preparation/SOURCE_MANIFEST.json').read_text())['candidate_files'].items()}
    assert expected==m['candidate_files']
    expected_base={name.removeprefix('engine/'):digest for name,digest in json.loads((HERE/'snapshots/preparation/base91_SOURCE_MANIFEST.json').read_text())['candidate_files'].items()}
    assert expected_base==m['base91_files']
    changed=[name for name in m['candidate_files'] if m['candidate_files'][name]!=m['base91_files'][name]]
    assert changed==['recycle.py']
    trees=[ast.parse((HERE/'snapshots'/label/'recycle.py').read_text()) for label in ('base91','candidate')]
    priors=[]
    for tree in trees:
        assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='RANK_PRIOR' for t in n.targets))
        priors.append(ast.literal_eval(assignment.value));assignment.value=ast.Constant(None)
    assert ast.dump(trees[0])==ast.dump(trees[1])
    assert priors==[[.55,.16,.08,.05,.035,.025,.02,.015],[.40,.20,.12,.08,.06,.05,.04,.03]]
    device=HERE/'device.py';extracted=spans(device)
    assert sha(device.read_bytes())==m['kernel_sha256']
    assert set(extracted)=={'_pair_slot','_draft_kernel','_publish_pairs_kernel','accept_kernel','_compact_kernel'}
    for name,record in json.loads((HERE/'SOURCE_SPANS.json').read_text()).items():
        for label in ('base91','candidate'):
            original=HERE/'snapshots'/label/record['file']
            assert spans(original)[name]==extracted[name]
        assert sha(extracted[name].encode())==record['decorated_function_sha256']
    archive=HERE/'krxfty_flat_prior_on91.tar.gz'
    assert sha(archive.read_bytes())==m['archive_sha256']=='938a935b45e990cea918ee24bfd693ecdf54ed475b86190c52ded3af5b22e66f'
    with tarfile.open(fileobj=io.BytesIO(archive.read_bytes()),mode='r:gz') as tf:
        assert tf.getnames()==list(m['candidate_files'])
        for member in tf.getmembers():assert member.isfile() and sha(tf.extractfile(member).read())==m['candidate_files'][member.name]
    for name,digest in json.loads((HERE/'AUDIT_MANIFEST.json').read_text())['files'].items():assert sha((HERE/name).read_bytes())==digest,name
    return m

def native(persist=True):
    import triton
    import triton.runtime.jit as jit
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import make_backend
    expected=json.loads((HERE/'NATIVE_EXPECTED.json').read_text())
    assert triton.__version__=='3.1.0'
    assert os.environ.get('DISABLE_MMA_V3','0').lower() in ('','0','false')
    assert os.environ.get('TRITON_DEBUG','0')!='1'
    assert sha(Path(inspect.getsourcefile(jit)).read_bytes())==expected['primary_sha256']['jit.py']
    backend=make_backend(GPUTarget('cuda',90,32))
    assert sha(Path(backend.parse_options.__func__.__code__.co_filename).read_bytes())==expected['primary_sha256']['nvidia_compiler.py']
    spec=importlib.util.spec_from_file_location('frozen_flat_prior',HERE/'device.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    actual={};receipts={};all_candidate={};selected={}
    for label,key in [('base91','baseline'),('candidate','candidate')]:
        calls=capture_launches(label);rows=[]
        for call in calls:
            row,build=record_native(call,getattr(module,call['kernel']),backend);rows.append(row)
            if key=='candidate':all_candidate[(row['kernel'],row['B_context'])]=build
        compare_all(rows,expected[key]);actual[key]=rows;receipts[key]=launch_receipt(calls)
    basekeys={r['complete_compile_key_sha256'] for r in actual['baseline']}
    candidatekeys={r['complete_compile_key_sha256'] for r in actual['candidate']}
    assert candidatekeys-basekeys==set(expected['new_keys'])=={r['complete_compile_key_sha256'] for r in SCHEDULE}
    assert candidatekeys&basekeys==set(expected['excluded_unchanged_keys']) and len(candidatekeys-basekeys)==56
    for item in SCHEDULE:
        build=all_candidate[item['kernel'],item['representative_B']]
        assert build[-1]['complete_compile_key_sha256']==item['complete_compile_key_sha256']
        selected[item['case']]=build
    assert list(selected)==CASES
    if persist:write('NATIVE_BINDER_ACTUAL.json',dict(baseline=actual['baseline'],candidate=actual['candidate'],wrapper_launch_receipts=receipts,all_recorded_fields_equal=True,native_records_compared=512,source_set_sha256=expected['source_set_sha256'],cuda_execution=False))
    assert 'torch' not in sys.modules
    return selected

def inventory(ptx):
    patterns={'global_load':r'\bld\.global[^;]*;','global_store':r'\bst\.global[^;]*;','atomic':r'\batom\.[^;]*;','barrier':r'\bbar\.sync[^;]*;','shared_load':r'\bld\.shared[^;]*;','shared_store':r'\bst\.shared[^;]*;','local_load':r'\bld\.local[^;]*;','local_store':r'\bst\.local[^;]*;','mma':r'\b(?:wgmma\.mma_async|mma\.sync)[^;]*;','shuffle':r'\bshfl\.[^;]*;'}
    return {k:dict(static_count=len(v),forms=sorted(set(v))) for k,p in patterns.items() for v in [re.findall(p,ptx)]}
def resources(log):
    regs=re.search(r'Used (\d+) registers',log);stack=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',log)
    assert regs and stack,log
    return dict(registers_per_thread=int(regs[1]),stack_bytes=int(stack[1]),spill_store_bytes=int(stack[2]),spill_load_bytes=int(stack[3]))

def child(name):
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import _path_to_binary
    verify_source()
    kernel,signature,constants,attrs,options,record=native(persist=False)[name]
    stem=OUT/name
    result=dict(case=name,native=record,parsed_options=options,runtime_grid=record['grid'],cuda_execution=False,started_at=utc())
    write(name+'.config.json',result)
    assert not (OUT/(name+'.compile-start.json')).exists(),'No retries or resumed builds'
    write(name+'.compile-start.json',dict(case=name,started_at=utc(),no_retry=True))
    begin=time.monotonic()
    compiled=triton.compile(ASTSource(kernel,signature,constants,attrs),target=GPUTarget('cuda',90,32),options=options)
    result['compile_seconds']=time.monotonic()-begin;result['metadata']=compiled.metadata._asdict()
    assert result['metadata']['debug'] is None
    for ext in ('ttir','ttgir','llir','ptx'):stem.with_suffix('.'+ext).write_text(compiled.asm[ext])
    stem.with_suffix('.cubin').write_bytes(compiled.asm['cubin'])
    ptxas,version=_path_to_binary('ptxas');begin=time.monotonic()
    assembly=subprocess.run([ptxas,'-v','--gpu-name=sm_90a',str(stem.with_suffix('.ptx')),'-o',str(stem.with_suffix('.verbose.cubin'))],capture_output=True,text=True,timeout=25)
    log=assembly.stdout+assembly.stderr;stem.with_suffix('.ptxas.log').write_text(log)
    result.update(ptxas_seconds=time.monotonic()-begin,ptxas_version=version,ptxas_returncode=assembly.returncode,ptxas_log=log,resources=resources(log) if assembly.returncode==0 else None,inventory=inventory(compiled.asm['ptx']))
    result['artifact_sha256']={ext:sha(stem.with_suffix('.'+ext).read_bytes()) for ext in ('ttir','ttgir','llir','ptx','cubin','verbose.cubin','ptxas.log') if stem.with_suffix('.'+ext).exists()}
    inv=result['inventory'];r=result['resources']
    result['success']=bool(assembly.returncode==0 and r and result['metadata']['shared']==0 and not any(r[k] for k in ('stack_bytes','spill_store_bytes','spill_load_bytes')) and all(inv[k]['static_count']==0 for k in ('mma','atomic','local_load','local_store')))
    result['runtime_release']='BLOCKED_PENDING_SEPARATE_CONTROLLER_INSTRUCTION_ORDERING_AND_BOUNDS_AUDIT'
    result['completed_at']=utc();result['torch_imported']='torch' in sys.modules
    assert not result['torch_imported'];verify_source();write(name+'.json',result)
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
    raise TimeoutError("178-second internal deadline;180-second absolute bound")

def main():
    OUT.mkdir(exist_ok=True)
    if len(sys.argv)>1:
        assert sys.argv[1] in CASES
        child(sys.argv[1]);return
    started=time.monotonic()
    signal.signal(signal.SIGALRM, deadline_alarm)
    signal.setitimer(signal.ITIMER_REAL, 178)
    manifest=verify_source()
    assert not list(OUT.glob('*.compile-start.json')), 'No retries or resumed builds'
    native() # All512 baseline/candidate records must match BEFORE the first of56 builds.
    write('NATIVE_PREFLIGHT.json',dict(status='PASS',launches=512,compile_cases_planned=56,compile_calls_before_preflight=0,source_set_sha256=manifest['source_set_sha256'],expected_sha256=sha((HERE/'NATIVE_EXPECTED.json').read_bytes()),cuda_execution=False))
    results=[]
    for name in CASES:
        remaining=178-(time.monotonic()-started)
        assert remaining>0
        begin=time.monotonic();result=dict(case=name,started_at=utc())
        try:
            run=run_bounded([sys.executable,'-B',__file__,name],min(25,remaining))
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
    verify_source();assert len(results)==len(list(OUT.glob('*.compile-start.json')))==56
    write('COMPLETION.json',dict(status='PASS',compile_calls=56,total_seconds=time.monotonic()-started,source_verified_before_after=True,cuda_execution=False,completed_at=utc()))
    signal.setitimer(signal.ITIMER_REAL,0)
if __name__=='__main__':main()
