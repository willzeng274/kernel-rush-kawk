"""Read/validate source and freeze review records; never compiles or runs GPU."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BASE=ROOT/'work/candidates/krxfty_prefix_splitloop_guarded/engine'
OUT=ROOT/'work/candidates/krxfty_top8/engine'
hashfile=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base_paths=sorted(str(p.relative_to(BASE)) for p in BASE.rglob('*') if p.is_file())
out_paths=sorted(str(p.relative_to(OUT)) for p in OUT.rglob('*') if p.is_file())
assert len(base_paths)==15 and len(out_paths)==16 and all(x.endswith('.py') for x in out_paths)
for line in (BASE.parent/'SOURCE_SHA256.txt').read_text().splitlines():
    digest,name=line.split(maxsplit=1)
    assert hashfile(BASE/name)==digest
changed=[p for p in base_paths if hashfile(BASE/p)!=hashfile(OUT/p)]
added=sorted(set(out_paths)-set(base_paths))
assert changed==['model.py','recycle.py'] and added==['kernels/top8.py']
for p in out_paths:ast.parse((OUT/p).read_text())
functions=[x for x in ast.parse((OUT/'kernels/top8.py').read_text()).body if isinstance(x,ast.FunctionDef)]
frozen=[x for x in ast.parse((HERE/'device_top8.py').read_text()).body if isinstance(x,ast.FunctionDef)]
assert [ast.dump(x) for x in functions]==[ast.dump(x) for x in frozen]
bodyhash=hashlib.sha256('\n'.join(ast.dump(x) for x in functions).encode()).hexdigest()
results=json.loads((HERE/'CPU_RESULTS.json').read_text())
assert results['status']=='PASS' and results['kernel_sha256']==hashfile(OUT/'kernels/top8.py')
assert results['policy_cases']==33 and results['integration_cases']==15
lines=''.join(hashfile(OUT/p)+'  '+p+'\n' for p in out_paths)
(OUT.parent/'SOURCE_SHA256.txt').write_text(lines)
manifest=dict(status='FROZEN_PENDING_ROOT_REVIEW_AND_COMPILATION',base_revision=80,base_commit='c3e4bfd6c65585c53e8552743f50f7285effee6d',
 changed_base_files=changed,added_files=added,unchanged_base_files=[p for p in base_paths if p not in changed],
 engine_files={p:hashfile(OUT/p) for p in out_paths},source_sha256_lines_sha256=hashlib.sha256(lines.encode()).hexdigest(),
 engine_top8_sha256=hashfile(OUT/'kernels/top8.py'),device_top8_sha256=hashfile(HERE/'device_top8.py'),device_function_ast_sha256=bodyhash,
 compile_harness_sha256=hashfile(HERE/'compile_top8.py'),cpu_harness_sha256=hashfile(HERE/'check_top8.py'),cpu_results_sha256=hashfile(HERE/'CPU_RESULTS.json'),
 cpu_execution='Torch2.5.1 CPU source-body emulation and host tests; no Triton compiler/GPU',
 cpu_invocation='PYTHONPATH="$PWD/work/candidates/prefill_research/test_deps" /Users/polyuser/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 work/research/krxfty_top8/check_top8.py',
 final_host_rerun=results.get('final_host_rerun'),
 compile_cases=[dict(name='top8_partials',signature=['*bf16','*i32','*i32'],constants=dict(V=151936,PARTS=75,BLOCK=2048),grid=['N',75],divisible_by_16=[0,1,2,3,5]),dict(name='top8_merge',signature=['*i32','*i32','*i32'],constants=dict(PARTS=75,MERGE_BLOCK=1024),grid=['N'],divisible_by_16=[0,1,2,4])],
 compile_options=dict(num_warps=4,num_stages=3,enable_fp_fusion=False),compilation_executed=False,gpu_executed=False,ci_executed=False,submitted=False,
 report_sha256=hashfile(HERE/'REPORT.md'),report_words=len((HERE/'REPORT.md').read_text().split()))
(HERE/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
(OUT.parent/'STATUS.json').write_text(json.dumps({k:manifest[k] for k in ('status','base_revision','base_commit','compilation_executed','gpu_executed','ci_executed','submitted')},indent=2)+'\n')
diff=''
for p in changed+added:
    before=(BASE/p).read_text().splitlines(keepends=True) if p in base_paths else []
    diff+=''.join(difflib.unified_diff(before,(OUT/p).read_text().splitlines(keepends=True),fromfile='base80/'+p,tofile='top8/'+p))
(HERE/'integration.diff').write_text(diff)
print(json.dumps({k:manifest[k] for k in ('status','base_commit','engine_top8_sha256','device_top8_sha256','compile_harness_sha256','source_sha256_lines_sha256','report_words')},indent=2))
