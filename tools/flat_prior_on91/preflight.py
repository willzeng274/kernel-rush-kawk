"""CPU source/native-Python/timeout checks only; no Torch/Triton compiler imports."""
import json,shutil,subprocess,sys
from pathlib import Path
from compile_flat_prior import HERE,verify_source,sha
from derive_native import derive
from check_timeout import check
manifest=verify_source()
native_check,receipts=derive()
assert native_check['native_records_compared']==512 and native_check['new_keys']==56
tracked=None
if '--require-tracked' in sys.argv:
    repo=HERE.parents[1];audit=json.loads((HERE/'AUDIT_MANIFEST.json').read_text())
    paths=[HERE/name for name in audit['files']]+[HERE/'AUDIT_MANIFEST.json']
    expected_paths={p.relative_to(repo).as_posix() for p in paths}
    actual=set(subprocess.check_output(['git','ls-files','--',str(HERE.relative_to(repo))],cwd=repo,text=True).splitlines())
    assert expected_paths==actual,dict(missing=sorted(expected_paths-actual),extra=sorted(actual-expected_paths))
    for p in paths:
        assert subprocess.check_output(['git','show',':'+p.relative_to(repo).as_posix()],cwd=repo)==p.read_bytes(),str(p)
    tracked=dict(required_count=len(expected_paths),all_required_tracked=True,staged_blobs_identical=True)
timeout_checks=check()
assert not {'torch','triton'} & set(sys.modules)
out=HERE/'compile-results';out.mkdir(exist_ok=True)
for name in ('SOURCE_MANIFEST.json','AUDIT_MANIFEST.json','NATIVE_EXPECTED.json','CASES.json','SOURCE_SPANS.json','POSTCOMPILE_AUDIT.md'):
    shutil.copyfile(HERE/name,out/name)
(out/'TIMEOUT_PREFLIGHT.json').write_text(json.dumps(timeout_checks,indent=2)+'\n')
(out/'NATIVE_PYTHON_PREFLIGHT.json').write_text(json.dumps(native_check,indent=2)+'\n')
(out/'WRAPPER_LAUNCHES.json').write_text(json.dumps(receipts,indent=2)+'\n')
result=dict(status='PASS',source_set_sha256=manifest['source_set_sha256'],engine_files_each=21,compile_calls_planned=56,excluded_unchanged_keys=8,native_records_compared=512,all_recorded_fields_equal=True,compiler_imported=False,compile_calls=0,cuda_execution=False,tracked_inputs=tracked,timeout_checks=timeout_checks['status'])
(out/'PREFLIGHT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
