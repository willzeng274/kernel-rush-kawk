"""Source and timeout checks only; imports neither Triton nor Torch."""
import json
import shutil
import subprocess
import sys
from pathlib import Path
from compile_down_s4 import HERE,verify_source,sha
from check_timeout import check

manifest=verify_source()
expected=json.loads((HERE/'NATIVE_EXPECTED.json').read_text())
assert len(expected['launches'])==2 and expected['compile_calls']==0
from launch_inputs import capture_launches,launch_receipt
assert launch_receipt(capture_launches())==expected['wrapper_launch_receipt']
tracked=None
if '--require-tracked' in sys.argv:
    repo=HERE.parents[1]
    audit=json.loads((HERE/'AUDIT_MANIFEST.json').read_text())
    paths=[HERE/name for name in audit['files']]+[HERE/'AUDIT_MANIFEST.json']
    expected_paths={p.relative_to(repo).as_posix() for p in paths}
    actual=set(subprocess.check_output(['git','ls-files','--',str(HERE.relative_to(repo))],cwd=repo,text=True).splitlines())
    assert expected_paths<=actual, sorted(expected_paths-actual)
    tracked=dict(required_count=len(expected_paths),all_required_tracked=True)
timeout_checks=check()
assert not {'torch','triton'} & set(sys.modules)
out=HERE/'compile-results';out.mkdir(exist_ok=True)
for name in ('SOURCE_MANIFEST.json','AUDIT_MANIFEST.json','NATIVE_EXPECTED.json'):
    shutil.copy2(HERE/name,out/name)
(out/'TIMEOUT_PREFLIGHT.json').write_text(json.dumps(timeout_checks,indent=2)+'\n')
result=dict(status='PASS',source_set_sha256=manifest['source_set_sha256'],engine_files=23,compile_calls_planned=2,
            compiler_imported=False,cuda_execution=False,tracked_inputs=tracked,timeout_checks=timeout_checks['status'])
(out/'PREFLIGHT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
