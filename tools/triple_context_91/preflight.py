"""Source/audit checks only; imports neither Triton nor Torch."""
import json
import shutil
import subprocess
import sys
from pathlib import Path
from common import HERE, sha, verify_source

manifest, plan = verify_source()
audit = json.loads((HERE / 'AUDIT_MANIFEST.json').read_text())
for name, digest in audit['files'].items():
    assert sha((HERE / name).read_bytes()) == digest, name
expected = json.loads((HERE / 'NATIVE_EXPECTED.json').read_text())
assert len(expected['launches']) == 16 and expected['compile_calls'] == 0
assert not any(name in sys.modules for name in ('torch', 'triton'))
out = HERE / 'compile-results'
out.mkdir(exist_ok=True)
for name in ('SOURCE_MANIFEST.json', 'AUDIT_MANIFEST.json', 'NATIVE_EXPECTED.json', 'COMPILE_PLAN.json'):
    shutil.copy2(HERE / name, out / name)
# CI must receive every exact input; ignored archive packaging is not used.
tracked = None
if '--require-tracked' in sys.argv:
    repo = HERE.parents[1]
    paths = [HERE / p for p in audit['files']] + [HERE / 'AUDIT_MANIFEST.json']
    for label in ('engine', 'preparation', 'primary'):
        paths += [HERE / 'snapshots' / label / p for p in manifest[label + '_files']]
    expected_paths = sorted(set(p.relative_to(repo).as_posix() for p in paths))
    actual = subprocess.check_output(['git', 'ls-files', '--', str(HERE.relative_to(repo))], cwd=repo, text=True).splitlines()
    assert set(expected_paths) <= set(actual), sorted(set(expected_paths) - set(actual))
    tracked = dict(required_count=len(expected_paths), all_required_tracked=True)
result = dict(status='PASS', source_set_sha256=manifest['candidate_source_set_sha256'], engine_files=18,
              compile_calls_planned=16, native_key_dedup=expected['dedup'], compiler_imported=False,
              cuda_execution=False, tracked_inputs=tracked)
(out / 'PREFLIGHT.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
