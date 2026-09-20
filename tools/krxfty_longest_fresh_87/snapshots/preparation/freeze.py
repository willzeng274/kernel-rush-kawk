"""Freeze source-only candidate and targeted compiler proposal; never compile."""
import difflib
import gzip
import hashlib
import io
import json
import runpy
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CAND = ROOT/'work/candidates/krxfty_longest_verified'
BASE = ROOT/'work/candidates/krxfty_pair_cache_on82/engine'
sha = lambda data: hashlib.sha256(data).hexdigest()
env = runpy.run_path(str(HERE/'check_longest.py'))
Template = env['Template']
base, candidate, diffs = {}, {}, []
files = sorted((CAND/'engine').rglob('*.py'))
for file in files:
    name = str(file.relative_to(CAND/'engine'))
    a, b = (BASE/name).read_bytes(), file.read_bytes()
    base[name], candidate[name] = sha(a), sha(b)
    if a != b:
        diffs.extend(difflib.unified_diff(a.decode().splitlines(True), b.decode().splitlines(True),
                                        fromfile='base83/'+name, tofile='longest_verified/'+name))
(HERE/'CANDIDATE.diff').write_text(''.join(diffs))
raw = io.BytesIO()
with tarfile.open(fileobj=raw, mode='w', format=tarfile.PAX_FORMAT) as tar:
    for file in files:
        data = file.read_bytes()
        info = tarfile.TarInfo(str(file.relative_to(CAND/'engine')))
        info.size, info.mode, info.mtime = len(data), 0o644, 0
        tar.addfile(info, io.BytesIO(data))
archive = gzip.compress(raw.getvalue(), mtime=0)
(CAND/'engine.tar.gz').write_bytes(archive)
manifest = dict(base_label='exact #83', base_path=str(BASE), candidate_path=str(CAND/'engine'),
                base_files=base, candidate_files=candidate,
                base_tree_sha256=sha(json.dumps(base, sort_keys=True).encode()),
                candidate_tree_sha256=sha(json.dumps(candidate, sort_keys=True).encode()),
                archive_sha256=sha(archive), archive_bytes=len(archive), file_count=len(files),
                changed_files=[name for name in base if base[name] != candidate[name]],
                source_diff_sha256=sha((HERE/'CANDIDATE.diff').read_bytes()))
(HERE/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
matrix = []
for r in (4,8,16,32,42,64):
    t = Template.build(r,8)
    matrix.append(dict(R=r, C=r-1, P=1<<r.bit_length(), MAXA=max(1,t.max_depth), GUARD=2*r,
                       longest_selector_compiled_in=t.max_depth<r-1))
plan = dict(status='proposal only; not executed', scope='accept_kernel only, exact baseline and candidate',
            target='sm90, platform Triton 3.1.0', num_warps=1,
            source_paths=[str(BASE/'kernels/accept.py'),str(CAND/'engine/kernels/accept.py')],
            matrix=matrix, signature='derive from actual wrappers: masks i64 pointer, depth i32 pointer; other arguments unchanged',
            checks=['Both versions compile for all listed shapes without launching CUDA.',
                    'Record compile diagnostics, registers, spills/local memory and shared memory.',
                    'Inspect candidate uint64 mask accumulation, bit63, padded guards and gain branch.',
                    'Confirm R4 static chain exclusion removes additional selection and reconstruction.',
                    'Retain source hashes and complete compiler artifacts; do not infer runtime speed or correctness.'],
            excluded=['model compilation','draft/publisher/compact recompilation','GPU invocation','CI dispatch','official submission'])
(HERE/'COMPILE_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
assert len((HERE/'REPORT.md').read_text().split()) <= 800
evidence = {p.name:sha(p.read_bytes()) for p in sorted(HERE.iterdir()) if p.is_file() and p.name!='EVIDENCE_MANIFEST.json'}
(HERE/'EVIDENCE_MANIFEST.json').write_text(json.dumps(evidence,indent=2)+'\n')
print(json.dumps({k:manifest[k] for k in ('candidate_tree_sha256','archive_sha256','archive_bytes','file_count','changed_files')},indent=2))
print('Report words:',len((HERE/'REPORT.md').read_text().split()))
