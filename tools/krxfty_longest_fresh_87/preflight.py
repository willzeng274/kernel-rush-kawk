import hashlib,json,shutil,sys
from pathlib import Path
from compile_accept import verify_source
HERE=Path(__file__).resolve().parent
audit=json.loads((HERE/'AUDIT_MANIFEST.json').read_text())
for name,digest in audit['files'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,name
manifest=json.loads((HERE/'SOURCE_MANIFEST.json').read_text())
verify_source(manifest)
assert not any(n in sys.modules for n in ('torch','triton'))
result=dict(status='PASS',source_tree_sha256=manifest['source_tree_sha256'],decorated_accept_sha256=manifest['decorated_accept_sha256'],compile_calls=6,cuda_execution=False,compiler_imported=False)
out=HERE/'compile-results';out.mkdir(exist_ok=True)
for name in ('SOURCE_MANIFEST.json','AUDIT_MANIFEST.json'):shutil.copy2(HERE/name,out/name)
(out/'PREFLIGHT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
