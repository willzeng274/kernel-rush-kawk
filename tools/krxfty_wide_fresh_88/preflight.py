import hashlib,json,shutil,sys
from pathlib import Path
from compile_wide import verify_source
HERE=Path(__file__).resolve().parent
manifest=verify_source()
audit=json.loads((HERE/'AUDIT_MANIFEST.json').read_text())
for name,digest in audit['files'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,name
assert not any(n in sys.modules for n in ('torch','triton'))
out=HERE/'compile-results';out.mkdir(exist_ok=True)
for name in ('SOURCE_MANIFEST.json','AUDIT_MANIFEST.json'):shutil.copy2(HERE/name,out/name)
result=dict(status='PASS',source_set_sha256=manifest['source_set_sha256'],engine_files=19,preparation_files=23,compile_calls_planned=7,cuda_execution=False,compiler_imported=False)
(out/'PREFLIGHT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
