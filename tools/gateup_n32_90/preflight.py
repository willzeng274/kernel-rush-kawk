import hashlib,json,shutil,sys
from pathlib import Path
from compile_n32 import verify_source
HERE=Path(__file__).resolve().parent
manifest=verify_source()
audit=json.loads((HERE/'AUDIT_MANIFEST.json').read_text())
for name,digest in audit['files'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,name
assert not any(n in sys.modules for n in ('torch','triton'))
OUT=HERE/'compile-results';OUT.mkdir(exist_ok=True)
for name in ('SOURCE_MANIFEST.json','AUDIT_MANIFEST.json','NATIVE_EXPECTED.json'):shutil.copy2(HERE/name,OUT/name)
result=dict(status='PASS',source_set_sha256=manifest['source_set_sha256'],engine_files=20,compile_calls_planned=2,cuda_execution=False,compiler_imported=False)
(OUT/'PREFLIGHT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
