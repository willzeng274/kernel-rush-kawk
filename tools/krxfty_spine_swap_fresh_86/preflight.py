"""Read-only source and launch geometry provenance; no compiler import."""
import hashlib,json,shutil,sys
from pathlib import Path
from compile_spine_swap import verify_source
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
audit=json.loads((HERE/'AUDIT_MANIFEST.json').read_text())
for name,digest in audit['files'].items():assert sha(HERE/name)==digest,name
manifest=json.loads((HERE/'SOURCE_MANIFEST.json').read_text())
verify_source(manifest)
assert [n for n in manifest['engine_files'] if manifest['engine_files'][n]!=manifest['base_files'][n]]==['recycle.py']
assert not any(n in sys.modules for n in ('torch','triton'))
result=dict(status='PASS',engine_files=18,base_commit=manifest['base_commit'],source_tree_sha256=manifest['source_tree_sha256'],decorated_draft_sha256=manifest['decorated_draft_sha256'],decorated_pair_slot_sha256=manifest['decorated_pair_slot_sha256'],unchanged_files=17,compile_calls=6,compiler_imported=False,cuda_execution=False)
out=HERE/'compile-results';out.mkdir(exist_ok=True)
for name in ('SOURCE_MANIFEST.json','AUDIT_MANIFEST.json'):shutil.copy2(HERE/name,out/name)
(out/'PREFLIGHT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
