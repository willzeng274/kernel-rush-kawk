"""Read-only provenance preflight; no device imports, compilation or execution."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys

HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def functions(text):
    return [n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)]
def exact(text,node):
    first=min([node.lineno]+[x.lineno for x in node.decorator_list])
    return ''.join(text.splitlines(keepends=True)[first-1:node.end_lineno])
audit=json.loads((HERE/'AUDIT_MANIFEST.json').read_text())
for name,digest in audit['files'].items():assert sha(HERE/name)==digest,(name,sha(HERE/name),digest)
manifest=json.loads((HERE/'SOURCE_MANIFEST.json').read_text())
engine=HERE/'snapshots/engine'
assert sorted(p.relative_to(engine).as_posix() for p in engine.rglob('*') if p.is_file())==sorted(manifest['engine_files'])
for name,digest in manifest['engine_files'].items():assert sha(engine/name)==digest
lines=''.join(manifest['engine_files'][name]+'  '+name+'\n' for name in sorted(manifest['engine_files']))
assert (HERE/'snapshots/SOURCE_SHA256.txt').read_text()==lines
assert hashlib.sha256(lines.encode()).hexdigest()==manifest['source_sha256_lines_sha256']
source=(engine/'kernels/top8.py').read_text();device=(HERE/'device_top8.py').read_text()
sf,df=functions(source),functions(device)
assert [n.name for n in sf]==['_bf16_key','_best_pair','top8_partials','top8_merge']==[n.name for n in df]
assert [ast.dump(n) for n in sf]==[ast.dump(n) for n in df]
assert [exact(source,n) for n in sf]==[exact(device,n) for n in df]
assert hashlib.sha256('\n'.join(ast.dump(n) for n in sf).encode()).hexdigest()==manifest['device_function_ast_sha256']
assert sha(HERE/'compile_top8.py')==manifest['compile_harness_sha256']
assert sha(HERE/'device_top8.py')==manifest['device_top8_sha256']
assert sha(engine/'kernels/top8.py')==manifest['engine_top8_sha256']
for path in ('compile_top8.py','device_top8.py'):
    compile((HERE/path).read_text(),str(HERE/path),'exec')
assert not any(name in sys.modules for name in ('torch','triton'))
result=dict(status='PASS',engine_file_count=len(manifest['engine_files']),full_source_snapshot_verified=True,
    exact_decorated_device_functions=True,device_function_ast_sha256=manifest['device_function_ast_sha256'],
    source_sha256_lines_sha256=manifest['source_sha256_lines_sha256'],
    engine_top8_sha256=manifest['engine_top8_sha256'],device_top8_sha256=manifest['device_top8_sha256'],
    compile_harness_sha256=manifest['compile_harness_sha256'],audit_manifest_sha256=sha(HERE/'AUDIT_MANIFEST.json'),
    preflight_sha256=sha(HERE/'preflight.py'),compiler_imported=False,cuda_execution=False,
    cpu_evidence_limit=audit['cpu_evidence_limit'])
out=HERE/'compile-results';out.mkdir(exist_ok=True)
for name in ('AUDIT_MANIFEST.json','SOURCE_MANIFEST.json'):
    shutil.copy2(HERE/name,out/name)
(out/'PREFLIGHT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
