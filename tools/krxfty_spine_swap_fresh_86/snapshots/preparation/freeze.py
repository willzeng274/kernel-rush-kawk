"""Freeze the isolated base83 candidate; does not compile or execute CUDA."""
import ast
import difflib
import gzip
import hashlib
import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CAND = ROOT / 'work/candidates/krxfty_spine_swap'
ENGINE = CAND / 'engine'
BASE = ROOT / 'work/candidates/krxfty_pair_cache_on82/engine'
sha = lambda data: hashlib.sha256(data).hexdigest()
manifest = lambda root: {str(p.relative_to(root)): sha(p.read_bytes()) for p in sorted(root.rglob('*.py'))}
before, after = manifest(BASE), manifest(ENGINE)
assert before == json.loads((BASE.parent / 'SOURCE_MANIFEST.json').read_text())['files']
assert after == json.loads((HERE / 'CPU_RESULTS.json').read_text())['source_sha256']
changed = [p for p in after if after[p] != before.get(p)]
assert changed == ['recycle.py'] and before.keys() == after.keys()
lines = ''.join(f'{value}  {key}\n' for key, value in after.items())
(CAND / 'SOURCE_SHA256.txt').write_text(lines)
diff = ''.join(difflib.unified_diff((BASE / 'recycle.py').read_text().splitlines(True),
                                    (ENGINE / 'recycle.py').read_text().splitlines(True),
                                    fromfile='base83/engine/recycle.py', tofile='spine_swap/engine/recycle.py'))
(HERE / 'DIFF.patch').write_text(diff)
module = ast.parse((ENGINE / 'recycle.py').read_text())
body = [n for n in module.body if getattr(n, 'name', None) == 'TreeTemplate' or
        isinstance(n, ast.Assign) and any(getattr(x, 'id', None) == 'RANK_PRIOR' for x in n.targets)]
env = dict(dataclass=dataclass, ancestor_masks=lambda parent: [0] * len(parent))
exec(compile(ast.Module(body=body, type_ignores=[]), '<actual TreeTemplate>', 'exec'), env)
draft = next(n for n in module.body if getattr(n, 'name', None) == '_draft_kernel')
source = (ENGINE / 'recycle.py').read_text().splitlines(True)
draft_source = ''.join(source[draft.decorator_list[0].lineno - 1:draft.end_lineno])
(HERE / 'DRAFT_KERNEL_FROZEN.txt').write_text(draft_source)
cases = []
for R in (4, 8, 16, 32, 42, 64):
    tree = env['TreeTemplate'].build(R, 8)
    S = max(1, len(tree.spine))
    cases.append(dict(R=R, K=8, S=S, SP=S + max(1, tree.max_depth) + 1, VOCAB=151936))
pointer_names = ['root_ptr', 'table_ptr', 'parent_ptr', 'rank_ptr', 'spine_slot_ptr',
                 'spine_child_ptr', 'spine_ptr', 'nseen_ptr', 'anchor_ptr', 'blk_ptr',
                 'root_prev_ptr', 'node_keys_ptr', 'pair_keys_ptr', 'pair_values_ptr']
pointer_types = ['i64', 'i32', 'i32', 'i32', 'i32', 'i32', 'i64', 'i32', 'i32', 'i64',
                 'i64', 'i64', 'i64', 'i32']
assert [a.arg for a in draft.args.args] == pointer_names + ['K', 'R', 'S', 'SP', 'VOCAB']
plan = dict(
    status='PROPOSED_ONLY_NOT_RUN', target=dict(triton='3.1.0', cuda_arch=90, warp_size=32, ptxas='sm_90a'),
    base_revision=83, base_commit='9a3e7ad5d4e763287a1e9b0de25a55281fab998d',
    source_tree_sha256=sha(lines.encode()),
    source_sha256={name: after[name] for name in ['recycle.py', 'pair_cache.py']},
    frozen_decorated_kernel_sha256=sha(draft_source.encode()),
    kernel='_draft_kernel', exact_compile_calls=6, cases=cases,
    pointer_signature=dict(zip(pointer_names, pointer_types)),
    constexpr=['K', 'R', 'S', 'SP', 'VOCAB'],
    launch=dict(grid='(B,)', backend_options={}, effective_num_warps=4, effective_num_stages=3,
                pointer_attributes='Derive with Triton JIT _get_config using aligned allocation-base pointer stubs; do not invent attributes.'),
    limits=dict(process_seconds=180, per_case_seconds=40, retries=0),
    scope='Only changed draft: R4 chain/elision, R8/16/32 normal, R42 fallback B3, R64 maximum. K8 and vocab151936. No batch sweep; B is not constexpr. No unchanged publisher/model/accept compilation. Short-loop R2/R3 and custom K are outside this minimum experiment.',
    procedure=[
        'Verify source-tree, recycle.py, pair_cache.py and decorated-kernel hashes before extraction.',
        'Extract exact decorated draft bytes; import the unchanged _pair_slot from the exact frozen pair_cache.py. Do not import or allocate the CUDA model.',
        'One CPU-only compile process; six cases in listed order, one compile call per case, existing empty backend options; retain TTIR, TTGIR, LLVM IR, PTX, cubin and ptxas diagnostics.',
        'Inspect R4 to confirm compile-time branch elimination. Confirm parent/spine metadata reads are within R; pool addresses stay clamped; restoration loads are duplicate-predicated; pair rank0 additionally requires exact-key equality.',
        'Confirm 0<=original0<VOCAB and distinct/equality/fresh/later-child predicates survive lowering; first-match accept is not part of this build and remains byte-identical.',
        'Inspect scalar lane ownership and inherited draft load ordering without claiming GPU memory safety. No added blk read exists for the swap.',
        'Record ptxas registers, shared memory, stack/spills, warnings and elapsed compile seconds per case. Compare retained base83 evidence where the specialization matches; R42 has no retained base83 compiler comparison.',
        'Verify frozen hashes unchanged after the run. No GPU execution, official run, push or submission is authorized by this plan.'
    ])
(HERE / 'COMPILE_PLAN.json').write_text(json.dumps(plan, indent=2) + '\n')
raw = io.BytesIO()
with tarfile.open(fileobj=raw, mode='w') as tar:
    for name in after:
        data = (ENGINE / name).read_bytes()
        entry = tarfile.TarInfo(name)
        entry.size, entry.mode, entry.mtime = len(data), 0o644, 0
        tar.addfile(entry, io.BytesIO(data))
archive = CAND / 'krxfty_spine_swap.tar.gz'
with archive.open('wb') as target:
    with gzip.GzipFile(fileobj=target, mode='wb', mtime=0, filename='') as zipped:
        zipped.write(raw.getvalue())
with tarfile.open(archive, 'r:gz') as tar:
    assert {entry.name: sha(tar.extractfile(entry).read()) for entry in tar.getmembers()} == after
status = dict(status='FROZEN_CPU_PASS_COMPILATION_PROPOSED_ONLY',
              base_revision=83, base_commit=plan['base_commit'], changed_files=changed,
              base_source_tree_sha256=sha((BASE.parent / 'SOURCE_SHA256.txt').read_bytes()),
              source_tree_sha256=sha(lines.encode()), archive_sha256=sha(archive.read_bytes()),
              diff_sha256=sha(diff.encode()), compile_plan_sha256=sha((HERE / 'COMPILE_PLAN.json').read_bytes()),
              cpu_results_sha256=sha((HERE / 'CPU_RESULTS.json').read_bytes()),
              cpu_test_sha256=sha((HERE / 'check_spine_swap.py').read_bytes()),
              decorated_kernel_sha256=sha(draft_source.encode()),
              compiler_execution=False, gpu_execution=False, ci_execution=False,
              main_edited=False, pushed=False, submitted=False)
(CAND / 'STATUS.json').write_text(json.dumps(status, indent=2) + '\n')
complete = dict(base=before, candidate=after, status=status)
for path in (CAND / 'SOURCE_MANIFEST.json', HERE / 'SOURCE_MANIFEST.json'):
    path.write_text(json.dumps(complete, indent=2) + '\n')
print(json.dumps(status, indent=2))
