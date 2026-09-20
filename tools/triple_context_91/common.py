"""Pure-Python source guards and complete native launch records; no compiler import."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE_ID = 'ea2f5459a7496c1ec4f94c25d0bb53f544a77f6ca59ab43b6a6fabae0e25806a'
ARCHIVE_ID = '13f47bbe4841f79c882e59687c72a4a65aba5c8ae54c132cfa18a76e0604163c'
TARGET = dict(backend='cuda', arch=90, warp_size=32)
BOUNDS = dict(single_worker=True, total_seconds=240, case_seconds=40, retry=False, expand_search=False)
sha = lambda b: hashlib.sha256(b).hexdigest()
canonical = lambda obj: json.dumps(obj, sort_keys=True, separators=(',', ':'))


def function_text(path, name):
    raw = path.read_text()
    fn = next(n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == name)
    start = min([fn.lineno] + [n.lineno for n in fn.decorator_list])
    return ''.join(raw.splitlines(True)[start - 1:fn.end_lineno])


def verify_source():
    manifest = json.loads((HERE / 'SOURCE_MANIFEST.json').read_text())
    assert manifest['candidate_source_set_sha256'] == SOURCE_ID
    assert manifest['candidate_archive_sha256'] == ARCHIVE_ID
    assert manifest['bounds'] == BOUNDS and manifest['target'] == TARGET
    for label in ('engine', 'preparation', 'primary'):
        folder = HERE / 'snapshots' / label
        found = {p.relative_to(folder).as_posix(): sha(p.read_bytes()) for p in folder.rglob('*') if p.is_file()}
        assert found == manifest[label + '_files'], label
    assert len(manifest['engine_files']) == 18
    source = json.loads((HERE / 'snapshots/preparation/SOURCE_MANIFEST.json').read_text())
    assert source['candidate_files'] == {'engine/' + p: h for p, h in manifest['engine_files'].items()}
    listing = ''.join(h + '  engine/' + p + '\n' for p, h in sorted(manifest['engine_files'].items()))
    assert listing.encode() == (HERE / 'snapshots/preparation/SOURCE_SHA256.txt').read_bytes()
    assert sha(listing.encode()) == SOURCE_ID
    assert sha((HERE / 'snapshots/preparation/SOURCE_MANIFEST.json').read_bytes()) == manifest['candidate_source_manifest_sha256']
    assert sha((HERE / 'COMPILE_PLAN.json').read_bytes()) == manifest['compile_plan_sha256']
    assert (HERE / 'COMPILE_PLAN.json').read_bytes() == (HERE / 'snapshots/preparation/COMPILE_PLAN.json').read_bytes()
    plan = json.loads((HERE / 'COMPILE_PLAN.json').read_text())
    assert plan['candidate_source_set_sha256'] == SOURCE_ID
    assert [c['id'] for c in plan['cases']] == manifest['cases']
    assert len(plan['cases']) == len(set(manifest['cases'])) == 16
    assert {c['constexpr']['R'] for c in plan['cases']} == {2, 3, 4, 8, 16, 32, 42, 64}
    assert {c['kernel'] for c in plan['cases']} == {'_draft_kernel', '_publish_pairs_kernel'}
    assert all(c['all_pointers_allocation_bases'] for c in plan['cases'])
    assert sha((HERE / 'triple_kernels.py').read_bytes()) == manifest['module_sha256']
    for name, source in manifest['function_sources'].items():
        original = function_text(HERE / 'snapshots' / source['file'], name)
        assert original == function_text(HERE / 'triple_kernels.py', name)
        assert sha(original.encode()) == source['sha256'], name
    return manifest, plan


def verify_audit():
    audit = json.loads((HERE / 'AUDIT_MANIFEST.json').read_text())
    for name, digest in audit['files'].items():
        assert sha((HERE / name).read_bytes()) == digest, name
    return audit


class Pointer:
    def __init__(self, signature):
        self.dtype = {'*i64': 'torch.int64', '*i32': 'torch.int32'}[signature]

    def data_ptr(self):
        return 0x100000  # Native aligned allocation-base specialization; never dereferenced.


def values_for(case):
    return {**{name: Pointer(sig) for name, sig in case['signature'].items()}, **case['constexpr']}


def make_record(case, kernel, attrs, bound, sigspec, constvals, extra, explicit, parsed):
    signature = {kernel.params[i].name: v for i, v in zip(kernel.non_constexpr_indices, sigspec)}
    constants = {p.name: v for p, v in zip(kernel.params, bound.values())
                 if p.is_constexpr or p.num in attrs.equal_to_1 or v is None}
    assert signature == case['signature'], (case['id'], signature)
    assert constants == case['constexpr'], (case['id'], constants)
    assert not {'B', 'T'} & set(constants), 'No invented batch or prompt specialization'
    for key, value in case['expected_default_options'].items():
        assert getattr(parsed, key) == value, (case['id'], key)
    assert parsed.num_ctas == 1
    normalized = dict(parsed.__dict__)
    assert len(parsed.extern_libs) == 1 and parsed.extern_libs[0][0] == 'libdevice'
    assert Path(parsed.extern_libs[0][1]).name == 'libdevice.10.bc'
    normalized['extern_libs'] = [['libdevice', '<TRITON_LIBDEVICE>']]
    # Exact JITFunction.run cache key. Function identity is a separate part of
    # the complete key because runtime keeps a separate cache per JITFunction.
    native_key = ''.join(sigspec) + str((constvals, extra))
    manifest = json.loads((HERE / 'SOURCE_MANIFEST.json').read_text())
    helpers = {n: manifest['function_sources'][n]['sha256'] for n in ('_pair_slot', '_triple_slot')}
    identity = dict(kernel=case['kernel'], native_key_sha256=sha(native_key.encode()),
                    function_sha256=case['source_function_sha256'], helper_sha256=helpers,
                    module_sha256=manifest['module_sha256'], target=TARGET,
                    signature=signature, constants=constants,
                    attrs=dict(divisible_by_16=sorted(attrs.divisible_by_16), equal_to_1=sorted(attrs.equal_to_1)),
                    options_explicit=explicit, parsed_options=normalized)
    return dict(case=case['id'], native_key_text=native_key, compile_identity=identity,
                dedup_key_sha256=sha(canonical(identity).encode()), representative_grid=case['representative_grid'])


def check_unique(records):
    assert len(records) == 16
    assert len({r['dedup_key_sha256'] for r in records}) == 16
    assert len({r['compile_identity']['native_key_sha256'] for r in records}) == 16
    return dict(planned=16, native_keys=16, complete_keys=16, builds_after_dedup=16,
                grid_not_in_key=True, batch_prompt_sweep=False)
