"""Targeted actual-AST CPU checks; no engine import, Triton compilation or GPU.

The old pair fixture supplies fake streams/pointers and its independent serial
prefix oracle. None of its original suites are run. New metadata is built by
the actual candidate Recycler constructor, rather than duplicated in fixtures.
"""
import ast
import hashlib
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / 'work/candidates/krxfty_pair_cache_on82/engine'
CAND = ROOT / 'work/candidates/krxfty_spine_swap/engine'
fixture = runpy.run_path(str(ROOT / 'work/research/krxfty_pair_cache/check_pair_cache.py'))
old = fixture['old']
TL, Tensor, Ptr = (fixture[x] for x in ('TL', 'Tensor', 'Ptr'))
load = fixture['load']
env = fixture['env']
load(CAND / 'kernels/attention.py', {'ancestor_masks'}, env)
load(CAND / 'recycle.py', {'TreeTemplate', 'RANK_PRIOR', '_draft_kernel', 'Recycler'}, env)
actual_draft = env['_draft_kernel']
env['_draft_kernel'] = fixture['Kernel'](actual_draft)
before_env = dict(env)
load(BASE / 'recycle.py', {'_draft_kernel'}, before_env)
before_draft = before_env['_draft_kernel']
STATS = dict(directed_cases=0, random_cases=0, ancestry_rows=0,
             accepted_prefix_cases=0, serial_requests=0, negative_controls=0,
             restored_siblings=0, metadata_reuse_checks=0)


def fake_torch(stream):
    def wrap(values, dtype, device):
        return Tensor(np.asarray(values, dtype=dtype), stream, True)
    return SimpleNamespace(
        int32=np.int32, int64=np.int64,
        tensor=lambda values, dtype, device: wrap(values, dtype, device),
        full=lambda shape, value, dtype, device: wrap(np.full(shape, value), dtype, device),
        empty=lambda shape, dtype, device: wrap(np.empty(shape, dtype=dtype), dtype, device),
        zeros=lambda shape, dtype, device: wrap(np.zeros(shape), dtype, device))


def make_rec(R, B=1, vocab=96):
    stream = old['Stream']()
    env['torch'] = fake_torch(stream)
    rec = env['Recycler'](vocab, B, R, 8, 'cpu-fixture')
    rec.root.data.fill(5)
    rec.root_prev.data.fill(3)
    rec.table.data[:] = np.array([4, 24, 25, 26, 27, 28, 29, 30])
    rec.pair_values.data.fill(-1)
    nseen = Tensor(np.full(B, 20, np.int32), stream, True)
    rec.spine_anchor.data.fill(20)
    return rec, stream, nseen


def run_kernel(rec, nseen, old_source=False):
    if not old_source:
        rec.draft(nseen)
        return rec.blk.data.copy()
    out = np.full_like(rec.blk.data, -777)
    keys = np.full_like(rec.node_keys.data, -777)
    for b in range(rec.B):
        TL.pid = (b, 0, 0)
        values = [rec.root.data, rec.table.data, rec.parent.data, rec.rank.data,
                  rec.spine_slot.data, rec.spine.data, nseen.data, rec.spine_anchor.data,
                  out, rec.root_prev.data, keys, rec.pair_keys.data, rec.pair_values.data]
        before_draft(*(Ptr(x) for x in values), K=rec.k, R=rec.R, S=rec.S, SP=rec.SP)
    return out


def pack(a, b):
    return (int(a) << 32) | int(b)


def set_pair(rec, b, previous, token, values, wrong_key=False):
    key = pack(previous, token)
    slot = b * 4096 + fixture['hash_ref'](key)
    rec.pair_keys.data[slot] = key ^ (1 << 40) if wrong_key else key
    rec.pair_values.data[slot] = values


def expected_draft(rec, nseen):
    # Independent lists/dictionaries and ancestry reconstruction, no device
    # metadata reads or translation of the new kernel's indexing expressions.
    rows, changed = [], []
    t = rec.template
    for b in range(rec.B):
        root = int(rec.root.data[b])
        offset = int(nseen.data[b] - rec.spine_anchor.data[b])
        pool = rec.spine.data[b].tolist()
        fresh = 0 <= offset <= rec.SP - rec.S
        if fresh and offset:
            fresh = pool[offset - 1] == root
        overrides = {node: pool[offset + depth] for depth, node in enumerate(t.spine)
                     if fresh and pool[offset + depth] >= 0}
        spine_by_parent = {t.parent[node]: node for node in overrides}
        branch = [[int(rec.root_prev.data[b]), root]]
        out = [root]
        for i in range(1, rec.R):
            p, rank = t.parent[i], t.rank[i]
            token = out[p]
            key = pack(*branch[p][-2:])
            slot = b * 4096 + fixture['hash_ref'](key)
            pair = rec.pair_values.data[slot] if rec.pair_keys.data[slot] == key else [-1] * rec.k
            regular = int(pair[rank]) if pair[rank] >= 0 else max(int(rec.table.data[token, rank]), 0)
            if i in overrides:
                value = overrides[i]
            else:
                value = regular
                other = spine_by_parent.get(p)
                if other is not None and other < i and value == out[other]:
                    original = int(pair[0]) if pair[0] >= 0 else int(rec.table.data[token, 0])
                    if 0 <= original < rec.table.shape[0] and original != out[other]:
                        value = original
                        changed.append((b, i))
            out.append(value)
            branch.append(branch[p] + [value])
        rows.append(out)
    return np.asarray(rows, np.int64), changed


def assert_draft(rec, nseen, baseline_equal=False):
    expected, changed = expected_draft(rec, nseen)
    actual = run_kernel(rec, nseen)
    assert np.array_equal(actual, expected), (actual.tolist(), expected.tolist())
    if baseline_equal:
        assert np.array_equal(actual, run_kernel(rec, nseen, True)), 'inactive draft changed'
    for b in range(rec.B):
        for i in range(rec.R):
            lineage, j = [], i
            while j >= 0:
                lineage.append(int(actual[b, j]))
                j = rec.template.parent[j]
            full = [int(rec.root_prev.data[b])] + lineage[::-1]
            assert int(rec.node_keys.data[b, i]) == pack(*full[-2:]), ('ancestry', b, i)
            STATS['ancestry_rows'] += 1
    STATS['restored_siblings'] += len(changed)
    return actual


def metadata_checks(rec):
    t = rec.template
    wanted = [-1] * rec.R
    for node in t.spine:
        wanted[t.parent[node]] = node
        assert all(node < sibling for sibling in t.children[t.parent[node]] if sibling != node)
    assert rec.spine_child.data.tolist() == wanted
    assert rec.spine_child.data.dtype == np.int32


def directed_checks():
    for R in (4, 8, 16, 32, 42, 64):
        rec, _, nseen = make_rec(R)
        metadata_checks(rec)
        sibling = next((i for i in rec.template.children[0] if rec.template.rank[i] == 1), None)
        # A valid sparse pair rank0 takes precedence over different unigram0;
        # unknown rank1 falls back independently and may duplicate the override.
        for u0, p0, p1, override, wrong_key in [
            (7, 11, -1, 23, False), (7, -1, 23, 23, False),
            (7, 11, 23, 23, False), (7, 11, 23, 23, True),
            (-1, -1, 23, 23, False), (23, 23, 23, 23, False),
            (0, -1, -1, 23, False), (95, -1, 23, 23, False),
            (96, -1, 23, 23, False), (7, 96, 23, 23, False),
            (7, 11, 22, 23, False)]:
            rec.table.data[:] = np.array([4, 24, 25, 26, 27, 28, 29, 30])
            rec.table.data[5, :2] = [u0, 23]
            rec.pair_keys.data.fill(-1)
            set_pair(rec, 0, 3, 5, [p0, p1] + [-1] * 6, wrong_key)
            rec.spine.data.fill(-1)
            rec.spine.data[0, 0] = override
            out = assert_draft(rec, nseen)
            effective0 = p0 if p0 >= 0 and not wrong_key else u0
            rank1 = p1 if p1 >= 0 and not wrong_key else 23
            if sibling is not None:
                wanted = effective0 if rank1 == override and 0 <= effective0 < 96 and effective0 != override else rank1
                assert out[0, sibling] == wanted
            STATS['directed_cases'] += 1
        # Existing partial rows may already duplicate rank0 at multiple ranks.
        # Swapping override duplicates does not promise distinct children.
        rec.table.data[:] = np.array([4, 24, 25, 26, 27, 28, 29, 30])
        rec.table.data[5, :3] = [7, 23, 11]
        set_pair(rec, 0, 3, 5, [11, -1, -1, -1, -1, -1, -1, -1])
        rec.spine.data.fill(-1)
        rec.spine.data[0, 0] = 23
        out = assert_draft(rec, nseen)
        if R >= 16:
            root_children = [out[0, c] for c in rec.template.children[0]]
            assert root_children.count(11) >= 2
        STATS['directed_cases'] += 1
        # Missing, stale (prior-token mismatch), negative and expired pools
        # retain the base83 tree exactly; matching offset remains active.
        for offset, previous_match, override in [(0, True, -1), (1, False, 23),
                                                (-1, True, 23), (rec.SP - rec.S + 1, True, 23),
                                                (2, True, 23)]:
            nseen.data.fill(20 + offset)
            rec.spine.data.fill(-1)
            if 0 <= offset < rec.SP:
                rec.spine.data[0, offset] = override
            if 0 < offset <= rec.SP:
                rec.spine.data[0, offset - 1] = 5 if previous_match else 6
            assert_draft(rec, nseen, baseline_equal=offset != 2)
            STATS['directed_cases'] += 1
        # All-unknown rows remain zero fallback, including active override zero:
        # raw -1 is not treated as a valid displaced token zero.
        rec.table.data.fill(-1)
        rec.pair_keys.data.fill(-1)
        nseen.data.fill(20)
        rec.spine.data.fill(-1)
        rec.spine.data[0, 0] = 0
        assert_draft(rec, nseen, baseline_equal=True)
        STATS['directed_cases'] += 1
        # Exercise every eligible parent, not only the root. Off-spine rank0
        # branches have no map entry and retain their own pair/unigram result.
        rec.table.data[:] = np.array([7, 23, 25, 26, 27, 28, 29, 30])
        rec.pair_keys.data.fill(-1)
        rec.spine.data.fill(23)
        for previous, token in [(3, 5), (5, 23), (23, 23)]:
            set_pair(rec, 0, previous, token, [11] + [-1] * 7)
        out = assert_draft(rec, nseen)
        for parent in range(R):
            spine_child = int(rec.spine_child.data[parent])
            for child in rec.template.children[parent]:
                if spine_child >= 0 and rec.template.rank[child] == 1:
                    assert out[0, child] == 11
                if spine_child < 0 and rec.template.rank[child] == 0:
                    assert out[0, child] != 23
        STATS['directed_cases'] += 1

    # Real vocabulary boundaries and packed keys with large predecessor/token
    # IDs; an invalid raw rank0 cannot be restored into a later sibling.
    rec, _, nseen = make_rec(8, vocab=151936)
    rec.root.data.fill(151935)
    rec.root_prev.data.fill(151934)
    rec.table.data[151935, :2] = [7, 151932]
    rec.spine.data[0, 0] = 151932
    for original in (0, 151935, 151936):
        set_pair(rec, 0, 151934, 151935, [original] + [-1] * 7)
        out = assert_draft(rec, nseen)
        assert out[0, 4] == (original if original < 151936 else 151932)
        STATS['directed_cases'] += 1


def random_checks():
    rng = np.random.default_rng(86083)
    for R in (4, 8, 16, 32, 42, 64):
        rec, _, nseen = make_rec(R, B=3)
        identity = (id(rec.spine_child.data), rec.spine_child.data.__array_interface__['data'][0])
        immutable = rec.spine_child.data.copy()
        for trial in range(24):
            rec.table.data[:] = rng.integers(-1, 24, rec.table.shape)
            rec.pair_keys.data.fill(-1)
            rec.spine.data[:] = rng.integers(-1, 24, rec.spine.shape)
            rec.root.data[:] = rng.integers(0, 24, rec.B)
            rec.root_prev.data[:] = rng.integers(0, 24, rec.B)
            offsets = [0, 2, rec.SP - rec.S + 1]
            nseen.data[:] = np.array(offsets) + 20
            for b in range(rec.B):
                token = int(rec.root.data[b])
                rec.spine.data[b, 0] = int(rec.table.data[token, 1])
                rec.spine.data[b, 1] = rec.root.data[b]
                if b == 1:
                    rec.spine.data[b, 2] = int(rec.table.data[token, 1])
                for previous, tok in [(int(rec.root_prev.data[b]), token)] + [tuple(x) for x in rng.integers(0, 24, (12, 2))]:
                    values = rng.integers(-1, 24, rec.k)
                    values[rng.random(rec.k) < .6] = -1
                    set_pair(rec, b, previous, tok, values)
            assert_draft(rec, nseen)
            assert (id(rec.spine_child.data), rec.spine_child.data.__array_interface__['data'][0]) == identity
            assert np.array_equal(rec.spine_child.data, immutable)
            STATS['metadata_reuse_checks'] += 1
            STATS['random_cases'] += rec.B


def acceptance_checks():
    load(CAND / 'kernels/accept.py', {'accept_kernel'}, env)
    for R in (8, 16, 32, 42, 64):
        rec, _, nseen = make_rec(R)
        rec.table.data[5, :2] = [7, 23]
        set_pair(rec, 0, 3, 5, [11] + [-1] * 7)
        rec.spine.data[0, 0] = 23
        original = run_kernel(rec, nseen, True)
        repaired = assert_draft(rec, nseen)
        sibling = next(c for c in rec.template.children[0] if rec.template.rank[c] == 1)
        first = rec.template.spine[0]
        for greedy in (11, 23):
            outcomes = []
            for blk in (original, repaired):
                arr = old['acceptance_arrays'](rec.template, 1)
                arr['blk'][:] = blk
                arr['cand'].fill(94)
                arr['cand'][0, 0] = greedy
                old['run_accept'](arr, kernel=env['accept_kernel'])
                path = arr['idx'][0, :arr['lens'][0]].tolist()
                tokens = arr['tokens'][0, :arr['cnt'][0]].tolist()
                # Each emitted child equals its actual parent's greedy ID.
                parent = 0
                for child in path:
                    assert int(blk[0, child]) == int(arr['cand'][0, parent])
                    parent = child
                assert tokens[-1] == int(arr['cand'][0, parent])
                outcomes.append(path)
                STATS['accepted_prefix_cases'] += 1
            if greedy == 11:
                assert outcomes[0] == [] and outcomes[1] == [sibling], 'lost pair-rank0 not recovered'
            else:
                assert outcomes[0] == outcomes[1] == [first], 'earlier spine first-match changed'


def serial_checks():
    fixture['make_fixture'].__globals__['CAND'] = CAND
    for B, R in [(16, 4), (8, 8), (4, 16), (2, 32), (3, 42), (1, 64)]:
        g, stream, state = fixture['make_fixture'](B, R, 25, 'mixed')
        # Construct new metadata with candidate __init__, then retain that same
        # allocation across the fixture's captured rounds and request reuse.
        constructed, _, _ = make_rec(R, B)
        g.recycler.spine_child = Tensor(constructed.spine_child.data, stream, True)
        captured = g.recycler.spine_child.data
        for repeat in range(2):
            prompts = [[(i * (repeat + 1) + b + repeat * 9) % 31 for i in range(g.T)] for b in range(B)]
            expected = list(map(list, zip(*(old['serial'](p, 25, 'mixed') for p in prompts))))
            assert list(g.run_recycle(prompts, 25)) == expected
            assert stream.idle()
            assert g.recycler.spine_child.data is captured
            assert np.array_equal(captured, constructed.spine_child.data)
            STATS['serial_requests'] += 1
            STATS['metadata_reuse_checks'] += 1


def negative_controls():
    source = (CAND / 'recycle.py').read_text()
    mutants = {
        'restore_unigram_instead_of_pair': source.replace('original0 = tl.where(pair0 >= 0, pair0, original0)', 'original0 = original0'),
        'restore_without_duplicate': source.replace('(cand == override)', '(cand >= 0)'),
        'accept_unknown_clamp_as_original': source.replace('original0 = tl.where(pair0 >= 0, pair0, original0)', 'original0 = tl.maximum(tl.where(pair0 >= 0, pair0, original0), 0)'),
    }
    saved = env['_draft_kernel']
    for name, source in mutants.items():
        mutated = dict(env)
        load(CAND / 'recycle.py', {'_draft_kernel'}, mutated, source)
        env['_draft_kernel'] = fixture['Kernel'](mutated['_draft_kernel'])
        rec, _, nseen = make_rec(8)
        rec.table.data[5, :2] = [7, 23]
        rec.spine.data[0, 0] = 23
        set_pair(rec, 0, 3, 5, [11] + [-1] * 7)
        if name == 'restore_without_duplicate':
            rec.table.data[5, 1] = 22
        elif name == 'accept_unknown_clamp_as_original':
            rec.table.data[5, 0] = -1
            rec.pair_keys.data.fill(-1)
        try:
            try:
                assert_draft(rec, nseen)
            except AssertionError:
                STATS['negative_controls'] += 1
            else:
                raise RuntimeError(f'negative control escaped: {name}')
        finally:
            env['_draft_kernel'] = saved


def source_checks():
    changed = []
    for path in sorted(CAND.rglob('*.py')):
        ast.parse(path.read_text())
        if path.read_bytes() != (BASE / path.relative_to(CAND)).read_bytes():
            changed.append(str(path.relative_to(CAND)))
    assert changed == ['recycle.py'], changed
    baseline = ast.parse((BASE / 'recycle.py').read_text())
    candidate = ast.parse((CAND / 'recycle.py').read_text())
    def named(tree, name):
        return next(x for x in tree.body if getattr(x, 'name', None) == name)
    def same(a, b):
        return ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)
    assert same(named(baseline, 'TreeTemplate'), named(candidate, 'TreeTemplate'))
    for name in ('update', 'publish_pairs', 'accept'):
        assert same(named(named(baseline, 'Recycler'), name), named(named(candidate, 'Recycler'), name))
    assert not any(isinstance(n, ast.Constant) and isinstance(n.value, float)
                   for n in ast.walk(named(candidate, '_draft_kernel')))
    return changed


if __name__ == '__main__':
    changed = source_checks()
    directed_checks()
    random_checks()
    acceptance_checks()
    serial_checks()
    negative_controls()
    assert STATS['restored_siblings'] > 0 and STATS['negative_controls'] == 3
    result = dict(status='PASS', source_extraction='actual AST', changed_files=changed,
                  device_execution=False, compiler_execution=False, stats=STATS,
                  pair_fixture_rounds=fixture['STATS']['rounds'],
                  source_sha256={str(p.relative_to(CAND)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(CAND.rglob('*.py'))})
    (HERE / 'CPU_RESULTS.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
