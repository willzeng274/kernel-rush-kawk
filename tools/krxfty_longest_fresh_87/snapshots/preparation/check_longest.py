"""Actual-source AST CPU semantics; no Torch, Triton compiler, GPU, or CI.

Expected paths are enumerated by independent DFS using child adjacency and
per-parent predictions, never the candidate's ancestor masks or depth scan.
"""
import ast
import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CAND = ROOT / 'work/candidates/krxfty_longest_verified/engine'
BASE = ROOT / 'work/candidates/krxfty_pair_cache_on82/engine'
STATS = dict(cases=0, live=0, frozen=0, gains=0, clipped_retained=0,
             compaction=0, publication=0, mixed_batch_lanes=0, mutants={}, shapes={})


class V(np.ndarray):
    def __new__(cls, value):
        return np.asarray(value).view(cls)

    def to(self, dtype):
        return V(self.astype(dtype))


class Ptr:
    def __init__(self, data, offset=0):
        self.data, self.offset = data.reshape(-1), offset

    def __add__(self, offset):
        return Ptr(self.data, self.offset + offset)


class TL:
    pid = (0, 0, 0)
    trace_target, trace = None, []
    int32, uint64 = np.int32, np.uint64
    program_id = staticmethod(lambda axis: V(TL.pid[axis]))
    arange = staticmethod(lambda a, b: V(np.arange(a, b, dtype=np.int32)))
    full = staticmethod(lambda shape, value, dtype: V(np.full(shape, value, dtype)))
    sum = staticmethod(lambda a, axis: V(np.sum(a, axis=axis)))
    min = staticmethod(lambda a, axis: V(np.min(a, axis=axis)))
    max = staticmethod(lambda a, axis: V(np.max(a, axis=axis)))
    where = staticmethod(lambda p, a, b: V(np.where(p, a, b)))
    minimum = staticmethod(lambda a, b: V(np.minimum(a, b)))
    maximum = staticmethod(lambda a, b: V(np.maximum(a, b)))
    static_range = staticmethod(range)

    @staticmethod
    def load(ptr, mask=True, other=0):
        offsets, active = np.broadcast_arrays(np.asarray(ptr.offset, int), np.asarray(mask, bool))
        ids = offsets[active]
        assert np.all((ids >= 0) & (ids < len(ptr.data))), ('read bounds', ids)
        result = np.full(offsets.shape, other, dtype=ptr.data.dtype)
        result[active] = ptr.data[ids]
        return V(result)

    @staticmethod
    def store(ptr, value, mask=True):
        offsets, values, active = np.broadcast_arrays(np.asarray(ptr.offset, int), value, np.asarray(mask, bool))
        ids = offsets[active]
        assert np.all((ids >= 0) & (ids < len(ptr.data))), ('write bounds', ids)
        ptr.data[ids] = values[active]
        if TL.trace_target is not None and np.shares_memory(ptr.data, TL.trace_target) and len(ids):
            TL.trace.append(values[active].tolist())


def extract(path, names, env, source=None):
    module = ast.parse(path.read_text() if source is None else source)
    body = [n for n in module.body if getattr(n, 'name', None) in names or
            isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    for n in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(n, ast.FunctionDef):
            n.decorator_list = [d for d in n.decorator_list if isinstance(d, ast.Name) and d.id in ('classmethod', 'property')]
            for arg in n.args.args + n.args.kwonlyargs:
                arg.annotation = None
            n.returns = None
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), 'exec'), env)
    return env


ENV = dict(tl=TL, dataclass=dataclass)
extract(CAND / 'kernels/attention.py', {'ancestor_masks'}, ENV)
extract(CAND / 'kernels/accept.py', {'flat_children', 'accept_kernel'}, ENV)
extract(CAND / 'recycle.py', {'RANK_PRIOR', 'TreeTemplate', 'Recycler'}, ENV)
extract(CAND / 'kernels/compact.py', {'_compact_kernel'}, ENV)
extract(CAND / 'pair_cache.py', {'PAIR_SLOTS', '_pair_slot', '_publish_pairs_kernel'}, ENV)
Template, kernel = ENV['TreeTemplate'], ENV['accept_kernel']


def custom(parent):
    children = [[] for _ in parent]
    for c, p in enumerate(parent):
        if p >= 0:
            children[p].append(c)
    return Template(parent, [0] * len(parent), children, ENV['ancestor_masks'](parent))


def oracle(t, blk, cand, remaining):
    old, p = [], 0
    while True:
        kids = [c for c in t.children[p] if blk[c] == cand[p]]
        if not kids:
            break
        p = kids[0]
        old.append(p)
    # Exhaustive recursive path enumeration. Only verified edges are visited.
    paths = [[]]
    def visit(p, prefix):
        for c in t.children[p]:
            if blk[c] == cand[p]:
                path = prefix + [c]
                paths.append(path)
                visit(c, path)
    visit(0, [])
    best = min(paths, key=lambda path: (-len(path), path[-1] if path else 0))
    chosen = best if t.size <= 64 and min(len(best) + 1, remaining) > min(len(old) + 1, remaining) else old
    endpoint = chosen[-1] if chosen else 0
    return [int(blk[c]) for c in chosen] + [int(cand[endpoint])], chosen, old, best


def arrays(t, blk, cand, remaining=99, frozen=False, cap=1024):
    r, a = t.size, max(1, t.max_depth)
    start, flat, par = ENV['flat_children'](t.children)
    assert sorted(flat) == list(range(1, r)), 'unique nonroot child bits'
    return dict(blk=np.array([blk], np.int64), cand=np.array([cand], np.int64),
                start=np.array(start, np.int32), flat=np.array(flat or [0], np.int32),
                par=np.array(par or [0], np.int32), masks=np.array(t.masks, np.int64),
                depth=np.array(t.depth, np.int32), done=np.array([frozen], np.int32),
                nseen=np.array([7], np.int32), pos=np.array([20], np.int32),
                limit=np.array([7 + remaining], np.int32), root=np.array([777], np.int64),
                idx=np.full((1, a), -999, np.int32), lens=np.array([100], np.int32),
                tokens=np.full((1, a + 1), -999, np.int64), cnt=np.array([100], np.int32))


def execute(arr, fn=kernel, cap=1024):
    _, r = arr['blk'].shape
    keys = ('blk', 'cand', 'start', 'flat', 'par', 'masks', 'depth', 'done', 'nseen',
            'pos', 'limit', 'root', 'idx', 'lens', 'tokens', 'cnt')
    TL.pid = (0, 0, 0)
    fn(*(Ptr(arr[k]) for k in keys), cap, r, len(arr['flat']),
       1 << max(r, len(arr['flat']) - 1).bit_length(), arr['idx'].shape[1], 2 * r)


def verify_case(t, arr, fn=kernel, cap=1024, downstream=False, count=True):
    before = {k: v.copy() for k, v in arr.items()}
    remaining = int(arr['limit'][0] - arr['nseen'][0])
    tokens, path, old, best = oracle(t, arr['blk'][0], arr['cand'][0], remaining)
    live = not bool(arr['done'][0])
    execute(arr, fn, cap)
    assert int(arr['lens'][0]) == (len(path) if live else -1), ('path length', t.size, path, arr['lens'])
    assert int(arr['cnt'][0]) == (len(tokens) if live else 0), 'full accepted count'
    assert int(arr['nseen'][0]) == 7 + (len(tokens) if live else 0), 'unclipped nseen'
    assert int(arr['root'][0]) == (tokens[-1] if live else 777), 'root'
    expected_done = (not live or 7 + len(tokens) >= int(arr['limit'][0]) or
                     20 + len(tokens) + 2 * t.size >= cap)
    assert bool(arr['done'][0]) == expected_done, 'done'
    if live:
        assert arr['idx'][0, :len(path)].tolist() == path, ('path', path, arr['idx'])
        assert arr['tokens'][0, :len(tokens)].tolist() == tokens, 'ordered tokens'
        assert all(a < b for a, b in zip([0] + path, path)), 'compaction-safe path order'
        for parent, child in zip([0] + path, path):
            assert int(arr['blk'][0, child]) == int(arr['cand'][0, parent]), 'fully verified own ancestor edge'
    else:
        assert np.array_equal(arr['idx'], before['idx']), 'frozen path storage'
        assert np.array_equal(arr['tokens'][0, 1:], before['tokens'][0, 1:]), 'frozen token tail'
    rec = object.__new__(ENV['Recycler'])
    rec.template = t
    assert rec.accept(before['blk'][0].tolist(), before['cand'][0].tolist(), remaining) == (tokens, path)
    if downstream:
        verify_downstream(t, arr, path if live else None, count=count)
    if count:
        STATS['cases'] += 1
        STATS['live' if live else 'frozen'] += 1
        STATS['gains'] += int(live and len(path) > len(old))
        STATS['clipped_retained'] += int(live and len(best) > len(old) and path == old)
        STATS['shapes'][str(t.size)] = STATS['shapes'].get(str(t.size), 0) + 1


def verify_downstream(t, arr, path, count=True):
    # Actual compact kernel, with distinctive source rows and overlap.
    a, r, cap, d = arr['idx'].shape[1], t.size, 256, 4
    cache = np.arange(cap * d, dtype=np.float32).reshape(1, 1, 1, cap, d)
    value = cache + 0.5
    old_k, old_v = cache.copy(), value.copy()
    TL.pid = (0, 0, 0)
    ENV['_compact_kernel'](*(Ptr(x) for x in (cache, value, arr['pos'], arr['idx'], arr['lens'])), cap, 1, 1, a, d)
    expected_k, expected_v = old_k.copy(), old_v.copy()
    if path is not None:
        for i, row in enumerate(path):
            expected_k[0, 0, 0, 21+i] = old_k[0, 0, 0, 20+row]
            expected_v[0, 0, 0, 21+i] = old_v[0, 0, 0, 20+row]
    assert np.array_equal(cache, expected_k) and np.array_equal(value, expected_v), 'actual compaction'
    assert 20 + int(arr['lens'][0]) + 1 == (20 if path is None else 21 + len(path)), 'next root position'
    # All rows deliberately collide in one hash slot. Actual publisher must
    # leave the latest consumed row, excluding siblings and the bonus token.
    slot_keys = np.full(4096, -1, np.int64)
    slot_values = np.full((4096, 8), -99, np.int32)
    node_keys = np.zeros((1, r), np.int64)
    top = np.arange(r * 8, dtype=np.int32).reshape(r, 8)
    previous = np.array([987], np.int64)
    TL.trace_target, TL.trace = slot_values, []
    try:
        ENV['_publish_pairs_kernel'](*(Ptr(x) for x in (slot_keys, slot_values, node_keys, top,
            arr['blk'], previous, arr['idx'], arr['lens'])), 8, r, a, 8)
    finally:
        TL.trace_target = None
    assert TL.trace == ([] if path is None else [top[row].tolist() for row in [0] + path]), 'every publication in path order'
    if path is None:
        assert np.all(slot_keys == -1) and np.all(slot_values == -99) and previous[0] == 987
    else:
        final = path[-1] if path else 0
        assert slot_keys[0] == 0 and np.array_equal(slot_values[0], top[final]), 'chronological publication'
        assert np.all(slot_keys[1:] == -1) and np.all(slot_values[1:] == -99)
        assert previous[0] == arr['blk'][0, final], 'consumed endpoint predecessor'
    if count:
        STATS['compaction'] += 1
        STATS['publication'] += 1


def directed():
    out = []
    t = Template.build(8, 8)
    assert t.parent[1] == t.parent[4] == 0 and t.parent[6] == 4
    blk, cand = [9]*8, [8]*8
    blk[1] = blk[4] = cand[0] = 1
    cand[1] = cand[4] = blk[6] = 2
    cand[6] = 3
    for remaining in (-2, 0, 1, 2, 3, 4, 99):
        out.append((f'r8_remaining_{remaining}', t, arrays(t, blk, cand, remaining)))
    out.append(('r8_frozen', t, arrays(t, blk, cand, 99, True)))
    # Strict equal-useful-length retention even if min endpoint prefers sibling.
    t = custom([-1, 0, 0, 2, 1])
    out.append(('equal_depth_keeps_old', t, arrays(t, [1]*5, [1]*5)))
    # Two longer alternatives tie: endpoint 6 is deterministically chosen.
    t = custom([-1, 0, 0, 0, 2, 3, 4, 5])
    out.append(('equal_alternatives', t, arrays(t, [1]*8, [1]*8)))
    # R64 endpoint 63, including valid bit63 and invalid bit63. Short old child1.
    t = custom([-1, 0, 0] + list(range(2, 63)))
    for wrong in (None, 63, 32, 2):
        blk = [1]*64
        if wrong is not None:
            blk[wrong] = 4
        out.append((f'bit63_invalid_{wrong}', t, arrays(t, blk, [1]*64)))
    for r in (4, 8, 16, 32, 42, 64):
        t = Template.build(r, 8)
        out.append((f'all_match_r{r}', t, arrays(t, [1]*r, [1]*r)))
        out.append((f'zero_match_r{r}', t, arrays(t, [1]*r, [2]*r)))
        out.append((f'frozen_r{r}', t, arrays(t, [1]*r, [1]*r, frozen=True)))
    # Unsupported mask width explicitly retains original acceptance.
    t = custom([-1, 0, 0] + list(range(2, 64)))
    t.masks = [0] * 65
    out.append(('r65_fallback', t, arrays(t, [1]*65, [1]*65)))
    return out


def preservation():
    changed = [str(p.relative_to(CAND)) for p in CAND.rglob('*.py')
               if p.read_bytes() != (BASE / p.relative_to(CAND)).read_bytes()]
    assert sorted(changed) == ['engine.py', 'kernels/accept.py', 'recycle.py']
    source = (CAND / 'engine.py').read_text().replace('                     rec.masks, rec.depth,\n', '')
    assert source == (BASE / 'engine.py').read_text(), 'callsite only; graph chronology unchanged'
    before, after = ast.parse((BASE / 'recycle.py').read_text()), ast.parse((CAND / 'recycle.py').read_text())
    for tree in (before, after):
        recycler = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Recycler')
        recycler.body = [n for n in recycler.body if getattr(n, 'name', None) != 'accept']
    assert ast.dump(before) == ast.dump(after), 'host reference only'
    for p in CAND.rglob('*.py'):
        ast.parse(p.read_text())
    STATS['changed_files'] = sorted(changed)


def mutants(cases):
    original = (CAND / 'kernels/accept.py').read_text()
    replacements = {
        'skip_longest': ('if R <= 64 and MAXA < R - 1:', 'if R < 0:'),
        'accept_bad_ancestors': ('((masks & bad) == 0)', '(masks == masks)'),
        'child_prediction_instead_of_parent': ('b * R + cp, mask=j < C', 'b * R + cl, mask=j < C'),
        'parent_bits_instead_of_child': ('<< cl.to(tl.uint64)', '<< cp.to(tl.uint64)'),
        'drop_invalid_bit63': ('        masks = tl.load', '        bad = bad & 9223372036854775807\n        masks = tl.load'),
        'padded_root_bit_poison': ('(j < C) & (cl > 0) & ~hit', '~hit'),
        'unclipped_choice': ('tl.minimum(best_depth + 1, remaining)', '(best_depth + 1)'),
        'replace_ties': ('tl.minimum(best_depth + 1, remaining) >', 'tl.minimum(best_depth + 1, remaining) >='),
        'reverse_path_stores': ('offset = tl.maximum(d - 1, 0)', 'offset = step'),
        'wrong_bonus_row': ('j == endpoint, cv', 'j == 0, cv'),
        'clip_count': ('cnt = tl.where(live, alen + 1, 0)', 'cnt = tl.where(live, tl.minimum(alen + 1, limit - n_prev), 0)'),
        'overwrite_frozen_root': ('tok, mask=live)', 'tok)'),
    }
    for label, (old, new) in replacements.items():
        assert old in original
        env = {'tl': TL}
        extract(CAND / 'kernels/accept.py', {'accept_kernel'}, env, original.replace(old, new))
        detected = None
        for name, t, arr in cases:
            try:
                verify_case(t, copy.deepcopy(arr), env['accept_kernel'], count=False)
            except (AssertionError, IndexError, TypeError, ValueError) as e:
                detected = dict(case=name, reason=str(e)[:180])
                break
        assert detected, ('surviving mutant', label)
        STATS['mutants'][label] = detected

    for label, filename, name, old, new in (
        ('compact_wrong_source', 'kernels/compact.py', '_compact_kernel', '(base + pos + src)', '(base + pos + 1 + j)'),
        ('publisher_omits_endpoint', 'pair_cache.py', '_publish_pairs_kernel', '(step <= plen)', '(step < plen)'),
    ):
        saved = ENV[name]
        source = (CAND / filename).read_text()
        assert old in source
        extract(CAND / filename, {name}, ENV, source.replace(old, new))
        detected = None
        try:
            for case, t, arr in cases[:8]:
                try:
                    verify_case(t, copy.deepcopy(arr), downstream=True, count=False)
                except AssertionError as e:
                    detected = dict(case=case, reason=str(e)[:180])
                    break
        finally:
            ENV[name] = saved
        assert detected, ('surviving downstream mutant', label)
        STATS['mutants'][label] = detected


def mixed_batch(cases):
    # Three lanes share metadata: deeper gain, clipped retention, and frozen.
    selected = [cases[i] for i in (4, 3, 7)]
    t = selected[0][1]
    metadata = {'start', 'flat', 'par', 'masks', 'depth', 'limit'}
    arr = {key: values.copy() if key in metadata else
           np.concatenate([case[2][key] for case in selected], axis=0)
           for key, values in selected[0][2].items()}
    arr['limit'][0] = 10
    arr['nseen'][:] = [7, 8, 7]
    keys = ('blk', 'cand', 'start', 'flat', 'par', 'masks', 'depth', 'done', 'nseen',
            'pos', 'limit', 'root', 'idx', 'lens', 'tokens', 'cnt')
    for b in range(3):
        TL.pid = (b, 0, 0)
        kernel(*(Ptr(arr[k]) for k in keys), 1024, 8, 7, 16, arr['idx'].shape[1], 16)
    assert arr['lens'].tolist() == [2, 1, -1]
    assert arr['cnt'].tolist() == [3, 2, 0]
    assert arr['nseen'].tolist() == [10, 10, 7]
    assert arr['root'].tolist() == [3, 2, 777]
    assert arr['idx'][0, :2].tolist() == [4, 6] and arr['idx'][1, 0] == 1
    STATS['mixed_batch_lanes'] += 3


def main():
    np.seterr(over='ignore')
    preservation()
    cases = directed()
    for _, t, arr in cases:
        verify_case(t, copy.deepcopy(arr), downstream=True)
    mixed_batch(cases)
    rng = np.random.default_rng(870083)
    random_cases = []
    for r in (4, 8, 16, 32, 42, 64):
        t = Template.build(r, 8)
        for trial in range(160):
            blk = rng.integers(0, 4, r).tolist()
            cand = rng.integers(0, 4, r).tolist()
            if trial % 4 == 0:
                # Create a deep exact branch; duplicates elsewhere are random.
                endpoint = int(rng.choice([i for i, kids in enumerate(t.children) if not kids]))
                p = endpoint
                while p > 0:
                    blk[p] = cand[t.parent[p]]
                    p = t.parent[p]
            arr = arrays(t, blk, cand, int(rng.choice([-1, 0, 1, 2, 3, 5, 100])), trial % 11 == 0)
            cap = 20 + 2*r + int(rng.choice([1, 2, 4, 100]))
            verify_case(t, copy.deepcopy(arr), cap=cap, downstream=trial % 16 == 0)
            random_cases.append((f'random_r{r}_{trial}', t, arr))
    mutants(cases + random_cases)
    STATS['actual_ast_sha256'] = {name: hashlib.sha256((CAND/name).read_bytes()).hexdigest()
                                 for name in ('kernels/accept.py', 'kernels/compact.py', 'pair_cache.py', 'recycle.py', 'engine.py')}
    (HERE / 'CPU_RESULTS.json').write_text(json.dumps(STATS, indent=2) + '\n')
    print(json.dumps(STATS, indent=2))


if __name__ == '__main__':
    main()
