"""CPU-only actual host-code controls. No Torch/Triton import or device compile."""
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
BASE = WORK / 'candidates/krxfty_median5_on90/engine'
CAND = WORK / 'candidates/krxfty_down_s4_v2_on91/engine'
DEVICE = CAND / 'kernels/down_s4.py'
SOURCE = CAND / 'kernels/m64_down_s4_selection.py'


class Tensor:
    counter = 0
    def __init__(self, shape=(2,), values=(100., .001), variant='original', dtype='bf16', device='cuda:0'):
        Tensor.counter += 1
        self.pointer = Tensor.counter * 4096
        self.shape, self.values, self.variant = tuple(shape), list(values), variant
        self.dtype, self.device, self.is_cuda = dtype, device, True
        self.contiguous = True
    def is_contiguous(self): return self.contiguous
    def data_ptr(self): return self.pointer
    def stride(self, i): return self.shape[1] if i == 0 else 1
    def float(self): return Tensor(self.shape, self.values, self.variant, 'fp32')
    def clone(self): return Tensor(self.shape, self.values, self.variant, self.dtype)
    def item(self): return self.values[0]
    def abs(self): return Tensor(values=[abs(v) for v in self.values])
    def max(self): return Tensor(values=[max(self.values)])
    def all(self): return Tensor(values=[all(self.values)])
    def binary(self, other, fn):
        rhs = other.values if isinstance(other, Tensor) else [other] * len(self.values)
        return Tensor(values=[fn(a, b) for a, b in zip(self.values, rhs)])
    def __sub__(self, other): return self.binary(other, lambda a, b: a-b)
    def __rmul__(self, other): return self.binary(other, lambda a, b: a*b)
    def __add__(self, other): return self.binary(other, lambda a, b: a+b)
    def __le__(self, other): return self.binary(other, lambda a, b: a<=b)


class Context:
    def __enter__(self): return self
    def __exit__(self, *_): return False
    def wait_stream(self, _): pass
    def replay(self): pass
    def record(self): pass
    def elapsed_time(self, _): return 108.


def load(state):
    cuda = SimpleNamespace(get_device_capability=lambda _: state.get('cap', (9, 0)),
        is_current_stream_capturing=lambda: state.get('capturing', False),
        synchronize=lambda: None, Stream=Context, current_stream=Context,
        stream=lambda _: Context(), CUDAGraph=Context, graph=lambda _: Context(),
        Event=lambda **_: Context())
    ns = {'math': math, 'os': SimpleNamespace(environ=state.get('env', {})),
        'time': SimpleNamespace(monotonic=lambda: state.get('now', 0.)),
        'budget': SimpleNamespace(remaining=lambda: state.get('remaining', 120.)),
        '_QUARANTINED': [],
        'torch': SimpleNamespace(bfloat16='bf16', float32='fp32', cuda=cuda,
            isfinite=lambda t: Tensor(values=[math.isfinite(v) for v in t.values]),
            equal=lambda a,b: a.values == b.values,
            randn=lambda shape,**kw: Tensor(shape, dtype=kw.get('dtype', 'bf16')),
            randn_like=lambda t: Tensor(t.shape),
            empty=lambda shape,**kw: Tensor(shape, dtype=kw.get('dtype', 'bf16'), device=kw.get('device', 'cuda:0')),
            empty_like=lambda t: Tensor(t.shape))}
    state['launches'] = []
    class Kernel:
        def __init__(self, kind): self.kind = kind
        def __getitem__(self, grid):
            def call(*args, **opts):
                state['launches'].append((self.kind, grid, args, opts))
                if self.kind == 'producer':
                    if state.get('mode') == 'throw': raise RuntimeError('candidate sentinel')
                    args[2].variant = 'candidate'
                    if state.get('mode') == 'act_mutation': args[0].values = [8., 9.]
                else:
                    out = args[1]
                    out.variant = 'candidate'
                    mode = state.get('mode')
                    if state.get('reject_layer') is not None and state['layer'] != state['reject_layer']:
                        mode = None
                    out.values = ([float('nan'), .001] if mode == 'nan' else
                                  [float('inf'), .001] if mode == 'inf' else
                                  [100., 1.] if mode == 'pointwise' else [100., .001])
            return call
    ns['_down_s4_kernel'], ns['_sum_kernel'] = Kernel('producer'), Kernel('merge')
    cls = next(n for n in ast.parse(DEVICE.read_text()).body if isinstance(n, ast.ClassDef))
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(DEVICE), 'exec'), ns)
    tree = ast.parse(SOURCE.read_text())
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    exec(compile(tree, str(SOURCE), 'exec'), ns)
    def norm(residual, branch, wn, eps, xout):
        assert len({residual.pointer, branch.pointer, xout.pointer}) == 3
        assert wn is state['next_wn'] and eps == 1e-6
        state.setdefault('norm_weights', []).append(wn)
        xout.values = [a+b for a,b in zip(residual.values, branch.values)]
        out = Tensor(residual.shape, branch.values, branch.variant)
        if state.get('mode') == 'next_norm' and branch.variant == 'candidate': out.values = [100., 1.]
        if state.get('mode') == 'next_residual' and branch.variant == 'candidate': xout.values = [1000., 1.]
        if state.get('mode') == 'residual_mutation' and branch.variant == 'candidate': residual.values = [1., 2.]
        return out
    ns['add_rms_norm'] = norm
    return ns


def model_and_original(state):
    layers = [SimpleNamespace(in_norm=Tensor((2560,)), post_norm=Tensor((2560,)),
                             wgu=Tensor((19456, 2560)), wd=Tensor((2560, 9728))) for _ in range(36)]
    model = SimpleNamespace(layers=layers, final_norm=Tensor((2560,)), device='cuda:0', cfg=SimpleNamespace(eps=1e-6))
    if state.get('bad_layout'): layers[-1].wd.contiguous = False
    if state.get('bad_align'): layers[-1].wd.pointer += 2
    if state.get('bad_next_norm'): model.final_norm.shape = (128,)
    calls = []
    def gu(x, y, wn, residual, w):
        i = next(i for i,l in enumerate(layers) if l.wgu is w)
        state['layer'] = i
        state['next_wn'] = layers[i+1].in_norm if i+1 < 36 else model.final_norm
        assert wn is layers[i].post_norm
        assert len({x.pointer, y.pointer, residual.pointer}) == 3
        calls.append(i)
        residual.values = [7., 7.]
        if state.get('mode') == 'input_mutation': x.values = [8., 9.]
        if state.get('timed_mutation') and len(calls) > 36: y.values = [8., 9.]
        return Tensor((64, 9728))
    def down(act, weight):
        assert act.shape == (64, 9728) and weight is layers[state['layer']].wd
        return Tensor((64, 2560), [float('nan'), .001] if state.get('ref_nan') else [100., .001])
    return model, dict(gu=gu, d=down, qkv=object(), o=object(), lm=object()), calls


def check_launches(state):
    for kind, grid, args, opts in state['launches']:
        assert grid == (160,)
        common = dict(num_warps=4, num_stages=3, num_ctas=1, enable_fp_fusion=True)
        if kind == 'producer':
            assert args[3:] == (9728, 9728)
            assert [t.shape for t in args[:3]] == [(64,9728),(2560,9728),(4,64,2560)]
            assert args[2].dtype == 'fp32' and len({t.pointer for t in args[:3]}) == 3
            assert opts == dict(common, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, SPLIT_K=4, K_PER_SPLIT=2432)
        else:
            assert args[2:] == (163840,) and args[0].shape == (4,64,2560)
            assert args[1].shape == (64,2560) and args[1].dtype == 'bf16'
            assert args[0].pointer != args[1].pointer
            assert opts == dict(common, SPLIT_K=4, BLOCK=1024)


def policy(name, expected, **overrides):
    state = dict(remaining=120., now=0., env={}); state.update(overrides)
    ns = load(state); model, original, calls = model_and_original(state)
    if state.get('short_model'): model.layers.pop()
    timings, logs = [], []
    def timer(fn, records, owners):
        expected_records = [(l.post_norm, l.wgu, l.wd, model.layers[i+1].in_norm if i+1<36 else model.final_norm)
                            for i,l in enumerate(model.layers)]
        assert records == expected_records
        allocations = []; outputs = [fn(r, allocations) for r in records]
        assert len(allocations) == 72
        variant = outputs[0].variant; timings.append(variant)
        assert all(t.shape == (64,9728) for t in allocations[::2])
        if state.get('expire_at') == len(timings): state['remaining'] = 11.9
        if state.get('deadline_at') == len(timings): state['now'] = 8.
        if state.get('timer_error'): raise RuntimeError('timer sentinel')
        if variant == 'original': return 1.
        if 'bad_time' in state: return state['bad_time']
        idx = sum(v == 'candidate' for v in timings)-1
        return state.get('ratios', [.9,.9])[idx]
    ns['_owned_time'] = timer
    if state.get('expiry_validation'):
        real = ns['_validate']
        def validation(*args):
            result = real(*args); state['remaining'] = 11.9; return result
        ns['_validate'] = validation
    result = ns['select_m64_down_s4'](original, model, logs.append)
    chosen = 'original' if result is original else 'candidate'
    assert chosen == expected, (name, chosen, expected, logs)
    if expected == 'candidate':
        assert isinstance(result['d'], ns['DownS4Matmul'])
        assert all(result[k] is original[k] for k in original if k != 'd')
        assert timings == ['original','candidate','candidate','original']
        assert calls[:36] == list(range(36))
        assert len(state['norm_weights']) == 72
        assert state['norm_weights'][-2:] == [model.final_norm]*2
    if expected == 'original' and not (state.get('mode')=='throw' or state.get('timer_error')):
        assert not any('optional error' in log for log in logs), (name, logs)
    check_launches(state)
    return dict(name=name, passed=True, selection=chosen, timing_order=timings, main_launches=len(state['launches'])//2)


def lifetime():
    results = []
    ns = load({}); owners = []; counts = []
    samples = iter([108*9,108*1,108*3,108*2,108*8])
    class Event(Context):
        def elapsed_time(self, _): return next(samples)
    ns['torch'].cuda.Event = lambda **_: Event()
    def fn(record, allocations):
        allocations.extend((Tensor((64,9728)), Tensor((64,2560))))
        return allocations[-1]
    def sync():
        assert len(owners) == 1
        owner = owners[0]; counts.append(len(owner['allocations']))
        assert 'stream' in owner and 'fn' in owner and 'records' in owner
        if len(owner['allocations']) == 78: assert 'graph' in owner and 'capture' in owner
    ns['torch'].cuda.synchronize = sync
    assert ns['_owned_time'](fn, list(range(36)), owners) == 3.
    assert not owners and counts == [6,78,78,78,78,78,78]
    results.append(dict(name='actual median5 timer retains 78 gu/down allocations and graph through all seven drains', passed=True))
    for phase in ('inner','final'):
        ns = load({}); marker = object()
        def select(original, model, owners, log):
            owners.append(marker)
            if phase == 'inner': raise RuntimeError('work sentinel')
            return original
        ns['_select_m64_down_s4'] = select
        ns['torch'].cuda.synchronize = lambda: (_ for _ in ()).throw(RuntimeError('drain sentinel'))
        try: ns['select_m64_down_s4']({}, object())
        except BaseException as exc: assert isinstance(exc, ns['DownS4DrainFailed']) and not isinstance(exc, Exception)
        else: raise AssertionError('failed drain swallowed')
        assert marker in ns['_QUARANTINED'][0][0]
        try: ns['select_m64_down_s4']({}, object())
        except ns['DownS4DrainFailed']: pass
        else: raise AssertionError('quarantined state reused')
        results.append(dict(name=phase+' failed drain quarantines owners and prevents ordinary fallback/reuse', passed=True))
    ns = load({'capturing':True})
    ns['torch'].cuda.synchronize = lambda: (_ for _ in ()).throw(AssertionError('capture drain'))
    original = {}; assert ns['select_m64_down_s4'](original, object()) is original
    results.append(dict(name='capture no-op allocates nothing and never drains', passed=True))
    ns = load({}); model, original, _ = model_and_original({}); created = []
    def alloc(shape, **kw):
        out = Tensor(shape); created.append(out); return out
    ns['torch'].randn = alloc
    ns['torch'].randn_like = lambda _: (_ for _ in ()).throw(RuntimeError('allocation sentinel'))
    caught = []
    def sync_partial():
        import inspect
        outer = inspect.currentframe().f_back.f_locals
        assert any(isinstance(o,dict) and o.get('x') is created[0] for o in outer['owners'])
        caught.append(True)
    ns['torch'].cuda.synchronize = sync_partial
    assert ns['select_m64_down_s4'](original, model) is original and caught
    results.append(dict(name='partial private allocation error retains first allocation until successful drain', passed=True))
    # Real timer interrupted after its graph was created: its owner stays live.
    ns = load({}); owners = []; drain_count = [0]
    def fail_timer():
        drain_count[0] += 1
        if drain_count[0] == 3: raise RuntimeError('timed drain sentinel')
    ns['torch'].cuda.synchronize = fail_timer
    try: ns['_owned_time'](fn, list(range(36)), owners)
    except RuntimeError: pass
    else: raise AssertionError('timer failure swallowed')
    assert len(owners) == 1 and len(owners[0]['allocations']) == 78 and 'graph' in owners[0]
    results.append(dict(name='interrupted actual timer keeps graph and all outputs available for outer quarantine', passed=True))
    return results


def wrapper():
    state = {}; ns = load(state); op = ns['DownS4Matmul']('cuda:0')
    a, w = Tensor((64,9728)), Tensor((2560,9728)); outputs = []
    c1 = op(a,w,allocation_owners=outputs); c2 = op(a,w,allocation_owners=outputs)
    assert outputs == [c1,c2] and c1.pointer != c2.pointer
    assert all(args[2] is op.part for kind,_,args,_ in state['launches'] if kind == 'producer')
    check_launches(state)
    rejected = 0
    for change in ('shape','dtype','layout','alignment','device','cpu'):
        bad = Tensor((64,9728))
        if change == 'shape': bad.shape = (63,9728)
        elif change == 'dtype': bad.dtype = 'fp32'
        elif change == 'layout': bad.contiguous = False
        elif change == 'alignment': bad.pointer += 2
        elif change == 'device': bad.device = 'cuda:1'
        else: bad.is_cuda = False
        before = len(state['launches'])
        try: op(bad,w)
        except ValueError: rejected += 1
        else: raise AssertionError(change+' guard absent')
        assert len(state['launches']) == before
    assert rejected == 6
    return dict(name='actual wrapper exact two-launch ABI, FP32 owned scratch, fresh BF16 outputs and six off-shape/layout guards', passed=True)


def geometry():
    for f in BASE.rglob('*.py'):
        rel = f.relative_to(BASE)
        if rel != Path('model.py'): assert f.read_bytes() == (CAND/rel).read_bytes(), rel
    base = ast.parse((BASE/'model.py').read_text()); new = ast.parse((CAND/'model.py').read_text())
    verify = next(n for n in new.body if isinstance(n,ast.ClassDef) and n.name=='VerifyPlan')
    init = next(n for n in verify.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    hook = init.body[-1]
    assert ast.unparse(hook.test) == 'tree and recycler is not None and (B * R == 64)'
    assert ast.unparse(hook.body[-1]) == 'self.mm = select_m64_down_s4(self.mm, m, log)'
    assert ast.unparse(hook.body[-3]) == 'self.mm = select_m64_gateup_n32(self.mm, m, log)'
    hook.body = hook.body[:-2]
    assert ast.dump(new) == ast.dump(base)
    assert sorted(f.relative_to(CAND).as_posix() for f in CAND.rglob('*.py') if not (BASE/f.relative_to(CAND)).exists()) == ['kernels/down_s4.py','kernels/m64_down_s4_selection.py']
    kernel = next(n for n in ast.parse(DEVICE.read_text()).body if isinstance(n,ast.FunctionDef))
    def structural(fn):
        loops = [n for n in ast.walk(fn) if isinstance(n,ast.For)]
        assert len(loops)==1 and ast.unparse(loops[0].iter)=='range(0, K_PER_SPLIT, BLOCK_K)'
        calls = [ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n,ast.Call)]
        assert calls.count('tl.program_id')==1 and calls.count('tl.dot')==1
        assert calls.count('tl.load')==2 and calls.count('tl.store')==1
        assert not [n for n in ast.walk(fn) if isinstance(n,(ast.If,ast.While))]
        assert not any(s.startswith('tl.atomic') or s=='tl.num_programs' for s in calls)
        assert not [n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='to']
        store = next(n for n in ast.walk(fn) if isinstance(n,ast.Call) and ast.unparse(n.func)=='tl.store')
        assert ast.unparse(store.args[0])=='part_ptr + (split * 64 + rm[:, None]) * 2560 + rn[None, :]'
        assert ast.unparse(store.args[1])=='acc'
        return store
    structural(kernel)
    static_asserts = [n for n in ast.walk(kernel) if isinstance(n, ast.Call)
                      and ast.unparse(n.func) == 'tl.static_assert']
    assert [ast.unparse(n.args[0]) for n in static_asserts] == [
        'BLOCK_M == 64', 'BLOCK_N == 64', 'BLOCK_K == 64',
        'SPLIT_K == 4', 'K_PER_SPLIT == 2432']
    assert not any(isinstance(n, ast.BoolOp) for call in static_asserts for n in ast.walk(call))
    for mutation in ('outer_loop','bf16_partial','wrong_plane'):
        broken = copy.deepcopy(kernel)
        if mutation=='outer_loop': broken.body.append(copy.deepcopy(next(n for n in ast.walk(broken) if isinstance(n,ast.For))))
        else:
            store = next(n for n in ast.walk(broken) if isinstance(n,ast.Call) and ast.unparse(n.func)=='tl.store')
            if mutation=='bf16_partial': store.args[1]=ast.parse('acc.to(tl.bfloat16)',mode='eval').body
            else: store.args[0]=ast.parse('part_ptr + rm[:, None] * 2560 + rn[None, :]',mode='eval').body
        try: structural(broken)
        except AssertionError: pass
        else: raise AssertionError(mutation+' escaped')
    # Interpret scalarized address expressions from the actual AST. Each rm/rn
    # pair stands for one element of the vector store; this checks addresses,
    # not GPU arithmetic or compiler memory ordering.
    assigns = {n.targets[0].id:n.value for n in ast.walk(kernel) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    split_expr = compile(ast.Expression(assigns['split']), '<split>', 'eval')
    covered = set(); intervals = []
    for pid in range(160):
        split = eval(split_expr,{},dict(tile=pid,SPLIT_K=4))
        col0 = (pid//4)*64
        interval = [split*2432+k0+r for k0 in range(0,2432,64) for r in range(64)]
        assert interval==list(range(split*2432,(split+1)*2432))
        if pid < 4: intervals.extend(interval)
        for row in range(64):
            for col in range(col0,col0+64):
                address=(split*64+row)*2560+col
                assert address not in covered
                covered.add(address)
    assert intervals==list(range(9728)) and covered==set(range(4*64*2560))
    assert intervals[:-64]!=list(range(9728)) and intervals+[0]!=list(range(9728))
    assert covered-{0}!=set(range(4*64*2560))
    return dict(name='exact base parity and late M64 hook; 655360 unique FP32 stores, exact K coverage and structural negative controls',passed=True)


def source_address_controls():
    """Evaluate address expressions extracted from the actual kernel source."""
    kernel = next(n for n in ast.parse(DEVICE.read_text()).body if isinstance(n,ast.FunctionDef))
    def verify(fn):
        assignments = {n.targets[0].id:n.value for n in ast.walk(fn)
                       if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
        assert ast.unparse(assignments['rm']) == 'tl.arange(0, BLOCK_M)'
        assert ast.unparse(assignments['rk']) == 'tl.arange(0, BLOCK_K)'
        loop = next(n for n in ast.walk(fn) if isinstance(n,ast.For))
        store = next(n for n in ast.walk(fn) if isinstance(n,ast.Call) and ast.unparse(n.func)=='tl.store')
        class Scalarize(ast.NodeTransformer):
            def visit_Subscript(self, node): return self.visit(node.value)
        address = ast.fix_missing_locations(Scalarize().visit(copy.deepcopy(store.args[0])))
        programs = {name:compile(ast.Expression(assignments[name]), '<kernel-address>', 'eval')
                    for name in ('split','rn','kk')}
        programs['address'] = compile(ast.Expression(address), '<kernel-address>', 'eval')
        programs['kloop'] = compile(ast.Expression(loop.iter), '<kernel-address>', 'eval')
        cfg = dict(BLOCK_M=64,BLOCK_N=64,BLOCK_K=64,SPLIT_K=4,K_PER_SPLIT=2432,part_ptr=0)
        seen = bytearray(4*64*2560)
        expected_plane_k = [set() for _ in range(4)]
        first_last = []
        for tile in range(160):
            env = dict(cfg,tile=tile)
            split = eval(programs['split'],{},env); env['split'] = split
            assert split in range(4)
            ks = set()
            for k0 in eval(programs['kloop'],{},env):
                for rk in range(64):
                    env.update(k0=k0,rk=rk)
                    kk = eval(programs['kk'],{},env)
                    assert 0 <= kk < 9728 and kk not in ks
                    ks.add(kk)
            assert len(ks)==2432
            if tile < 4: expected_plane_k[split] = ks
            assert ks == expected_plane_k[split]
            for offset in range(64):
                env['tl'] = SimpleNamespace(arange=lambda _start,_end:offset)
                rn = eval(programs['rn'],{},env); env['rn'] = rn
                assert 0 <= rn < 2560
                for row in range(64):
                    env['rm'] = row
                    address_value = eval(programs['address'],{},env)
                    assert 0 <= address_value < len(seen) and seen[address_value]==0
                    seen[address_value]=1
            first_last.append((min(ks),max(ks)))
        assert all(seen)
        union = set()
        for plane in expected_plane_k:
            assert not (union & plane)
            union.update(plane)
        assert union == set(range(9728))
        return dict(partial_addresses=len(seen),k_intervals=first_last[:4])
    result = verify(kernel)
    mutations = {'missing_plane':('split','tile % 3'),
                 'repeated_channels':('rn','tl.arange(0, BLOCK_N)'),
                 'overlapping_K':('kk','split * (K_PER_SPLIT - 64) + k0 + rk'),
                 'K_gap_or_oob':('kk','split * (K_PER_SPLIT + 64) + k0 + rk')}
    rejected = []
    for label,(name,expression) in mutations.items():
        altered = copy.deepcopy(kernel)
        target = next(n for n in ast.walk(altered) if isinstance(n,ast.Assign)
                      and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name)
        target.value=ast.parse(expression,mode='eval').body
        try: verify(altered)
        except AssertionError: rejected.append(label)
        else: raise AssertionError(label+' source mutation escaped')
    assert len(rejected)==4
    return dict(name='actual AST address interpretation catches four source-level split/channel/K mutants',passed=True,negative_controls=rejected,**result)


def main():
    checks = [geometry(),wrapper(),source_address_controls()]
    cases = [
      ('candidate wins both orders','candidate',{}),
      ('one order misses threshold','original',{'ratios':[.9,.99]}),
      ('exact three percent is insufficient','original',{'ratios':[.97,.97]}),
      ('both orders slower','original',{'ratios':[1.1,1.1]}),
      ('finite pointwise near-zero check','original',{'mode':'pointwise'}),
      ('last layer validated','original',{'mode':'pointwise','reject_layer':35}),
      ('candidate NaN','original',{'mode':'nan'}),('candidate infinity','original',{'mode':'inf'}),
      ('reference NaN','original',{'ref_nan':True}),
      ('next norm output mismatch','original',{'mode':'next_norm'}),
      ('next residual mismatch','original',{'mode':'next_residual'}),
      ('common gu activation mutation','original',{'mode':'act_mutation'}),
      ('common residual mutation','original',{'mode':'residual_mutation'}),
      ('validation input mutation','original',{'mode':'input_mutation'}),
      ('timed input mutation','original',{'timed_mutation':True}),
      ('candidate error drained fallback','original',{'mode':'throw'}),
      ('timer error drained fallback','original',{'timer_error':True}),
      ('nonfinite timer rejected','original',{'bad_time':float('nan')}),
      ('zero timer rejected','original',{'bad_time':0.}),
      ('negative timer rejected','original',{'bad_time':-1.}),
      ('entry below20 seconds','original',{'remaining':19.9}),
      ('entry NaN budget','original',{'remaining':float('nan')}),
      ('forced cuBLAS','original',{'env':{'ENGINE_FORCE_CUBLAS':'1'}}),
      ('norm fusion optout','original',{'env':{'ENGINE_NORM_FUSED':'0'}}),
      ('bad weight alignment','original',{'bad_align':True}),
      ('bad weight layout','original',{'bad_layout':True}),
      ('wrong final next norm','original',{'bad_next_norm':True}),
      ('unsupported device','original',{'cap':(8,0)}),
      ('incomplete layers','original',{'short_model':True}),
      ('expires during validation','original',{'expiry_validation':True}),
      ('reserve expires after first timing','original',{'expire_at':1}),
      ('reserve expires after final timing','original',{'expire_at':4}),
      ('deadline expires after final timing','original',{'deadline_at':4})]
    checks.extend(policy(name, expected, **kw) for name,expected,kw in cases)
    checks.extend(lifetime())
    for f in CAND.rglob('*.py'): ast.parse(f.read_text())
    result = dict(checks=checks,count=len(checks),all_passed=True,gpu_execution=False,triton_compilation=False,
                  limits='Actual host code under CPU stand-ins; no device arithmetic, compiler ordering, runtime correctness or speed claim.')
    (HERE/'HOST_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}))


if __name__=='__main__': main()
