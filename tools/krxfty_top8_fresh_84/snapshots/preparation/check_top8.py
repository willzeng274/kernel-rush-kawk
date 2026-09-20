"""Actual source-body checks on CPU; no Triton compiler or GPU execution."""
import ast
import contextlib
import hashlib
import heapq
import json
import math
from pathlib import Path
import struct
import sys
import types

import torch

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parents[1] / 'candidates/krxfty_top8/engine'
BASE = HERE.parents[1] / 'candidates/krxfty_prefix_splitloop_guarded/engine'
SOURCE = ENGINE / 'kernels/top8.py'
torch.set_num_threads(1)


class Pointer:
    def __init__(self, data, offset=0, name=''):
        self.data, self.offset, self.name = data.reshape(-1), offset, name
    def __add__(self, offset):
        return Pointer(self.data, self.offset + offset, self.name)


class TL:
    pids = (0, 0)
    int16, int32 = torch.int16, torch.int32
    writes = {}
    program_id = staticmethod(lambda axis: torch.tensor(TL.pids[axis], dtype=torch.int32))
    arange = staticmethod(lambda a,b: torch.arange(a,b,dtype=torch.int32))
    static_range = staticmethod(range)
    where = staticmethod(lambda c,a,b: torch.where(torch.as_tensor(c), torch.as_tensor(a), torch.as_tensor(b)))
    @staticmethod
    def reduce(pair, axis, combine):
        a,b = torch.broadcast_tensors(*pair)
        assert axis == 0
        while a.numel() > 1:
            a,b = combine(a[::2],b[::2],a[1::2],b[1::2])
        return a[0],b[0]
    @staticmethod
    def load(p, mask=True, other=0):
        offset, mask = torch.broadcast_tensors(torch.as_tensor(p.offset).long(), torch.as_tensor(mask))
        selected = offset[mask]
        assert bool(((selected >= 0) & (selected < p.data.numel())).all()), 'OOB load'
        out = torch.full(offset.shape, other, dtype=p.data.dtype)
        out[mask] = p.data[selected]
        return out
    @staticmethod
    def store(p, values, mask=True):
        offset, values, mask = torch.broadcast_tensors(torch.as_tensor(p.offset).long(), torch.as_tensor(values), torch.as_tensor(mask))
        selected = offset[mask]
        assert bool(((selected >= 0) & (selected < p.data.numel())).all()), 'OOB store'
        counts = TL.writes.setdefault(p.name, torch.zeros(p.data.numel(), dtype=torch.int32))
        counts.index_add_(0,selected,torch.ones_like(selected,dtype=torch.int32))
        p.data[selected] = values[mask].to(p.data.dtype)


class Bitcast(ast.NodeTransformer):
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and node.func.attr=='to' and node.keywords:
            assert len(node.keywords)==1 and node.keywords[0].arg=='bitcast' and node.keywords[0].value.value is True
            return ast.copy_location(ast.Call(func=ast.Name(id='_bitcast',ctx=ast.Load()),args=[node.func.value,node.args[0]],keywords=[]),node)
        return node


def bodies(text=None):
    tree=ast.parse(SOURCE.read_text() if text is None else text)
    nodes=[]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in ('_bf16_key','_best_pair','top8_partials','top8_merge'):
            node.decorator_list=[]
            for arg in node.args.args: arg.annotation=None
            nodes.append(node)
    tree=Bitcast().visit(ast.Module(body=nodes,type_ignores=[]))
    ns={'tl':TL,'_bitcast':lambda x,dtype: x.view(dtype)}
    exec(compile(ast.fix_missing_locations(tree),str(SOURCE),'exec'),ns)
    return ns


def order_key(item):
    idx, val = item
    if math.isnan(val): return (2,0.0,0,-idx)
    # Numeric ordering plus +0 above -0, independently of the bit-key formula.
    return (1,val,int(val==0 and math.copysign(1,val)>0),-idx)


def oracle(logits):
    return torch.tensor([[item[0] for item in heapq.nlargest(8,enumerate(row),key=order_key)]
                         for row in logits.float().tolist()],dtype=torch.int32)


def run_source(logits, text=None, buffers=None):
    ns=bodies(text)
    n,v=logits.shape
    assert v==151936
    raw=logits.view(torch.int16).clone()
    greedy=logits.argmax(-1).clone()
    if buffers is None:
        buffers=(torch.full((n,75,8),-991,dtype=torch.int32),torch.full((n,75,8),-991,dtype=torch.int32),torch.full((n,8),-991,dtype=torch.int32))
    keys,ids,out=buffers
    TL.writes={}
    for row in range(n):
        for part in range(75):
            TL.pids=(row,part)
            ns['top8_partials'](Pointer(logits),Pointer(keys,name='keys'),Pointer(ids,name='ids'),v,75,2048)
        TL.pids=(row,0)
        ns['top8_merge'](Pointer(keys),Pointer(ids),Pointer(out,name='out'),75,1024)
    assert torch.equal(out,oracle(logits)), 'rank/ID oracle mismatch'
    assert torch.equal(raw,logits.view(torch.int16)), 'input bits changed'
    assert torch.equal(greedy,logits.argmax(-1)), 'greedy changed'
    assert set(TL.writes)=={'keys','ids','out'}
    for counts in TL.writes.values(): assert bool((counts==1).all()), 'unwritten/duplicate output'
    return buffers


def kernel_checks():
    torch.manual_seed(8319)
    bits=torch.arange(65536,dtype=torch.int32).to(torch.int16)
    values=bits.view(torch.bfloat16)
    keys=bodies()['_bf16_key'](values)
    # All encodings are checked, not merely the eight maxima of one exhaustive row.
    expected=sorted(enumerate(values.float().tolist()),key=order_key,reverse=True)
    actual=torch.argsort(keys,descending=True,stable=True).tolist()
    assert actual==[x[0] for x in expected], 'all-encoding order mismatch'
    assert keys.min().item()>0 and keys.max().item()==65536
    cases=[('randomN64',torch.randn(64,151936).bfloat16()),
           ('randomN192',torch.randn(192,151936).bfloat16()),
           ('dense-ties',torch.randint(-2,3,(2,151936)).bfloat16()),
           ('negative-infinity',torch.full((1,151936),-float('inf'),dtype=torch.bfloat16)),
           ('positive-infinity',torch.full((1,151936),float('inf'),dtype=torch.bfloat16)),
           ('nan',torch.full((1,151936),float('nan'),dtype=torch.bfloat16)),
           ('all-encodings',values.repeat(3)[:151936].reshape(1,-1)),
           ('signed-zero',torch.zeros((1,151936),dtype=torch.bfloat16))]
    cases[-1][1][:,::2]=-0.0
    edges=torch.full((3,151936),-float('inf'),dtype=torch.bfloat16)
    edge_ids=((0,2047,2048,4095,4096,151551,151552,151935),
              (151935,151934,151933,151932,151931,151930,151929,151928),
              (151552,151553,151554,151555,151556,151557,151558,151559))
    for r,ids in enumerate(edge_ids):
        for rank,idx in enumerate(ids): edges[r,idx]=16-rank
    cases.append(('tails',edges))
    for name,x in cases:
        run_source(x)
        print('PASS kernel',name,flush=True)
    buffers=run_source(edges)
    run_source(torch.flip(edges,[1]),buffers=buffers)
    text=SOURCE.read_text()
    mutants={
      'tail-load':text.replace('token < V, other=0','True, other=0'),
      'tail-live':text.replace('tl.where(token < V, _bf16_key(value), 0)','tl.where(token < V - 384, _bf16_key(value), 0)'),
      'tie-direction':text.replace('(id_a < id_b)','(id_a > id_b)'),
      'duplicate-winner':text.replace('key = tl.where(token == best_id, 0, key)','key = key'),
      'local-id':text.replace('tl.store(PartialIds + off, best_id)','tl.store(PartialIds + off, best_id % BLOCK)'),
      'merge-padding':text.replace('slot < PARTS * 8, other=0','True, other=0'),
      'nan-key':text.replace('65535, bits ^ mask','1, bits ^ mask'),
    }
    detected=[]
    for name, mutant in mutants.items():
        assert mutant!=text
        x=cases[2][1][:1] if name=='tie-direction' else (cases[6][1] if name=='nan-key' else edges[:1])
        try: run_source(x,mutant)
        except AssertionError: detected.append(name)
        else: raise AssertionError('undetected mutation '+name)
    return dict(kernel_cases=len(cases)+2,kernel_rows=sum(len(x) for _,x in cases)+6,
                all_bf16_encodings=65536,detected_mutations=detected)


class FakeBudget:
    seconds=100.0
    @classmethod
    def remaining(cls): return cls.seconds


class FakeCuda:
    capturing=False
    fail_sync=False
    durations=[]
    graphs=[]
    is_current_stream_capturing=classmethod(lambda cls: cls.capturing)
    @classmethod
    def synchronize(cls):
        if cls.fail_sync: raise RuntimeError('injected drain failure')
    @classmethod
    @contextlib.contextmanager
    def graph(cls,g):
        cls.capturing=True
        try: yield
        finally: cls.capturing=False
    class CUDAGraph:
        def __init__(self): self.replays=0; FakeCuda.graphs.append(self)
        def replay(self): self.replays+=1
    class Event:
        def __init__(self,enable_timing): pass
        def record(self): pass
        def synchronize(self): FakeCuda.synchronize()
        def elapsed_time(self,end): return FakeCuda.durations.pop(0)*8


def host_namespace():
    tree=ast.parse(SOURCE.read_text())
    nodes=[x for x in tree.body if isinstance(x,ast.ClassDef)]
    fake_torch=types.SimpleNamespace(**{n:getattr(torch,n) for n in ('empty','empty_like','int16','int32','topk','isfinite','equal')},cuda=FakeCuda)
    ns={'torch':fake_torch,'time':types.SimpleNamespace(monotonic=lambda:0.0),'budget':FakeBudget,'sys':sys,'_FAILED_TRIALS':[]}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(SOURCE),'exec'),ns)
    return ns


def policy_checks():
    ns=host_namespace(); cls=ns['VerifierTop8']
    logits=torch.randn(2,151936).bfloat16(); tokens=torch.tensor([5,5]); ref=oracle(logits)
    count=0
    def fresh(durations=(10,8,8,10)):
        FakeCuda.capturing=False;FakeCuda.fail_sync=False;FakeCuda.durations=list(durations);FakeCuda.graphs=[];FakeBudget.seconds=100
        item=cls(151936,2,'cpu'); item._top=lambda x: ref
        return item,torch.full((151936,8),-1,dtype=torch.int32)
    for durations,expected in [((10,8,8,10),True),((10,9.5,9.5,10),False),((10,9,9,8),False),((10,11,11,10),False)]:
        item,table=fresh(durations);item.update(table,tokens,logits)
        assert item.chosen and item.enabled==expected and not item.trial_graphs
        assert item.trial_inputs is None and not item.failed
        assert len(FakeCuda.graphs)==2 and [g.replays for g in FakeCuda.graphs]==[5,5]
        previous=len(FakeCuda.graphs);item.update(table,tokens,logits)
        assert len(FakeCuda.graphs)==previous and not FakeCuda.durations
        count+=1
    for kind in ('budget','capturing','invalid','duplicate','wrong-rank','nonfinite','deadline','drain'):
        item,table=fresh()
        x=logits.clone()
        if kind=='budget':FakeBudget.seconds=41
        if kind=='capturing':FakeCuda.capturing=True
        if kind=='invalid':item._top=lambda x: torch.full((2,8),151936,dtype=torch.int32)
        if kind=='duplicate':item._top=lambda x: torch.zeros((2,8),dtype=torch.int32)
        if kind=='wrong-rank':item._top=lambda x: torch.arange(8,dtype=torch.int32).expand(2,-1)
        if kind=='nonfinite':x[0,0]=float('nan')
        if kind=='deadline':
            def top(x): FakeBudget.seconds=29;return ref
            item._top=top
        if kind=='drain':
            original_check=item._check_deadline
            calls=[0]
            def check(deadline):
                calls[0]+=1
                if len(item.trial_graphs)==1:
                    FakeCuda.fail_sync=True
                    raise RuntimeError('injected after capture')
                original_check(deadline)
            item._check_deadline=check
            try:item.update(table,tokens,x)
            except RuntimeError:pass
            else:raise AssertionError('failed drain hidden')
            assert item.trial_graphs, 'failed drain released graph owners'
            assert item.failed and not item.enabled
            assert item in ns['_FAILED_TRIALS'], 'failed parent abandonment loses ownership'
            assert item.trial_inputs[0] is table and item.trial_inputs[2] is x
            FakeCuda.fail_sync=False
            try:item.update(table,tokens,x)
            except RuntimeError:pass
            else:raise AssertionError('failed owner reused')
        else:
            item.update(table,tokens,x)
            assert item.chosen and not item.enabled and not item.trial_graphs
            assert torch.equal(table[5],torch.topk(x,8,dim=-1).indices[-1].int())
        count+=1
    for stop_at in range(1,20):
        item,table=fresh()
        checks=[0]
        original_check=item._check_deadline
        def check(deadline):
            checks[0]+=1
            if checks[0]==stop_at: raise ns['_PickerDeadline']()
            original_check(deadline)
        item._check_deadline=check
        item.update(table,tokens,logits)
        assert checks[0]==stop_at and not item.enabled and not item.failed
        assert not item.trial_graphs and item.trial_inputs is None
        count+=1
    for kind in ('input-mutation','compile-error'):
        item,table=fresh();x=logits.clone()
        def bad_top(x):
            if kind=='compile-error': raise RuntimeError('injected compile error')
            x[0,0]=x[0,0]+1
            return ref
        item._top=bad_top
        try:item.update(table,tokens,x)
        except RuntimeError:pass
        else:raise AssertionError('unsafe failure hidden')
        assert item.failed and not item.enabled and not item.trial_graphs and item.trial_inputs is None
        count+=1
    return dict(policy_cases=count)


def get_method(path,clsname,method):
    tree=ast.parse(path.read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==clsname)
    node=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name==method)
    node.decorator_list=[]
    for arg in node.args.args:arg.annotation=None
    node.returns=None
    return node


def integration_checks():
    recycle=ENGINE/'recycle.py';model=ENGINE/'model.py'
    # Exact source preservation of prompt path, table lifecycle and greedy return.
    assert ast.dump(get_method(recycle,'Recycler','update'))==ast.dump(get_method(BASE/'recycle.py','Recycler','update'))
    assert ast.dump(get_method(model,'Plan','prefill'))==ast.dump(get_method(BASE/'model.py','Plan','prefill'))
    assert ast.dump(get_method(model,'Plan','_warm_table'))==ast.dump(get_method(BASE/'model.py','Plan','_warm_table'))
    init=get_method(recycle,'Recycler','__init__');base_init=get_method(BASE/'recycle.py','Recycler','__init__')
    assert ast.dump(ast.Module(body=init.body[:-1],type_ignores=[]))==ast.dump(ast.Module(body=base_init.body,type_ignores=[]))
    assert ast.unparse(init.body[-1].value.test)=='k == 8 and vocab == 151936'
    current=get_method(model,'VerifyPlan','verify');base=get_method(BASE/'model.py','VerifyPlan','verify')
    assert ast.dump(current.body[-1])==ast.dump(base.body[-1])
    modified=ast.parse(ast.unparse(current))
    for node in ast.walk(modified):
        if isinstance(node,ast.Attribute) and node.attr=='update_verifier':node.attr='update'
    assert ast.dump(modified.body[0])==ast.dump(base)
    methods=[get_method(recycle,'Recycler',n) for n in ('update','update_verifier')]
    ns={'torch':torch};exec(compile(ast.fix_missing_locations(ast.Module(body=methods,type_ignores=[])),str(recycle),'exec'),ns)
    logits=torch.randn(3,151936).bfloat16();tokens=torch.tensor([4,6,4]);table=torch.full((151936,8),-1,dtype=torch.int32)
    calls=[]
    rec=types.SimpleNamespace(k=8,table=table,verifier_top8=types.SimpleNamespace(update=lambda *a:calls.append(a)))
    rec.update=types.MethodType(ns['update'],rec);rec.update_verifier=types.MethodType(ns['update_verifier'],rec)
    rec.update(tokens,logits);assert not calls
    assert torch.equal(table[4],torch.topk(logits,8).indices[-1].int())
    rec.update_verifier(tokens,logits);assert len(calls)==1 and calls[0][0] is table and calls[0][2] is logits
    rec.verifier_top8=None;rec.update_verifier(tokens,logits)
    # Execute actual verifier function; layer producer is injected, not simulated model evidence.
    B,R=1,3
    plan=types.SimpleNamespace(B=B,model=types.SimpleNamespace(embed=torch.zeros((151936,1),dtype=torch.bfloat16)),rope_fused=False)
    rec.depth=None
    verify=types.SimpleNamespace(plan=plan,R=R,blk=tokens.view(B,R),mm=None,q=None,attn_out=None,attention=None,pos=None,tree=False,recycler=rec)
    env={'torch':torch,'F':torch.nn.functional,'run_layers':lambda *a,**kw: logits}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[current],type_ignores=[])),str(model),'exec'),env)
    before=logits.view(torch.int16).clone()
    for k in (1,2,4,8):
        rec.k=k;rec.table=torch.full((151936,k),-1,dtype=torch.int32)
        got=env['verify'](verify)
        assert torch.equal(got,logits.argmax(-1).view(B,R))
        assert torch.equal(before,logits.view(torch.int16))
    helper=host_namespace()['VerifierTop8'](151936,3,'cpu')
    helper.chosen=True;helper.enabled=True
    source_calls=[]
    def actual_top(x):
        source_calls.append(x)
        return run_source(x)[2]
    helper._top=actual_top
    rec.k=8;rec.verifier_top8=helper
    rec.update(tokens,logits)  # Same row count still stays on the prefill path.
    assert not source_calls
    got=env['verify'](verify)
    assert source_calls==[logits] and torch.equal(got,logits.argmax(-1).view(B,R))
    assert torch.equal(rec.table[4],oracle(logits)[-1])
    assert torch.equal(before,logits.view(torch.int16))
    # The real launch wrapper is checked against the prepared compile signature.
    launches=[]
    class Launch:
        def __init__(self,name):self.name=name
        def __getitem__(self,grid):
            return lambda *args,**kw:launches.append((self.name,grid,args,kw))
    host=host_namespace()
    host['top8_partials']=Launch('top8_partials');host['top8_merge']=Launch('top8_merge')
    for n in (1,64,128,192):
        obj=host['VerifierTop8'](151936,n,'cpu');token=object()
        assert obj._top(token) is obj.out
        partial,merge=launches[-2:]
        assert partial[1]==(n,75) and merge[1]==(n,)
        assert partial[2][0] is token and partial[2][1] is obj.keys and partial[2][2] is obj.ids
        assert partial[2][3:]==(151936,75,2048) and merge[2][3:]==(75,1024)
        assert merge[2][0] is obj.keys and merge[2][1] is obj.ids and merge[2][2] is obj.out
        assert partial[3]==merge[3]==dict(num_warps=4,num_stages=3,enable_fp_fusion=False)
    return dict(integration_cases=15,greedy_method_preserved=True,prefill_methods_preserved=True)


if __name__=='__main__':
    result={'status':'PASS','python':sys.version,'executable':sys.executable,'torch':torch.__version__,'execution':'CPU actual source-body emulation; no Triton compilation/GPU',
            'kernel_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            **kernel_checks(),**policy_checks(),**integration_checks()}
    (HERE/'CPU_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
