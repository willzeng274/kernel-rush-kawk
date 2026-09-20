"""Pure CPU policy/lifetime/launch checks. No device arithmetic or CUDA import."""
import contextlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[3]
CAND = ROOT / 'work/candidates/krxfty_wide64/engine'
HERE = Path(__file__).resolve().parent
CHECKS = []

class Tensor:
    next_ptr = 0x100000
    def __init__(self, shape, values=None, dtype='bf16', name=''):
        self.shape, self.values, self.dtype, self.name = tuple(shape), list(values or [1.,2.,3.,4.]), dtype, name
        self.is_cuda, self.device = True, 'cuda:0'
        self.ptr = Tensor.next_ptr; Tensor.next_ptr += 16
        self.norm_name = None
    def data_ptr(self): return self.ptr
    def is_contiguous(self): return True
    def stride(self, i): return self.shape[1] if i == 0 else 1
    def float(self): return self.copy(dtype='fp32')
    def clone(self): return self.copy()
    def copy(self, dtype=None):
        r=Tensor(self.shape,self.values,dtype or self.dtype,self.name);r.norm_name=self.norm_name;return r
    def __sub__(self, other): return Tensor(self.shape,[a-b for a,b in zip(self.values,other.values)],'fp32')
    def __mul__(self, other): return Tensor(self.shape,[a*other for a in self.values],'fp32')
    __rmul__ = __mul__
    def __add__(self, other): return Tensor(self.shape,[a+other for a in self.values],'fp32')
    def __le__(self, other): return Tensor(self.shape,[a<=b for a,b in zip(self.values,other.values)],'bool')
    def abs(self): return Tensor(self.shape,[abs(v) for v in self.values],'fp32')
    def max(self): return Scalar(max(self.values))
    def all(self): return Scalar(all(self.values))
    def argmax(self, dim): return Tensor((64,),[max(range(4),key=self.values.__getitem__)],'i64')
class Scalar:
    def __init__(self,v):self.v=v
    def item(self):return self.v

class State:
    def __init__(self):
        self.now=0.;self.remaining=120.;self.capture=False;self.syncs=0;self.fail_sync=None
        self.candidate_calls=[];self.norm_calls=[];self.timing_calls=[];self.bad=None
        self.last_kind='original';self.last_op=None;self.time_step=0.;self.ratios={};self.streams=[];self.graphs=[]
    def sync(self):
        self.syncs+=1
        if self.fail_sync and self.fail_sync(self.syncs):raise RuntimeError('injected CUDA drain failure')

class Stream:
    def __init__(self,state):self.state=state;state.streams.append(self)
    def wait_stream(self,other):pass
class Graph:
    def __init__(self,state):self.state=state;self.replays=0;state.graphs.append(self)
    def replay(self):self.replays+=1
class Event:
    def record(self):pass
    def elapsed_time(self,other):return 108.


def install(state, actual_gemm=False):
    torch=types.ModuleType('torch');torch.bfloat16='bf16';torch.float32='fp32';torch.Tensor=Tensor
    torch.empty=lambda shape,dtype,device:Tensor(shape,dtype=dtype)
    torch.empty_like=lambda t:Tensor(t.shape,dtype=t.dtype)
    torch.isfinite=lambda t:Tensor(t.shape,[math.isfinite(v) for v in t.values],'bool')
    torch.equal=lambda a,b:a.shape==b.shape and a.values==b.values
    main=Stream(state)
    torch.cuda=types.SimpleNamespace(synchronize=state.sync,is_current_stream_capturing=lambda:state.capture,
        get_device_capability=lambda dev:(9,0),get_device_properties=lambda dev:types.SimpleNamespace(multi_processor_count=132),
        Stream=lambda:Stream(state),current_stream=lambda:main,stream=lambda s:contextlib.nullcontext(),
        CUDAGraph=lambda:Graph(state),graph=lambda g:contextlib.nullcontext(),Event=lambda **kw:Event())
    budget=types.ModuleType('budget');budget.remaining=lambda:state.remaining-state.now;budget.expired=lambda:budget.remaining()<0
    kernels=types.ModuleType('kernels');kernels.__path__=[]
    norm=types.ModuleType('kernels.add_rmsnorm')
    def addnorm(x,y,wn,eps,xout):
        state.norm_calls.append(wn.name)
        xout.values=[a+b for a,b in zip(x.values,y.values)]
        t=x.copy();t.norm_name=wn.name;return t
    norm.add_rms_norm=addnorm
    sys.modules.update(torch=torch,budget=budget,kernels=kernels)
    sys.modules['kernels.add_rmsnorm']=norm
    if not actual_gemm:
        gemm=types.ModuleType('kernels.gemm')
        class FakeMatmul:
            def __init__(self,M,N,K,device,**cfg):self.M,self.N,self.K,self.cfg=M,N,K,cfg;self.BLOCK_M=32
            def __call__(self,a,w,norm=None):
                cfg='C' if self.cfg['split_k']==2 else ('B' if self.cfg['block_n']==128 else 'A')
                op={(6144,2560):'qkv',(2560,4096):'o',(2560,9728):'d',(151936,2560):'lm'}[(self.N,self.K)]
                assert self.BLOCK_M==64 and norm is None
                if op in ('qkv','lm'):assert a.norm_name==w.name.replace('w','n',1),(a.norm_name,w.name)
                state.candidate_calls.append((op,cfg,w.name,a.norm_name))
                state.last_kind,state.last_op=cfg,op
                out=Tensor((64,self.N),w.values)
                if state.bad:state.bad(op,cfg,w,out)
                return out
        gemm.SkinnyMatmul=FakeMatmul;sys.modules['kernels.gemm']=gemm
    else:
        triton=types.ModuleType('triton');triton.cdiv=lambda a,b:(a+b-1)//b
        class JIT:
            def __init__(self,fn):self.fn=fn;self.calls=[]
            def __getitem__(self,grid):
                def launch(*args,**kwargs):self.calls.append((grid,args,kwargs))
                return launch
        triton.jit=JIT;tl=types.ModuleType('triton.language');tl.constexpr=object
        triton.language=tl;sys.modules.update(triton=triton);sys.modules['triton.language']=tl
        load('kernels.gemm',CAND/'kernels/gemm.py')
    mod=load('test_wide_selector',CAND/'kernels/m64_matmul_selection.py')
    mod.time=types.SimpleNamespace(monotonic=lambda:state.now)
    return mod

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def fixture(state):
    layers=[]
    for i in range(36):
        layers.append(types.SimpleNamespace(wqkv=Tensor((6144,2560),name=f'wqkv{i}'),in_norm=Tensor((2560,),name=f'nqkv{i}'),
          wo=Tensor((2560,4096),name=f'wo{i}'),wgu=Tensor((19456,2560),name=f'wgu{i}'),
          post_norm=Tensor((2560,),name=f'ngu{i}'),wd=Tensor((2560,9728),name=f'wd{i}')))
    model=types.SimpleNamespace(layers=layers,lm_head=Tensor((151936,2560),name='wlm'),final_norm=Tensor((2560,),name='nlm'),cfg=types.SimpleNamespace(eps=1e-6))
    def baseline(op,normed):
        def call(*args):
            w=args[-1]
            if normed:
                x,y,wn,xout,w=args;sys.modules['kernels.add_rmsnorm'].add_rms_norm(x,y,wn,1e-6,xout)
            state.last_kind,state.last_op='original',op
            return Tensor((64,w.shape[0]),w.values)
        return call
    original={name:baseline(name,name in ('qkv','gu','lm')) for name in ('qkv','o','gu','d','lm')}
    return original,model,Tensor((64,2560)),Tensor((64,2560)),Tensor((64,4096)),Tensor((64,9728))

def fake_timer(state):
    def timer(fn,records,owners):
        fn(records[0])
        kind,op=state.last_kind,state.last_op
        state.timing_calls.append((op,kind,len(records)))
        state.now+=state.time_step
        val=state.ratios.get((op,kind),{'original':1.,'A':.94,'B':.92,'C':.90}[kind])
        if isinstance(val,list):return val.pop(0)
        return val
    return timer

def policy(name,change=None,verify=None):
    state=State();mod=install(state);args=fixture(state);mod._owned_time=fake_timer(state)
    if change:change(state,mod,args)
    result=mod.select_m64_projections(*args)
    if verify:verify(state,mod,args,result)
    CHECKS.append(name)
    return state,mod,args,result

def originals(s,m,a,r):assert r is a[0]

def winners(s,m,a,r):
    assert r['gu'] is a[0]['gu']
    for name in ('qkv','o','d','lm'):assert r[name] is not a[0][name]
    assert {n for n,k,l in s.timing_calls if l==36}=={'qkv','o','d'}
    assert all(l==(1 if n=='lm' else 36) for n,k,l in s.timing_calls)
    for op in ('qkv','o','d'):
        for cfg in 'ABC':assert len({w for n,c,w,_ in s.candidate_calls if n==op and c==cfg})==36

policy('all three configs, all 36 layer weights, norm identity, gate/up preserved',verify=winners)
policy('fewer than 42 seconds retains original',lambda s,m,a:setattr(s,'remaining',41.99),originals)
policy('exact 42-second admission is allowed',lambda s,m,a:setattr(s,'remaining',42.),winners)
policy('off-shape retains original',lambda s,m,a:setattr(a[2],'shape',(63,2560)),originals)
policy('misaligned weight retains original',lambda s,m,a:setattr(a[1].layers[-1].wd,'ptr',3),originals)
s,m,a,r=policy('capture no-op retains original',lambda s,m,a:setattr(s,'capture',True),originals);assert s.syncs==0
policy('local deadline after partial winners discards entire suite',lambda s,m,a:setattr(s,'time_step',1.),originals)
policy('shared 24-second reserve aborts entire suite',lambda s,m,a:(setattr(s,'remaining',42.),setattr(s,'time_step',1.)),originals)
policy('all slower alternatives retain original call objects',lambda s,m,a:s.ratios.update({(n,k):1.05 for n in ('qkv','o','d','lm') for k in 'ABC'}),lambda s,m,a,r: all(r[n] is a[0][n] for n in a[0]) or (_ for _ in ()).throw(AssertionError()))

def corrupt(value):
    def modify(s,m,a):
        def bad(op,cfg,w,out):out.values[0]=value
        s.bad=bad
    return modify
for val in (float('nan'),float('inf'),100.):
    policy('nonfinite/error rejection '+str(val),corrupt(val),lambda s,m,a,r: all(r[n] is a[0][n] for n in a[0]) or (_ for _ in ()).throw(AssertionError()))

# An error near zero must fail the pointwise tolerance despite a large maximum.
def near_zero(s,m,a):
    for layer in a[1].layers:layer.wo.values=[0.,1000.,3.,4.]
    s.bad=lambda op,cfg,w,out:out.values.__setitem__(0,.01) if op=='o' else None
policy('pointwise check rejects local error hidden by global max',near_zero,lambda s,m,a,r: r['o'] is a[0]['o'] or (_ for _ in ()).throw(AssertionError()))

def lm_flip(s,m,a):
    a[1].lm_head.values=[1.,1.001,0.,0.]
    s.bad=lambda op,cfg,w,out:out.values.__setitem__(0,1.002) if op=='lm' else None
policy('LM argmax flip rejects numerically close candidate',lm_flip,lambda s,m,a,r: r['lm'] is a[0]['lm'] or (_ for _ in ()).throw(AssertionError()))

def residual_bad(s,m,a):
    candidate=m._candidate
    def factory(*args):
        f=candidate(*args)
        if not args[4]:return f
        def wrong(x,y,wn,xout,w):
            out=f(x,y,wn,xout,w);xout.values[0]+=1;return out
        return wrong
    m._candidate=factory
policy('residual mismatch rejects normed candidate',residual_bad,lambda s,m,a,r: all(r[n] is a[0][n] for n in ('qkv','lm')) or (_ for _ in ()).throw(AssertionError()))

def verify_cfg(state,args,result,expected):
    for name in ('qkv','o','d','lm'):
        source={'qkv':args[2],'o':args[4],'d':args[5],'lm':args[2]}[name]
        w={'qkv':args[1].layers[0].wqkv,'o':args[1].layers[0].wo,
           'd':args[1].layers[0].wd,'lm':args[1].lm_head}[name]
        if name in ('qkv','lm'):
            wn=args[1].layers[0].in_norm if name=='qkv' else args[1].final_norm
            result[name](source,args[3],wn,Tensor((64,2560)),w)
        else:result[name](source,w)
        assert state.last_kind==expected[name],(name,state.last_kind,expected[name])

policy('lowest worst-ratio candidate wins each operation',verify=lambda s,m,a,r:verify_cfg(s,a,r,dict.fromkeys(('qkv','o','d','lm'),'C')))

def one_bad_order(s,m,a):
    s.ratios.update({(op,'C'):[.85,1.01] for op in ('qkv','o','d','lm')})
policy('one losing order rejects candidate despite fast other order',one_bad_order,
       lambda s,m,a,r:verify_cfg(s,a,r,dict.fromkeys(('qkv','o','d','lm'),'B')))

def exact_margin(s,m,a):
    s.ratios.update({(op,cfg):.97 for op in ('qkv','o','d','lm') for cfg in 'ABC'})
policy('exact three-percent boundary does not clear strict gain requirement',exact_margin,
       lambda s,m,a,r:verify_cfg(s,a,r,dict.fromkeys(('qkv','o','d','lm'),'original')))
policy('nonfinite timing after earlier winners abandons entire suite',
       lambda s,m,a:s.ratios.update({('lm','C'):float('nan')}),originals)

# Real owned-timer control flow: streams/capture/outputs exist, fake CUDA executes nothing.
s=State();m=install(s);owners=[];outputs=[]
ms=m._owned_time(lambda record: outputs.append(Tensor((64,2))) or outputs[-1],[1,2],owners)
assert ms==1. and len(outputs)==39 and not owners and s.graphs[0].replays==4 and s.syncs==3
CHECKS.append('owned timer releases graph/output owners only after three drains')
for sticky in (False,True):
    s=State();m=install(s);args=fixture(s)
    s.fail_sync=(lambda n:n>=3) if sticky else (lambda n:n==3)
    def inner(*args):
        owners=args[6]
        return m._owned_time(lambda record:Tensor((64,2)),[1],owners)
    m._select_m64_projections=inner
    try:m.select_m64_projections(*args)
    except BaseException as exc:
        if sticky:
            assert isinstance(exc,m.SelectionDrainFailed) and not isinstance(exc,Exception)
            assert m._QUARANTINED
            kept=m._QUARANTINED[0][0]
            assert any(isinstance(v,dict) and {'graph','stream','last_output','capture'}<=set(v) for v in kept)
            count=s.syncs
            try:m.select_m64_projections(*args)
            except m.SelectionDrainFailed:pass
            else:raise AssertionError('quarantine reuse succeeded')
            assert s.syncs==count
        else:
            assert isinstance(exc,RuntimeError) and not m._QUARANTINED and s.syncs==4
    else:raise AssertionError('failure swallowed')
    CHECKS.append('sticky failure quarantines graph/stream/output and blocks fallback/reuse' if sticky else 'successful recovery drain propagates original error')

s=State();m=install(s);args=fixture(s);s.fail_sync=lambda n:True;m._select_m64_projections=lambda *a:a[0]
try:m.select_m64_projections(*args)
except m.SelectionDrainFailed:assert m._QUARANTINED
else:raise AssertionError('final drain error swallowed')
CHECKS.append('final drain failure quarantines original and selected ownership')

# Real unchanged host launcher, mocked device functions: exact matrix and FP32 reduction.
s=State();m=install(s,actual_gemm=True);gemm=sys.modules['kernels.gemm'];launches=[]
for name,cfg in m.CONFIGS:
    for n,k in ((6144,2560),(2560,4096),(2560,9728),(151936,2560)):
        mm=m._Wide64Matmul(n,k,'cuda:0',cfg);a=Tensor((64,k));w=Tensor((n,k));out=mm(a,w)
        grid,args,kw=gemm._skinny_kernel.calls[-1]
        assert args[4:9]==(64,n,k,k,k) and kw['BLOCK_M']==64 and kw['NORM'] is False
        assert kw['BLOCK_N']==cfg['block_n'] and kw['BLOCK_K']==64
        assert out.shape==(64,n) and out.dtype=='bf16'
        if cfg['split_k']==2:
            assert mm.part.dtype=='fp32' and mm.part.shape==(2,64,n)
            rg,ra,rkw=gemm._sum_kernel.calls[-1]
            assert ra[0] is mm.part and ra[1] is out and ra[2]==64*n and rkw['SPLIT_K']==2
        else:assert mm.part is None and args[3] is out
        launches.append({'config':name,'shape':[64,n,k],'grid':grid,'part_dtype':None if mm.part is None else mm.part.dtype})
CHECKS.append('12 exact host launches retain BF16 inputs/outputs and four FP32 split reductions')

result={'status':'passed','checks':CHECKS,'check_count':len(CHECKS),'launches':launches,'device_arithmetic_tested':False,'gpu_execution':False,'compiler_execution':False,'limitations':'Mock tensors test control policy and ownership only; unchanged real host launcher is exercised with recorded device calls. No BF16 GEMM correctness or actual GPU timing evidence.'}
(HERE/'CPU_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':'passed','check_count':len(CHECKS),'launch_cases':len(launches),'CUDA_or_compiler':False}))
