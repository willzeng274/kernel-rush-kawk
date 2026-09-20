"""Exact frozen baseline/candidate wrappers with shape-only CPU stand-ins."""
from __future__ import annotations
import ast,dataclasses,json,math
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent
SCHEDULE=json.loads((HERE/'CASES.json').read_text());CASES=[r['case'] for r in SCHEDULE]
class Tensor:
    def __init__(self,shape,data=None,dtype='int64',fill=0):
        self.shape,self.data,self.dtype,self.fill=tuple(shape),data,dtype,fill
    def numel(self):return math.prod(self.shape)
    def data_ptr(self):return 4096
class Torch:
    int32,int64,bfloat16='int32','int64','bfloat16'
    cuda=SimpleNamespace(Event=lambda:object(),CUDAGraph=object,empty_cache=lambda:None)
    @staticmethod
    def empty(shape,**kw):return Tensor(shape,dtype=kw.get('dtype','int64'))
    zeros=empty
    @staticmethod
    def full(shape,value,**kw):return Tensor(shape,dtype=kw.get('dtype','int64'),fill=value)
    @staticmethod
    def tensor(data,**kw):return Tensor((len(data),),list(data),kw.get('dtype','int64'))
class Recorder:
    def __init__(self,name,calls):self.name,self.calls=name,calls
    def __getitem__(self,grid):
        def record(*args,**kwargs):self.calls.append(dict(kernel=self.name,grid=list(grid),args=args,kwargs=kwargs))
        return record

def extract(path,names,env):
    nodes=[n for n in ast.parse(path.read_text()).body if getattr(n,'name',None) in names]
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),env)

def load_host(path):
    ns=dict(dataclass=dataclasses.dataclass,torch=Torch,os=SimpleNamespace(environ={}),
            Plan=lambda *a:SimpleNamespace(cap=896),VerifyPlan=lambda *a,**kw:SimpleNamespace(),
            PromptModelSeed=lambda *a:SimpleNamespace(),PAIR_SLOTS=4096,__name__=__name__)
    nodes=[]
    for file,names in [('kernels/attention.py',{'ancestor_masks'}),('kernels/accept.py',{'flat_children'}),('recycle.py',{'TreeTemplate','Recycler'}),('engine.py',{'GraphPlan','Engine'})]:
        for node in ast.parse((path/file).read_text()).body:
            if getattr(node,'name',None) not in names:continue
            if isinstance(node,ast.ClassDef) and node.name in ('Recycler','GraphPlan','Engine'):
                allowed={'Recycler':{'__init__','draft','publish_pairs'},'GraphPlan':{'__init__'},'Engine':{'_plan'}}[node.name]
                node.body=[m for m in node.body if isinstance(m,ast.FunctionDef) and m.name in allowed]
            nodes.append(node)
    prior=next(n for n in ast.parse((path/'recycle.py').read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='RANK_PRIOR' for t in n.targets))
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),prior]+nodes,type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(path/'host_geometry'),'exec'),ns)
    return ns

def default_engine(ns,path):
    engine=object.__new__(ns['Engine'])
    cls=next(n for n in ast.parse((path/'engine.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Engine')
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    names={'spec_k','spec_max_rows','recycle','recycle_k','tree_rows_by_batch','tau_floor_by_batch','tau_floor_default'}
    selected=[n for n in init.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr in names for t in n.targets)]
    assert len(selected)==len(names)
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),'actual_default_assignments','exec'),dict(ns,self=engine))
    engine.plans={};engine.model=SimpleNamespace(cfg=SimpleNamespace(vocab=151936,hidden=2560),device='mock');engine.use_graphs=False
    return engine

def serialize(value):
    return dict(shape=list(value.shape),dtype=value.dtype,allocation_base_alignment=16) if isinstance(value,Tensor) else value

def capture_launches(label):
    assert label in ('base91','candidate')
    path=HERE/'snapshots'/label;env=load_host(path);calls=[]
    env['triton']=SimpleNamespace(next_power_of_2=lambda n:1<<(n-1).bit_length())
    for name in ('_draft_kernel','_publish_pairs_kernel','accept_kernel','_compact_kernel'):env[name]=Recorder(name,calls)
    extract(path/'kernels/accept.py',{'accept_paths'},env)
    extract(path/'kernels/compact.py',{'compact_paths'},env)
    engine=default_engine(env,path)
    model=ast.parse((path/'model.py').read_text());plan=next(x for x in model.body if isinstance(x,ast.ClassDef) and x.name=='Plan');init=next(x for x in plan.body if isinstance(x,ast.FunctionDef) and x.name=='__init__')
    formula=next(x for x in init.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='cap' for t in x.targets))
    fake=SimpleNamespace();exec(compile(ast.Module(body=[formula],type_ignores=[]),'actual_CAP_formula','exec'),dict(self=fake,T=512,max_new=128));cap=fake.cap
    assert cap==896 and cap%64==0
    for b in range(1,65):
        graph=engine._plan(b,512,128);rec=graph.recycler
        rec.draft(graph.nseen);rec.publish_pairs(graph.path_idx,graph.path_len)
        pos=Tensor((b,),dtype='int32')
        env['accept_paths'](rec.blk,graph.cand,rec.child_start,rec.child_list,rec.child_par,rec.masks,rec.depth,graph.done,graph.nseen,pos,graph.limit,rec.root,graph.path_idx,graph.path_len,graph.acc_tokens,graph.acc_count,cap,graph.guard)
        k=Tensor((36,b,8,cap,128),dtype='bfloat16');v=Tensor(k.shape,dtype='bfloat16')
        env['compact_paths'](k,v,pos,graph.path_idx,graph.path_len)
        for call in calls[-4:]:call.update(B_context=b,R_context=rec.R)
    assert len(calls)==256
    return calls

def launch_receipt(calls):
    return [dict(kernel=c['kernel'],B_context=c['B_context'],R_context=c['R_context'],grid=c['grid'],args=[serialize(v) for v in c['args']],kwargs=c['kwargs']) for c in calls]
