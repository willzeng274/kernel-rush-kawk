"""Execute exact host constructors/wrappers against allocation and launch records.

This creates a compile PLAN only. Native Triton specialization attributes must
be confirmed by its real _get_config at a separately authorized compile step.
"""
from __future__ import annotations
import ast
import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CAND=ROOT/'work/candidates/krxfty_triple_context/engine'
BASE=ROOT/'work/candidates/krxfty_longest_verified/engine'


class Tensor:
    def __init__(self,shape,dtype,data=None,device=None,pin_memory=False):
        self.shape=tuple(shape);self.dtype=dtype;self.data=data
        self.device=device;self.pin_memory=pin_memory
    def numel(self):
        import math
        return math.prod(self.shape)


class Torch:
    int32,int64='i32','i64'
    cuda=SimpleNamespace(Event=lambda:object(),CUDAGraph=object)
    @staticmethod
    def empty(shape,**kw):return Tensor(shape,**kw)
    zeros=empty
    @staticmethod
    def full(shape,value,**kw):return Tensor(shape,data=value,**kw)
    @staticmethod
    def tensor(data,**kw):return Tensor((len(data),),data=list(data),**kw)


class Launch:
    def __init__(self,name,calls):self.name=name;self.calls=calls
    def __getitem__(self,grid):
        def call(*args,**kw):self.calls.append((self.name,grid,args,kw))
        return call


def load_host(path):
    nodes=[]
    wanted={'kernels/attention.py':{'ancestor_masks'},'kernels/accept.py':{'flat_children'},
            'recycle.py':{'RANK_PRIOR','TreeTemplate','Recycler'},'engine.py':{'GraphPlan'}}
    for filename,names in wanted.items():
        for node in ast.parse((path/filename).read_text()).body:
            if getattr(node,'name',None) in names or (isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in node.targets)):
                if isinstance(node,ast.ClassDef) and node.name in ('Recycler','GraphPlan'):
                    node.body=[n for n in node.body if isinstance(n,ast.FunctionDef) and n.name in ('__init__','draft','publish_pairs')]
                nodes.append(node)
    calls=[]
    ns=dict(dataclass=dataclasses.dataclass,torch=Torch,os=SimpleNamespace(environ={}),
            Plan=lambda *a:SimpleNamespace(),VerifyPlan=lambda *a,**kw:SimpleNamespace(),
            PAIR_SLOTS=4096,TRIPLE_SLOTS=4096,
            triton=SimpleNamespace(next_power_of_2=lambda n:1<<(n-1).bit_length()),
            _draft_kernel=Launch('draft',calls),_publish_pairs_kernel=Launch('publisher',calls))
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(path/'host_shapes'),'exec'),ns)
    return ns,calls


def path_to(parent,node):
    path=[]
    while node>0:path.append(node);node=parent[node]
    return [0]+path[::-1]


def source_function(path,name):
    source=path.read_text();node=next(n for n in ast.parse(source).body if getattr(n,'name',None)==name)
    start=min([node.lineno]+[d.lineno for d in node.decorator_list])
    text=''.join(source.splitlines(keepends=True)[start-1:node.end_lineno])
    return node,hashlib.sha256(text.encode()).hexdigest()


def main():
    ns,calls=load_host(CAND);base,_=load_host(BASE)
    rows=[];cases={};checks=0
    # One extra batch value proves no batch-dependent constexpr/options; B is grid.
    batches={2:64,3:42,4:16,8:8,16:4,32:2,42:3,64:1}
    kernels={kind:source_function(CAND/file,fn) for kind,file,fn in [
        ('draft','recycle.py','_draft_kernel'),('publisher','pair_cache.py','_publish_pairs_kernel')]}
    for r,b in batches.items():
        for batch in (b,b+1):
            for t in (1,2,3,4,512,5000):
                model=SimpleNamespace(cfg=SimpleNamespace(vocab=151936),device='mock')
                g=ns['GraphPlan'](model,batch,t,128,recycle_rows=r,recycle_k=8)
                bg=base['GraphPlan'](model,batch,t,128,recycle_rows=r,recycle_k=8)
                rec=g.recycler;bt=bg.recycler.template;tree=rec.template
                assert tree.parent==bt.parent and tree.rank==bt.rank and tree.masks==bt.masks
                paths=[path_to(tree.parent,i) for i in range(r)]
                assert tree.depth==[len(p)-1 for p in paths]
                for i,p in enumerate(paths):
                    assert set(p)=={bit for bit in range(64) if ((tree.masks[i]&((1<<64)-1))>>bit)&1}
                    assert len(p)-1<=rec.maxa
                assert rec.S==len(tree.spine) and rec.SP==rec.S+rec.maxa+1
                expected={'node_oldest':((batch,r),'i32'),'root_prevprev':((batch,),'i32'),
                          'triple_keys':((batch*4096,),'i64'),'triple_oldest':((batch*4096,),'i32'),
                          'triple_values':((batch*4096,8),'i32')}
                for name,(shape,dtype) in expected.items():
                    tensor=getattr(rec,name);assert tensor.shape==shape and tensor.dtype==dtype and tensor.device=='mock'
                assert rec.root_prevprev.data==rec.node_oldest.data==rec.triple_oldest.data==-1
                n=max(1,batch*min(4096,max(0,t-3)))
                for field,dtype in [('slots','i64'),('keys','i64'),('oldest','i32'),('next','i32')]:
                    shape=(n,8) if field=='next' else (n,)
                    for owner in ('host','seed'):
                        tensor=getattr(g,owner+'_triple_'+field)
                        assert tensor.shape==shape and tensor.dtype==dtype
                        assert tensor.pin_memory==(owner=='host') and tensor.device==('mock' if owner=='seed' else None)
                assert g.path_idx.shape==(batch,rec.maxa) and g.acc_tokens.shape==(batch,rec.maxa+1)
                assert g.host_pool.shape==(2,batch,rec.SP) and g.guard==2*r
                # Baseline tensor allocation shapes/dtypes and scalar controllers stay equal.
                for owner,new in ((bg,g),(bg.recycler,rec)):
                    for name,value in vars(owner).items():
                        if isinstance(value,Tensor):
                            got=getattr(new,name);assert got.shape==value.shape and got.dtype==value.dtype
                for name in ('tau_floor','guard','spine_min_match','maxa','depth_in_flight'):
                    assert getattr(g,name)==getattr(bg,name)
                calls.clear();rec.draft(g.nseen);rec.publish_pairs(g.path_idx,g.path_len)
                assert len(calls)==2 and [c[0] for c in calls]==['draft','publisher']
                draft_fields=['root','table','parent','rank','spine_slot','spine',None,'spine_anchor','blk',
                              'root_prev','node_keys','pair_keys','pair_values','root_prevprev','node_oldest',
                              'triple_keys','triple_oldest','triple_values']
                expected_ptrs=[g.nseen if f is None else getattr(rec,f) for f in draft_fields]
                assert list(calls[0][2])==expected_ptrs
                expected_ptrs=[rec.pair_keys,rec.pair_values,rec.node_keys,rec.top,rec.blk,rec.root_prev,
                               g.path_idx,g.path_len,rec.triple_keys,rec.triple_oldest,rec.triple_values,
                               rec.node_oldest,rec.root_prevprev]
                assert list(calls[1][2])==expected_ptrs
                for kind,grid,args,kwargs in calls:
                    node,source_sha=kernels[kind]
                    assert grid==(batch,)
                    count=len(args);names=[a.arg for a in node.args.args]
                    signature={names[i]:'*'+arg.dtype for i,arg in enumerate(args)}
                    constexpr={k:v for k,v in kwargs.items() if k!='num_warps'}
                    assert set(names[count:])==set(constexpr)
                    options={k:v for k,v in kwargs.items() if k=='num_warps'}
                    key=json.dumps([kind,source_sha,signature,constexpr,options],sort_keys=True)
                    item={'id':kind+'_R'+str(r),'kernel':node.name,'source_function_sha256':source_sha,
                          'signature':signature,'constexpr':constexpr,'requested_options':options,
                          'expected_default_options':{'num_warps':4 if kind=='draft' else 1,'num_stages':3,'enable_fp_fusion':True},
                          'representative_grid':grid,'observed_batch_grids':[],'all_pointers_allocation_bases':True}
                    prior=cases.setdefault(key,item)
                    if batch not in prior['observed_batch_grids']:prior['observed_batch_grids'].append(batch)
                checks+=1
        rows.append({'R':r,'representative_B':b,'S':rec.S,'MAXA':rec.maxa,'SP':rec.SP,'parent':tree.parent,
                     'spine':tree.spine,'rank':tree.rank,'unchanged_from87':True})
    assert len(cases)==16
    helper_hashes={name:source_function(CAND/'pair_cache.py',name)[1] for name in ('_pair_slot','_triple_slot')}
    result={'status':'PASS','host_constructors_checked':checks,'geometries':rows,
            'compile_plan_only':True,'native_key_dedup_executed':False,
            'native_key_dedup_plan':'At authorized compile, use the exact frozen decorated functions and transitive helpers. Check every pointer/constexpr against this actual-host-launch record; call Triton3.1 kernel._get_config with aligned allocation-base pointer stand-ins and actual constexpr values. Include resulting native attrs, full runtime signature, constexpr, requested/default options, target CUDA SM90 warp32 and helper/source hashes in dedup key. Assert 16 distinct draft/publisher specializations, and no B/T scalar specialization. Do not invent B sweeps, alignment tiers, or options. If native attrs diverge, stop and report instead of silently reusing artifacts.',
            'helper_function_sha256':helper_hashes,'cases':list(cases.values()),
            'reuse':'Unchanged model/kernels/accept/compact code and launch constexprs retain existing compile evidence only when exact source closure, signature, all actual native attributes, constexprs, options and target match. Both modified kernels need all 16 listed cases; helpers inline without extra standalone compile.',
            'inspect_after_compile':'PTXAS resources/spills and exact draft lane predicates/dependency stores and loads. Existing pair-draft audit reported scalar stores only tid0 and redundant other-lane dependent loads without barriers; this CPU audit cannot prove that behavior safe or that added oldest propagation avoids it.',
            'test_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (HERE/'HOST_LAUNCH_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'PASS','host_constructors_checked':checks,'compile_cases':len(cases),'geometry':[{k:x[k] for k in ('R','S','MAXA','SP')} for x in rows]},indent=2))


if __name__=='__main__':main()
