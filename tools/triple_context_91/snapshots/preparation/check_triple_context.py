"""Actual-source CPU check; no candidate imports, compilation or GPU execution.

Reuses only the old NumPy pointer/queued-stream substrate and fixture constructor.
All context expectations come from full branch histories, independently of saved
pair/oldest buffers. No preceding test suite is run.
"""
import ast
import hashlib
import json
import runpy
from pathlib import Path
from types import MethodType, SimpleNamespace
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / 'work/candidates/krxfty_longest_verified/engine'
CAND = ROOT / 'work/candidates/krxfty_triple_context/engine'
old = runpy.run_path(str(ROOT/'work/research/krxfty_controller_fresh_78/check_controller.py'))
longest = runpy.run_path(str(ROOT/'work/research/krxfty_longest_verified/check_longest.py'))
V, Ptr, TL, Tensor = map(old.get, ('V', 'Ptr', 'TL', 'Tensor'))
TL.uint64 = np.uint64
TL.max = staticmethod(lambda a, axis: V(np.max(a, axis=axis)))
TL.full = staticmethod(lambda shape, value, dtype: V(np.full(shape, value, dtype)))
load = old['load_nodes']
np.seterr(over='ignore')
STATS = dict(hash_cases=0, prompt_cases=0, ancestry_nodes=0, rounds=0,
             requests=0, frozen=0, zero=0, full=0, partial=0,
             alternate=0, direct_drafts=0, negative_controls=0)


def pack(a, b): return (int(a) << 32) | int(b)


def mul64(a, b):
    # Independent multiplication with four 32-bit limbs.
    lo = (a & 0xffffffff) * (b & 0xffffffff)
    hi = ((lo >> 32) + (a >> 32) * (b & 0xffffffff) +
          (a & 0xffffffff) * (b >> 32)) & 0xffffffff
    return (hi << 32) | (lo & 0xffffffff)


def pair_hash(key):
    mixed = mul64(key ^ (key >> 23), 6364136223846793005)
    return (mixed ^ (mixed >> 32)) & 4095


def triple_hash(context):
    a, b, c = map(int, context)
    return pair_hash(pack(b, c) ^ mul64(a + 1, 11400714819323198485))


def seed_oracle(prompts, k, width):
    result = {}
    for b, prompt in enumerate(prompts):
        latest, successors = {}, {}
        # Forward history grouping; newest owner and newest distinct successors.
        for i in range(width, len(prompt)):
            context = tuple(prompt[i-width:i])
            latest[context] = i
            successors.setdefault(context, {})[prompt[i]] = i
        owners = {}
        for ctx in sorted(latest, key=latest.get):
            slot = triple_hash(ctx) if width == 3 else pair_hash(pack(*ctx))
            owners[b*4096+slot] = ctx
        for slot, ctx in owners.items():
            values = sorted(successors[ctx], key=successors[ctx].get, reverse=True)[:min(k, 8)]
            result[slot] = (ctx, values+[-1]*(k-len(values)))
    return result


env = dict(old['ENV'])
load(CAND/'kernels/attention.py', {'ancestor_masks'}, env)
load(CAND/'pair_cache.py', {'PAIR_SLOTS', 'TRIPLE_SLOTS', 'pair_slot', 'triple_slot',
     'prompt_pairs', 'prompt_triples', '_pair_slot', '_triple_slot', '_publish_pairs_kernel'}, env)
load(CAND/'recycle.py', {'RANK_PRIOR', 'TreeTemplate', 'Recycler', '_draft_kernel'}, env)
load(CAND/'kernels/accept.py', {'accept_kernel'}, env)
Template = env['TreeTemplate']
draft, publish, accept = map(env.get, ('_draft_kernel', '_publish_pairs_kernel', 'accept_kernel'))


class Kernel:
    def __init__(self, fn): self.fn = fn
    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            kwargs.pop('num_warps', None)
            for b in range(grid[0]):
                TL.pid = (b, 0, 0)
                self.fn(*(Ptr(x.data) if isinstance(x, Tensor) else x for x in args), **kwargs)
        return launch


env['_draft_kernel'] = Kernel(draft)
env['_publish_pairs_kernel'] = Kernel(publish)
env['triton'] = SimpleNamespace(next_power_of_2=lambda n: 1 << (n-1).bit_length())
Tensor.shape = property(lambda self: self.data.shape)


def index_copy(self, dim, indices, source):
    assert dim == 0
    def fn(): self.data[indices.data] = source.data
    if self.stream.executing: fn()
    else: self.stream.add(fn)
    return self


Tensor.index_copy_ = index_copy


def prompt_checks():
    rng = np.random.default_rng(9101)
    cases = [([], 8), ([[]], 8)]
    cases += [([list(range(t))], 8) for t in range(1, 5)]
    cases += [([[1,2,3,4,1,2,3,5,1,2,3,4], [1,2,3,9]], 8),
              ([list(range(9000))], 8)]
    for _ in range(160):
        b, t, k = int(rng.integers(1,5)), int(rng.integers(0,100)), int(rng.choice([1,2,8,12]))
        cases.append((rng.integers(0,17,(b,t)).tolist(), k))
    for prompts, k in cases:
        slots, keys, oldest, values = env['prompt_triples'](prompts, k)
        got = {slot: ((a, key>>32, key&0xffffffff), row)
               for slot,key,a,row in zip(slots,keys,oldest,values)}
        assert len(slots) == len(set(slots)) and len(slots) <= len(prompts)*4096
        assert got == seed_oracle(prompts, k, 3)
        assert len(slots) <= sum(min(4096, max(0,len(p)-3)) for p in prompts)
        STATS['prompt_cases'] += 1
    contexts = [(0,0,0),(151935,151935,151935),(0,151935,0)]
    contexts += [tuple(map(int,c)) for c in rng.integers(0,151936,(1600,3))]
    for ctx in contexts:
        key = pack(*ctx[-2:]); a = ctx[0]
        assert env['triple_slot'](key,a) == triple_hash(ctx) == int(env['_triple_slot'](V(np.int64(key)),V(np.int32(a))))
        STATS['hash_cases'] += 1
    # Same tail, different oldest; same oldest, different tail collisions.
    witnesses = []
    for kind in ('oldest','tail'):
        seen = {}
        for n in range(151936):
            ctx = (n,11,17) if kind == 'oldest' else (7,n,17)
            h = triple_hash(ctx)
            if h in seen:
                witnesses.append((seen[h],ctx,h)); break
            seen[h] = ctx
        assert len(witnesses) == (1 if kind == 'oldest' else 2)
    return witnesses


def make_fixture(B, R, N, mode='mixed', T=12):
    factory = old['make_host']; factory.__globals__['ENGINE'] = CAND
    factory.__globals__['Template'] = Template
    g, s = factory(B,R,N,mode)
    g.T = T
    rec, t, K = g.recycler, g.recycler.template, 8
    ten = lambda data, device=True: Tensor(np.asarray(data),s,device)
    g.plan.ids = ten(np.zeros((B,T),np.int64))
    rec.B, rec.R, rec.k, rec.maxa = B,R,K,g.maxa
    for name,data in [('parent',np.array(t.parent,np.int32)), ('rank',np.array(t.rank,np.int32)),
                      ('masks',np.array(t.masks,np.int64)), ('depth',np.array(t.depth,np.int32)),
                      ('spine_slot',np.array([t.spine.index(i) if i in t.spine else -1 for i in range(R)],np.int32)),
                      ('table',np.full((151936,K),-1,np.int32)),
                      ('pair_keys',np.full(B*4096,-1,np.int64)), ('pair_values',np.full((B*4096,K),-123,np.int32)),
                      ('node_keys',np.zeros((B,R),np.int64)), ('root_prev',np.full(B,-999,np.int64)),
                      ('root_prevprev',np.full(B,-1,np.int32)), ('node_oldest',np.full((B,R),-1,np.int32)),
                      ('triple_keys',np.full(B*4096,-123,np.int64)), ('triple_oldest',np.full(B*4096,-1,np.int32)),
                      ('triple_values',np.full((B*4096,K),-123,np.int32)), ('top',np.zeros((B*R,K),np.int32))]:
        setattr(rec,name,ten(data))
    for prefix,width in [('pair',2),('triple',3)]:
        cap = max(1,B*min(4096,max(0,T-width)))
        fields = [('slots',np.int64,(cap,)), ('keys',np.int64,(cap,)), ('next',np.int32,(cap,K))]
        if width == 3: fields += [('oldest',np.int32,(cap,))]
        for name,dtype,shape in fields:
            setattr(g,'host_'+prefix+'_'+name,ten(np.empty(shape,dtype),False))
            setattr(g,'seed_'+prefix+'_'+name,ten(np.empty(shape,dtype)))
    ns = g.run_recycle.__globals__
    ns.update({name:env[name] for name in ('prompt_pairs','prompt_triples','PAIR_SLOTS','TRIPLE_SLOTS')})
    rec.draft = MethodType(env['Recycler'].draft,rec)
    actual_publish = MethodType(env['Recycler'].publish_pairs,rec)
    state = {'tables':{}, 'prefix':None, 'prev':None, 'oldest':None, 'check':True, 'histories':None}
    def prefill():
        s.acc_waits = 0
        def fn():
            prompts = g.plan.ids.data.tolist()
            state['tables'] = {w:seed_oracle(prompts,K,w) for w in (2,3)}
            state['histories'] = [p+[old['oracle'](p,mode)] for p in prompts]
            g.plan.pos.data[:] = T
            for b,p in enumerate(prompts):
                g.plan.tok.data[b] = old['oracle'](p,mode)
                g.plan.k_cache.data[0,b,0,:T,0] = p
                g.plan.v_cache.data[0,b,0,:T,0] = p
                for i,token in enumerate(p):
                    first = old['oracle'](p[:i+1],mode)
                    rec.table.data[token] = [(first+rank)%31 for rank in range(K)]
        s.add(fn)
    g._step_prefill = prefill
    old_verify = g.verify.verify
    def verify():
        state['prefix'] = [g.plan.k_cache.data[0,b,0,:int(g.verify.pos.data[b]),0].tolist() for b in range(B)]
        state['prev'] = rec.root_prev.data.copy(); state['oldest'] = rec.root_prevprev.data.copy()
        for b,prefix in enumerate(state['prefix']):
            # The KV prefix may stop before a frozen root; test root state against
            # independently tracked consumed history, not against physical KV rows.
            history = state['histories'][b]
            assert rec.root.data[b] == history[-1]
            assert rec.root_prev.data[b] == history[-2]
            assert rec.root_prevprev.data[b] == (history[-3] if len(history)>=3 else -1)
            for i in range(R):
                path=[]; node=i
                while node>=0:
                    path.append(int(rec.blk.data[b,node])); node=t.parent[node]
                full = history[:-1]+list(reversed(path))
                assert rec.node_keys.data[b,i] == pack(*full[-2:]), ('pair ancestry',b,i,full)
                assert rec.node_oldest.data[b,i] == (full[-3] if len(full)>=3 else -1), ('oldest ancestry',b,i,full)
                STATS['ancestry_nodes'] += 1
        out = old_verify()
        for b in range(B):
            for row in range(R):
                top = [(int(out.data[b,row])+rank)%31 for rank in range(K)]
                rec.top.data[b*R+row] = top
                rec.table.data[int(rec.blk.data[b,row])] = top
        return out
    g.verify.verify = verify
    def accept_wrapper(*args):
        tensors,cap,guard = args[:-2],args[-2],args[-1]
        for b in range(B):
            before_done = bool(g.done.data[b]); remaining = int(g.limit.data[0]-g.nseen.data[b])
            tokens,path,first,best = longest['oracle'](t,rec.blk.data[b],g.cand.data[b],remaining)
            TL.pid = (b,0,0)
            accept(*(Ptr(x.data) for x in tensors),cap,R,R-1,1<<R.bit_length(),g.maxa,guard)
            assert int(g.path_len.data[b]) == (-1 if before_done else len(path))
            if not before_done:
                assert g.path_idx.data[b,:len(path)].tolist() == path
                assert g.acc_tokens.data[b,:len(tokens)].tolist() == tokens
                STATS['alternate'] += path != first
    ns['accept_paths'] = accept_wrapper
    def publish_and_check(idx,lens):
        actual_publish(idx,lens)
        if not state['check']: return
        for b in range(B):
            count = int(lens.data[b]); history = state['histories'][b]
            if count < 0:
                assert rec.root_prev.data[b] == state['prev'][b]
                assert rec.root_prevprev.data[b] == state['oldest'][b]
                STATS['frozen'] += 1
                continue
            STATS['zero'] += count == 0; STATS['full'] += count == g.maxa
            STATS['partial'] += 0 < count < g.maxa
            consumed = history[:-1].copy()
            for row in [0]+idx.data[b,:count].tolist():
                consumed.append(int(rec.blk.data[b,row]))
                want = old['oracle'](consumed,mode)
                values = [(want+rank)%31 for rank in range(K)]
                for width in (2,3):
                    if len(consumed)<width: continue
                    ctx = tuple(consumed[-width:])
                    slot = b*4096+(triple_hash(ctx) if width==3 else pair_hash(pack(*ctx)))
                    state['tables'][width][slot] = (ctx,values)
            assert rec.root_prev.data[b] == consumed[-1]
            assert rec.root_prevprev.data[b] == consumed[-2]
            assert rec.root.data[b] == old['oracle'](consumed,mode)
            state['histories'][b] = consumed+[int(rec.root.data[b])]
        for width in (2,3):
            tables = state['tables'][width]
            valid = rec.pair_keys.data >= 0 if width == 2 else rec.triple_oldest.data >= 0
            assert set(np.flatnonzero(valid).tolist()) == set(tables), 'unexpected/unaccepted publication'
            for slot,(ctx,values) in tables.items():
                keys = rec.pair_keys if width==2 else rec.triple_keys
                vals = rec.pair_values if width==2 else rec.triple_values
                assert int(keys.data[slot]) == pack(*ctx[-2:])
                assert vals.data[slot].tolist() == values, ('publication',width,slot)
                if width==3: assert int(rec.triple_oldest.data[slot]) == ctx[0]
        STATS['rounds'] += 1
    rec.publish_pairs = publish_and_check
    return g,s,state


def direct_draft_checks(witnesses):
    for R in (2,3,4,8,16,32,42,64):
        g,s,state = make_fixture(2,R,33); rec=g.recycler; t=rec.template
        state['check']=False; s.executing=True
        rec.table.data.fill(4); rec.spine.data.fill(-1); rec.spine_anchor.data.fill(1); g.nseen.data.fill(1)
        child = next(i for i in t.children[0] if t.rank[i]==0)
        for wrong,ctx,slot in witnesses:
            rec.root.data[:] = ctx[2]; rec.root_prev.data[:] = ctx[1]; rec.root_prevprev.data[:] = ctx[0]
            rec.pair_keys.data.fill(-1); rec.triple_oldest.data.fill(-1)
            pairslot=pair_hash(pack(*ctx[-2:]))
            for b in range(2):
                rec.pair_keys.data[b*4096+pairslot]=pack(*ctx[-2:]); rec.pair_values.data[b*4096+pairslot]=23
                tag=wrong if b==0 else ctx
                rec.triple_keys.data[b*4096+slot]=pack(*tag[-2:]);rec.triple_oldest.data[b*4096+slot]=tag[0]
                rec.triple_values.data[b*4096+slot]=27
            rec.draft(g.nseen)
            assert rec.blk.data[:,child].tolist()==[23,27], 'both exact tag components required'
            rec.triple_values.data[4096+slot,0]=-1; rec.draft(g.nseen)
            assert rec.blk.data[:,child].tolist()==[23,23], 'triple missing rank fallback'
            rec.pair_values.data[pairslot,0]=-1; rec.draft(g.nseen)
            assert rec.blk.data[:,child].tolist()==[4,23], 'pair missing rank fallback'
            rec.spine.data[:,0]=29; rec.draft(g.nseen)
            assert rec.blk.data[:,child].tolist()==[29,29], 'spine priority'
            rec.spine.data.fill(-1)
            STATS['direct_drafts'] += 4
        # Missing oldest cannot expose any old triple slot contents.
        rec.root_prevprev.data.fill(-1);rec.triple_oldest.data.fill(-1);rec.triple_values.data.fill(30)
        rec.draft(g.nseen)
        assert rec.node_oldest.data[:,0].tolist()==[-1,-1]
        for b in range(2):
            for i in range(1,R):
                ancestors=[];node=i
                while node>=0:ancestors.append(int(rec.blk.data[b,node]));node=t.parent[node]
                full=[int(rec.root_prev.data[b])]+ancestors[::-1]
                assert rec.node_oldest.data[b,i]==full[-3]


def request_checks():
    for B,R,mode,T in [(1,64,'easy',1),(1,32,'mixed',2),(4,16,'mixed',3),(4,8,'easy',4),
                        (1,64,'mixed',12),(16,4,'easy',12),(2,42,'mixed',4)]:
        g,s,state=make_fixture(B,R,29,mode,T)
        for repeat in range(2):
            prompts=[[(i*(1+repeat)+b*7+repeat*11)%31 for i in range(T)] for b in range(B)]
            expected=list(map(list,zip(*(old['serial'](p,29,mode) for p in prompts))))
            assert list(g.run_recycle(prompts,29))==expected
            assert s.idle(); STATS['requests']+=1
    for cutoff in (1,2,5):
        g,s,state=make_fixture(4,16,29,T=4)
        gen=g.run_recycle([[1,2,3,4]]*4,29)
        for _ in range(cutoff):next(gen)
        gen.close();assert s.idle()
        prompts=[[9,8,7,b] for b in range(4)]
        assert list(g.run_recycle(prompts,29))==list(map(list,zip(*(old['serial'](p,29,'mixed') for p in prompts))))
        assert s.idle(); STATS['requests']+=1
    g,s,state=make_fixture(1,8,1,T=4)
    assert list(g.run_recycle([[1,2,3,4]],0))==[] and not s.ops
    assert len(list(g.run_recycle([[1,2,3,4]],1)))==1 and s.idle()
    assert np.count_nonzero(g.recycler.triple_oldest.data>=0)==1
    g._seed_triple_table([[7,8,9]]);s.drain()
    assert np.all(g.recycler.triple_oldest.data==-1), 'empty request seed reset'
    ns=g.run_recycle.__globals__;saved=ns['prompt_triples']
    def fail_seed(*args):raise ValueError('injected triple seed failure')
    ns['prompt_triples']=fail_seed
    try:
        try:list(g.run_recycle([[1,2,3,4]],1));raise AssertionError('seed exception swallowed')
        except ValueError:assert s.idle()
    finally:ns['prompt_triples']=saved
    g,s,state=make_fixture(4,16,29,T=4)
    gen=g.run_recycle([[1,2,3,4]]*4,29);next(gen);next(gen)
    assert not s.idle();g.finish_event.fail=True;gen.close()
    assert g.recycle_cleanup_failed
    count=len(s.ops)
    try:list(g.run_recycle([[1,2,3,4]]*4,29));raise AssertionError('reused failed drain')
    except RuntimeError as exc:assert 'failed drain' in str(exc)
    assert len(s.ops)==count
    STATS['ownership']={'max_inflight':old['STATS']['max_inflight'],'cancellations':3,'seed_failure_drained':True,'failed_drain_reuse_rejected':True,'empty_seed_reset':True}


def directed_publication():
    # Explicit alternate longest branch with duplicate root-child tokens.
    g,s,state=make_fixture(2,8,33,T=4);rec=g.recycler;t=rec.template;s.executing=True
    state['check']=False
    blk=np.full((2,8),9,np.int64);cand=np.full((2,8),8,np.int64)
    blk[:,1]=blk[:,4]=cand[:,0]=1;cand[:,1]=cand[:,4]=blk[:,6]=2;cand[:,6]=3
    assert t.parent[1]==t.parent[4]==0 and t.parent[6]==4
    rec.blk.data[:]=blk;g.cand.data[:]=cand;g.nseen.data[:]=[1,32];g.limit.data[:]=33
    g.done.data[:]=0;g.verify.pos.data[:]=4
    for b in range(2):
        for row in range(8):
            node=row;path=[]
            while node>=0:path.append(int(blk[b,node]));node=t.parent[node]
            full=[5,6,7,8]+path[::-1]
            rec.node_keys.data[b,row]=pack(*full[-2:]);rec.node_oldest.data[b,row]=full[-3]
            rec.top.data[b*8+row]=np.arange(8)+100+row*10
    args=(rec.blk,g.cand,rec.child_start,rec.child_list,rec.child_par,rec.masks,rec.depth,
          g.done,g.nseen,g.verify.pos,g.limit,rec.root,g.path_idx,g.path_len,g.acc_tokens,g.acc_count,g.plan.cap,g.guard)
    g._round.__globals__['accept_paths'](*args)
    assert g.path_idx.data[0,:2].tolist()==[4,6]
    assert g.path_idx.data[1,:1].tolist()==[1], 'clipped progress retains first path'
    rec.publish_pairs(g.path_idx,g.path_len)
    for b,path in enumerate(([0,4,6],[0,1])):
        expected={}
        for row in path:
            full=[5,6,7,8];branch=[];node=row
            while node>=0:branch.append(int(blk[b,node]));node=t.parent[node]
            full+=branch[::-1];ctx=tuple(full[-3:]);expected[b*4096+triple_hash(ctx)]=(ctx,row)
        actual=set(np.flatnonzero(rec.triple_oldest.data[b*4096:(b+1)*4096]>=0)+b*4096)
        assert actual==set(expected)
        for slot,(ctx,row) in expected.items():
            assert rec.triple_values.data[slot].tolist()==(np.arange(8)+100+row*10).tolist()
        assert rec.root_prev.data[b]==blk[b,path[-1]]
        full=[5,6,7,8]+[int(blk[b,row]) for row in path]
        assert rec.root_prevprev.data[b]==full[-2]
    STATS['direct_alternate_longest']=True
    # Deliberately poison unused path entries, then zero-consumed and frozen.
    rec.root_prev.data[:]=[8,222];rec.root_prevprev.data[:]=[7,111]
    g.path_idx.data.fill(-999);g.path_len.data[:]=[0,-1]
    before=[x.data.copy() for x in (rec.triple_oldest,rec.triple_keys,rec.triple_values)]
    rec.publish_pairs(g.path_idx,g.path_len)
    assert rec.root_prev.data.tolist()==[9,222] and rec.root_prevprev.data.tolist()==[8,111]
    for arr,oldarr in zip((rec.triple_oldest,rec.triple_keys,rec.triple_values),before):
        assert np.array_equal(arr.data[4096:],oldarr[4096:])
    STATS['direct_zero_frozen']=True


def chronological_collision():
    # Two genuinely different contexts on one consumed branch alias one slot.
    # Contexts and final owners derive only from the full sequential history.
    rng=np.random.default_rng(9103)
    for _ in range(10000):
        history=rng.integers(0,151936,11).tolist()
        contexts=[tuple(history[i-2:i+1]) for i in range(2,11)]
        hashes=[triple_hash(c) for c in contexts]
        if len(set(hashes))<len(hashes) and len(set(contexts))==len(contexts):break
    else:raise AssertionError('no chronological collision witness')
    g,s,state=make_fixture(1,64,33,T=2);s.executing=True;state['check']=False
    rec=g.recycler;path=[0]+rec.template.spine
    assert len(path)==9
    for n,row in enumerate(path):
        rec.blk.data[0,row]=history[n+2]
        rec.node_keys.data[0,row]=pack(history[n+1],history[n+2])
        rec.node_oldest.data[0,row]=history[n]
        rec.top.data[row]=np.arange(8)+1000+row*10
    g.path_idx.data[0,:]=path[1:];g.path_len.data[:]=8
    rec.publish_pairs(g.path_idx,g.path_len)
    expected={}
    for ctx,row in zip(contexts,path):expected[triple_hash(ctx)]=(ctx,row)
    assert set(np.flatnonzero(rec.triple_oldest.data>=0).tolist())==set(expected)
    for slot,(ctx,row) in expected.items():
        assert int(rec.triple_oldest.data[slot])==ctx[0]
        assert int(rec.triple_keys.data[slot])==pack(*ctx[-2:])
        assert rec.triple_values.data[slot].tolist()==(np.arange(8)+1000+row*10).tolist()
    assert int(rec.root_prev.data[0])==history[-1] and int(rec.root_prevprev.data[0])==history[-2]
    STATS['direct_chronological_collision']={'contexts':contexts,'slots':hashes,'path':path,'later_consumed_owner_verified':True}


def negative_controls(witnesses):
    original=env['_draft_kernel']
    source=(CAND/'recycle.py').read_text()
    for before,after in [
        ('(stored_oldest == oldest)','(stored_oldest >= 0)'),
        ('valid_oldest & (stored_tail == key)','valid_oldest'),
        ('(key.to(tl.uint64) >> 32).to(tl.int32)','ptok.to(tl.int32)')]:
        assert before in source
        mutated=dict(env);load(CAND/'recycle.py',{'_draft_kernel'},mutated,source.replace(before,after))
        env['_draft_kernel']=Kernel(mutated['_draft_kernel'])
        try:
            try:direct_draft_checks(witnesses);raise RuntimeError('negative control survived')
            except AssertionError:STATS['negative_controls']+=1
        finally:env['_draft_kernel']=original
    original=env['_publish_pairs_kernel'];source=(CAND/'pair_cache.py').read_text()
    for before,after in [
        ('(last_key >> 32).to(tl.int32)','previous.to(tl.int32)'),
        ('(step <= plen)','(step <= MAXA)')]:
        assert before in source
        mutated=dict(env);load(CAND/'pair_cache.py',{'_publish_pairs_kernel'},mutated,source.replace(before,after))
        env['_publish_pairs_kernel']=Kernel(mutated['_publish_pairs_kernel'])
        try:
            try:directed_publication();raise RuntimeError('negative control survived')
            except AssertionError:STATS['negative_controls']+=1
        finally:env['_publish_pairs_kernel']=original
    # Missing drain must leave pending pinned-buffer work observable.
    g,s,state=make_fixture(1,8,9,T=4);g._drain_recycle=lambda:None
    gen=g.run_recycle([[1,2,3,4]],9)
    for _ in range(9):next(gen)
    assert not s.idle();gen.close();s.drain();STATS['negative_controls']+=1


def source_checks():
    files={str(p.relative_to(BASE)) for p in BASE.rglob('*') if p.is_file()}
    assert files=={str(p.relative_to(CAND)) for p in CAND.rglob('*') if p.is_file()}
    changed=sorted(f for f in files if (BASE/f).read_bytes()!=(CAND/f).read_bytes())
    assert changed==['engine.py','pair_cache.py','recycle.py']
    def node(path,name,method=None):
        n=next(n for n in ast.parse(path.read_text()).body if getattr(n,'name',None)==name)
        if method:n=next(m for m in n.body if getattr(m,'name',None)==method)
        return ast.dump(n,include_attributes=False)
    for name in ('pair_slot','prompt_pairs','_pair_slot'):
        assert node(BASE/'pair_cache.py',name)==node(CAND/'pair_cache.py',name)
    for name in ('TreeTemplate',):assert node(BASE/'recycle.py',name)==node(CAND/'recycle.py',name)
    for method in ('accept','update'):assert node(BASE/'recycle.py','Recycler',method)==node(CAND/'recycle.py','Recycler',method)
    assert node(BASE/'engine.py','Engine')==node(CAND/'engine.py','Engine')
    for method in ('_round','_launch_round','_write_spine','_drain_recycle','_seed_pair_table','_warm_eager','capture'):
        assert node(BASE/'engine.py','GraphPlan',method)==node(CAND/'engine.py','GraphPlan',method)
    for p in CAND.rglob('*.py'):compile(p.read_text(),str(p),'exec')
    return changed


if __name__=='__main__':
    changed=source_checks();witnesses=prompt_checks();direct_draft_checks(witnesses)
    request_checks();directed_publication();chronological_collision();negative_controls(witnesses)
    assert STATS['zero'] and STATS['full'] and STATS['partial'] and STATS['frozen']
    result={'status':'PASS','gpu':False,'stats':STATS,'changed':changed,
            'collision_witnesses':witnesses,'source_sha256':{str(p.relative_to(CAND)):hashlib.sha256(p.read_bytes()).hexdigest() for p in CAND.rglob('*') if p.is_file()},
            'test_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'limits':['Single-program scalar CPU semantics only; no GPU lane execution, memory ordering, compiler, graph capture, numeric model or timing claim.','Synthetic full-history greedy model and exhaustive tree ancestry; sampled prompt/hash inputs are not exhaustive vocabulary combinations.']}
    (HERE/'CPU_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'PASS','stats':STATS},indent=2))
