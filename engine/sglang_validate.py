"""Real-prefix full-vector/cache validation and selected-baseline attention pricing."""
import time
import math
import torch
import torch.nn.functional as F
from sglang_lifetime import CandidateRejected,Temporaries
from sglang_runtime import bind
from sglang_decode_port import launch


def close(a,b,name):
    if not torch.isfinite(a).all() or not torch.isfinite(b).all() or not torch.allclose(a,b,rtol=.02,atol=.03):
        raise CandidateRejected('numerical mismatch: '+name)


def bits(a,b):
    return torch.equal(a.view(torch.int16),b.view(torch.int16))


def restore(e,prompts):
    c=e.sg_control;c.healthy();at=time.perf_counter()
    e.prefill_input.copy_(torch.tensor(prompts,dtype=torch.int64),non_blocking=False)
    e.prefill_graph.replay();e.position.fill_(e.prompt);c.drain()
    first=e.ids.clone();c.drain()
    c.restore=max(c.restore,time.perf_counter()-at)
    if c.base:c.reserve()
    return first


def prefix(e,a,prompts,target):
    if not e.prompt<=target<e.capacity:raise ValueError('invalid real prefix')
    bind(e,a);restore(e,prompts);pos=e.prompt
    chunk=a.native_chunks[4] if a.native_chunks is not None else a.chunks
    while pos+4<=target and 4 in chunk.graphs:
        e.sg_control.live('prefix',5.);chunk.graphs[4].replay();pos+=4
    while pos<target:
        e.sg_control.live('prefix',5.);e._step();pos+=1
    e.sg_control.drain()


def raw_attention(e,a,port,prompts):
    c=e.sg_control
    prefix(e,a,prompts,e.capacity-1)
    e._sg_memory_guard(validation=True)
    owner=Temporaries(c)
    old_qkv=e.qkv
    old_position=e.position.clone();owner.hold(old_position)
    old_keys,old_values=list(e.keys),list(e.values)
    try:
        e.qkv=owner.hold(old_qkv.clone())
        original_qkv=owner.hold(old_qkv.clone())
        for idx in (0,len(e.layers)-1):
            pairs=[(owner.hold(old_keys[idx].clone()),owner.hold(old_values[idx].clone())) for _ in range(2)]
            reference=owner.hold(torch.empty_like(e.attention))
            positions=sorted({0,min(30,e.capacity-1),min(31,e.capacity-1),min(32,e.capacity-1),
                              min(63,e.capacity-1),min(64,e.capacity-1),min(255,e.capacity-1),
                              min(256,e.capacity-1),e.prompt,e.capacity-1})
            for scale in (.25,1.,4.):
                e.qkv.copy_((original_qkv.float()*scale).to(torch.bfloat16))
                for pos in positions:
                    c.live('attention_numerics',5.)
                    e.position.fill_(pos)
                    for k,v in pairs:
                        k.copy_(old_keys[idx]);v.copy_(old_values[idx])
                        k[:,:,pos+1:].fill_(float('nan'));v[:,:,pos+1:].fill_(float('nan'))
                    e.keys[idx],e.values[idx]=pairs[0]
                    a.attention.run(idx);reference.copy_(e.attention);c.drain()
                    e.keys[idx],e.values[idx]=pairs[1]
                    port.mid.fill_(float('nan'));port.lse.fill_(float('nan'))
                    port.run(idx);c.drain()
                    close(e.attention,reference,'all attention channels')
                    # A separate full BF16 SDPA computation consumes exactly
                    # the just-written current slot and all valid history.
                    expected=owner.hold(F.scaled_dot_product_attention(
                        e.query.unsqueeze(2),pairs[1][0][:,:,:pos+1],pairs[1][1][:,:,:pos+1],
                        enable_gqa=True).reshape(e.batch,4096))
                    close(e.attention,expected,'BF16 SDPA attention')
                    if any(not bits(x,y) for x,y in zip(pairs[0],pairs[1])):
                        raise CandidateRejected('writer/cache bits differ')
                    for k,v in pairs:
                        if (not bits(k[:,:,:pos],old_keys[idx][:,:,:pos]) or
                            not bits(v[:,:,:pos],old_values[idx][:,:,:pos]) or
                            not torch.isnan(k[:,:,pos+1:]).all() or not torch.isnan(v[:,:,pos+1:]).all()):
                            raise CandidateRejected('off-path writer cache mutation')
                    # Attention-only replay may not write any cache byte.
                    port_snapshot=(owner.hold(pairs[1][0].clone()),owner.hold(pairs[1][1].clone()))
                    launch(e.query,pairs[1][0],pairs[1][1],e.position,port.mid,port.lse,
                           e.attention,e.capacity,port.sm_count)
                    c.drain()
                    if any(not bits(x,y) for x,y in zip(pairs[1],port_snapshot)):
                        raise CandidateRejected('attention modified KV')
                    # Release these two synchronous snapshots immediately, not
                    # after the entire matrix of numerical probes.
                    owner.items.pop();owner.items.pop();port_snapshot=None
            e.keys[idx],e.values[idx]=old_keys[idx],old_values[idx]
            # Arrays are drained; keep allocation scope bounded to one layer.
            c.drain()
            owner.items[:]=[x for x in owner.items if x is old_position or x is e.qkv or x is original_qkv]
            pairs=reference=None
    finally:
        try:c.drain()
        finally:
            # Python references can restore even under poison; no device writes
            # or owner release is attempted until a successful drain.
            e.qkv=old_qkv;e.keys[:]=old_keys;e.values[:]=old_values
        if not c.poisoned:e.position.copy_(old_position)
        owner.close()


class AttentionWitness:
    def __init__(self,e,operation,owner,reference=None):
        self.e,self.operation,self.owner,self.reference=e,operation,owner,reference
        self.outputs=[]
    def run(self,idx):
        self.e.sg_control.live('full_numerics',5.)
        self.operation.run(idx)
        if self.reference is None:self.outputs.append(self.owner.hold(self.e.attention.clone()))
        else:close(self.e.attention,self.reference[idx],'layer attention')


def full_step(e,a,port,prompts,pos):
    c=e.sg_control;e._sg_memory_guard(validation=True)
    owner=Temporaries(c)
    try:
        prefix(e,a,prompts,pos)
        for k,v in zip(e.keys,e.values):
            k[:,:,pos+1:].fill_(123.);v[:,:,pos+1:].fill_(-123.)
        witness=AttentionWitness(e,a.attention,owner)
        e.fused_cache_attention=witness;e._step();c.drain()
        expected=owner.hold(e.logits.clone())
        cache=[(owner.hold(k.clone()),owner.hold(v.clone())) for k,v in zip(e.keys,e.values)]
        prefix(e,a,prompts,pos)
        for k,v in zip(e.keys,e.values):
            k[:,:,pos+1:].fill_(123.);v[:,:,pos+1:].fill_(-123.)
        port.mid.fill_(float('nan'));port.lse.fill_(float('nan'))
        e.fused_cache_attention=AttentionWitness(e,port,owner,witness.outputs)
        e._step();c.drain()
        close(e.logits,expected,'full vocabulary logits')
        if not torch.equal(e.logits.argmax(-1),expected.argmax(-1)):
            raise CandidateRejected('same-prefix greedy ID differs')
        for (k,v),(ok,ov) in zip(zip(e.keys,e.values),cache):
            c.live()
            close(k[:,:,:pos+1],ok[:,:,:pos+1],'all-layer K')
            close(v[:,:,:pos+1],ov[:,:,:pos+1],'all-layer V')
            if (not bits(k[:,:,:pos],ok[:,:,:pos]) or not bits(v[:,:,:pos],ov[:,:,:pos]) or
                    not bits(k[:,:,pos+1:],ok[:,:,pos+1:]) or not bits(v[:,:,pos+1:],ov[:,:,pos+1:])):
                raise CandidateRejected('off-path complete-step KV mutation')
    finally:
        e.fused_cache_attention=a.attention
        owner.close()


def validate(e,a,port,prompts):
    raw_attention(e,a,port,prompts)
    for pos in sorted({e.prompt,min(e.prompt+3,e.capacity-1),e.capacity-1}):
        full_step(e,a,port,prompts,pos)
    bind(e,a);restore(e,prompts)


class LayerGraph:
    def __init__(self,e,operation):
        c=e.sg_control;self.control=c;c.register(self)
        self.stream=self.graph=None
        c.live('attention_capture',5.)
        self.stream=torch.cuda.Stream(device=e.ids.device)
        c.drain()
        with torch.cuda.stream(self.stream):
            for idx in range(len(e.layers)):
                c.live();operation.run(idx)
        c.wait(self.stream.synchronize)
        c.live()
        self.graph=torch.cuda.CUDAGraph()
        c.live();e.sg_captures+=1
        with torch.cuda.graph(self.graph,stream=self.stream):
            for idx in range(len(e.layers)):operation.run(idx)
        c.live();self.graph.replay();c.drain();c.live()


def time_graph(e,graph):
    c=e.sg_control
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    samples=[]
    for _ in range(3):
        c.live('attention_price',5.)
        start.record()
        for _ in range(8):graph.replay()
        end.record();c.wait(end.synchronize)
        elapsed=start.elapsed_time(end)/8
        if not math.isfinite(elapsed) or elapsed<=0:
            raise CandidateRejected('invalid CUDA attention timing')
        samples.append(elapsed)
        c.live()
    return sorted(samples)[1]


def micro_admission(e,a,port,prompts):
    c=e.sg_control
    props=torch.cuda.get_device_properties(e.ids.device)
    l2=getattr(props,'L2_cache_size',0)
    if type(l2)is not int or l2<=0:
        raise CandidateRejected('reported device L2 size unavailable')
    fixtures=[]
    try:
        for pos in sorted({e.prompt,e.capacity-1}):
            pool_bytes=2*len(e.layers)*e.batch*8*(pos+1)*128*2
            if pool_bytes<=l2:
                raise CandidateRejected('actual all-layer cache pool fits L2')
            prefix(e,a,prompts,pos)
            e.qkv.copy_(e.prefill_qkv.view(e.batch,e.prompt,6144)[:,-1])
            if not fixtures:
                fixtures.append(LayerGraph(e,a.attention))
                fixtures.append(LayerGraph(e,port))
            e.position.fill_(pos)
            for x in fixtures:x.graph.replay()
            c.drain()
            timings=[time_graph(e,fixtures[index].graph) for index in (0,1,1,0)]
            if not max(timings[1:3])<.92*min(timings[0],timings[3]):
                raise CandidateRejected('selected attention complete-pipeline gain insufficient')
    finally:
        c.drain()
        for owner in fixtures:c.release(owner)
        fixtures.clear()
