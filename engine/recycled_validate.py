"""Full-vector/all-layer W4 validation and paid valid-history round pricing."""
import time
import torch
from recycled_host import RequestState, Prices, DebtPolicy
from suffix_lifetime import CandidateRejected, Temporaries


def close(actual,expected,name):
    if (not torch.isfinite(actual).all() or not torch.isfinite(expected).all() or
            not torch.allclose(actual,expected,rtol=.02,atol=.03)):
        raise CandidateRejected("numerical mismatch: "+name)


def logits(actual,expected):
    close(actual,expected,'full logits')
    if not torch.equal(actual.argmax(-1),expected.argmax(-1)):
        raise CandidateRejected("same-prefix greedy ID mismatch")


def bits(actual,expected):
    return torch.equal(actual.view(torch.int16),expected.view(torch.int16))


def restore(e,prompts):
    c = e.suffix_control
    c.healthy()
    started = time.perf_counter()
    e.prefill_input.copy_(torch.tensor(prompts,dtype=torch.int64),non_blocking=False)
    e.prefill_graph.replay()
    e.position.fill_(e.prompt)
    c.drain()
    first = e.ids.tolist()
    c.restore = max(c.restore,time.perf_counter()-started)
    if c.base:
        c.reserve()
    return first


def native_four(e):
    # B1 retained #32 prepares native_chunks and its bounded verifier, whereas
    # larger batches prepare chunks. Do not create a different comparator.
    return e.native_chunks[4] if e.native_chunks is not None else e.chunks


def prefix(e,prompts,target):
    if not e.prompt <= target < e.capacity:
        raise ValueError("valid consumed-prefix target required")
    pending = restore(e,prompts)
    histories = [list(p) for p in prompts]
    position = e.prompt
    chunk = native_four(e)
    while position+4 <= target:
        e.suffix_control.live('prefix',5.)
        chunk.graphs[4].replay()
        rows = chunk.outputs[4].tolist()
        for b,h in enumerate(histories):
            h.extend([pending[b]]+[row[b] for row in rows[:3]])
        pending,position = rows[-1],position+4
    while position < target:
        e.suffix_control.live('prefix',5.)
        for h,y in zip(histories,pending):
            h.append(y)
        e._step()
        pending = e.ids.tolist()
        position += 1
    e.suffix_control.drain()
    return histories,pending


def inactive_zero(graph,active):
    for b,count in enumerate(active):
        for k,v in zip(graph.keys,graph.values):
            if torch.count_nonzero(k[b,:,count:]) or torch.count_nonzero(v[b,:,count:]):
                raise CandidateRejected("inactive scratch was not zero")


def serial_case(e,graph,inputs,base,active):
    c = e.suffix_control
    c.live('numerics',5.)
    started = time.perf_counter()
    graph.replay(inputs,[base]*e.batch,active)
    c.drain()
    inactive_zero(graph,active)
    all_logits = graph.logits.view(e.batch,graph.width,-1)
    if not torch.isfinite(all_logits).all():
        raise CandidateRejected("nonfinite verifier logits")
    for row in range(graph.width):
        c.live()
        e.position.fill_(base+row)
        e.ids.copy_(torch.tensor([x[row] for x in inputs],dtype=torch.int64),non_blocking=False)
        e._step()
        selected = [b for b in range(e.batch) if row < active[b]]
        if selected:
            logits(all_logits[selected,row],e.logits[selected])
            for sk,sv,k,v in zip(graph.keys,graph.values,e.keys,e.values):
                close(sk[selected,:,row],k[selected,:,base+row],'K')
                close(sv[selected,:,row],v[selected,:,base+row],'V')
    graph.discard()
    c.observed('numerics',time.perf_counter()-started)


def full_guard(e,graph,inputs,lengths,active,counts):
    """Every main-cache byte, all layers; rejected tails deliberately poisoned."""
    c = e.suffix_control
    c.live('guard',5.)
    e._memory_guard()
    owner = Temporaries(c)
    try:
        for k,v in zip(e.keys,e.values):
            for b,length in enumerate(lengths):
                k[b,:,length:min(e.capacity,length+graph.width)].fill_(123.)
                v[b,:,length:min(e.capacity,length+graph.width)].fill_(-123.)
        before = [(owner.hold(k.clone()),owner.hold(v.clone())) for k,v in zip(e.keys,e.values)]
        graph.replay(inputs,lengths,active)
        c.drain()
        if graph.width == 4:
            if any(not bits(k,old[0]) or not bits(v,old[1])
                   for k,v,old in zip(e.keys,e.values,before)):
                raise CandidateRejected("W4 replay wrote main cache")
        graph.compact(counts)
        c.drain()
        for (k,v,sk,sv),(ek,ev) in zip(zip(e.keys,e.values,graph.keys,graph.values),before):
            c.live()
            for b,count in enumerate(counts):
                start = lengths[b]
                ek[b,:,start:start+count].copy_(sk[b,:,:count])
                ev[b,:,start:start+count].copy_(sv[b,:,:count])
            if not bits(k,ek) or not bits(v,ev):
                raise CandidateRejected("off-path cache write or compaction mismatch")
    finally:
        owner.close()


def poison_future(e,graph,inputs,lengths,active):
    c = e.suffix_control
    c.live('poison',5.)
    e._memory_guard()
    owner = Temporaries(c)
    saved = []
    try:
        saved = [(owner.hold(k.clone()),owner.hold(v.clone())) for k,v in zip(e.keys,e.values)]
        graph.replay(inputs,lengths,active)
        c.drain()
        old_logits = owner.hold(graph.logits.clone())
        scratch = [(owner.hold(k.clone()),owner.hold(v.clone())) for k,v in zip(graph.keys,graph.values)]
        graph.discard()
        for k,v in zip(e.keys,e.values):
            for b,length in enumerate(lengths):
                k[b,:,length:].fill_(123.)
                v[b,:,length:].fill_(-123.)
        graph.replay(inputs,lengths,active)
        c.drain()
        logits(graph.logits,old_logits)
        for (k,v),(ok,ov) in zip(zip(graph.keys,graph.values),scratch):
            close(k,ok,'future-masked K')
            close(v,ov,'future-masked V')
        graph.discard()
    finally:
        if not c.poisoned:
            for (k,v),(ok,ov) in zip(zip(e.keys,e.values),saved):
                k.copy_(ok); v.copy_(ov)
        owner.close()


def divergent_case(e,graph,prompts):
    """At most four offset groups, each one linear W4 serial reference."""
    c = e.suffix_control
    histories,pending = prefix(e,prompts,e.prompt+3)
    lengths = [e.prompt+b%4 for b in range(e.batch)]
    active = [0 if e.batch > 1 and b == e.batch-1 else graph.width for b in range(e.batch)]
    roots = [histories[b][lengths[b]] if lengths[b] < len(histories[b]) else pending[b]
             for b in range(e.batch)]
    inputs = tuple(tuple([roots[b]]+[prompts[b][-j] for j in range(1,graph.width)])
                   for b in range(e.batch))
    owner = Temporaries(c)
    try:
        saved = [(owner.hold(k[:,:,e.prompt:e.prompt+3].clone()),
                  owner.hold(v[:,:,e.prompt:e.prompt+3].clone())) for k,v in zip(e.keys,e.values)]
        graph.replay(inputs,lengths,active)
        c.drain()
        inactive_zero(graph,active)
        if not torch.isfinite(graph.logits).all():
            raise CandidateRejected("divergent nonfinite logits")
        graph.discard()
        all_logits = graph.logits.view(e.batch,graph.width,-1)
        for base in sorted(set(lengths)):
            for (k,v),(ok,ov) in zip(zip(e.keys,e.values),saved):
                k[:,:,e.prompt:e.prompt+3].copy_(ok)
                v[:,:,e.prompt:e.prompt+3].copy_(ov)
            for row in range(graph.width):
                c.live('numerics',5.)
                e.position.fill_(base+row)
                e.ids.copy_(torch.tensor([x[row] for x in inputs],dtype=torch.int64),non_blocking=False)
                e._step()
                selected = [b for b in range(e.batch) if lengths[b] == base and row < active[b]]
                if selected:
                    logits(all_logits[selected,row],e.logits[selected])
                    for sk,sv,k,v in zip(graph.keys,graph.values,e.keys,e.values):
                        close(sk[selected,:,row],k[selected,:,base+row],'divergent K')
                        close(sv[selected,:,row],v[selected,:,base+row],'divergent V')
    finally:
        owner.close()
    # Rebuild real prefixes before main-cache mutation checks.
    prefix(e,prompts,e.prompt+3)
    counts = [a if graph.width == 1 else min(a,b%5) for b,a in enumerate(active)]
    full_guard(e,graph,inputs,lengths,active,counts)


def validate_numerics(e,graphs,prompts):
    c = e.suffix_control
    four,one = graphs[4],graphs[1]
    first = restore(e,prompts)
    inputs = tuple(tuple([first[b]]+[prompts[b][-j] for j in range(1,4)]) for b in range(e.batch))
    serial_case(e,four,inputs,e.prompt,[4]*e.batch)
    count_sets = [[min(4,b%5) for b in range(e.batch)]] if e.batch > 1 else [[0],[1],[4]]
    for counts in count_sets:
        restore(e,prompts)
        full_guard(e,four,inputs,[e.prompt]*e.batch,[4]*e.batch,counts)
    restore(e,prompts)
    poison_future(e,four,inputs,[e.prompt]*e.batch,[4]*e.batch)
    for graph in (one,four):
        divergent_case(e,graph,prompts)
    _,pending = prefix(e,prompts,e.capacity-4)
    late = tuple(tuple([pending[b]]+list(inputs[b][1:])) for b in range(e.batch))
    tail_sets = [[1+b%4 for b in range(e.batch)]] if e.batch > 1 else [[1],[2],[3],[4]]
    for tails in tail_sets:
        prefix(e,prompts,e.capacity-4)
        serial_case(e,four,late,e.capacity-4,tails)
        full_guard(e,four,late,[e.capacity-4]*e.batch,tails,tails)
    inactive = tuple((y,) for y in pending)
    full_guard(e,one,inactive,[e.capacity-1]*e.batch,[0]*e.batch,[0]*e.batch)
    if not torch.isfinite(one.logits).all():
        raise CandidateRejected("inactive nonfinite logits")
    inactive_zero(one,[0]*e.batch)
    # Distinct request writes must not leak between prompts.
    restore(e,prompts)
    e._memory_guard()
    owner = Temporaries(c)
    try:
        saved = [(owner.hold(k[:,:,:e.prompt].clone()),owner.hold(v[:,:,:e.prompt].clone()))
                 for k,v in zip(e.keys,e.values)]
        alternate = [list(reversed(p)) for p in prompts]
        for b,p in enumerate(prompts):
            if alternate[b] == p:
                alternate[b][0] = (p[0]+1)%e.model.config.vocab_size
        current = restore(e,alternate)
        if all(bits(k[:,:,:e.prompt],old[0]) and bits(v[:,:,:e.prompt],old[1])
               for k,v,old in zip(e.keys,e.values,saved)):
            raise CandidateRejected("distinct prompt did not rewrite KV")
        serial_case(e,one,tuple((y,) for y in current),e.prompt,[1]*e.batch)
        restore(e,prompts)
        if any(not bits(k[:,:,:e.prompt],old[0]) or not bits(v[:,:,:e.prompt],old[1])
               for k,v,old in zip(e.keys,e.values,saved)):
            raise CandidateRejected("prompt restoration differs")
    finally:
        owner.close()


def measure_prices(e,graphs,prompts,output,p0,d0):
    c = e.suffix_control
    one_times,four_times,teardowns,pairs = [],[],[],[]
    for base in dict.fromkeys((e.prompt,e.capacity-4)):
        local_one,local_native = [],[]
        for _ in range(2):
            for width,times in ((1,one_times),(4,four_times)):
                histories,pending = prefix(e,prompts,base)
                state = RequestState(histories,5,e.model.config.vocab_size)
                state.bind_first(pending)
                # Drop aliases before timed destruction; SAM owns its copy.
                histories = pending = None
                if width == 1:
                    state.disable_wide()
                c.live('price',5.)
                at = time.perf_counter()
                # Include actual policy arithmetic, with ample fixed fixture
                # allowance; force width only in this non-emitting price probe.
                fixture_policy = DebtPolicy(5,Prices(p0,d0,d0,d0))
                fixture_policy.allow(0.,state.produced)
                plan = state.next_plan(width)
                graph = graphs[width]
                graph.replay(plan.inputs,plan.lengths,plan.active)
                ys = graph.output.tolist()
                counts = state.commit(plan,ys)
                graph.compact(counts)
                state.take_rows()
                c.drain()
                elapsed = time.perf_counter()-at
                times.append(elapsed)
                if width == 1:
                    local_one.append(elapsed)
                c.observed('price',elapsed)
                # Price maximal compaction without treating hypotheses as
                # accepted greedy outputs. Next fixture restores actual KV.
                if width == 4:
                    graph.replay(plan.inputs,plan.lengths,plan.active)
                    graph.output.tolist()
                    at = time.perf_counter()
                    graph.compact(plan.active)
                    c.drain()
                    # Conservative extra charge: retain the entire measured
                    # host/model round and add one full-count compaction.
                    four_times.append(elapsed+time.perf_counter()-at)
                plan = ys = counts = None
                at = time.perf_counter()
                state.dispose()
                state = None
                teardowns.append(time.perf_counter()-at)
            prefix(e,prompts,base)
            c.live('native_price',5.)
            chunk = native_four(e)
            at = time.perf_counter()
            chunk.graphs[4].replay()
            chunk.outputs[4].tolist()
            c.drain()
            elapsed = time.perf_counter()-at
            local_native.append(elapsed/4)
            c.observed('native_price',elapsed)
        pairs.append((max(local_one),min(local_native)))
    c1,c4 = max(one_times),max(four_times)
    if any(one > 1.04*native for one,native in pairs):
        raise CandidateRejected("W1 matched-endpoint price exceeds ceiling")
    if c4 >= 3.5*c1:
        raise CandidateRejected("W4 price exceeds width ceiling")
    return Prices(p0,d0,c1,c4),max(teardowns)
