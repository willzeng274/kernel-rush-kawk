"""Paid fresh request execution and actual retained #32 whole-call admission."""
import time
from dataclasses import replace
import torch
from recycled_host import RequestState, DebtPolicy
from suffix_lifetime import CandidateRejected


def request(e, prompts, output, prices, evidence=None, use_suffix=True, initialization=None):
    c, state = e.suffix_control, None
    c.healthy()
    try:
        started = time.perf_counter()
        e.prefill_input.copy_(torch.tensor(prompts,dtype=torch.int64),non_blocking=False)
        e.prefill_graph.replay()
        # Fresh CPU indices overlap only the current complete prefill replay.
        state = RequestState(prompts,output,e.model.config.vocab_size,
                             use_suffix=use_suffix,trace=evidence is not None)
        first = e.ids.tolist()
        state.bind_first(first)
        first_at = time.perf_counter()
        setup_debt = max(0.,first_at-started-prices.p0)
        if output == 1:
            c.drain()
            state.dispose()
            state = None
        yield first
        if output <= 1:
            return
        policy = DebtPolicy(output,prices)
        while min(state.produced) < output:
            # This exact endpoint includes first-yield/resume bookkeeping.
            boundary = time.perf_counter()
            elapsed = setup_debt+boundary-first_at
            if initialization is not None:
                initialization['initial'] = elapsed
                return  # Probe never drafts or verifies a speculative row.
            width = 4
            if state.fallback or not policy.allow(elapsed,state.produced):
                if not state.fallback:
                    state.disable_wide()
                width = 1
            # One pre-draft price reservation only. Never subtract actual host
            # time from the frozen device price or refill allowance after a miss.
            plan = state.next_plan(width)
            graph = e.suffix_graphs[plan.width]
            graph.replay(plan.inputs,plan.lengths,plan.active)
            predictions = graph.output.tolist()
            counts = state.commit(plan,predictions)
            graph.compact(counts)
            rows = state.take_rows()
            final = state.emitted == output
            if final:
                c.drain()
                if evidence is not None:
                    evidence.update(state.trace)
                dispose_at = time.perf_counter()
                state.dispose()
                state = None
                if evidence is not None:
                    evidence['cleanup'] = time.perf_counter()-dispose_at
            yield from rows
            if final:
                return
    finally:
        # Early close, output-consumer failure and exceptions preserve owners.
        # A failed drain poisons the engine; no further request is permitted.
        try:
            c.drain()
        finally:
            # CPU proposal structures are never DMA sources; graph owners stay
            # registered even on a failed drain. CPU release is always safe.
            if state is not None:
                at = time.perf_counter()
                state.dispose()
                state = None
                if initialization is not None:
                    initialization['cleanup'] = time.perf_counter()-at


def timed_complete(e, factory, phase):
    c = e.suffix_control
    c.live(phase,5.)
    c.drain()
    started, first, rows = time.perf_counter(), None, []
    gen = factory()
    try:
        for row in gen:
            if first is None:
                first = time.perf_counter()-started
            rows.append(tuple(row))
            c.live()
    finally:
        gen.close()
        c.drain()
    elapsed = time.perf_counter()-started
    c.observed(phase,elapsed)
    return tuple(rows),elapsed,first if first is not None else elapsed


def require_stream(rows,batch,output,vocab):
    if len(rows) != output or any(len(row) != batch for row in rows):
        raise CandidateRejected("complete stream shape mismatch")
    if any(type(y) is not int or not 0 <= y < vocab for row in rows for y in row):
        raise CandidateRejected("complete stream contains invalid ID")


def calibrate_initial(e,prompts,output,prices,cleanup):
    initial, tears = [], [cleanup]
    for _ in range(2):
        c = e.suffix_control
        c.live('initial',5.)
        probe = {}
        # Identical first-row consumer and trace allocation as actual B trials:
        # tuple conversion, append, live check, then generator resume.
        sampled = timed_complete(e,lambda:request(e,prompts,output,prices,
                                  evidence={},initialization=probe),'initial')
        require_stream(sampled[0],e.batch,1,e.model.config.vocab_size)
        initial.append(probe['initial'])
        tears.append(probe['cleanup'])
    return replace(prices,initial=max(initial),cleanup=max(tears))


def whole_call_admission(e,prompts,output,prices,retained_factory):
    """Strict actual retained A/B/B/A; no optional control calls consume reserve."""
    traces = [{},{}]
    a0 = timed_complete(e,retained_factory,'base')
    b0 = timed_complete(e,lambda:request(e,prompts,output,prices,traces[0]),'candidate')
    b1 = timed_complete(e,lambda:request(e,prompts,output,prices,traces[1]),'candidate')
    a1 = timed_complete(e,retained_factory,'base')
    runs = (a0,b0,b1,a1)
    for run in runs:
        require_stream(run[0],e.batch,output,e.model.config.vocab_size)
    if not (a0[0] == b0[0] == b1[0] == a1[0]):
        raise CandidateRejected("complete greedy streams differ")
    if any(not (t['wide'] and t['overrides'] and t['accepted_changed']) for t in traces):
        raise CandidateRejected("paid accepted changed suffix proposal not executed")
    if max(b0[1],b1[1]) >= .94*min(a0[1],a1[1]):
        raise CandidateRejected("actual retained complete-call gain insufficient")
    # C0/C1 are omitted in this first candidate to protect mandatory setup.
    # Passing proves the complete hybrid's benefit, not SAM-only causal gain.
    return prices
