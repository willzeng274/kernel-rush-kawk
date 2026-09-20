"""Strict full-vector/all-layer checks and paid complete-round prices."""
import time
from dataclasses import replace
import torch
from lookahead_host import Request, PARENTS, DEPTHS, ancestry_path, DebtPolicy
from lookahead_lifetime import CandidateRejected, Temporaries


def close(actual, expected, name):
    if (not torch.isfinite(actual).all() or not torch.isfinite(expected).all()
            or not torch.allclose(actual, expected, rtol=.02, atol=.03)):
        raise CandidateRejected("numerical mismatch: " + name)


def logits(actual, expected):
    close(actual, expected, "full logits")
    if not torch.equal(actual.argmax(-1), expected.argmax(-1)):
        raise CandidateRejected("same-prefix argmax mismatch")


def bits(actual, expected):
    return torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def restore(e, prompts):
    """Fresh prefill; the returned first IDs are current-call results only."""
    control = e.la_control
    control.healthy()
    started = time.perf_counter()
    e.prefill_input.copy_(torch.tensor(prompts, dtype=torch.int64), non_blocking=False)
    e.prefill_graph.replay()
    e.position.fill_(e.prompt)
    control.drain()
    first = e.ids.tolist()
    elapsed = time.perf_counter() - started
    # Recovery restoration must complete even if the optional deadline expired.
    control.restore = max(control.restore, elapsed)
    if control.base:
        control.reserve()
    return first


def prefix(e, prompts, target, with_history=False):
    pending = restore(e, prompts)
    histories = [list(p) for p in prompts] if with_history else None
    for pos in range(e.prompt, target):
        e.la_control.live("numerics", 5.0)
        if histories is not None:
            for h, token in zip(histories, e.ids.tolist()):
                h.append(token)
        e._step()
    e.la_control.drain()
    pending = e.ids.tolist()
    return (pending, histories) if with_history else pending


def full_guard(e, graph, inputs, lengths, active, paths=None):
    """Check every main-cache byte outside the consumed path, every layer."""
    c = e.la_control
    c.live("numerics", 5.0)
    e._memory_guard()
    owner = Temporaries(c)
    try:
        before = [(owner.hold(k.clone()), owner.hold(v.clone())) for k, v in zip(e.keys, e.values)]
        graph.replay(inputs, lengths, active)
        c.drain()
        if paths is not None:
            graph.compact(paths)
            c.drain()
        for layer, (k, v, sk, sv) in enumerate(zip(e.keys, e.values, graph.keys, graph.values)):
            c.live()
            ek, ev = before[layer]
            expected_paths = paths if paths is not None else tuple((0,) if a else () for a in active)
            if graph.width == 12 and paths is None:
                expected_paths = [()] * e.batch
            for b, path in enumerate(expected_paths):
                for offset, row in enumerate(path):
                    ek[b, :, lengths[b] + offset].copy_(sk[b, :, row])
                    ev[b, :, lengths[b] + offset].copy_(sv[b, :, row])
            if not bits(k, ek) or not bits(v, ev):
                raise CandidateRejected("off-path main KV write")
    finally:
        owner.close()
    e._memory_guard()


def compare_case(e, graph, nodes, lengths, active):
    """Verify each active node on its own complete causal prefix.

    Main KV before min(lengths) is restored by caller. Divergent lengths must
    refer to the same actual serial prefixes already materialized by caller.
    """
    c = e.la_control
    c.live("numerics", 5.0)
    started = time.perf_counter()
    graph.replay(nodes, lengths, active)
    c.drain()
    width = graph.width
    if len(set(lengths)) != 1:
        raise ValueError("batched branch reference requires a common base")
    if not torch.isfinite(graph.logits).all():
        raise CandidateRejected("nonfinite verifier logits")
    for row in range(width):
        c.live()
        selected = [b for b in range(e.batch) if active[b] & (1 << row)]
        inactive = [b for b in range(e.batch) if not active[b] & (1 << row)]
        if inactive:
            for sk, sv in zip(graph.keys, graph.values):
                if torch.count_nonzero(sk[inactive, :, row]) or torch.count_nonzero(sv[inactive, :, row]):
                    raise CandidateRejected("inactive scratch is not zero")
        if not selected:
            continue
        path = ancestry_path(row) if width == 12 else (0,)
        for offset, ancestor in enumerate(path):
            c.live()
            e.position.fill_(lengths[0] + offset)
            e.ids.copy_(torch.tensor([nodes[b][ancestor] for b in range(e.batch)], dtype=torch.int64))
            e._step()
        logits(graph.logits.view(e.batch, width, -1)[selected, row], e.logits[selected])
        for sk, sv, k, v in zip(graph.keys, graph.values, e.keys, e.values):
            close(sk[selected, :, row], k[selected, :, lengths[0] + len(path) - 1], "K")
            close(sv[selected, :, row], v[selected, :, lengths[0] + len(path) - 1], "V")
    c.drain()
    c.observed("numerics", time.perf_counter() - started)


def poison_future(e, graph, nodes, lengths, active):
    """Paired exact replay: unused main-cache capacity must never be read."""
    c = e.la_control
    c.live("numerics", 5.0)
    e._memory_guard()
    owner = Temporaries(c)
    try:
        main = [(owner.hold(k.clone()), owner.hold(v.clone())) for k, v in zip(e.keys, e.values)]
        graph.replay(nodes, lengths, active)
        c.drain()
        old_logits = owner.hold(graph.logits.clone())
        scratch = [(owner.hold(k.clone()), owner.hold(v.clone()))
                   for k, v in zip(graph.keys, graph.values)]
        for k, v in zip(e.keys, e.values):
            c.live()
            for b, length in enumerate(lengths):
                k[b, :, length:].fill_(123.)
                v[b, :, length:].fill_(-123.)
        graph.replay(nodes, lengths, active)
        c.drain()
        logits(graph.logits, old_logits)
        for (k, v), (old_k, old_v) in zip(zip(graph.keys, graph.values), scratch):
            close(k, old_k, "future-masked K")
            close(v, old_v, "future-masked V")
    finally:
        # Main may still contain uninitialized NaN bytes; preserve storage bits
        # with copies rather than comparing float values or normalizing tails.
        if not c.poisoned and "main" in locals():
            for (k, v), (old_k, old_v) in zip(zip(e.keys, e.values), main):
                k.copy_(old_k); v.copy_(old_v)
        owner.close()
    c.live()


def validate(e, graphs, prompts, output):
    c, wide, one = e.la_control, graphs[12], graphs[1]
    # Each batch lane has its own prompt; every hypothetical token is from it.
    nodes = tuple(tuple(p[-1 - j % len(p)] for j in range(12)) for p in prompts)
    first = restore(e, prompts)
    poison_inputs = tuple((first[b],) + nodes[b][1:] for b in range(e.batch))
    poison_future(e, wide, poison_inputs, [e.prompt] * e.batch, [4095] * e.batch)
    for base in dict.fromkeys((e.prompt, e.capacity - 5)):
        first = prefix(e, prompts, base)
        inputs = tuple((first[b],) + nodes[b][1:] for b in range(e.batch))
        # Test full steady, paid fill, every tail mask, no candidates/root-only,
        # completed lanes. Rebuild real base before each serial comparison.
        masks = (4095, 143, 255, 1023, 1, 1281, 769, 3841, 0)
        for mask in masks:
            prefix(e, prompts, base)
            compare_case(e, wide, inputs, [base] * e.batch, [mask] * e.batch)
        # Shared first candidate inputs retain separate ancestry and second IDs.
        shared = tuple(row[:10] + (row[8], row[11]) for row in inputs)
        prefix(e, prompts, base)
        compare_case(e, wide, shared, [base] * e.batch, [4095] * e.batch)
        # Every permitted compaction route, including no writes.
        for path in ((), (0,), (0, 8), (0, 8, 9), (0, 10), (0, 10, 11)):
            prefix(e, prompts, base)
            full_guard(e, wide, inputs, [base] * e.batch, [4095] * e.batch,
                       [path] * e.batch)
        # W1 independent length/active semantics: generate genuine serial KV,
        # use earlier positions in it, and compare against the same prefix.
        pending = prefix(e, prompts, base + 3)
        lengths = [base + b % 3 for b in range(e.batch)]
        singles = tuple((pending[b],) for b in range(e.batch))
        active = [0 if e.batch > 1 and b == e.batch - 1 else 1 for b in range(e.batch)]
        full_guard(e, one, singles, lengths, active)
        # Lane-wise reference above would overwrite other divergent prefixes;
        # clone the initialized prefix and restore it for each target lane.
        owner = Temporaries(c)
        try:
            e._memory_guard()
            saved = [(owner.hold(k.clone()), owner.hold(v.clone())) for k, v in zip(e.keys, e.values)]
            one.replay(singles, lengths, active)
            c.drain()
            for b in range(e.batch):
                c.live()
                if not active[b]:
                    continue
                for (k, v), (sk, sv) in zip(zip(e.keys, e.values), saved):
                    k.copy_(sk); v.copy_(sv)
                e.position.fill_(lengths[b]); e.ids.fill_(singles[b][0]); e._step()
                logits(one.logits[b], e.logits[b])
                for sk, sv, k, v in zip(one.keys, one.values, e.keys, e.values):
                    close(sk[b, :, 0], k[b, :, lengths[b]], "divergent K")
                    close(sv[b, :, 0], v[b, :, lengths[b]], "divergent V")
        finally:
            owner.close()
        full_guard(e, one, singles, [e.capacity - 1] * e.batch, [0] * e.batch)
        # W12 also consumes vector lengths. Independent W1 alone cannot catch
        # an accidental scalar base in W12 attention/RoPE. All referenced past
        # slots are actual serial tokens, restored before every target path.
        pending = prefix(e, prompts, base + 3)
        lengths = [base + b % 3 for b in range(e.batch)]
        mixed = [0 if e.batch > 1 and b == e.batch - 1 else 785 for b in range(e.batch)]
        inp = tuple((pending[b],) + nodes[b][1:] for b in range(e.batch))
        e._memory_guard()
        owner = Temporaries(c)
        try:
            saved = [(owner.hold(k.clone()), owner.hold(v.clone())) for k, v in zip(e.keys, e.values)]
            wide.replay(inp, lengths, mixed)
            c.drain()
            for b in range(e.batch):
                for row in range(12):
                    c.live()
                    if not mixed[b] & (1 << row):
                        for sk, sv in zip(wide.keys, wide.values):
                            if torch.count_nonzero(sk[b, :, row]) or torch.count_nonzero(sv[b, :, row]):
                                raise CandidateRejected("divergent inactive scratch")
                        continue
                    for (k, v), (sk, sv) in zip(zip(e.keys, e.values), saved):
                        k.copy_(sk); v.copy_(sv)
                    path = ancestry_path(row)
                    for offset, j in enumerate(path):
                        e.position.fill_(lengths[b] + offset); e.ids.fill_(inp[b][j]); e._step()
                    logits(wide.logits.view(e.batch, 12, -1)[b, row], e.logits[b])
                    for sk, sv, k, v in zip(wide.keys, wide.values, e.keys, e.values):
                        close(sk[b, :, row], k[b, :, lengths[b] + len(path) - 1], "divergent W12 K")
                        close(sv[b, :, row], v[b, :, lengths[b] + len(path) - 1], "divergent W12 V")
        finally:
            owner.close()
        e._memory_guard()
    # Consecutive distinct requests must rewrite prompt KV, then restoring the
    # original must reproduce it bit-for-bit. No generated trial IDs are reused.
    restore(e, prompts)
    e._memory_guard()
    owner = Temporaries(c)
    try:
        saved = [(owner.hold(k[:, :, :e.prompt].clone()), owner.hold(v[:, :, :e.prompt].clone()))
                 for k, v in zip(e.keys, e.values)]
        alternate = [list(reversed(p)) for p in prompts]
        for b, p in enumerate(prompts):
            if alternate[b] == p:
                alternate[b][0] = (p[0] + 1) % e.model.config.vocab_size
        fresh_first = restore(e, alternate)
        if all(bits(k[:, :, :e.prompt], old[0]) and bits(v[:, :, :e.prompt], old[1])
               for k, v, old in zip(e.keys, e.values, saved)):
            raise CandidateRejected("distinct request did not change prompt KV")
        compare_case(e, one, tuple((y,) for y in fresh_first), [e.prompt] * e.batch, [1] * e.batch)
        restore(e, prompts)
        if any(not bits(k[:, :, :e.prompt], old[0]) or not bits(v[:, :, :e.prompt], old[1])
               for k, v, old in zip(e.keys, e.values, saved)):
            raise CandidateRejected("fresh request prefix reset mismatch")
    finally:
        owner.close()


def prices(e, graphs, prompts, output, d0):
    """Full active W12 round at both endpoints, including host and compaction."""
    c = e.la_control
    wide_times, one_times = [], []
    for base in dict.fromkeys((e.prompt, e.capacity - 5)):
        for width in (1, 12):
            for _ in range(2):
                first, histories = prefix(e, prompts, base, with_history=True)
                state = Request(histories, first, output, e.model.config.vocab_size)
                if width == 1:
                    state.disable_wide()
                else:
                    # A pricing fixture, never emitted: all twelve genuine
                    # logical rows active, including both verification branches.
                    state.upper = [tuple(p[-1 - j % len(p)] for j in range(4)) for p in prompts]
                    state.lower = [u[1:] for u in state.upper]
                c.live("price", 5.0)
                started = time.perf_counter()
                plan = state.next_plan()
                if width == 12:
                    pairs = tuple(((u[0], u[1]), (u[2], u[3])) for u in state.upper)
                    inp = tuple(row[:8] + pairs[b][0] + pairs[b][1] for b, row in enumerate(plan.inputs))
                    plan = replace(plan, inputs=inp, active=(4095,) * e.batch, candidates=pairs)
                    state.outstanding = plan
                    graphs[width].replay(plan.inputs, plan.lengths, plan.active)
                    ys = graphs[width].output.tolist()
                else:
                    graphs[width].replay(tuple((f,) for f in first), [base] * e.batch, [1] * e.batch)
                    ys = [[row[0]] + [-1] * 11 for row in graphs[width].output.tolist()]
                paths = state.commit(plan, ys)
                graphs[width].compact(paths)
                state.take_rows()
                c.drain()
                elapsed = time.perf_counter() - started
                (wide_times if width == 12 else one_times).append(elapsed)
                c.observed("price", elapsed)
    if max(one_times) > 1.04 * d0:
        raise CandidateRejected("independent W1 price exceeds retained ceiling")
    started = time.perf_counter()
    first = restore(e, prompts)
    # Restore is setup; the separately measured initialization starts below.
    started = time.perf_counter()
    Request(prompts, first, output, e.model.config.vocab_size)
    init = time.perf_counter() - started
    costs = (d0, max(wide_times))
    if DebtPolicy(output, *costs, init).disabled:
        raise CandidateRejected("three paid startup misses do not fit fixed budget")
    return costs
