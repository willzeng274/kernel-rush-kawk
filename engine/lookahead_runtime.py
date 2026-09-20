"""Paid request-local trajectory loop and whole-call timing helpers."""
import time
import torch
from lookahead_host import Request, DebtPolicy
from lookahead_lifetime import CandidateRejected


def request(e, prompts, output, costs, evidence=None):
    """Recompute prefill and every output; never reuse setup-trial answers."""
    control = e.la_control
    control.healthy()
    try:
        # CPU temporary is consumed by a blocking copy, with no host-DMA race.
        e.prefill_input.copy_(torch.tensor(prompts, dtype=torch.int64), non_blocking=False)
        e.prefill_graph.replay()
        first = e.ids.tolist()
        yield first
        if output <= 1:
            return
        started = time.perf_counter()  # BEFORE request/pool initialization.
        state = Request(prompts, first, output, e.model.config.vocab_size,
                        trace=evidence is not None)
        policy = DebtPolicy(output, costs[0], costs[1], time.perf_counter() - started)
        while min(state.produced) < output:
            remaining = [output - p for p in state.produced]
            wants_wide = not state.fallback and any(
                r >= 5 or (r >= 2 and state.upper[b] is not None)
                for b, r in enumerate(remaining))
            if wants_wide and not policy.allow(time.perf_counter() - started, state.produced):
                state.disable_wide()
            plan = state.next_plan()
            width = 12 if any(m in ("fill", "steady", "tail") for m in plan.modes) else 1
            graph = e.la_graphs[width]
            if width == 12:
                graph.replay(plan.inputs, plan.lengths, plan.active)
                ys = graph.output.tolist()
            else:
                graph.replay(tuple((row[0],) for row in plan.inputs), plan.lengths,
                             tuple(mask & 1 for mask in plan.active))
                ys = [[row[0]] + [-1] * 11 for row in graph.output.tolist()]
            paths = state.commit(plan, ys)
            graph.compact(paths)
            rows = state.take_rows()
            # No next-round replay is launched ahead of output delivery. A
            # final drain precedes the last row, and finally covers early close.
            if state.emitted == output:
                control.drain()
            yield from (list(row) for row in rows)
        if evidence is not None:
            evidence.update(state.trace)
    finally:
        control.drain()


def timed_complete(e, factory, phase):
    control = e.la_control
    control.live(phase, 5.0)
    control.drain()
    started, first_elapsed, rows = time.perf_counter(), None, []
    gen = factory()
    try:
        for row in gen:
            if first_elapsed is None:
                first_elapsed = time.perf_counter() - started
            rows.append(tuple(row))
            control.live()  # Stop before another warmup-trial generator advance.
    finally:
        gen.close()
        control.drain()
    elapsed = time.perf_counter() - started
    control.observed(phase, elapsed)
    return tuple(rows), elapsed, elapsed - (first_elapsed or elapsed)


def require_stream(rows, batch, output, vocab):
    if len(rows) != output or any(len(row) != batch for row in rows):
        raise CandidateRejected("complete stream shape mismatch")
    if any(type(y) is not int or not 0 <= y < vocab for row in rows for y in row):
        raise CandidateRejected("complete stream invalid token")


def whole_call_admission(e, prompts, output, costs, retained_factory):
    """ABBA: true retained generator, candidate, candidate, true retained."""
    traces = [{}, {}]
    a0 = timed_complete(e, retained_factory, "base")
    b0 = timed_complete(e, lambda: request(e, prompts, output, costs, traces[0]), "candidate")
    b1 = timed_complete(e, lambda: request(e, prompts, output, costs, traces[1]), "candidate")
    a1 = timed_complete(e, retained_factory, "base")
    for run in (a0, b0, b1, a1):
        require_stream(run[0], e.batch, output, e.model.config.vocab_size)
    if not (a0[0] == a1[0] == b0[0] == b1[0]):
        raise CandidateRejected("complete greedy streams differ")
    if any(not (t.get("fill") and t.get("steady") and t.get("trajectory_selected")) for t in traces):
        raise CandidateRejected("paid trajectory mechanism not executed")
    if max(b0[1], b1[1]) >= .94 * min(a0[1], a1[1]):
        raise CandidateRejected("complete request gain insufficient")
    # The policy used in the trials must not later receive cheaper or more
    # permissive prices. Keep the original conservative fixed pair unchanged.
    return costs
