"""Fail-closed tree validation against every own-prefix causal branch."""
import time
import torch
from recycled_validate import (CandidateRejected, _live, _close, _same_bits,
                               _logits, _direct_replay_checked, _prefix_to)
from tree_host import TreeState, DEPTHS
from top2_validate import check_model_top2, validate_top2_helper

PATHS = ((), (0,), (0, 1), (0, 1, 2), (0, 1, 2, 3), (0, 1, 2, 3, 4),
         (0, 5), (0, 5, 6), (0, 5, 6, 7))


def _serial_compare(e, graph, inputs, base, active, deadline):
    b = e.batch
    graph.replay(inputs, [base] * b, active)
    torch.cuda.synchronize()
    check_model_top2(graph)
    logits = graph.logits.view(b, 8, -1)
    # Restart causal input position at the root for each branch. Serial calls
    # rewrite all preceding branch slots, so the alternative never reads the
    # primary branch's rejected K/V. Committed slots below base stay untouched.
    for path in ((0, 1, 2, 3, 4), (0, 5, 6, 7)):
        for depth, row in enumerate(path):
            _live(deadline)
            if base + depth >= e.capacity:
                continue
            e.position.fill_(base + depth)
            e.ids.copy_(torch.tensor([x[row] for x in inputs], device=e.ids.device))
            e._step()
            selected = [j for j in range(b) if active[j] & (1 << row)]
            if selected:
                _logits(logits[selected, row], e.logits[selected])
                for sk, sv, k, v in zip(graph.keys, graph.values, e.keys, e.values):
                    _close(sk[selected, :, row], k[selected, :, base + depth], "tree K")
                    _close(sv[selected, :, row], v[selected, :, base + depth], "tree V")
    for j in range(b):
        for row in range(8):
            if not active[j] & (1 << row):
                for sk, sv in zip(graph.keys, graph.values):
                    if torch.count_nonzero(sk[j, :, row]) or torch.count_nonzero(sv[j, :, row]):
                        raise CandidateRejected("inactive tree scratch row was not zeroed")
    if not torch.isfinite(logits).all():
        raise CandidateRejected("tree inactive logits are not finite")


def _compaction(e, graph, base, deadline):
    start, stop = max(0, base - 1), min(e.capacity, base + 6)
    original = [(k[:, :, start:stop].clone(), v[:, :, start:stop].clone())
                for k, v in zip(e.keys, e.values)]
    scratch = [(k.clone(), v.clone()) for k, v in zip(graph.keys, graph.values)]
    active = graph.meta[:, 9].tolist()
    for proposed in PATHS:
        _live(deadline)
        paths = [tuple(row for row in proposed if active[b] & (1 << row))
                 for b in range(e.batch)]
        # Inactive masks are ancestor closed, so filtering any valid branch
        # path yields its prefix. Poison every rejected scratch node explicitly.
        for (sk, sv), (old_k, old_v) in zip(zip(graph.keys, graph.values), scratch):
            sk.copy_(old_k)
            sv.copy_(old_v)
            for b, path in enumerate(paths):
                for row in range(8):
                    if row not in path:
                        sk[b, :, row].fill_(123.0 + row)
                        sv[b, :, row].fill_(-123.0 - row)
        for k, v in zip(e.keys, e.values):
            k[:, :, base:stop].fill_(456.0)
            v[:, :, base:stop].fill_(-456.0)
        expected = [(k[:, :, start:stop].clone(), v[:, :, start:stop].clone())
                    for k, v in zip(e.keys, e.values)]
        graph.compact(paths)
        torch.cuda.synchronize()
        for layer, (k, v, sk, sv) in enumerate(zip(e.keys, e.values, graph.keys, graph.values)):
            for b, path in enumerate(paths):
                expected_k, expected_v = expected[layer][0][b], expected[layer][1][b]
                for rank, row in enumerate(path):
                    expected_k[:, base - start + rank].copy_(sk[b, :, row])
                    expected_v[:, base - start + rank].copy_(sv[b, :, row])
                if (not _same_bits(k[b, :, start:stop], expected_k) or
                        not _same_bits(v[b, :, start:stop], expected_v)):
                    raise CandidateRejected("tree visited-path compaction or boundary mismatch")
    for (sk, sv), (old_k, old_v) in zip(zip(graph.keys, graph.values), scratch):
        sk.copy_(old_k)
        sv.copy_(old_v)
    for (k, v), (old_k, old_v) in zip(zip(e.keys, e.values), original):
        k[:, :, start:stop].copy_(old_k)
        v[:, :, start:stop].copy_(old_v)
    torch.cuda.synchronize()


def _divergent_one(e, graph, inputs, base, deadline):
    # The alternative serial path was most recently installed by comparison.
    # Recreate the primary prefix so each divergent W1 row has a known context.
    b = e.batch
    for depth in range(5):
        _live(deadline)
        e.position.fill_(base + depth)
        e.ids.copy_(torch.tensor([x[depth] for x in inputs], device=e.ids.device))
        e._step()
    offsets = [j % 5 for j in range(b)]
    ones = tuple((inputs[j][offsets[j]],) for j in range(b))
    active = [0 if b > 1 and j == b - 1 else 1 for j in range(b)]
    _direct_replay_checked(e, graph, ones, [base + d for d in offsets], active)
    for row in range(5):
        _live(deadline)
        e.position.fill_(base + row)
        e.ids.copy_(torch.tensor([x[row] for x in inputs], device=e.ids.device))
        e._step()
        selected = [j for j, d in enumerate(offsets) if d == row and active[j]]
        if selected:
            _logits(graph.logits[selected], e.logits[selected])
            for sk, sv, k, v in zip(graph.keys, graph.values, e.keys, e.values):
                _close(sk[selected, :, 0], k[selected, :, base + row], "divergent W1 K")
                _close(sv[selected, :, 0], v[selected, :, base + row], "divergent W1 V")


def validate_tree_numerics(e, graphs, input_ids, first, deadline):
    b, prompt = e.batch, e.prompt
    tree, one = graphs[8], graphs[1]
    validate_top2_helper(tree, deadline)
    first_host = first.tolist()
    inputs = tuple(tuple([first_host[j]] + [input_ids[j][-(r % prompt + 1)] for r in range(7)])
                   for j in range(b))
    _serial_compare(e, tree, inputs, prompt, [255] * b, deadline)
    _compaction(e, tree, prompt, deadline)
    _divergent_one(e, one, inputs, prompt, deadline)
    # Every tail depth 1..5 is exercised even for B1. This also exercises
    # alternative slots physically beyond CAP's virtual prefix index near end.
    for remaining in range(1, 6):
        _live(deadline, 2.0)
        base = e.capacity - remaining
        _, pending = _prefix_to(e, input_ids, first, base, deadline)
        late = tuple(tuple([pending[j]] + list(inputs[j][1:])) for j in range(b))
        active = [sum(1 << row for row, depth in enumerate(DEPTHS)
                      if depth < max(0, remaining - (j % 3))) for j in range(b)]
        _serial_compare(e, tree, late, base, active, deadline)
        _compaction(e, tree, base, deadline)
    # No finished member reads an invalid RoPE position or changes main KV.
    _direct_replay_checked(e, one, tuple((x,) for x in first_host),
                           [e.capacity] * b, [0] * b)
    if not torch.isfinite(one.logits).all():
        raise CandidateRejected("inactive W1 numerical state is not finite")
    for sk, sv in zip(one.keys, one.values):
        if torch.count_nonzero(sk) or torch.count_nonzero(sv):
            raise CandidateRejected("finished W1 members wrote scratch cache")
    tree.replay(inputs, [e.capacity] * b, [0] * b)
    torch.cuda.synchronize()
    if not torch.isfinite(tree.logits).all():
        raise CandidateRejected("finished tree numerical state is not finite")
    check_model_top2(tree)
    for sk, sv in zip(tree.keys, tree.values):
        if torch.count_nonzero(sk) or torch.count_nonzero(sv):
            raise CandidateRejected("finished tree members wrote scratch cache")
    e.prefill_graph.replay()
    e.position.fill_(prompt)
    e.ids.copy_(first)
    torch.cuda.synchronize()


def _compact(graph, paths):
    graph.compact(tuple(len(path) for path in paths) if graph.width == 1 else paths)


def _commit_output(state, plan, graph):
    if plan.width == 8:
        # One synchronization/transfer replaces the old argmax-only transfer.
        # Column zero is the original torch.argmax copied by the helper.
        ranked = graph.proposal_view.tolist()
        predictions = [[pair[0] for pair in rows] for rows in ranked]
        return state.commit(plan, predictions, ranked=ranked)
    return state.commit(plan, graph.output.tolist())


def run_tree_request(e, graphs, input_ids, first_row, output, one_cost, tree_cost):
    state = TreeState(input_ids, first_row, output, one_cost, tree_cost)
    plan = state.next_plan()
    if plan is not None:
        graphs[plan.width].replay(plan.inputs, plan.lengths, plan.active)
    while plan is not None:
        graph = graphs[plan.width]
        paths = _commit_output(state, plan, graph)
        _compact(graph, paths)
        rows = state.take_rows()
        following = state.next_plan()
        if following is not None:
            graphs[following.width].replay(following.inputs, following.lengths, following.active)
        else:
            torch.cuda.current_stream().synchronize()
        yield from rows
        plan = following


def measure_tree_and_admit(e, graphs, input_ids, first, output, deadline):
    _live(deadline, 8.0)
    first_host = first.tolist()
    one_times, tree_times, endpoint_pairs = [], [], []
    for base in dict.fromkeys((e.prompt, e.capacity - 5)):
        endpoint_one, endpoint_native = [], []
        for _ in range(3):
            for width, times in ((1, one_times), (8, tree_times)):
                _live(deadline)
                prefix, pending = _prefix_to(e, input_ids, first, base, deadline)
                state = TreeState(prefix, pending, 6)
                start = time.perf_counter()
                plan = state.next_plan(force_width=width)
                graph = graphs[width]
                graph.replay(plan.inputs, plan.lengths, plan.active)
                paths = _commit_output(state, plan, graph)
                _compact(graph, paths)
                state.take_rows()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                if width == 1:
                    endpoint_one.append(elapsed)
            _prefix_to(e, input_ids, first, base, deadline)
            start = time.perf_counter()
            e.chunks.graphs[4].replay()
            e.chunks.outputs[4].tolist()
            endpoint_native.append((time.perf_counter() - start) / 4)
        endpoint_pairs.append((max(endpoint_one), min(endpoint_native)))
    cost_one, cost_tree = max(one_times), max(tree_times)
    if (any(one > native * 1.04 for one, native in endpoint_pairs)
            or cost_tree >= cost_one * 4.5):
        return None
    # Refuse an impossible first attempt explicitly. Request initialization is
    # measured in each TreeState and added to its debt before issuing a plan.
    probe = TreeState(input_ids, first_host, output, cost_one, cost_tree)
    if probe.next_plan().width != 8:
        return None
    elapsed_native, elapsed_candidate = [], []
    for _ in range(2):
        _live(deadline, 3.0)
        e.prefill_graph.replay()
        e.position.fill_(e.prompt)
        e.ids.copy_(first)
        torch.cuda.synchronize()
        start = time.perf_counter()
        expected = list(e.chunks.generate())
        elapsed_native.append(time.perf_counter() - start)
        e.prefill_graph.replay()
        torch.cuda.synchronize()
        start = time.perf_counter()
        actual = list(run_tree_request(e, graphs, input_ids, first_host, output, cost_one, cost_tree))
        elapsed_candidate.append(time.perf_counter() - start)
        if actual != expected:
            raise CandidateRejected("complete tree request differs from serial greedy reference")
    if max(elapsed_candidate) >= min(elapsed_native) * 0.94:
        return None
    return cost_one, cost_tree
