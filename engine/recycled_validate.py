"""Untimed, fail-closed admission of the exact candidate actually executed.

No claim that one warmup predicts acceptance on new prompts or sample spread.
Validation uses retained sequential same-prefix model logits and every layer's
K/V, divergent W1 lengths, active tails, compaction and complete request cost.
"""
import statistics
import time
import torch
from recycled_host import RequestState


class CandidateRejected(Exception):
    """A completed numerical check rejects this optional candidate."""


def _live(deadline, reserve=1.0):
    if time.monotonic() + reserve >= deadline:
        raise TimeoutError("recycling warmup deadline reserve")


def _close(actual, expected, name):
    if (not torch.isfinite(actual).all() or
            not torch.allclose(actual, expected, rtol=0.02, atol=0.03)):
        raise CandidateRejected("recycling numerical mismatch: " + name)


def _same_bits(actual, expected):
    # Uninitialized masked cache tails may contain NaNs; unchanged NaN payloads
    # must compare by storage bits rather than IEEE floating equality.
    return torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def _logits(actual, expected):
    _close(actual, expected, "full logits")
    if not torch.equal(actual.argmax(-1), expected.argmax(-1)):
        raise CandidateRejected("recycling argmax differs from serial same-prefix argmax")


def _serial_compare(e, graph, inputs, base, active, deadline):
    b, w = e.batch, graph.width
    graph.replay(inputs, [base] * b, active)
    torch.cuda.synchronize()
    logits = graph.logits.view(b, w, -1)
    for row in range(w):
        _live(deadline)
        e.position.fill_(base + row)
        e.ids.copy_(torch.tensor([x[row] for x in inputs], device=e.ids.device))
        e._step()
        selected = [j for j in range(b) if row < active[j]]
        if selected:
            _logits(logits[selected, row], e.logits[selected])
            for sk, sv, k, v in zip(graph.keys, graph.values, e.keys, e.values):
                _close(sk[selected, :, row], k[selected, :, base + row], "K")
                _close(sv[selected, :, row], v[selected, :, base + row], "V")
        for j in range(b):
            if row >= active[j]:
                for sk, sv in zip(graph.keys, graph.values):
                    if torch.count_nonzero(sk[j, :, row]) or torch.count_nonzero(sv[j, :, row]):
                        raise CandidateRejected("inactive scratch row was not zeroed")


def _compaction(e, graph, base):
    start, stop = max(0, base - 1), min(e.capacity, base + graph.width + 1)
    original = [(k[:, :, start:stop].clone(), v[:, :, start:stop].clone())
                for k, v in zip(e.keys, e.values)]
    active = graph.meta[:, graph.width + 1].tolist()
    count_sets = [[min(active[b], (b + 1) % (graph.width + 1))
                   for b in range(e.batch)]]
    if e.batch == 1:
        count_sets = [[0], [min(1, active[0])], [active[0]]]
    for counts in count_sets:
        # Poison the uncommitted tail. Otherwise a mistaken rejected-row copy
        # could be invisible when scratch and serial KV happened to be equal.
        for k, v in zip(e.keys, e.values):
            k[:, :, base:base + graph.width].fill_(123.0)
            v[:, :, base:base + graph.width].fill_(-123.0)
        expected = [(k[:, :, start:stop].clone(), v[:, :, start:stop].clone())
                    for k, v in zip(e.keys, e.values)]
        graph.compact(counts)
        torch.cuda.synchronize()
        for layer, (k, v, sk, sv) in enumerate(zip(e.keys, e.values, graph.keys, graph.values)):
            for b, count in enumerate(counts):
                expected_k, expected_v = expected[layer][0][b], expected[layer][1][b]
                expected_k[:, base - start:base - start + count].copy_(sk[b, :, :count])
                expected_v[:, base - start:base - start + count].copy_(sv[b, :, :count])
                if (not _same_bits(k[b, :, start:stop], expected_k) or
                        not _same_bits(v[b, :, start:stop], expected_v)):
                    raise CandidateRejected("accepted-row compaction or boundary guard mismatch")
    # Later divergent-position comparisons need the original teacher-forced KV.
    for (k, v), (old_k, old_v) in zip(zip(e.keys, e.values), original):
        k[:, :, start:stop].copy_(old_k)
        v[:, :, start:stop].copy_(old_v)
    torch.cuda.synchronize()


def _direct_replay_checked(e, graph, inputs, lengths, active):
    # Check the actual W1 direct stores before any serial reference overwrites
    # them. Each sequence may mutate exactly its own active current slot.
    start = max(0, min(lengths) - 1)
    stop = min(e.capacity, max(lengths) + 2)
    before = [(k[:, :, start:stop].clone(), v[:, :, start:stop].clone())
              for k, v in zip(e.keys, e.values)]
    graph.replay(inputs, lengths, active)
    torch.cuda.synchronize()
    for layer, (k, v, sk, sv) in enumerate(zip(e.keys, e.values, graph.keys, graph.values)):
        expected_k, expected_v = before[layer]
        for b, count in enumerate(active):
            if count:
                offset = lengths[b] - start
                expected_k[b, :, offset].copy_(sk[b, :, 0])
                expected_v[b, :, offset].copy_(sv[b, :, 0])
        if (not _same_bits(k[:, :, start:stop], expected_k) or
                not _same_bits(v[:, :, start:stop], expected_v)):
            raise CandidateRejected("W1 direct write changed an unconsumed cache slot")


def validate_numerics(e, graphs, input_ids, first, deadline):
    b, prompt = e.batch, e.prompt
    four, one = graphs[4], graphs[1]
    first_host = first.tolist()
    inputs = tuple(tuple([first_host[j]] + [input_ids[j][-(r + 1)] for r in range(3)])
                   for j in range(b))
    _serial_compare(e, four, inputs, prompt, [4] * b, deadline)
    _compaction(e, four, prompt)
    # Main now contains four teacher-forced rows for each sequence. Different
    # W1 lengths point into the same initialized prefix at different positions.
    offsets = [j % 4 for j in range(b)]
    ones = tuple((inputs[j][offsets[j]],) for j in range(b))
    active = [0 if b > 1 and j == b - 1 else 1 for j in range(b)]
    _direct_replay_checked(e, one, ones, [prompt + d for d in offsets], active)
    for row in range(4):
        _live(deadline)
        e.position.fill_(prompt + row)
        e.ids.copy_(torch.tensor([x[row] for x in inputs], device=e.ids.device))
        e._step()
        selected = [j for j, d in enumerate(offsets) if d == row and active[j]]
        if selected:
            _logits(one.logits[selected], e.logits[selected])
            for sk, sv, k, v in zip(one.keys, one.values, e.keys, e.values):
                _close(sk[selected, :, 0], k[selected, :, prompt + row], "divergent K")
                _close(sv[selected, :, 0], v[selected, :, prompt + row], "divergent V")
    # Independently grow a real serial prefix to the last four safe positions.
    e.prefill_graph.replay()
    e.position.fill_(prompt)
    e.ids.copy_(first)
    target = e.capacity - 4
    position = prompt
    while position + 4 <= target:
        _live(deadline)
        e.chunks.graphs[4].replay()
        position += 4
    while position < target:
        _live(deadline)
        e._step()
        position += 1
    pending = e.ids.tolist()
    late = tuple(tuple([pending[j]] + list(inputs[j][1:])) for j in range(b))
    tails = [1 + (j % 4) for j in range(b)]
    _serial_compare(e, four, late, target, tails, deadline)
    _compaction(e, four, target)
    # W1 completed members must use safe position zero and never grow main KV.
    _direct_replay_checked(e, one, tuple((x,) for x in pending),
                           [e.capacity - 1] * b, [0] * b)
    if not torch.isfinite(one.logits).all():
        raise CandidateRejected("inactive member numerical state is not finite")
    for sk, sv in zip(one.keys, one.values):
        if torch.count_nonzero(sk) or torch.count_nonzero(sv):
            raise CandidateRejected("finished members wrote scratch cache")
    # Prompt is the only persistent content a new request may expose.
    e.prefill_graph.replay()
    e.position.fill_(prompt)
    e.ids.copy_(first)
    torch.cuda.synchronize()


def run_request(e, graphs, input_ids, first_row, output, one_cost, four_cost):
    state = RequestState(input_ids, first_row, output, one_cost, four_cost)
    plan = state.next_plan()
    if plan is not None:
        graphs[plan.width].replay(plan.inputs, plan.lengths, plan.active)
    while plan is not None:
        graph = graphs[plan.width]
        predictions = graph.output.tolist()
        counts = state.commit(plan, predictions)
        graph.compact(counts)
        rows = state.take_rows()
        following = state.next_plan()
        if following is not None:
            graphs[following.width].replay(following.inputs, following.lengths, following.active)
        else:
            # No pending GPU work may escape the final sample token.
            torch.cuda.current_stream().synchronize()
        yield from rows
        plan = following


def _prefix_to(e, input_ids, first, target, deadline):
    """Grow an actual serial prefix and its host history for endpoint timing."""
    e.prefill_graph.replay()
    e.position.fill_(e.prompt)
    e.ids.copy_(first)
    histories = [list(p) for p in input_ids]
    pending = first.tolist()
    position = e.prompt
    while position + 4 <= target:
        _live(deadline)
        e.chunks.graphs[4].replay()
        rows = e.chunks.outputs[4].tolist()
        for b in range(e.batch):
            histories[b].extend([pending[b]] + [rows[j][b] for j in range(3)])
        pending = rows[-1]
        position += 4
    while position < target:
        _live(deadline)
        for b in range(e.batch):
            histories[b].append(pending[b])
        e._step()
        pending = e.ids.tolist()
        position += 1
    torch.cuda.synchronize()
    return histories, pending


def measure_and_admit(e, graphs, input_ids, first, output, deadline):
    _live(deadline, 8.0)
    first_host = first.tolist()
    one_times, four_times, endpoint_pairs = [], [], []
    # Full wall-clock round cost includes drafting, metadata transfer, result
    # transfer, recycling, compaction and its completion. Check both the short
    # and long prefix endpoints; each trial begins on an actual serial prefix.
    # Prompt indexing is counted by the complete request trials below.
    for base in dict.fromkeys((e.prompt, e.capacity - 4)):
        endpoint_one, endpoint_native = [], []
        for _ in range(3):
            for width, times in ((1, one_times), (4, four_times)):
                _live(deadline)
                prefix, pending = _prefix_to(e, input_ids, first, base, deadline)
                state = RequestState(prefix, pending, 5)
                start = time.perf_counter()
                plan = state.next_plan(force_width=width)
                graph = graphs[width]
                graph.replay(plan.inputs, plan.lengths, plan.active)
                counts = state.commit(plan, graph.output.tolist())
                graph.compact(counts)
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
    cost_one = max(one_times)
    cost_four = max(four_times)
    # Compare the same prefix endpoints: longer KV naturally costs more, even
    # for identical paths. The global maxima below price the request debt model;
    # they do not establish a retained-relative bound for every intermediate L.
    # Once sequences diverge, only the independently positioned W1 can fallback.
    if (any(one > native * 1.04 for one, native in endpoint_pairs)
            or cost_four >= cost_one * 3.5):
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
        actual = list(run_request(e, graphs, input_ids, first_host, output, cost_one, cost_four))
        elapsed_candidate.append(time.perf_counter() - start)
        if actual != expected:
            raise CandidateRejected("complete request differs from serial greedy reference")
    if max(elapsed_candidate) >= min(elapsed_native) * 0.94:
        return None
    return cost_one, cost_four
