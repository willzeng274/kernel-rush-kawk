"""Compact request-local N=3, W=4, G=2 host state; no GPU imports.

The caller supplies full-model argmax values for the immutable plan. ``commit``
models successful ordered scratch compaction, then publishes logical progress.
Only the graph runtime executes model operations.
"""
from collections import OrderedDict, deque
from dataclasses import dataclass
import math

PARENTS = (-1, 0, 1, 2, 0, 1, 2, 3, 0, 8, 0, 10)
DEPTHS = (0, 1, 2, 3, 1, 2, 3, 4, 1, 2, 1, 2)
ANCESTORS = (1, 3, 7, 15, 17, 35, 71, 143, 257, 769, 1025, 3073)
FILL_ROWS = (1, 2, 3, 7)
UPPER_ROWS = (4, 5, 6, 7)
VERIFY_PATHS = ((8, 9), (10, 11))


@dataclass
class DebtPolicy:
    """Fixed 4% retained-relative stop rule; timing arguments are real costs.

    D0 is retained #32 deployed post-prefill time per output rank, including
    its own B1 speculative path when selected; it is not chunk-only cost. V12
    includes a full wide round, host work and compaction. No startup allowance
    is added. Actual elapsed post-prefill time includes initialization.
    """
    limit: int
    d0: float
    v12: float
    init_cost: float
    disabled: bool = False

    def __post_init__(self):
        if self.limit < 1 or any(not math.isfinite(x) or x <= 0
                                 for x in (self.d0, self.v12)):
            raise ValueError("positive finite costs required")
        if not math.isfinite(self.init_cost) or self.init_cost < 0:
            raise ValueError("nonnegative finite initialization cost required")
        self.allowance = 0.04 * (self.limit - 1) * self.d0
        # At least fill, tuple generation, and subsequent retrieval must fit
        # three consecutive one-output misses at the measured wide price.
        self.disabled = (self.limit < 7 or self.init_cost
                         + 3 * max(0.0, self.v12 - self.d0) > self.allowance)

    def allow(self, actual_elapsed, produced):
        if not math.isfinite(actual_elapsed) or actual_elapsed < 0 or not produced:
            raise ValueError("valid elapsed and per-member progress required")
        debt = actual_elapsed - (min(produced) - 1) * self.d0
        if debt + max(0.0, self.v12 - self.d0) > self.allowance:
            self.disabled = True
        return not self.disabled


class TrigramPool:
    """Bounded update-recency LRU; oldest keys/tuples evicted first.

    Reading does not change recency. Distinct tuples sharing the same first
    continuation token remain distinct. All keys are full Python integer IDs.
    """
    def __init__(self, max_keys=256, per_key=2, trace=False):
        if max_keys < 1 or per_key < 1:
            raise ValueError("positive pool bounds required")
        self.max_keys, self.per_key = max_keys, per_key
        self.data = OrderedDict()
        self.revision = 0
        self.origins = {} if trace else None

    def insert(self, a, b, c, trajectory=False):
        pair = (b, c)
        values = self.data.pop(a, [])
        if pair in values:
            values.remove(pair)
        values.append(pair)
        removed = values[:-self.per_key]
        self.data[a] = values[-self.per_key:]
        evicted = None
        if len(self.data) > self.max_keys:
            evicted = self.data.popitem(last=False)
        if self.origins is not None:
            for old in removed:
                self.origins.pop((a, old), None)
            if evicted is not None:
                key, pairs = evicted
                for old in pairs:
                    self.origins.pop((key, old), None)
            self.origins[(a, pair)] = trajectory
        self.revision += 1

    def retrieve(self, key):
        return tuple(reversed(self.data.get(key, ())))[:2]

    def seed(self, history, limit=256):
        # Proposal-only bounded scan; it never changes full attention history.
        tail = history[-limit:]
        for i in range(len(tail) - 2):
            self.insert(*tail[i:i + 3])


@dataclass(frozen=True)
class Plan:
    owner: object
    epoch: int
    lengths: tuple
    pending: tuple
    produced: tuple
    modes: tuple
    lower: tuple
    upper: tuple
    revisions: tuple
    inputs: tuple
    active: tuple
    candidates: tuple


def ancestry_path(row):
    path = []
    while row >= 0:
        path.append(row)
        row = PARENTS[row]
    return tuple(reversed(path))


def positions(length, active):
    return tuple(length + depth if active & (1 << j) else 0
                 for j, depth in enumerate(DEPTHS))


def visible(length, active, row, key, capacity):
    """Conceptual key address: main slots [0,capacity), then 12 scratch slots."""
    if not (0 <= row < 12 and active & (1 << row)):
        return False
    if 0 <= key < capacity:
        return key < length
    node = key - capacity
    return (0 <= node < 12 and bool(active & (1 << node))
            and bool(ANCESTORS[row] & (1 << node)))


def choose_verified(inputs, ys, active, remaining, candidates):
    """Select one WHOLE verifier branch; trajectory nodes are ineligible.

    A last prediction is exact on the accepted path even if it has no matching
    child. It is the new pending token, never a consumed scratch input.
    """
    if remaining == 0:
        return (), ()
    best_path, best_outputs = (0,), (ys[0],)
    for candidate, branch in zip(candidates, VERIFY_PATHS):
        path, outputs = [0], [ys[0]]
        for token, row in zip(candidate, branch):
            if len(outputs) >= remaining or not active & (1 << row):
                break
            if token != outputs[-1]:
                break
            path.append(row)
            outputs.append(ys[row])
        if len(outputs) > len(best_outputs):
            best_path, best_outputs = tuple(path), tuple(outputs)
    return best_path, best_outputs


class Request:
    """Fresh state constructed after first prefill yield on every request.

    ``first`` already contains native/full-model greedy prefill predictions.
    Only two-token observed suffixes, bounded pools and output FIFOs persist.
    """
    def __init__(self, prompts, first, limit, vocab_size, pool_keys=256, trace=False):
        if (not prompts or len(first) != len(prompts) or limit < 1
                or not prompts[0] or any(len(p) != len(prompts[0]) for p in prompts)):
            raise ValueError("nonempty rectangular prompts and positive limit required")
        if vocab_size < 1 or any(type(t) is not int or not 0 <= t < vocab_size
                                 for p in prompts for t in p):
            raise ValueError("invalid prompt token")
        if any(type(t) is not int or not 0 <= t < vocab_size for t in first):
            raise ValueError("invalid first token")
        self.batch, self.prompt, self.limit = len(prompts), len(prompts[0]), limit
        self.capacity, self.vocab_size = self.prompt + limit, vocab_size
        self.owner, self.epoch, self.outstanding = object(), 0, None
        self.lengths = [self.prompt] * self.batch
        self.pending, self.produced = list(first), [1] * self.batch
        self.tails = [(p[-1], f) for p, f in zip(prompts, first)]
        self.trace = dict(fill=0, steady=0, trajectory_selected=0) if trace else None
        self.fifos = [deque() for _ in prompts]
        self.emitted = 1  # prefill result has already crossed the stream boundary
        self.pools = [TrigramPool(pool_keys, trace=trace) for _ in prompts]
        self.lower, self.upper = [], [None] * self.batch
        self.fallback = False
        for b, p in enumerate(prompts):
            self.pools[b].seed(p)
            if len(p) >= 2:
                self.pools[b].insert(p[-2], p[-1], first[b])
            self.lower.append(tuple(p[(len(p) - 4 + i) % len(p)] for i in range(4)))

    def disable_wide(self):
        if self.outstanding is not None:
            raise ValueError("cannot abandon an outstanding verification plan")
        self.fallback = True
        self.lower = [None] * self.batch
        self.upper = [None] * self.batch

    def next_plan(self):
        if self.outstanding is not None:
            raise ValueError("outstanding plan must commit first")
        if min(self.produced) == self.limit:
            return None
        all_inputs, all_active, modes, all_candidates = [], [], [], []
        for b in range(self.batch):
            r = self.limit - self.produced[b]
            nodes, active = [self.pending[b]] * 12, (1 if r else 0)
            candidates = ()
            if r == 0:
                mode = "done"
            elif self.fallback or r == 1 or (self.upper[b] is None and r < 5):
                mode = "one"
            elif self.upper[b] is None:
                mode = "fill"
                for row, token in zip(FILL_ROWS, self.lower[b]):
                    nodes[row] = token
                    active |= 1 << row
            else:
                mode = "steady" if r >= 5 else "tail"
                if mode == "steady":
                    nodes[1:4], nodes[4:8] = self.lower[b], self.upper[b]
                    active |= 255
                # Snapshot retrieval BEFORE current predictions/pool insertion.
                candidates = self.pools[b].retrieve(self.pending[b])
                if self.trace is not None:
                    self.trace['trajectory_selected'] += sum(
                        bool(self.pools[b].origins.get((self.pending[b], pair)))
                        for pair in candidates)
                for pair, branch in zip(candidates, VERIFY_PATHS):
                    for token, row in zip(pair, branch):
                        nodes[row] = token
                        if DEPTHS[row] < r:
                            active |= 1 << row
            all_inputs.append(tuple(nodes))
            all_active.append(active)
            all_candidates.append(candidates)
            modes.append(mode)
        plan = Plan(self.owner, self.epoch, tuple(self.lengths), tuple(self.pending),
                    tuple(self.produced), tuple(modes), tuple(self.lower), tuple(self.upper),
                    tuple(p.revision for p in self.pools), tuple(all_inputs),
                    tuple(all_active), tuple(all_candidates))
        self.outstanding = plan
        return plan

    def _guard(self, plan):
        if (plan is not self.outstanding or plan.owner is not self.owner
                or plan.epoch != self.epoch or plan.lengths != tuple(self.lengths)
                or plan.pending != tuple(self.pending) or plan.produced != tuple(self.produced)
                or plan.lower != tuple(self.lower) or plan.upper != tuple(self.upper)
                or plan.revisions != tuple(p.revision for p in self.pools)):
            raise ValueError("stale, foreign, replaced or mutated plan/state")

    def commit(self, plan, predictions):
        self._guard(plan)
        if len(predictions) != self.batch or any(len(p) != 12 for p in predictions):
            raise ValueError("one twelve-ID vector per batch member required")
        # Validate all members before any member mutates. Inactive sentinels
        # are allowed, and no inactive prediction may affect state.
        for ys, active in zip(predictions, plan.active):
            for row, y in enumerate(ys):
                if active & (1 << row) and (type(y) is not int or not 0 <= y < self.vocab_size):
                    raise ValueError("invalid active argmax ID")
        decisions = []
        for b in range(self.batch):
            decisions.append(choose_verified(plan.inputs[b], predictions[b], plan.active[b],
                                             self.limit - self.produced[b], plan.candidates[b]))
        for b, (path, outputs) in enumerate(decisions):
            if not outputs:
                continue
            ys, old_nodes = predictions[b], plan.inputs[b]
            # Symbolic scratch->main copy: only root + one verifier path.
            if any(row not in (0, 8, 9, 10, 11) for row in path):
                raise AssertionError("trajectory scratch cannot commit")
            for j, row in enumerate(path):
                if ancestry_path(row) != path[:j + 1]:
                    raise AssertionError("foreign branch compaction")
            if self.trace is not None and plan.modes[b] in ("fill", "steady"):
                self.trace[plan.modes[b]] += 1
            if plan.modes[b] == "fill":
                self.lower[b] = plan.lower[b][1:]
                self.upper[b] = tuple(ys[row] for row in FILL_ROWS)
            elif plan.modes[b] == "steady":
                keys = (plan.pending[b],) + plan.lower[b]
                upper_preds = tuple(ys[row] for row in UPPER_ROWS)
                for key, second, third in zip(keys, plan.upper[b], upper_preds):
                    self.pools[b].insert(key, second, third, trajectory=True)
                # ALWAYS_FWD_ONE: rotation is one ITERATION, regardless of c.
                self.lower[b] = plan.upper[b][1:]
                self.upper[b] = upper_preds
            for y in outputs:
                a, z = self.tails[b]
                self.pools[b].insert(a, z, y)
                self.tails[b] = (z, y)
            self.fifos[b].extend(outputs)
            self.lengths[b] += len(path)
            self.produced[b] += len(outputs)
            self.pending[b] = outputs[-1]
            if self.produced[b] == self.limit:
                self.lower[b] = self.upper[b] = None
        self.outstanding = None
        self.epoch += 1
        return tuple(path for path, _ in decisions)

    def take_rows(self):
        rows = []
        while self.emitted < self.limit and all(self.fifos):
            rows.append(tuple(f.popleft() for f in self.fifos))
            self.emitted += 1
        return tuple(rows)
