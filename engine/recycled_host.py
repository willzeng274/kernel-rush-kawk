"""Request-local exact W4 recycling. Pure Python; no CUDA/model dependency.

Each sequence owns its pending output, consumed cache length and output FIFO.
Verifier argmaxes at *all* active hypothetical rows are proposals for future
rounds; only a matched ancestor path can enter the committed output stream.
"""
from collections import OrderedDict, deque
from dataclasses import dataclass


class Successors:
    def __init__(self, prompt, capacity=8192):
        self.capacity = int(capacity)
        if self.capacity < 1 or not prompt:
            raise ValueError("nonempty prompt and positive capacity required")
        self.observed = OrderedDict()
        self.predicted = OrderedDict()
        for end in range(1, len(prompt)):
            self.record(prompt[max(0, end - 8):end], prompt[end], observed=True)

    def record(self, context, token, observed=False):
        table = self.observed if observed else self.predicted
        for size in ((8, 4, 2) if observed else (4, 2, 1)):
            if len(context) < size:
                continue
            key = tuple(context[-size:])
            old = table.pop(key, ())
            table[key] = (int(token),) + tuple(x for x in old if x != token)[:1]
            if len(table) > self.capacity:
                table.popitem(last=False)

    def successor(self, context):
        for size in (8, 4, 2, 1):
            if len(context) >= size:
                key = tuple(context[-size:])
                # Long observed contexts are the strongest prompt-local source;
                # model predictions populate the short-context cold-start gaps.
                for table in (self.observed, self.predicted):
                    values = table.get(key)
                    if values:
                        return values[0]
        return None


@dataclass(frozen=True)
class Plan:
    epoch: int
    nonce: object
    width: int
    lengths: tuple
    pending: tuple
    active: tuple
    inputs: tuple


def chain_positions(length, active, width):
    """Actual device formula: inactive rows use a safe initialized RoPE row."""
    return tuple(length + row if row < active else 0 for row in range(width))


def chain_visible(length, active, row, key, capacity):
    """Logical device mask over [committed capacity | chain scratch]."""
    if not 0 <= row < active:
        return False
    if key < capacity:
        return 0 <= key < length
    scratch = key - capacity
    return 0 <= scratch <= row and scratch < active


class RequestState:
    def __init__(self, prompts, first, limit, cost_one=1.0, cost_four=1.5,
                 exploration_fraction=0.04):
        if (not prompts or len(prompts) != len(first) or limit < 1 or
                any(not p or len(p) != len(prompts[0]) for p in prompts)):
            raise ValueError("nonempty equal prompts, one first token each and positive limit required")
        if cost_one <= 0 or cost_four <= 0 or not 0 <= exploration_fraction <= 0.10:
            raise ValueError("invalid measured costs or exploration bound")
        self.batch, self.prompt, self.limit = len(prompts), len(prompts[0]), int(limit)
        self.histories = [list(p) + [int(y)] for p, y in zip(prompts, first)]
        self.tables = [Successors(p) for p in prompts]
        for table, history in zip(self.tables, self.histories):
            table.record(history[-9:-1], history[-1], observed=True)
        self.produced = [1] * self.batch
        self.lengths = [self.prompt] * self.batch
        self.pending = [int(x) for x in first]
        self.fifos = [deque() for _ in prompts]  # First row already emitted.
        self.emitted = 1
        self.epoch, self.nonce = 0, object()
        self.cost_one, self.cost_four = float(cost_one), float(cost_four)
        self.allowance = exploration_fraction * (limit - 1) * self.cost_one
        self.spent = 0.0
        self.fallback = False
        self._outstanding = None

    def next_plan(self, force_width=None):
        if self._outstanding is not None:
            raise ValueError("commit the outstanding plan before drafting again")
        remaining = [self.limit - n for n in self.produced]
        if not max(remaining):
            return None
        # min(produced) measures actual outward progress, not the arithmetic
        # mean of accepted counts. Future failed rounds consume a fixed budget.
        debt = self.spent - (min(self.produced) - 1) * self.cost_one
        worst_extra = max(0.0, self.cost_four - self.cost_one)
        if debt + worst_extra > self.allowance:
            self.fallback = True
        width = 1 if self.fallback or max(remaining) < 2 else 4
        if force_width is not None:
            if force_width not in (1, 4):
                raise ValueError("width must be one or four")
            width = force_width
        active = tuple(min(width, r) for r in remaining)
        inputs = []
        for b, count in enumerate(active):
            chain = [self.pending[b]]
            context = self.histories[b][-8:].copy()
            while len(chain) < width:
                draft = self.tables[b].successor(context)
                # Valid deterministic token from this request, always verified.
                if draft is None:
                    draft = self.histories[b][-1]
                chain.append(int(draft))
                context = (context + [int(draft)])[-8:]
            inputs.append(tuple(chain))
        plan = Plan(self.epoch, self.nonce, width, tuple(self.lengths),
                    tuple(self.pending), active, tuple(inputs))
        self._outstanding = plan
        return plan

    def commit(self, plan, predictions):
        if (plan is not self._outstanding or plan.nonce is not self.nonce or
                plan.epoch != self.epoch or plan.lengths != tuple(self.lengths) or
                plan.pending != tuple(self.pending)):
            raise ValueError("stale, foreign or mutated verification plan")
        if len(predictions) != self.batch or any(len(p) != plan.width for p in predictions):
            raise ValueError("one prediction per batch verification row required")
        counts = []
        for b, (chain, ys, active) in enumerate(zip(plan.inputs, predictions, plan.active)):
            context = self.histories[b][-8:].copy()
            # Root is already at the end of the history. Later inputs append
            # hypothetical tokens, including rejected rows, to its own context.
            for row in range(active):
                if row:
                    context = (context + [chain[row]])[-8:]
                self.tables[b].record(context, ys[row], observed=False)
            count = 0
            if active:
                count = 1
                while count < active and int(ys[count - 1]) == chain[count]:
                    count += 1
            accepted = [int(x) for x in ys[:count]]
            for token in accepted:
                self.tables[b].record(self.histories[b][-8:], token, observed=True)
                self.histories[b].append(token)
            self.fifos[b].extend(accepted)
            self.lengths[b] += count
            self.produced[b] += count
            if count:
                self.pending[b] = accepted[-1]
            counts.append(count)
        self.spent += self.cost_four if plan.width == 4 else self.cost_one
        self.epoch += 1
        self._outstanding = None
        self.check_invariants()
        return tuple(counts)

    def take_rows(self):
        rows = []
        while self.emitted < min(self.produced):
            rows.append([fifo.popleft() for fifo in self.fifos])
            self.emitted += 1
        self.check_invariants()
        return rows

    def check_invariants(self):
        for b in range(self.batch):
            assert self.lengths[b] == self.prompt + self.produced[b] - 1
            assert len(self.histories[b]) == self.prompt + self.produced[b]
            assert self.pending[b] == self.histories[b][-1]
            assert len(self.fifos[b]) == self.produced[b] - self.emitted
            assert self.emitted <= self.produced[b] <= self.limit
