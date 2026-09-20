"""Exact independent W4 commitments with a fresh contiguous-suffix override."""
from collections import OrderedDict, deque
from dataclasses import dataclass
import math
from suffix_host import SAM


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
                for table in (self.observed, self.predicted):
                    values = table.get(key)
                    if values:
                        return values[0]
        return None


@dataclass(frozen=True)
class Prices:
    p0: float
    d0: float
    c1: float
    c4: float
    initial: float = 0.0
    cleanup: float = 0.0

    def __post_init__(self):
        if any(not math.isfinite(x) or x <= 0 for x in (self.p0, self.d0, self.c1, self.c4)):
            raise ValueError("positive finite reference and round prices required")
        if any(not math.isfinite(x) or x < 0 for x in (self.initial, self.cleanup)):
            raise ValueError("finite nonnegative one-time envelopes required")


class DebtPolicy:
    """Frozen paid first-miss floor. This is NOT a universal 4% bound."""
    def __init__(self, limit, prices):
        if type(limit) is not int or limit < 1:
            raise ValueError("positive integer output limit required")
        self.limit, self.prices, self.disabled = limit, prices, False
        n = limit - 1
        self.p1, self.p4 = max(0., prices.c1-prices.d0), max(0., prices.c4-prices.d0)
        self.allowance = max(.04*n*prices.d0, prices.initial+prices.cleanup+
                             max(n*self.p1, self.p4+max(0,n-1)*self.p1))

    def allow(self, elapsed, produced):
        if (not math.isfinite(elapsed) or elapsed < 0 or not produced or
                any(type(p) is not int or not 1 <= p <= self.limit for p in produced)):
            raise ValueError("valid paid elapsed and per-lane progress required")
        m = min(produced)
        debt = elapsed - (m-1)*self.prices.d0
        # Match the floor's grouping at an exactly paid first boundary. This
        # avoids a floating-summation false veto without adding an epsilon.
        required = (debt+self.prices.cleanup)+(self.p4+max(0,self.limit-m-1)*self.p1)
        if self.limit-m < 2 or required > self.allowance:
            self.disabled = True
        return not self.disabled


@dataclass(frozen=True)
class Plan:
    nonce: object
    epoch: int
    width: int
    lengths: tuple
    pending: tuple
    produced: tuple
    active: tuple
    inputs: tuple
    old_inputs: object


def chain_positions(length, active, width):
    return tuple(length+row if row < active else 0 for row in range(width))


def chain_visible(length, active, row, key, capacity):
    if not 0 <= row < active:
        return False
    if key < capacity:
        return 0 <= key < length
    return 0 <= key-capacity <= row and key-capacity < active


class RequestState:
    """Prompt-only construction overlaps prefill; bind current first IDs once."""
    def __init__(self, prompts, limit, vocab, use_suffix=True, trace=False):
        if (not prompts or not prompts[0] or type(limit) is not int or limit < 1 or
                any(len(p) != len(prompts[0]) for p in prompts)):
            raise ValueError("nonempty rectangular prompts and positive limit required")
        if type(vocab) is not int or vocab < 1 or any(
                type(y) is not int or not 0 <= y < vocab for p in prompts for y in p):
            raise ValueError("invalid prompt token")
        self.batch, self.prompt, self.limit, self.vocab = len(prompts), len(prompts[0]), limit, vocab
        self.nonce, self.epoch, self._outstanding = object(), 0, None
        self.fallback, self.bound, self.disposed = False, False, False
        self.sams = [SAM(p) for p in prompts] if use_suffix else []
        self.histories = [s.tokens for s in self.sams] if use_suffix else [list(p) for p in prompts]
        self.tables = [Successors(p) for p in prompts]
        self.pending, self.produced, self.lengths, self.fifos = [], [], [], []
        self.emitted = 0
        self.trace = dict(wide=0, one=0, overrides=0, accepted_override=0,
                          accepted_changed=0, matches={}, counts=[0]*5) if trace else None

    def bind_first(self, first):
        if self.bound or self.disposed or len(first) != self.batch or any(
                type(y) is not int or not 0 <= y < self.vocab for y in first):
            raise ValueError("one fresh valid first token per lane required")
        for b, token in enumerate(first):
            self.tables[b].record(self.histories[b][-8:], token, observed=True)
            if self.sams:
                self.sams[b].append([token])
            else:
                self.histories[b].append(token)
        self.pending, self.produced = list(first), [1]*self.batch
        self.lengths, self.fifos = [self.prompt]*self.batch, [deque() for _ in first]
        self.emitted, self.bound = 1, True
        self.check_invariants()

    def disable_wide(self):
        if self._outstanding is not None or self.disposed:
            raise ValueError("cannot freeze an outstanding or disposed request")
        self.fallback = True
        for sam in self.sams:
            sam.freeze()

    def next_plan(self, width=4):
        if self.disposed or not self.bound or self._outstanding is not None:
            raise ValueError("bound live request without outstanding plan required")
        if width not in (1,4) or (self.fallback and width != 1):
            raise ValueError("invalid width or revival of frozen proposals")
        remaining = [self.limit-p for p in self.produced]
        if not max(remaining):
            return None
        active = tuple(min(width,r) for r in remaining)
        inputs, old_inputs = [], []
        for b, count in enumerate(active):
            chain = [self.pending[b]]*width
            old = tuple(chain)
            if width == 4 and count:
                context = self.histories[b][-8:].copy()
                for j in range(1,4):
                    y = self.tables[b].successor(context)
                    if y is None:
                        y = self.histories[b][-1]
                    chain[j] = int(y)
                    context = (context+[int(y)])[-8:]
                old = tuple(chain)
                if self.sams:
                    draft, length = self.sams[b].choose(chain[1:])
                    chain[1:] = draft
                    if self.trace is not None:
                        matches = self.trace['matches']
                        matches[length] = matches.get(length,0)+1
            inputs.append(tuple(chain))
            if self.trace is not None:
                old_inputs.append(old)
        plan = Plan(self.nonce,self.epoch,width,tuple(self.lengths),tuple(self.pending),
                    tuple(self.produced),active,tuple(inputs),
                    tuple(old_inputs) if self.trace is not None else None)
        self._outstanding = plan
        return plan

    def commit(self, plan, predictions):
        if (self.disposed or plan is not self._outstanding or plan.nonce is not self.nonce or
                plan.epoch != self.epoch or plan.lengths != tuple(self.lengths) or
                plan.pending != tuple(self.pending) or plan.produced != tuple(self.produced)):
            raise ValueError("stale, foreign or mutated verification plan")
        if len(predictions) != self.batch or any(len(p) != plan.width for p in predictions):
            raise ValueError("one prediction per verifier row required")
        decisions = []
        for chain, ys, active in zip(plan.inputs,predictions,plan.active):
            if any(type(y) is not int or not 0 <= y < self.vocab for y in ys[:active]):
                raise ValueError("invalid active argmax")
            count = int(active > 0)
            while count < active and ys[count-1] == chain[count]:
                count += 1
            decisions.append(count)
        for b, count in enumerate(decisions):
            chain, ys, active = plan.inputs[b], predictions[b], plan.active[b]
            if not self.fallback:
                context = self.histories[b][-8:].copy()
                for row in range(active):
                    if row:
                        context = (context+[chain[row]])[-8:]
                    self.tables[b].record(context,ys[row],observed=False)
            accepted = ys[:count]
            for token in accepted:
                if not self.fallback:
                    self.tables[b].record(self.histories[b][-8:],token,observed=True)
                if self.sams and not self.fallback:
                    self.sams[b].append([token])
                else:
                    self.histories[b].append(token)
            self.fifos[b].extend(accepted)
            self.produced[b] += count
            self.lengths[b] += count
            if count:
                self.pending[b] = accepted[-1]
            if self.trace is not None:
                self.trace['counts'][count] += 1
                old = plan.old_inputs[b]
                if plan.width == 4 and chain[1:active] != old[1:active]:
                    self.trace['overrides'] += 1
                    self.trace['accepted_override'] += max(0,count-1)
                    self.trace['accepted_changed'] += sum(chain[j] != old[j] for j in range(1,count))
        if self.trace is not None:
            self.trace['wide' if plan.width == 4 else 'one'] += 1
        self.epoch += 1
        self._outstanding = None
        self.check_invariants()
        return tuple(decisions)

    def take_rows(self):
        rows = []
        while self.emitted < min(self.produced):
            rows.append([fifo.popleft() for fifo in self.fifos])
            self.emitted += 1
        self.check_invariants()
        return rows

    def check_invariants(self):
        for b in range(self.batch):
            assert self.lengths[b] == self.prompt+self.produced[b]-1
            assert len(self.histories[b]) == self.prompt+self.produced[b]
            assert self.pending[b] == self.histories[b][-1]
            assert len(self.fifos[b]) == self.produced[b]-self.emitted
            assert self.emitted <= self.produced[b] <= self.limit

    def dispose(self):
        # No cycles or external SAM/history aliases escape this request.
        self._outstanding = None
        self.sams.clear()
        self.tables.clear()
        self.histories.clear()
        self.fifos.clear()
        self.pending.clear()
        self.lengths.clear()
        self.produced.clear()
        self.trace = None
        self.disposed = True
