"""Exact request-local two-continuation tree and independent output progress."""
import time
from recycled_host import RequestState, Plan

PARENTS = (-1, 0, 1, 2, 3, 0, 5, 6)
DEPTHS = (0, 1, 2, 3, 4, 1, 2, 3)
ANCESTORS = (1, 3, 7, 15, 31, 33, 97, 225)


def successors(table, context):
    """Select up to two distinct *existing* edges, strongest context first."""
    result = []
    for size in (8, 4, 2, 1):
        if len(context) >= size:
            key = tuple(context[-size:])
            for source in (table.observed, table.predicted):
                for token in source.get(key, ()):
                    if token not in result:
                        result.append(token)
                        if len(result) == 2:
                            return result
    return result


def tree_positions(length, active):
    return tuple(length + depth if active & (1 << row) else 0
                 for row, depth in enumerate(DEPTHS))


def tree_visible(length, active, row, key, capacity):
    if not active & (1 << row):
        return False
    if key < capacity:
        return 0 <= key < length
    node = key - capacity
    return (0 <= node < 8 and bool(active & (1 << node))
            and bool(ANCESTORS[row] & (1 << node)))


class TreeState(RequestState):
    def __init__(self, prompts, first, limit, cost_one=1.0, cost_tree=1.5):
        setup_start = time.perf_counter()
        if (not prompts or len(prompts) != len(first) or limit < 1 or
                any(not p or len(p) != len(prompts[0]) for p in prompts)):
            raise ValueError("nonempty equal prompts, one first token each and positive limit required")
        # Bound proposal-only indexing across the batch. Every complete prompt
        # remains in history and in dense model attention/KV; this changes only
        # draft recall, which the full-request gain gate measures honestly.
        seed_width = min(len(prompts[0]), max(8, 2048 // len(prompts)))
        super().__init__([p[-seed_width:] for p in prompts], first, limit, cost_one, cost_tree)
        self.prompt = len(prompts[0])
        self.histories = [list(p) + [int(y)] for p, y in zip(prompts, first)]
        self.lengths = [self.prompt] * self.batch
        self.seed_tokens = seed_width * self.batch
        self.spent = time.perf_counter() - setup_start
        # A 4% request budget can suppress even the first trial. Permit only
        # the extra measured startup cost of ONE miss, capped by 8% overall.
        # Once that trial is spent, further rounds need actual earned progress.
        full_budget = (limit - 1) * self.cost_one
        worst_extra = max(0.0, self.cost_four - self.cost_one)
        self.allowance = min(0.08 * full_budget,
                             max(0.04 * full_budget, self.spent + worst_extra))

    def next_plan(self, force_width=None):
        if self._outstanding is not None:
            raise ValueError("commit outstanding plan before drafting again")
        remaining = [self.limit - n for n in self.produced]
        if not max(remaining):
            return None
        debt = self.spent - (min(self.produced) - 1) * self.cost_one
        if debt + max(0.0, self.cost_four - self.cost_one) > self.allowance + 1e-12:
            self.fallback = True
        width = 1 if self.fallback or max(remaining) < 2 else 8
        if force_width is not None:
            if force_width not in (1, 8):
                raise ValueError("tree width must be one or eight")
            width = force_width
        inputs, active = [], []
        for b, remaining_b in enumerate(remaining):
            nodes = [self.pending[b]] * width
            enabled = 1 if remaining_b else 0
            if width == 8 and remaining_b > 1:
                context = self.histories[b][-8:]
                choices = successors(self.tables[b], context)
                nodes[1] = choices[0] if choices else self.pending[b]
                enabled |= 2
                if len(choices) > 1:
                    nodes[5] = choices[1]
                    enabled |= 32
                contexts = {0: context}
                for row in range(1, 8):
                    parent = PARENTS[row]
                    if DEPTHS[row] >= remaining_b or not enabled & (1 << parent):
                        enabled &= ~(1 << row)
                        continue
                    if row not in (1, 5):
                        choices_next = successors(self.tables[b], contexts[parent])
                        nodes[row] = choices_next[0] if choices_next else nodes[parent]
                        enabled |= 1 << row
                    if enabled & (1 << row):
                        contexts[row] = (contexts[parent] + [nodes[row]])[-8:]
            inputs.append(tuple(nodes))
            active.append(enabled)
        plan = Plan(self.epoch, self.nonce, width, tuple(self.lengths),
                    tuple(self.pending), tuple(active), tuple(inputs))
        self._outstanding = plan
        return plan

    def commit(self, plan, predictions):
        if (plan is not self._outstanding or plan.nonce is not self.nonce or
                plan.epoch != self.epoch or plan.lengths != tuple(self.lengths) or
                plan.pending != tuple(self.pending)):
            raise ValueError("stale, foreign or mutated verification plan")
        if len(predictions) != self.batch or any(len(p) != plan.width for p in predictions):
            raise ValueError("one prediction per verifier node required")
        paths = []
        for b, (nodes, ys, active) in enumerate(zip(plan.inputs, predictions, plan.active)):
            contexts = {0: self.histories[b][-8:].copy()}
            for row in range(plan.width):
                if active & (1 << row):
                    if row:
                        contexts[row] = (contexts[PARENTS[row]] + [nodes[row]])[-8:]
                    self.tables[b].record(contexts[row], ys[row], observed=False)
            visited, accepted = [], []
            node = 0
            while active & (1 << node):
                visited.append(node)
                token = int(ys[node])
                accepted.append(token)
                children = [j for j in range(1, plan.width)
                            if PARENTS[j] == node and active & (1 << j)
                            and nodes[j] == token]
                if len(children) > 1:
                    raise ValueError("duplicate active sibling candidates")
                if not children:
                    break
                node = children[0]
            for token in accepted:
                self.tables[b].record(self.histories[b][-8:], token, observed=True)
                self.histories[b].append(token)
            self.fifos[b].extend(accepted)
            self.lengths[b] += len(visited)
            self.produced[b] += len(visited)
            if accepted:
                self.pending[b] = accepted[-1]
            paths.append(tuple(visited))
        self.spent += self.cost_four if plan.width == 8 else self.cost_one
        self.epoch += 1
        self._outstanding = None
        self.check_invariants()
        return tuple(paths)
