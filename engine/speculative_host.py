"""Pure-Python B1 scheduling for exact, bounded prompt-lookup speculation.

This module does not run a model and is not a submission engine. Each request
must construct a fresh ``RequestState`` AFTER emitting the prefill token.
Native fallback means an already-captured sequential decode graph, preferably
four steps. Speculative mode means one four-row causal model verification.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    kind: str
    position: int
    # In verify mode, inputs are [pending token, draft 1, draft 2, draft 3].
    # Native mode starts from the pending token and feeds its own predictions.
    inputs: tuple[int, ...]
    steps: int
    generation: int


class PromptLookup:
    """Most recent observed continuation of the longest matching suffix.

    Index only occurrences with ``draft_length`` fully observed successors.
    Contents are strictly request-local prompt and already-committed tokens.
    Neither rejected draft tokens nor previous requests enter the index.
    """

    def __init__(self, tokens, draft_length=3, ngram_lengths=(8, 4, 2)):
        if draft_length < 1 or not ngram_lengths or min(ngram_lengths) < 1:
            raise ValueError("positive draft length and n-gram lengths required")
        self.tokens = list(tokens)
        self.draft_length = int(draft_length)
        self.ngram_lengths = tuple(sorted(set(ngram_lengths), reverse=True))
        self.index = {size: {} for size in self.ngram_lengths}
        self.next_end = min(self.ngram_lengths)
        self._index_new()

    def _index_new(self):
        stop = len(self.tokens) - self.draft_length
        while self.next_end <= stop:
            end = self.next_end
            for size in self.ngram_lengths:
                if size <= end:
                    self.index[size][tuple(self.tokens[end-size:end])] = end
            self.next_end += 1

    def append(self, committed):
        self.tokens.extend(committed)
        self._index_new()

    def draft(self):
        for size in self.ngram_lengths:
            if size <= len(self.tokens):
                end = self.index[size].get(tuple(self.tokens[-size:]))
                if end is not None:
                    return tuple(self.tokens[end:end+self.draft_length])
        return ()


def verified_prefix(inputs, predictions):
    """Return only exact greedy tokens on the committed prefix.

    Row j consumes inputs[j] and predicts the token following it. Stop on the
    first disagreement between prediction j and draft input j+1; prediction j
    itself is valid and becomes the next pending token. If all drafts match,
    the last verification row provides one additional exact token.
    """
    if not inputs or len(inputs) != len(predictions):
        raise ValueError("one prediction is required per verification input")
    for row in range(len(inputs) - 1):
        if predictions[row] != inputs[row + 1]:
            return tuple(predictions[:row + 1])
    return tuple(predictions)


class RequestState:
    """Track output count and logical cache length for a single B1 request.

    ``first`` was produced by full-prompt prefill and has already been emitted.
    ``position`` is the physical cache slot at which the pending token must be
    consumed; positions strictly below it contain the valid committed prefix.
    A commit advances position by the number of verified tokens, not by four.

    Budget is based only on the requested output length. With denominator 24,
    speculation can remove at most 3*floor((output-1)/24) ordinary decode steps.
    This bounds favorable work variation but does NOT guarantee the runtime
    spread gate: GPU cost, drafting overhead and input-dependent prefill remain.
    """

    def __init__(self, prompt, first, max_new_tokens, budget_denominator=24):
        if not prompt or max_new_tokens < 1 or budget_denominator < 4:
            raise ValueError("nonempty prompt, positive output and denominator >=4 required")
        self.lookup = PromptLookup([*prompt, first])
        self.limit = int(max_new_tokens)
        self.produced = 1
        self.position = len(prompt)
        self.pending = int(first)
        self.budget = (self.limit - 1) // int(budget_denominator)
        self.attempts = 0
        self.generation = 0

    @property
    def remaining(self):
        return self.limit - self.produced

    def next_plan(self):
        if not self.remaining:
            return None
        if self.remaining >= 4 and self.attempts < self.budget:
            draft = self.lookup.draft()
            if len(draft) == 3:
                return Plan("verify", self.position,
                            (self.pending, *draft), 4, self.generation)
        return Plan("native", self.position, (self.pending,),
                    min(4, self.remaining), self.generation)

    def commit(self, plan, predictions):
        if (plan.generation != self.generation or
                plan.position != self.position or
                plan.inputs[0] != self.pending):
            raise ValueError("stale or inconsistent execution plan")
        if len(predictions) != plan.steps:
            raise ValueError("one output per executed row or native step required")
        if plan.kind == "verify":
            if plan.steps != 4 or len(plan.inputs) != 4:
                raise ValueError("only four-row verification is supported")
            committed = verified_prefix(plan.inputs, predictions)
            self.attempts += 1
        elif plan.kind == "native":
            committed = tuple(predictions)
        else:
            raise ValueError("unknown execution mode")
        if not committed or len(committed) > self.remaining:
            raise ValueError("execution would violate the requested output count")
        self.position += len(committed)
        self.pending = int(committed[-1])
        self.produced += len(committed)
        self.generation += 1
        self.lookup.append(committed)
        return committed
