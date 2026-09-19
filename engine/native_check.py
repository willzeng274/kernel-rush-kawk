"""Warmup-only native own-prefix evidence for otherwise equal-vector rows.

Never selects output tokens. No reference model is constructed on the existing
all-equal path. Native work is outside trial timers, inside the original deadline.
"""
import math
from dataclasses import dataclass
import torch
from recycled_validate import CandidateRejected, _close, _live
from native_shell import GEOMETRY, build_native

PRIMARY = (0, 1, 2, 3, 4)
ALTERNATIVE = (0, 5, 6, 7)
MARGIN = 1.0
LOGIT_CAP = 4 * 1024**3
MEMORY_FRACTION = 0.85
MAX_CALLS = 15


def matrix(rows, width=None):
    if (not isinstance(rows, (list, tuple)) or not rows or
            any(not isinstance(row, (list, tuple)) for row in rows)):
        raise CandidateRejected("native case requires a rectangular token matrix")
    n = len(rows[0]) if width is None else width
    vocab = GEOMETRY['vocab_size']
    if (n < 1 or not 1 <= len(rows) <= 8 or
            any(len(row) != n or any(type(x) is not int or not 0 <= x < vocab for x in row)
                for row in rows)):
        raise CandidateRejected("invalid native token matrix")
    return tuple(tuple(row) for row in rows)


@dataclass(frozen=True)
class Case:
    inputs: tuple
    kind: str
    positions: tuple

    def __post_init__(self):
        object.__setattr__(self, 'inputs', matrix(self.inputs))
        if not isinstance(self.positions, (list, tuple)):
            raise CandidateRejected("invalid native positions")
        object.__setattr__(self, 'positions', tuple(self.positions))
        self.validate()

    def validate(self):
        if (type(self.kind) is not str or not self.kind or not self.positions or
                any(type(p) is not int or not 0 <= p < len(self.inputs[0]) for p in self.positions) or
                len(set(self.positions)) != len(self.positions)):
            raise CandidateRejected("invalid native case identity")


def branch_case(prefixes, inputs, path, capacity):
    prefixes, inputs = matrix(prefixes), matrix(inputs, 8)
    if len(prefixes) != len(inputs) or path not in (PRIMARY, ALTERNATIVE):
        raise CandidateRejected("invalid native branch ancestry")
    base = len(prefixes[0])
    if type(capacity) is not int:
        raise CandidateRejected("invalid native capacity")
    count = min(len(path), capacity - base)
    if count < 1:
        raise CandidateRejected("no valid native branch rows")
    path = path[:count]
    return Case(tuple(prefix + tuple(nodes[row] for row in path)
                      for prefix, nodes in zip(prefixes, inputs)),
                'branch:' + str(path), tuple(range(base, base + count)))


def branch_job(prefixes, inputs, path, capacity, active, logits):
    case = branch_case(prefixes, inputs, path, capacity)
    if len(active) != len(case.inputs) or any(type(x) is not int or not 0 <= x < 256 for x in active):
        raise CandidateRejected("invalid native active branch")
    chosen = logits.argmax(-1).tolist()
    if len(chosen) != len(case.inputs) or any(len(row) != 8 for row in chosen):
        raise CandidateRejected("invalid native branch predictions")
    targets = {(j, pos): chosen[j][row]
               for j in range(len(case.inputs))
               for row, pos in zip(path, case.positions) if active[j] & (1 << row)}
    return case, targets


def one_job(prefixes, inputs, capacity, offsets, active, logits):
    case = branch_case(prefixes, inputs, PRIMARY, capacity)
    b = len(case.inputs)
    if (len(offsets) != b or len(active) != b or
            any(type(d) is not int or not 0 <= d < len(case.positions) for d in offsets) or
            any(type(x) is not int or x not in (0, 1) for x in active)):
        raise CandidateRejected("invalid divergent native W1 map")
    chosen = logits.argmax(-1).tolist()
    if len(chosen) != b or any(type(x) is not int for x in chosen):
        raise CandidateRejected("invalid divergent native W1 predictions")
    return case, {(j, case.positions[d]): chosen[j] for j, d in enumerate(offsets) if active[j]}


def complete_case(prompts, first, actual, output):
    prompts = matrix(prompts)
    b, s = len(prompts), len(prompts[0])
    if (type(output) is not int or output < 1 or
            not isinstance(first, (list, tuple)) or len(first) != b or
            not isinstance(actual, (list, tuple)) or len(actual) != output - 1 or
            any(not isinstance(row, (list, tuple)) or len(row) != b for row in actual)):
        raise CandidateRejected("invalid complete native stream shape")
    y = matrix([tuple([first[j]] + [row[j] for row in actual]) for j in range(b)], output)
    case = Case(tuple(prefix + values[:-1] for prefix, values in zip(prompts, y)),
                'complete', tuple(range(s - 1, s + output - 1)))
    return case, {(j, s - 1 + t): y[j][t] for j in range(b) for t in range(output)}


def memory_plan(case):
    """Conservative estimate only; native forward is not preemptible."""
    b, t = len(case.inputs), len(case.inputs[0])
    h, i, q, v = (GEOMETRY['hidden_size'], GEOMETRY['intermediate_size'],
                  GEOMETRY['num_attention_heads'] * GEOMETRY['head_dim'], GEOMETRY['vocab_size'])
    logits = 2 * b * t * v
    activation = b * t * (16 * h + 16 * q + 12 * i)
    attention = 16 * b * GEOMETRY['num_attention_heads'] * t * t
    reduction = 6 * 64 * v
    extra = logits + activation + attention + reduction + 512 * 1024**2
    return logits, extra


class NativeOwner:
    def __init__(self, engine, deadline):
        self.engine = engine
        self.deadline = deadline
        self.device = engine.model.lm_head.weight.device
        self.model = None
        self.constructing = []
        self.active = []
        self.calls = 0
        self.rescue_active = False
        self.closed = False
        # Register before shell construction, state assignment or device work.
        engine.native_check_owner = self
        engine.recycle_quarantine.append(self)

    def publish(self, value):
        self.constructing.append(value)
        return value

    def live(self, reserve=1.0):
        _live(self.deadline, reserve)

    def preflight(self, case):
        self.live()
        logits, extra = memory_plan(case)
        free, total = torch.cuda.mem_get_info(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        if (logits > LOGIT_CAP or total <= 0 or free < extra or
                reserved + extra > MEMORY_FRACTION * total):
            raise CandidateRejected("native full-position memory preflight")
        self.live()

    def peak_guard(self):
        self.live()
        _, total = torch.cuda.mem_get_info(self.device)
        values = (torch.cuda.memory_allocated(self.device), torch.cuda.memory_reserved(self.device),
                  torch.cuda.max_memory_allocated(self.device), torch.cuda.max_memory_reserved(self.device))
        if total <= 0 or any(value > MEMORY_FRACTION * total for value in values):
            raise CandidateRejected("native current or unreset peak memory guard")
        self.live()

    def check(self, case, targets):
        if self.closed or type(case) is not Case:
            raise CandidateRejected("invalid native owner/case lifetime")
        case.validate()
        b, t = len(case.inputs), len(case.inputs[0])
        v = GEOMETRY['vocab_size']
        if (not isinstance(targets, dict) or not targets or
                any(not isinstance(key, tuple) or len(key) != 2 or
                    type(key[0]) is not int or not 0 <= key[0] < b or
                    type(key[1]) is not int or key[1] not in case.positions or
                    type(y) is not int or not 0 <= y < v for key, y in targets.items())):
            raise CandidateRejected("invalid native target identity")
        if b != self.engine.batch or t > self.engine.capacity:
            raise CandidateRejected("native full-batch/capacity mismatch")
        self.active = [case, dict(targets)]
        try:
            with torch.inference_mode():
                self.live(8.0)
                self.preflight(case)
                if self.calls >= MAX_CALLS:
                    raise CandidateRejected("native forward call budget")
                if self.model is None:
                    self.live(8.0)
                    self.model = build_native(self.engine.model, self)
                self.live()
                # Publish device storage before copy queues work on it.
                ids = torch.empty((b, t), dtype=torch.long, device=self.device)
                self.active.append(ids)
                cpu_ids = torch.tensor(case.inputs, dtype=torch.long)
                self.active.append(cpu_ids)
                ids.copy_(cpu_ids)
                self.preflight(case)
                self.live(8.0)
                self.calls += 1
                # Native root forward and default ALL-position head, no cache.
                result = self.model(input_ids=ids, use_cache=False, past_key_values=None,
                                    return_dict=True, output_attentions=False, output_hidden_states=False)
                self.active.append(result)
                torch.cuda.synchronize(self.device)
                self.live()
                self.peak_guard()
                logits = result.logits
                if (tuple(logits.shape) != (b, t, v) or logits.dtype != torch.bfloat16 or
                        logits.device != self.device or not logits.is_contiguous()):
                    raise CandidateRejected("invalid native full-position output")
                flat = logits.view(-1, v)
                for start in range(0, b * t, 64):
                    self.live()
                    if not torch.isfinite(flat[start:start + 64]).all():
                        raise CandidateRejected("nonfinite native output")
                regrets = {}
                for key, token in targets.items():
                    self.live()
                    row = logits[key[0], key[1]].float()
                    regret = float(row.max() - row[token])
                    if not math.isfinite(regret):
                        raise CandidateRejected("nonfinite native regret")
                    regrets[key] = regret
                self.live()
                self.peak_guard()
                self.live()
                return regrets
        finally:
            # A failed drain retains self.active and partially constructed shell
            # in the engine registry. Never turn device failure into fallback.
            torch.cuda.synchronize(self.device)
            self.active.clear()

    def release_after_drain(self):
        self.model = None
        self.constructing.clear()
        self.active.clear()
        self.closed = True
        self.engine = None


def owner_for(engine, deadline):
    owner = getattr(engine, 'native_check_owner', None)
    if owner is None:
        owner = NativeOwner(engine, deadline)
    if owner.closed or owner.deadline != deadline or owner.engine is not engine:
        raise CandidateRejected("stale native owner")
    return owner


def compare_row(engine, actual, expected, selected_keys, job_factory, evidence, deadline):
    # Preserve the original vector check BEFORE relaxing only the ID veto.
    _close(actual, expected, "full logits")
    chosen_ids, serial_ids = actual.argmax(-1), expected.argmax(-1)
    if torch.equal(chosen_ids, serial_ids):
        return
    chosen, serial = chosen_ids.tolist(), serial_ids.tolist()
    if len(selected_keys) != len(chosen):
        raise CandidateRejected("native selected-row map mismatch")
    case, targets = job_factory()
    if evidence and (evidence['case'] != case or evidence['targets'] != targets):
        raise CandidateRejected("stale native evidence case/targets")
    owner = owner_for(engine, deadline)
    if not evidence:
        evidence.update(case=case, targets=dict(targets), regrets=owner.check(case, targets))
    for key, y, ref in zip(selected_keys, chosen, serial):
        if y == ref:
            continue
        if evidence['targets'].get(key) != y or key not in evidence['regrets']:
            raise CandidateRejected("stale native selected target")
        if evidence['regrets'][key] > MARGIN:
            raise CandidateRejected("native own-prefix regret exceeds 1.0")
    owner.rescue_active = True


def compare_complete_trials(engine, prompts, first, trials, output, deadline):
    if len(trials) != 2:
        raise CandidateRejected("both timed complete trials required")
    cases = []
    for actual, expected in trials:
        cases.append(complete_case(prompts, first, actual, output))
        complete_case(prompts, first, expected, output)
    owner = getattr(engine, 'native_check_owner', None)
    if not ((owner is not None and owner.rescue_active) or
            any(actual != expected for actual, expected in trials)):
        return
    owner = owner_for(engine, deadline)
    # BOTH timers are already stopped. Includes first output and retroactively
    # verifies trial one if only trial two diverged from serial greedy.
    for case, targets in cases:
        regrets = owner.check(case, targets)
        if any(value > MARGIN for value in regrets.values()):
            raise CandidateRejected("complete native own-prefix regret exceeds 1.0")
    owner.rescue_active = True
