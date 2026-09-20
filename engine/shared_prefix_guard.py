"""Owned, bounded admission of a request-local shared-prefix prefill graph.

All comparisons use the original full route on the identical supplied context.
The 270-second setup ceiling never changes any original 180-second selector.
"""
import math
import time
from types import SimpleNamespace
import torch
from torch.backends.cuda import SDPAParams, can_use_flash_attention
from torch.nn.attention import SDPBackend, sdpa_kernel
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources


class Rejected(Exception):
    pass


def guard_prefix(engine):
    if getattr(engine, '_prefix_pending', None) is not None:
        raise RuntimeError('shared-prefix owner has not drained')


def _admitted(trials):
    roles = ('parent', 'shared', 'shared', 'parent',
             'mismatch_parent', 'fallback', 'fallback', 'mismatch_parent')
    if len(trials) != 8 or tuple(t.role for t in trials) != roles:
        return False
    if not all(math.isfinite(t.ttft) and math.isfinite(t.total) and
               0 < t.ttft <= t.total for t in trials):
        return False
    if any(t.tokens != trials[0].tokens for t in trials[1:4]):
        return False
    if any(t.tokens != trials[4].tokens for t in trials[5:]):
        return False
    p, s = (trials[0], trials[3]), (trials[1], trials[2])
    f, q = (trials[5], trials[6]), (trials[4], trials[7])
    return (max(t.ttft for t in s) < .95 * min(t.ttft for t in p)
            and max(t.total for t in s) < min(t.total for t in p)
            and max(t.total for t in trials) <= 1.20 * min(t.total for t in trials)
            and max(t.ttft for t in f) <= 1.01 * min(t.ttft for t in q)
            and max(t.total for t in f) <= 1.01 * min(t.total for t in q))


class PrefixOwner:
    POISON = 0x7FC1  # BF16 quiet NaN, checked as exact int16 storage bits.

    def __init__(self, engine, output, deadline):
        # No allocation or device call before prepare_prefix publishes this.
        self.engine, self.output, self.deadline = engine, output, deadline
        self.device = engine.prefill_input.device
        self.prefix_length = engine.prompt // 2
        self.enabled = False
        self.prefix = self.suffix = self.suffix_input = None
        self.copies = ()
        self.graph = self.stream = self.current_input = self.host_input = None
        self.references, self.trials, self.first = [], [], None
        self.generator = None
        self.shared_ids = self.mismatch_ids = None
        self.drain_error = None
        self.capturing = False
        self.price = 0.0

    def live(self):
        # Fixed full capture must complete or fail; no mid-capture timeout throw.
        if not self.capturing and time.monotonic() >= self.deadline:
            raise Rejected('shared-prefix setup deadline')

    def drain(self):
        error = None
        if self.stream is not None:
            try:
                torch.cuda.current_stream(self.device).wait_stream(self.stream)
            except BaseException as exc:
                error = exc
        try:
            torch.cuda.synchronize(self.device)
        except BaseException as exc:
            if error is None:
                error = exc
        if error is not None and self.drain_error is None:
            self.drain_error = error
        if self.drain_error is not None:
            raise self.drain_error

    def _room(self, extra=0):
        self.live()
        free, total = torch.cuda.mem_get_info(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        peak = torch.cuda.max_memory_reserved(self.device)
        self.live()
        return (extra + (512 << 20) < free
                and max(reserved + extra, peak) < int(.85 * total))

    def _sizes(self):
        e = self.engine
        prompt_refs = 2 * len(e.layers) * e.batch * 8 * e.prompt * 128 * 2
        logits_ref = e.batch * e.model.config.vocab_size * 2
        comparison = max(e.batch * 8 * e.prompt * 128,
                         e.batch * e.model.config.vocab_size) * 32
        # Conservative reserve estimate, not a Flash/capture allocation bound.
        graph = (32768 * e.batch * (e.prompt-self.prefix_length) + (64 << 20))
        suffix = 8 * e.batch * (e.prompt-self.prefix_length)
        return prompt_refs + logits_ref + comparison + graph + suffix, comparison

    def _forecast(self, remaining, extra=0.0):
        self.live()
        # Include one final complete real generation still owed to the caller.
        cost = 1.25 * self.price * (remaining + 1) + extra
        if (not math.isfinite(cost) or cost <= 0
                or time.monotonic() + cost >= self.deadline):
            raise Rejected('insufficient complete-request setup time')

    def _copy_input(self, ids):
        self.live()
        self.current_input = torch.empty((self.engine.batch, self.engine.prompt),
                                        dtype=torch.int64, device=self.device)
        # Retain both ends before upload, including partially failed copies.
        self.host_input = torch.tensor(ids, dtype=torch.int64, device="cpu")
        self.live()
        self.current_input.copy_(self.host_input)
        self.live()
        self.engine.prefill_input.copy_(self.current_input)
        self.live()

    def _flash_supported(self):
        self.live()
        with torch.cuda.device(self.device), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            for span in (self.prefix, self.suffix):
                q = span.query_view
                for k, v in span.kv:
                    self.live()
                    if (tuple(q.shape) != (span.batch, 32, span.length, 128)
                            or tuple(k.shape) != (span.batch, 8, span.start+span.length, 128)
                            or tuple(v.shape) != tuple(k.shape)
                            or not all(t.is_cuda and t.device == self.device
                                       and t.dtype == torch.bfloat16 and t.stride(-1) == 1
                                       for t in (q, k, v))):
                        return False
                    params = SDPAParams(q, k, v, None, 0.0, span.start == 0, True)
                    if not can_use_flash_attention(params):
                        return False
        self.live()
        return True

    def _probe_price(self, base_setup):
        self.live()
        # Price a bounded, already-captured prefill and at most one decode chunk
        # before starting a possibly long complete-N generation.
        if (not math.isfinite(base_setup) or base_setup < 0
                or time.monotonic() + max(.05, base_setup) >= self.deadline):
            raise Rejected('insufficient bounded-pilot headroom')
        e = self.engine
        self._copy_input(self.shared_ids)
        start = time.monotonic()
        e.prefill_graph.replay()
        torch.cuda.synchronize(self.device)
        prefill = time.monotonic()-start
        self.live()
        decode = 0.0
        if self.output > 1:
            size = e.chunks.schedule[0]
            e.position.fill_(e.prompt)
            start = time.monotonic()
            e.chunks.graphs[size].replay()
            e.chunks.outputs[size].tolist()
            torch.cuda.synchronize(self.device)
            elapsed = time.monotonic()-start
            self.live()
            decode = elapsed * math.ceil((self.output-1)/size) * e.capacity/e.prompt
        self.price = max(.001, prefill + decode)
        self._forecast(8)

    def _trial(self, role, ids):
        self.live()
        e = self.engine
        self.drain()
        self.live()
        # The generator body, conversion, genuine detector and output reads are
        # all inside the clock. Original parent has no extra detection work.
        started = time.monotonic()
        parent = role in ('parent', 'mismatch_parent')
        self.generator = (e._parent_stream(ids, self.output) if parent
                          else e._candidate_stream(ids, self.output, self))
        rows, first_at = [], None
        try:
            for _ in range(self.output):
                self.live()
                try:
                    row = next(self.generator)
                except StopIteration as exc:
                    raise Rejected("missing generated row") from exc
                now = time.monotonic()
                if first_at is None:
                    first_at = now
                if (len(row) != e.batch or any(type(t) is not int or
                        not 0 <= t < e.model.config.vocab_size for t in row)):
                    raise Rejected('malformed generated row')
                rows.append(tuple(row))
            self.live()
            try:
                next(self.generator)
            except StopIteration:
                pass
            else:
                raise Rejected('surplus generated row')
            torch.cuda.synchronize(self.device)
            stopped = time.monotonic()
            self.live()
            trial = SimpleNamespace(role=role, ttft=first_at-started,
                                    total=stopped-started, tokens=tuple(rows))
            if not (math.isfinite(trial.ttft) and math.isfinite(trial.total)
                    and 0 < trial.ttft <= trial.total):
                raise Rejected('invalid complete-call clock')
            self.trials.append(trial)
            self.price = max(self.price, trial.total)
            return trial
        finally:
            # Closing does not cancel an already-enqueued following chunk.
            # Keep the generator reachable until a successful outer drain.
            self.generator.close()

    def _save_reference(self):
        if not self._room(self._sizes()[0]):
            raise Rejected("reference allocation memory headroom")
        self._copy_input(self.shared_ids)
        self.engine.prefill_graph.replay()
        self.live()
        for pair in zip(self.engine.keys, self.engine.values):
            for cache in pair:
                self.live()
                trial = SimpleNamespace(tensor=None)
                self.references.append(trial)
                view = cache[:, :, :self.engine.prompt, :]
                trial.tensor = torch.empty_like(view, memory_format=torch.contiguous_format)
                trial.tensor.copy_(view)
        self.live()
        self.first = torch.empty_like(self.engine.logits)
        self.first.copy_(self.engine.logits)
        self.drain()
        self.live()

    def _poison(self):
        for pair in zip(self.engine.keys, self.engine.values):
            for cache in pair:
                self.live()
                cache.view(torch.int16).fill_(self.POISON)

    def _close(self, actual, expected):
        self.live()
        if tuple(actual.shape) != tuple(expected.shape):
            raise Rejected('numerical shape mismatch')
        if not torch.isfinite(actual).all().item():
            raise Rejected('nonfinite shared values')
        self.live()
        if not torch.isfinite(expected).all().item():
            raise Rejected('nonfinite original values')
        self.live()
        if not torch.allclose(actual, expected, rtol=.02, atol=.03):
            raise Rejected('full-vector numerical mismatch')
        self.live()

    def _check(self):
        e = self.engine
        _, temporary = self._sizes()
        if not self._room(temporary):
            raise Rejected('comparison memory headroom')
        self._close(e.logits, self.first)
        if not torch.equal(e.logits.argmax(-1), self.first.argmax(-1)):
            raise Rejected('first greedy mismatch')
        index = 0
        for pair in zip(e.keys, e.values):
            for cache in pair:
                self.live()
                if not self._room(temporary):
                    raise Rejected('cache comparison memory headroom')
                self._close(cache[:, :, :e.prompt, :], self.references[index].tensor)
                index += 1
                prefix = cache[:, :, :self.prefix_length, :].view(torch.int16)
                self.live()
                if not torch.equal(prefix, prefix[:1].expand_as(prefix)):
                    raise Rejected('prefix copy storage mismatch')
                self.live()
                tail = cache[:, :, e.prompt:, :].view(torch.int16)
                if not (tail == self.POISON).all().item():
                    raise Rejected('future cache storage changed')
                self.live()
        if not self._room():
            raise Rejected('observed comparison peak')

    def _warm_capture(self):
        self.live()
        self.stream = torch.cuda.Stream(device=self.device)
        current = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(current)
        start = time.monotonic()
        try:
            with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
                self.engine._shared_prefill_eager(self)
        finally:
            current.wait_stream(self.stream)
        self.drain()
        self.live()
        checked = time.monotonic()
        self._check()
        check_cost = time.monotonic()-checked
        # Cold work above is already wall-charged. Price one fully warmed eager
        # pass separately so a paid compilation is not multiplied by all trials.
        self._forecast(7, extra=self.price+check_cost)
        start = time.monotonic()
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            self.engine._shared_prefill_eager(self)
        self.drain()
        steady = time.monotonic()-start
        self.live()
        self.price = max(self.price, steady)
        self._forecast(7, extra=2*self.price+check_cost)
        e = self.engine
        graph_extra = (32768*e.batch*(e.prompt-self.prefix_length)+(64 << 20)
                       + self._sizes()[1])
        if not self._room(graph_extra):
            raise Rejected("graph capture memory headroom")
        self.live()
        self.graph = torch.cuda.CUDAGraph()
        self.live()
        self.capturing = True
        try:
            with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
                with torch.cuda.graph(self.graph, stream=self.stream):
                    self.engine._shared_prefill_eager(self)
        except BaseException as exc:
            # Invalidated captures cannot be treated as an ordinary fallback.
            self.drain_error = exc
            raise
        finally:
            self.capturing = False
            current.wait_stream(self.stream)
        self.drain()
        self.live()
        self._poison()
        self.live()
        self.graph.replay()
        self.drain()
        self._check()

    def prepare(self, input_ids, base_setup):
        e = self.engine
        self.live()
        amount, _ = self._sizes()
        if not self._room(amount):
            raise Rejected('reference/graph memory headroom')
        # Recreate built-in integer objects per row where Python permits it.
        # This avoids pointer-identity shortcuts on synthetic matching tokens.
        self.shared_ids = []
        for row in input_ids:
            self.live()
            values = input_ids[0][:self.prefix_length] + row[self.prefix_length:]
            self.shared_ids.append([int(str(t)) for t in values])
        self.mismatch_ids = []
        for row in self.shared_ids:
            self.live()
            self.mismatch_ids.append([int(str(t)) for t in row])
        self.mismatch_ids[-1][self.prefix_length-1] = (
            self.mismatch_ids[-1][self.prefix_length-1]+1) % e.model.config.vocab_size
        self.live()
        e._make_prefix_spans(self)
        if not self._flash_supported():
            raise Rejected('unsupported lower-right Flash views')
        self._probe_price(base_setup)
        self._trial('parent', self.shared_ids)
        self._forecast(7)
        self._save_reference()
        self._poison()
        self._warm_capture()
        self.drain()
        self.references.clear()
        self.first = None
        self.live()
        for role, ids in (('shared', self.shared_ids), ('shared', self.shared_ids),
                          ('parent', self.shared_ids),
                          ('mismatch_parent', self.mismatch_ids),
                          ('fallback', self.mismatch_ids), ('fallback', self.mismatch_ids),
                          ('mismatch_parent', self.mismatch_ids)):
            self._forecast(8-len(self.trials))
            self._trial(role, ids)
        self._forecast(0)
        if not self._room() or not _admitted(self.trials):
            raise Rejected('complete-call performance/numerical admission')
        self.live()
        self.enabled = True

    def clear_temporary(self):
        self.references.clear()
        self.trials.clear()
        self.first = self.current_input = self.host_input = self.generator = None
        self.shared_ids = self.mismatch_ids = None
        if not self.enabled:
            self.graph = self.stream = self.prefix = self.suffix = self.suffix_input = None
            self.copies = ()


def retire_prefix(engine):
    guard_prefix(engine)
    owner = getattr(engine, '_prefix_owner', None)
    if owner is not None:
        engine._prefix_pending = owner
        owner.drain()
        engine._prefix_owner = None
        engine._prefix_pending = None


def prepare_prefix(engine, input_ids, output, base_setup):
    guard_prefix(engine)
    started = time.monotonic()
    deadline = min(engine._layout_deadline+90.0, started+30.0)-5.0
    if (started >= deadline or len(engine.layers) != 36 or len(engine.keys) != 36
            or len(engine.values) != 36 or engine.h != 2560 or engine.i != 9728
            or engine.model.config.vocab_size != 151936):
        return
    owner = PrefixOwner(engine, output, deadline)
    engine._prefix_pending = owner
    engine._prefix_owner = owner
    try:
        try:
            with torch.inference_mode():
                owner.prepare(input_ids, base_setup)
        except (Rejected, CompilationError, OutOfResources, torch.cuda.OutOfMemoryError):
            owner.enabled = False
    finally:
        # A failed drain/capture keeps every partial object engine-reachable.
        owner.drain()
        owner.clear_temporary()
        engine._prefix_pending = None
