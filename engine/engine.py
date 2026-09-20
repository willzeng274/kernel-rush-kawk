"""Retained #32 plus three optional licensed SGLang B1 BF16 GEMV families."""
import time
import torch
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources
from retained_engine import Engine as RetainedEngine
from full_prefill_engine import Engine as PrefillEngine
from gemv_lifetime import Control, CandidateRejected
from gemv_layout import OptionalGemvLayout
from gemv_capture import NativeChunks
from gemv_runtime import snapshot, bind, timed_complete, require_stream, whole_call_admission
from gemv_validate import restore, select_families, validate


class Engine(RetainedEngine):
    def __init__(self, model_path):
        self.gv_control = Control(time.monotonic(), lambda: torch.cuda.synchronize())
        self.gv_a = self.gv_b = self.gv_bound = self.gv_retained_layout = None
        self.gv_decided = self.gv_timing = False
        self.gv_captures = self.gv_generation = 0
        super().__init__(model_path)
        self.gv_capable = (self.h == 2560 and self.i == 9728 and len(self.layers) == 36
                           and self.model.config.vocab_size == 151936
                           and self.model.lm_head.weight.dtype == torch.bfloat16
                           and torch.cuda.get_device_capability() == (9, 0))

    def _allocate(self, batch, prompt, output):
        if self.gv_timing:
            raise RuntimeError('timed request attempted allocation')
        c = self.gv_control
        c.drain()
        if self.gv_retained_layout is not None:
            self.native_layout = self.gv_retained_layout
        self.gv_a = self.gv_b = self.gv_bound = None
        c.owners.clear()
        self.gv_decided = False
        self.gv_generation += 1
        super()._allocate(batch, prompt, output)
        self.gv_retained_layout = self.native_layout

    def _capture_prefill(self):
        if self.gv_timing:
            raise RuntimeError('timed request attempted prefill capture')
        self.gv_captures += 1
        return PrefillEngine._capture_prefill(self)

    def _capture_chunks(self, first, steps):
        if self.gv_timing:
            raise RuntimeError('timed request attempted ordinary capture')
        self.gv_captures += 1
        return PrefillEngine._capture_chunks(self, first, steps)

    def _capture_speculative(self, first):
        if self.gv_timing:
            raise RuntimeError('timed request attempted speculative capture')
        self.gv_captures += 1
        return RetainedEngine._capture_speculative(self, first)

    def _capture(self, first):
        if self.gv_timing:
            raise RuntimeError('timed request attempted base capture')
        self.gv_captures += 1
        return super()._capture(first)

    def _gv_memory_guard(self, extra_bytes=0):
        c = self.gv_control
        c.live()
        free, total = torch.cuda.mem_get_info()
        cache = sum(t.numel() * t.element_size() for t in self.keys + self.values)
        written = 2 * 36 * self.batch * 8 * 4 * 128 * 2
        logits = self.batch * self.model.config.vocab_size * 2
        # Existing A/B pools/layouts are already in current allocation. Budget
        # a full cache snapshot, written interval, logits, private tensors and
        # two GiB of graph/allocator headroom without resetting historical peak.
        required = cache + written + logits + int(extra_bytes) + 2 * 2**30
        if (extra_bytes < 0 or free < required
                or torch.cuda.memory_allocated() + required > .85 * total
                or torch.cuda.max_memory_allocated() >= .85 * total):
            raise CandidateRejected('optional GEMV memory preflight')

    def _run_retained(self, prompts, output):
        """Same cheap steady guards and final drain in trials and deployed calls."""
        c = self.gv_control
        c.healthy()
        cfg = self.gv_bound
        shape = (len(prompts), len(prompts[0]), output)
        if cfg is not None and self.shape == shape:
            if (cfg.generation != self.gv_generation or self.native_layout is not cfg.layout
                    or self.fused_cache_attention is not cfg.attention or self.chunks is not cfg.chunks
                    or self.native_chunks is not cfg.native_chunks or self.verifier is not cfg.verifier
                    or self.prefill_graph is not cfg.prefill or self.dense_prefill is not cfg.dense_prefill):
                raise RuntimeError('steady GEMV configuration identity changed')
        gen = None
        try:
            gen = RetainedEngine.generate(self, prompts, output)
            yield from gen
        finally:
            try:
                if gen is not None:
                    gen.close()
            finally:
                c.drain()

    def _prepare_gemv(self, prompts, output):
        c = self.gv_control
        # Cold selectors/capture are charged once to t0, never multiplied into
        # the reserve. They run outside the optional-error handler.
        cold = self._run_retained(prompts, output)
        try:
            for _ in cold:
                pass
        finally:
            try:
                cold.close()
            finally:
                c.drain()
        self.gv_a = snapshot(self)
        self.gv_bound = self.gv_a
        accepted = mandatory_failed = False
        def retained_stream():
            nonlocal mandatory_failed
            mandatory = self.gv_bound is self.gv_a
            try:
                yield from self._run_retained(prompts, output)
            except Exception:
                if mandatory:
                    mandatory_failed = True
                raise
        try:
            with torch.inference_mode():
                c.live('base', 5.)
                pilot = timed_complete(self, self.gv_a, retained_stream, 'base')
                require_stream(pilot[0], self.batch, output, self.model.config.vocab_size)
                pilot = None
                c.live('restore', 2.)
                restore(self, prompts)
                c.reserve()
                c.live()
                chosen = select_families(self, self.gv_retained_layout)
                layout = OptionalGemvLayout(self, self.gv_retained_layout, chosen)
                c.register(layout)
                first = restore(self, prompts)
                self.native_layout = layout
                if self.gv_a.native_chunks is not None:
                    native_chunks = {}
                    for size in (1, 2, 3, 4):
                        native_chunks[size] = NativeChunks(self, first, size, size)
                    self.native_chunks = native_chunks
                    self.chunks = self.gv_a.chunks
                else:
                    self.native_chunks = None
                    self.chunks = NativeChunks(self, first, output - 1, 4)
                self.gv_b = snapshot(self, port=True)
                bind(self, self.gv_a)
                validate(self, self.gv_a, self.gv_b, prompts)
                whole_call_admission(self, self.gv_a, self.gv_b, retained_stream)
                self._gv_memory_guard()
                c.live()
                accepted = True
        except (CandidateRejected, TimeoutError, CompilationError, OutOfResources, torch.cuda.OutOfMemoryError):
            if mandatory_failed:
                raise
        finally:
            c.drain()
            bind(self, self.gv_b if accepted else self.gv_a)
            with torch.inference_mode():
                restore(self, prompts)
            c.drain()
            if not accepted:
                self.gv_b = None
                c.owners.clear()
            self.gv_decided = True

    def generate(self, input_ids, max_new_tokens):
        if max_new_tokens <= 0:
            return
        self.gv_control.healthy()
        gen = None
        try:
            batch, prompt = len(input_ids), len(input_ids[0])
            eligible = self.gv_capable and batch == 1 and prompt >= 1 and max_new_tokens >= 2
            if eligible and (self.shape != (batch, prompt, max_new_tokens) or not self.gv_decided):
                self._prepare_gemv(input_ids, max_new_tokens)
            gen = self._run_retained(input_ids, max_new_tokens)
            yield from gen
        finally:
            try:
                if gen is not None:
                    gen.close()
            finally:
                self.gv_control.drain()
