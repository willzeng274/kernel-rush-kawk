"""Retained #32 plus one fail-closed W12 two-dimensional Lookahead option."""
import time
import torch
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources
from retained_engine import Engine as RetainedEngine
from lookahead_graph import LookaheadGraph
from lookahead_lifetime import Control, CandidateRejected
from lookahead_runtime import request, timed_complete, whole_call_admission
from lookahead_validate import restore, validate, prices


class Engine(RetainedEngine):
    def __init__(self, model_path):
        # RetainedEngine still owns its original, unmodified 180-second tuner.
        self.la_control = Control(time.monotonic(), lambda: torch.cuda.synchronize())
        self.la_graphs = self.la_costs = None
        self.la_decided = False
        super().__init__(model_path)

    def _allocate(self, batch, prompt, output):
        self.la_control.drain()
        self.la_graphs = self.la_costs = None
        self.la_control.owners.clear()  # Only after successful drain.
        self.la_decided = False
        super()._allocate(batch, prompt, output)

    def _memory_guard(self):
        self.la_control.live()
        free, total = torch.cuda.mem_get_info()
        cache = sum(x.numel() * x.element_size() for x in self.keys + self.values)
        # Both geometries, full-cache byte guards, cuBLAS/capture/allocator room.
        b, cap = self.batch, self.capacity
        splits12 = (cap + 11 + 127) // 128
        scratch = 147456 * b * 13
        logits = b * 13 * self.model.config.vocab_size * 2
        activations = b * 13 * 52000 * 2
        partial = b * 32 * (12 * splits12 + self.splits) * 130 * 4
        extra = scratch + logits + activations + partial + 2 * cache + 2 * 2**30
        if (torch.cuda.memory_allocated() + extra > .85 * total
                or torch.cuda.max_memory_allocated() > .85 * total
                or free < extra):
            raise CandidateRejected("optional memory preflight")

    def _prepare_lookahead(self, prompts, output):
        c = self.la_control
        accepted = False
        # Prepare the REAL retained generator, including B1 speculative/chunk
        # captures. Its complete first warmup work consumes the original budget.
        native = RetainedEngine.generate(self, prompts, output)
        try:
            for _ in native:
                pass
        finally:
            native.close()
            c.drain()
        try:
            c.live("base", 10.0)
            baseline = timed_complete(self, lambda: RetainedEngine.generate(self, prompts, output), "base")
            restore(self, prompts)
            c.reserve()
            c.live()
            self._memory_guard()
            c.live("constructor_one", 20.0)
            start = time.perf_counter()
            one = LookaheadGraph(self, 1)
            c.observed("constructor_one", time.perf_counter() - start)
            c.live("constructor_wide", 45.0)
            start = time.perf_counter()
            wide = LookaheadGraph(self, 12)
            c.observed("constructor_wide", time.perf_counter() - start)
            self.la_graphs = {1: one, 12: wide}
            validate(self, self.la_graphs, prompts, output)
            d0 = baseline[2] / (output - 1)
            costs = prices(self, self.la_graphs, prompts, output, d0)
            self.la_costs = whole_call_admission(
                self, prompts, output, costs,
                lambda: RetainedEngine.generate(self, prompts, output))
            self._memory_guard()
            c.live()
            accepted = True
        except (CandidateRejected, TimeoutError, CompilationError, OutOfResources,
                torch.cuda.OutOfMemoryError):
            # Only optional setup rejects. Runtime errors and failed drains
            # propagate; measured requests never catch their own failures.
            accepted = False
        finally:
            c.drain()
            if not accepted:
                self.la_graphs = self.la_costs = None
                c.owners.clear()
            # Always reset prompt KV/position/current IDs after setup trials.
            # No setup answer list is returned by the real warmup generation.
            restore(self, prompts)
            self.la_decided = True

    def generate(self, input_ids, max_new_tokens):
        if max_new_tokens <= 0:
            return
        self.la_control.healthy()
        try:
            batch, prompt = len(input_ids), len(input_ids[0])
            supported = 1 <= batch <= 16 and prompt >= 1 and max_new_tokens >= 7
            if not supported:
                yield from RetainedEngine.generate(self, input_ids, max_new_tokens)
                return
            with torch.inference_mode():
                if self.shape != (batch, prompt, max_new_tokens) or not self.la_decided:
                    self._prepare_lookahead(input_ids, max_new_tokens)
                if self.la_graphs is None:
                    yield from RetainedEngine.generate(self, input_ids, max_new_tokens)
                else:
                    yield from request(self, input_ids, max_new_tokens, self.la_costs)
        finally:
            self.la_control.drain()
