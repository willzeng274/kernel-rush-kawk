"""Exact request-local W4 prediction recycling with independent B<=16 progress.

Retained #32 is the fallback before any divergent state is created. Once this
path starts, its verified W1 graph preserves per-sequence logical positions.
There is no acceptance rollback, approximate attention, KV eviction or delay.
"""
import sys
import time
import torch
from retained_engine import Engine as RetainedEngine
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources
from recycled_graph import ChainGraph
from recycled_validate import CandidateRejected, validate_numerics, measure_and_admit, run_request


class Engine(RetainedEngine):
    def __init__(self, model_path):
        # The inherited 180s tuner deadline already includes model loading.
        # Leave 30s below the platform's 300s ceiling for recovery/generation.
        self.recycle_deadline = time.monotonic() + 270.0
        self.recycle_quarantine = []
        super().__init__(model_path)

    def _allocate(self, batch, prompt, output):
        super()._allocate(batch, prompt, output)
        self.recycle_decided = False
        self.recycle_graphs = None
        self.recycle_costs = None

    def _prepare_recycling(self, input_ids, first, output):
        self.recycle_decided = True
        if time.monotonic() + 50.0 >= self.recycle_deadline:
            return
        # Baseline failures propagate; only candidate construction/admission
        # below may be rejected in favor of the retained path.
        if self.chunks is None:
            self._capture_chunks(first, output - 1)
        if time.monotonic() + 40.0 >= self.recycle_deadline:
            return
        graphs = None
        try:
            graphs = {1: ChainGraph(self, 1)}
            if time.monotonic() + 30.0 >= self.recycle_deadline:
                return
            graphs[4] = ChainGraph(self, 4)
            validate_numerics(self, graphs, input_ids, first, self.recycle_deadline)
            costs = measure_and_admit(self, graphs, input_ids, first, output,
                                      self.recycle_deadline)
            if costs is not None and time.monotonic() < self.recycle_deadline:
                self.recycle_graphs, self.recycle_costs = graphs, costs
                print("recycling: validated W1/W4 and complete-request gain", file=sys.stderr)
        except (CandidateRejected, TimeoutError, CompilationError, OutOfResources) as error:
            # Safe only before the first emitted token and before any timed
            # divergent request. Candidate runtime exceptions are never caught.
            print("recycling disabled:", type(error).__name__, str(error)[:180], file=sys.stderr)
        finally:
            # Completed owners are pinned before an error can unwind this frame.
            # A failed drain propagates and keeps the quarantine intact; such a
            # device/capture failure is never treated as a recoverable fallback.
            owners = list(graphs.values()) if graphs is not None else []
            self.recycle_quarantine.extend(owners)
            torch.cuda.synchronize()
            for owner in owners:
                self.recycle_quarantine.remove(owner)
            if sys.exc_info()[0] is None:
                # Only normal completion or an explicitly handled candidate
                # rejection may restore and proceed to the retained engine.
                self.prefill_graph.replay()
                self.position.fill_(self.prompt)
                self.ids.copy_(first)
                torch.cuda.synchronize()

    def generate(self, input_ids, max_new_tokens):
        if max_new_tokens <= 0:
            return
        batch, prompt = len(input_ids), len(input_ids[0])
        if not (1 <= batch <= 16 and max_new_tokens >= 8 and prompt >= 3):
            yield from super().generate(input_ids, max_new_tokens)
            return
        with torch.inference_mode():
            if self.shape != (batch, prompt, max_new_tokens):
                self._allocate(batch, prompt, max_new_tokens)
            if self.recycle_decided and self.recycle_graphs is None:
                yield from super().generate(input_ids, max_new_tokens)
                return
            self.prefill_input.copy_(torch.tensor(input_ids, dtype=torch.int64, device="cuda:0"))
            if self.prefill_graph is None:
                self._capture_prefill()
            self.prefill_graph.replay()
            first = self.ids.clone()
            if not self.recycle_decided:
                self._prepare_recycling(input_ids, first, max_new_tokens)
            if self.recycle_graphs is None:
                yield from super().generate(input_ids, max_new_tokens)
                return
            first_row = first.tolist()
            yield first_row
            # Request tables and histories are created after the first token;
            # every call receives new tables, lengths, pending IDs and FIFOs.
            yield from run_request(self, self.recycle_graphs, input_ids, first_row,
                                   max_new_tokens, *self.recycle_costs)
