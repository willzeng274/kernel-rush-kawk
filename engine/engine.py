"""Captured fused prefill, native layout, and bounded exact B1 speculation.

All other batches and outputs too short for the fixed attempt budget delegate
to the passing fused-prefill/native-layout chunk engine unchanged.
"""
import torch

from full_prefill_engine import Engine as PrefillEngine
from chunk_graph import DecodeChunks
from speculative_host import RequestState
from verify_graph import VerifyGraph
from fused_cache_attention import FusedCacheAttention


class Engine(PrefillEngine):
    def __init__(self, model_path: str) -> None:
        super().__init__(model_path)

    def _allocate(self, batch, prompt, output):
        super()._allocate(batch, prompt, output)
        self.native_chunks = None
        self.verifier = None

    def _capture_speculative(self, first):
        self.fused_cache_attention = FusedCacheAttention(self, self._attention_deadline)
        def decode():
            self._step()
            return self.ids

        def reset():
            self.position.fill_(self.prompt)
            self.ids.copy_(first)

        # Acceptance changes the tail length, so every possible size is warmed
        # and captured before any measured request reuses this shape.
        self.native_chunks = {
            size: DecodeChunks(decode, self.ids, size, reset, chunk_size=size)
            for size in (1, 2, 3, 4)
        }
        self.verifier = VerifyGraph(self, first)
        reset()
        torch.cuda.synchronize(self.ids.device)

    def _launch_plan(self, plan):
        if plan.kind == "verify":
            self.verifier.replay(plan.inputs)
        else:
            self.native_chunks[plan.steps].graphs[plan.steps].replay()

    def _read_plan(self, plan):
        if plan.kind == "verify":
            return self.verifier.output.tolist()
        return [row[0] for row in
                self.native_chunks[plan.steps].outputs[plan.steps].tolist()]

    def _generate_speculative(self, prompt, first, output):
        # Called only after the prefill token has been emitted to the caller.
        state = RequestState(prompt, first, output)
        plan = state.next_plan()
        if plan is None:
            return
        self._launch_plan(plan)
        while plan is not None:
            committed = state.commit(plan, self._read_plan(plan))
            if plan.kind == "verify":
                # Speculative KV beyond this logical prefix remains masked and
                # is overwritten before being consumed by any future step.
                self.position.fill_(state.position)
                self.ids.fill_(state.pending)
            following = state.next_plan()
            if following is not None:
                self._launch_plan(following)
            elif plan.kind == "verify":
                # The host read finishes verification; finish the subsequent
                # position/token restoration too before the final yield.
                torch.cuda.current_stream().synchronize()
            for token in committed:
                yield [token]
            plan = following

    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        if max_new_tokens <= 0:
            return
        batch, prompt = len(input_ids), len(input_ids[0])
        if batch != 1 or max_new_tokens < 25:
            yield from super().generate(input_ids, max_new_tokens)
            return
        with torch.inference_mode():
            if self.shape != (batch, prompt, max_new_tokens):
                self._allocate(batch, prompt, max_new_tokens)
            current = torch.tensor(input_ids, device="cuda:0", dtype=torch.int64)
            self.prefill_input.copy_(current)
            if self.prefill_graph is None:
                self._capture_prefill()
            # Every replay rewrites this request's full prompt KV. Keep the
            # prefill result independent of native IDs, which graph warmup
            # mutates while preparing all four possible native tail lengths.
            self.prefill_graph.replay()
            first = self.ids.clone()
            if self.native_chunks is None:
                self._capture_speculative(first)
            self.position.fill_(prompt)
            self.ids.copy_(first)
            first_row = self.ids.tolist()
            yield first_row
            yield from self._generate_speculative(
                input_ids[0], first_row[0], max_new_tokens,
            )
