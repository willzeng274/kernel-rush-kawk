"""Captured prefill plus exact four-step decode chunks."""

import torch

from engine_base import Engine as BaseEngine, PrefillCache
from chunk_graph import DecodeChunks


class Engine(BaseEngine):
    def __init__(self, model_path: str) -> None:
        super().__init__(model_path)

    def _allocate(self, batch, prompt, output):
        super()._allocate(batch, prompt, output)
        self.chunks = None
        self.prefill_input = torch.empty((batch, prompt), dtype=torch.int64, device="cuda:0")
        self.prefill_graph = None

    def _prefill_eager(self):
        # A new adapter presents length zero while graph capture records all
        # full-prompt KV writes. Every replay rewrites those captured buffers.
        cache = PrefillCache(self.keys, self.values)
        result = self.model(
            input_ids=self.prefill_input, past_key_values=cache,
            use_cache=True, logits_to_keep=1, return_dict=True,
        )
        self.ids.copy_(result.logits[:, -1, :].argmax(dim=-1))

    def _capture_prefill(self):
        current = torch.cuda.current_stream()
        stream = torch.cuda.Stream()
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._prefill_eager()
        current.wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        # Use a separate private pool from the decode graph.
        with torch.cuda.graph(graph, stream=stream):
            self._prefill_eager()
        current.wait_stream(stream)
        self.prefill_graph = graph

    def _capture_chunks(self, first, steps):
        def decode():
            self._step()
            return self.ids

        def reset():
            self.position.fill_(self.prompt)
            self.ids.copy_(first)

        self.chunks = DecodeChunks(
            decode, self.ids, steps, reset, chunk_size=4
        )

    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        if max_new_tokens <= 0:
            return
        batch, prompt = len(input_ids), len(input_ids[0])
        with torch.inference_mode():
            if self.shape != (batch, prompt, max_new_tokens):
                self._allocate(batch, prompt, max_new_tokens)
            current = torch.tensor(input_ids, device="cuda:0", dtype=torch.int64)
            self.prefill_input.copy_(current)
            if self.prefill_graph is None:
                self._capture_prefill()
            self.prefill_graph.replay()
            first = self.ids.clone()
            if max_new_tokens > 1 and self.chunks is None:
                self._capture_chunks(first, max_new_tokens - 1)
            self.position.fill_(prompt)
            self.ids.copy_(first)
            yield self.ids.tolist()
            if max_new_tokens > 1:
                yield from self.chunks.generate()
