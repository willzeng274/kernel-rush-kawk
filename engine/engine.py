"""Exact custom decode with four-step captured chunks after the first token."""

import time

import torch

from engine_base import Engine as BaseEngine, PrefillCache
from chunk_graph import DecodeChunks
from native_layout import NativeLayout


class Engine(BaseEngine):
    def __init__(self, model_path: str) -> None:
        self._layout_deadline = time.monotonic() + 180.0
        self.native_layout = None
        super().__init__(model_path)

    def _allocate(self, batch, prompt, output):
        super()._allocate(batch, prompt, output)
        self.chunks = None
        # Select once on first warmup, after allocating the actual KV cache.
        # Later allocations/generations cannot trigger another benchmark.
        if self.native_layout is None:
            self.native_layout = NativeLayout(self, self._layout_deadline)

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
            cache = PrefillCache(self.keys, self.values)
            result = self.model(
                input_ids=current, past_key_values=cache, use_cache=True,
                logits_to_keep=1, return_dict=True,
            )
            first = result.logits[:, -1, :].argmax(dim=-1)
            del result
            if max_new_tokens > 1 and self.chunks is None:
                self._capture_chunks(first, max_new_tokens - 1)
            self.position.fill_(prompt)
            self.ids.copy_(first)
            yield self.ids.tolist()
            if max_new_tokens > 1:
                yield from self.chunks.generate()
