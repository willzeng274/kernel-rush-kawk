"""Model-based prompt seeds with one vocabulary projection per unique token.

The captured prefill copies its final hidden tail to stable owned storage.
Variable-row seeding runs eagerly on the same stream, before the first output
and draft. These predictions are proposals; the full model still verifies each
emitted token. Repeated token IDs explicitly use their latest prompt occurrence.
"""

from __future__ import annotations

import torch


def latest_prompt_rows(input_ids: list[list[int]], tail_limit: int = 1024):
    """Return offsets in the flattened hidden tails and their unique token IDs."""
    tail = min(len(input_ids[0]), tail_limit)
    last = {}
    for batch, row in enumerate(input_ids):
        for column, token in enumerate(row[-tail:]):
            last[token] = batch * tail + column
    pairs = sorted(last.items(), key=lambda pair: pair[1])
    return [offset for _, offset in pairs], [token for token, _ in pairs]


class PromptModelSeed:
    def __init__(self, batch: int, prompt: int, hidden: int, device):
        self.batch, self.prompt, self.width = batch, prompt, hidden
        self.tail = min(prompt, 1024)
        self.capacity = batch * self.tail
        # Allocated outside capture and never replaced. Eager self-checks can
        # copy into this storage without changing the graph's output address.
        self.hidden = torch.empty((batch, self.tail, hidden), dtype=torch.bfloat16, device=device)
        self.host_offsets = torch.empty((self.capacity,), dtype=torch.int64, pin_memory=True)
        self.host_tokens = torch.empty((self.capacity,), dtype=torch.int64, pin_memory=True)
        self.offsets = torch.empty((self.capacity,), dtype=torch.int64, device=device)
        self.tokens = torch.empty((self.capacity,), dtype=torch.int64, device=device)

    def capture_hidden(self, h: torch.Tensor) -> None:
        self.hidden.copy_(h.view(self.batch, self.prompt, self.width)[:, -self.tail:])

    @torch.inference_mode()
    def seed(self, input_ids: list[list[int]], lm_head: torch.Tensor, recycler) -> None:
        offsets, tokens = latest_prompt_rows(input_ids)
        count = len(tokens)
        if count > self.capacity:
            raise ValueError("prompt seed count exceeds the plan's capacity")
        if not count:
            return
        self.host_offsets.numpy()[:count] = offsets
        self.host_tokens.numpy()[:count] = tokens
        self.offsets[:count].copy_(self.host_offsets[:count], non_blocking=True)
        self.tokens[:count].copy_(self.host_tokens[:count], non_blocking=True)
        h = self.hidden.view(-1, self.width)
        # Keep the original 2048-row projection bound. No vocabulary tensor is
        # copied to the CPU and no graph or Triton specialization is created.
        for start in range(0, count, 2048):
            end = min(start + 2048, count)
            selected = h.index_select(0, self.offsets[start:end])
            logits = selected @ lm_head.t()
            recycler.update(self.tokens[start:end], logits)
