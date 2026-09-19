"""Exact fixed-size decode graph chunks, captured during workload warmup.

The model callback must perform ONE actual decode step, update all changing
CUDA state (token, position, KV cache) in place, and return the newly selected
token tensor. Its return may alias ``token``. Python counters and tensor
addresses must not change. No host reads or nested graph replay in callback.

``reset_state`` restores the already-prefilled CUDA state before each warmup
and capture. Usually it copies an independently cloned first-token tensor and
fills the position with prompt length; decode never changes the prompt's KV
slots, so the complete cache need not be copied. Reset again before emitting
the warmup generation itself. Never retain prompt content between generate
calls: the engine must prefill each new prompt over the same persistent cache.
"""

import torch


class DecodeChunks:
    """Capture a deterministic partition of the tokens AFTER the first token.

    ``token`` is a persistent CUDA int64 tensor with B elements. ``steps`` is
    max_new_tokens - 1, NOT max_new_tokens. Capture happens in the constructor;
    construct only during untimed workload warmup. Caller still emits the
    ordinary prefill result first, then ``yield from chunks.generate()``.

    No graph memory pool is shared: this avoids replay-order assumptions when
    a tail graph and the main graph have different temporary live ranges.
    """

    def __init__(self, decode, token, steps, reset_state, chunk_size=4):
        if steps < 0 or chunk_size < 1:
            raise ValueError("steps must be nonnegative and chunk_size positive")
        self.steps = int(steps)
        self.chunk_size = int(chunk_size)
        full_chunks, tail = divmod(self.steps, self.chunk_size)
        self.schedule = [self.chunk_size] * full_chunks
        if tail:
            self.schedule.append(tail)
        self.graphs = {}
        self.outputs = {}
        self.capture_stream = torch.cuda.Stream(device=token.device)

        with torch.inference_mode():
            for size in dict.fromkeys(self.schedule):
                output = torch.empty(
                    (size, token.numel()), dtype=token.dtype, device=token.device
                )
                # Explicit side-stream warmup initializes every callback path.
                torch.cuda.synchronize(token.device)
                with torch.cuda.stream(self.capture_stream):
                    reset_state()
                    for index in range(size):
                        output[index].copy_(decode().reshape(-1))
                self.capture_stream.synchronize()
                reset_state()
                torch.cuda.synchronize(token.device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=self.capture_stream):
                    for index in range(size):
                        output[index].copy_(decode().reshape(-1))
                # Instantiate and execute outside measured generations.
                graph.replay()
                torch.cuda.synchronize(token.device)
                self.graphs[size] = graph
                self.outputs[size] = output
            reset_state()
            torch.cuda.synchronize(token.device)

    def generate(self):
        """Yield precisely ``steps`` host lists from the caller's current state.

        Each chunk is copied to independent Python lists before replay can
        overwrite its GPU output buffer. The next chunk is queued before the
        current chunk's host lists are yielded, overlapping protocol writes
        with legitimate generation. All final GPU work is complete before the
        final yield. No speculative or surplus tokens are computed.
        """
        if not self.schedule:
            return
        self.graphs[self.schedule[0]].replay()
        for index, size in enumerate(self.schedule):
            host_rows = self.outputs[size].tolist()
            if index + 1 < len(self.schedule):
                self.graphs[self.schedule[index + 1]].replay()
            yield from host_rows
