"""Owned unchanged native steps; exact port launches proved inside each capture."""
import time
import torch


class NativeChunks:
    def __init__(self, e, first, steps, chunk_size=4):
        self.control = c = e.gv_control
        c.register(self)
        self.steps, self.chunk_size = int(steps), int(chunk_size)
        if self.steps < 1 or self.chunk_size < 1:
            raise ValueError('invalid native chunk schedule')
        full, tail = divmod(self.steps, self.chunk_size)
        self.schedule = [self.chunk_size] * full + ([tail] if tail else [])
        self.graphs, self.outputs, self.proofs = {}, {}, {}
        self.capture_stream = self.pending_graph = None
        self.layout = e.native_layout
        self.first = first
        with torch.inference_mode():
            c.live('native_capture', 20.)
            e._gv_memory_guard()
            self.capture_stream = torch.cuda.Stream(device=e.ids.device)
            for size in dict.fromkeys(self.schedule):
                c.live('native_capture', 20.)
                started = time.perf_counter()
                e._gv_memory_guard()
                self.outputs[size] = torch.empty((size, e.ids.numel()), dtype=e.ids.dtype, device=e.ids.device)
                output = self.outputs[size]
                c.drain()
                with torch.cuda.stream(self.capture_stream):
                    self._reset(e, self.first)
                    for index in range(size):
                        c.live()
                        e._step()
                        c.live()
                        output[index].copy_(e.ids.reshape(-1))
                c.wait(self.capture_stream.synchronize)
                self._reset(e, self.first)
                c.drain()
                c.live()
                self.pending_graph = torch.cuda.CUDAGraph()
                c.live()
                e.gv_captures += 1
                self.layout.begin_capture(self.pending_graph, size)
                try:
                    with torch.cuda.graph(self.pending_graph, stream=self.capture_stream):
                        for index in range(size):
                            e._step()
                            output[index].copy_(e.ids.reshape(-1))
                    proof = self.layout.finish_capture(self.pending_graph)
                finally:
                    self.layout.cancel_capture()
                c.live()
                self.graphs[size] = self.pending_graph
                self.proofs[size] = proof
                self.pending_graph = None
                self.graphs[size].replay()
                c.drain()
                e._gv_memory_guard()
                c.observed('native_capture', time.perf_counter() - started)
            self._reset(e, self.first)
            c.drain()

    @staticmethod
    def _reset(e, first):
        e.position.fill_(e.prompt)
        e.ids.copy_(first, non_blocking=False)

    def generate(self):
        self.graphs[self.schedule[0]].replay()
        for index, size in enumerate(self.schedule):
            host_rows = self.outputs[size].tolist()
            if index + 1 < len(self.schedule):
                self.graphs[self.schedule[index + 1]].replay()
            yield from host_rows
