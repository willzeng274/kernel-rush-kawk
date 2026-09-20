"""Owned optional native chunk capture with the retained schedule and output flow."""
import torch


class NativeChunks:
    def __init__(self,e,first,steps,chunk_size=4):
        self.control=e.sg_control
        self.control.register(self)
        self.steps,self.chunk_size=int(steps),int(chunk_size)
        if self.steps < 0 or self.chunk_size < 1:
            raise ValueError('invalid native chunk schedule')
        full,tail=divmod(self.steps,self.chunk_size)
        self.schedule=[self.chunk_size]*full+([tail] if tail else [])
        self.graphs,self.outputs={},{}
        self.capture_stream=self.pending_graph=None
        self.control.live()
        self.capture_stream=torch.cuda.Stream(device=e.ids.device)
        for size in dict.fromkeys(self.schedule):
            self.control.live('native_capture',5.)
            self.outputs[size]=torch.empty((size,e.ids.numel()),dtype=e.ids.dtype,device=e.ids.device)
            output=self.outputs[size]
            self.control.drain()
            with torch.cuda.stream(self.capture_stream):
                self._reset(e,first)
                for index in range(size):
                    self.control.live()
                    e._step()
                    output[index].copy_(e.ids.reshape(-1))
            self.control.wait(self.capture_stream.synchronize)
            self._reset(e,first)
            self.control.drain()
            self.control.live()
            self.pending_graph=torch.cuda.CUDAGraph()
            self.control.live()
            e.sg_captures+=1
            with torch.cuda.graph(self.pending_graph,stream=self.capture_stream):
                for index in range(size):
                    e._step()
                    output[index].copy_(e.ids.reshape(-1))
            self.control.live()
            self.graphs[size]=self.pending_graph
            self.pending_graph=None
            self.graphs[size].replay()
            self.control.drain()
            self.control.live()
        self._reset(e,first)
        self.control.drain()

    @staticmethod
    def _reset(e,first):
        e.position.fill_(e.prompt)
        e.ids.copy_(first,non_blocking=False)

    def generate(self):
        if not self.schedule:
            return
        self.graphs[self.schedule[0]].replay()
        for index,size in enumerate(self.schedule):
            host_rows=self.outputs[size].tolist()
            if index+1<len(self.schedule):
                self.graphs[self.schedule[index+1]].replay()
            yield from host_rows
