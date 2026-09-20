"""Retained #32 plus one exact request-local suffix/recycling W4 option."""
import time
import torch
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources
from retained_engine import Engine as RetainedEngine
from recycled_graph import ChainGraph
from suffix_lifetime import Control, CandidateRejected
from suffix_runtime import request, timed_complete, require_stream, calibrate_initial, whole_call_admission
from recycled_validate import restore, validate_numerics, measure_prices


class Engine(RetainedEngine):
    def __init__(self,model_path):
        self.suffix_control = Control(time.monotonic(),lambda:torch.cuda.synchronize())
        self.suffix_graphs = self.suffix_prices = None
        self.suffix_decided = False
        super().__init__(model_path)

    def _allocate(self,batch,prompt,output):
        c = self.suffix_control
        c.drain()
        self.suffix_graphs = self.suffix_prices = None
        c.owners.clear()  # Partial graph owners can release only after drain.
        self.suffix_decided = False
        super()._allocate(batch,prompt,output)

    def _memory_guard(self):
        c = self.suffix_control
        c.live()
        free,total = torch.cuda.mem_get_info()
        b,w = self.batch,5
        cache = sum(x.numel()*x.element_size() for x in self.keys+self.values)
        scratch = 2*len(self.layers)*b*8*w*128*2
        logits = b*w*self.model.config.vocab_size*2
        activations = b*w*(3*self.h+6144+4096+4096+3*self.i)*2
        partial = b*w*32*self.splits*130*4
        # One full-cache backup at a time; paired poison also owns logits and
        # scratch snapshots. Existing allocations include baseline graph pools.
        extra = 2*scratch+2*logits+activations+partial+cache+2*2**30
        if (torch.cuda.memory_allocated()+extra > .85*total or
                torch.cuda.max_memory_allocated() > .85*total or free < extra):
            raise CandidateRejected("optional memory preflight")

    def _prepare_suffix(self,prompts,output):
        c = self.suffix_control
        # Real retained preparation has priority, including B1 speculation.
        # Its original 180-second selector deadline is never changed.
        native = RetainedEngine.generate(self,prompts,output)
        try:
            for row in native:
                pass
        finally:
            native.close()
            c.drain()
        accepted = False
        retained_failed = False
        def retained_stream():
            # Only errors raised by actual retained generation are mandatory.
            # Deadline exits in the surrounding optional timing consumer remain
            # recoverable; never reinterpret a retained OOM/compiler failure.
            nonlocal retained_failed
            try:
                yield from RetainedEngine.generate(self,prompts,output)
            except Exception:
                retained_failed = True
                raise
        try:
            c.live('base',10.)
            baseline = timed_complete(self,retained_stream,'base')
            require_stream(baseline[0],self.batch,output,self.model.config.vocab_size)
            p0,d0 = baseline[2],(baseline[1]-baseline[2])/(output-1)
            baseline = None
            restore(self,prompts)
            c.reserve()
            self._memory_guard()
            c.live('constructor_one',20.)
            at = time.perf_counter()
            one = ChainGraph(self,1)
            c.observed('constructor_one',time.perf_counter()-at)
            c.live('constructor_four',30.)
            at = time.perf_counter()
            four = ChainGraph(self,4)
            c.observed('constructor_four',time.perf_counter()-at)
            self.suffix_graphs = {1:one,4:four}
            validate_numerics(self,self.suffix_graphs,prompts)
            prices,cleanup = measure_prices(self,self.suffix_graphs,prompts,output,p0,d0)
            prices = calibrate_initial(self,prompts,output,prices,cleanup)
            self.suffix_prices = whole_call_admission(self,prompts,output,prices,
                              retained_stream)
            self._memory_guard()
            c.live()
            accepted = True
        except (CandidateRejected,TimeoutError,CompilationError,OutOfResources,
                torch.cuda.OutOfMemoryError):
            # Only optional setup rejects. Unknown runtime/device failures and
            # any failed drain propagate and never become baseline fallback.
            if retained_failed:
                raise
            accepted = False
        finally:
            c.drain()
            if not accepted:
                self.suffix_graphs = self.suffix_prices = None
                c.owners.clear()
            # Reset all prompt KV/current IDs after private warmup trials.
            # Real generation below recomputes every first/output token.
            restore(self,prompts)
            self.suffix_decided = True

    def generate(self,input_ids,max_new_tokens):
        if max_new_tokens <= 0:
            return
        self.suffix_control.healthy()
        try:
            batch,prompt = len(input_ids),len(input_ids[0])
            supported = 1 <= batch <= 16 and prompt >= 3 and max_new_tokens >= 8
            if not supported:
                yield from RetainedEngine.generate(self,input_ids,max_new_tokens)
                return
            with torch.inference_mode():
                if self.shape != (batch,prompt,max_new_tokens) or not self.suffix_decided:
                    self._prepare_suffix(input_ids,max_new_tokens)
                if self.suffix_graphs is None:
                    yield from RetainedEngine.generate(self,input_ids,max_new_tokens)
                else:
                    yield from request(self,input_ids,max_new_tokens,self.suffix_prices)
        finally:
            self.suffix_control.drain()
