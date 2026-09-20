"""Retained #32 with one licensed SGLang BF16 native-attention option."""
import time
import torch
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources
from retained_engine import Engine as RetainedEngine
from full_prefill_engine import Engine as PrefillEngine
from sglang_attention import SGLangAttention
from sglang_capture import NativeChunks
from sglang_lifetime import Control,CandidateRejected
from sglang_runtime import Configuration,snapshot,bind,timed_complete,require_stream,whole_call_admission
from sglang_validate import restore,validate,micro_admission


class Engine(RetainedEngine):
    def __init__(self,model_path):
        self.sg_control=Control(time.monotonic(),lambda:torch.cuda.synchronize())
        self.sg_a=self.sg_b=self.sg_bound=None
        self.sg_decided=self.sg_timing=False
        self.sg_captures=0
        super().__init__(model_path)

    def _allocate(self,batch,prompt,output):
        if self.sg_timing:raise RuntimeError('timed request attempted allocation')
        c=self.sg_control;c.drain()
        self.sg_a=self.sg_b=self.sg_bound=None
        c.owners.clear();self.sg_decided=False
        super()._allocate(batch,prompt,output)

    def _capture_chunks(self,first,steps):
        if self.sg_timing:raise RuntimeError('timed request attempted ordinary capture')
        self.sg_captures+=1
        return PrefillEngine._capture_chunks(self,first,steps)

    def _capture_speculative(self,first):
        if self.sg_timing:raise RuntimeError('timed request attempted B1 capture')
        self.sg_captures+=1
        return RetainedEngine._capture_speculative(self,first)

    def _sg_memory_guard(self,validation=False):
        c=self.sg_control;c.live()
        free,total=torch.cuda.mem_get_info()
        cache=sum(x.numel()*x.element_size() for x in self.keys+self.values)
        one_layer=2*self.batch*8*self.capacity*128*2
        # At most one full-cache reference, six private layer-sized buffers,
        # all-layer attention witnesses, and ample graph/allocator workspace.
        extra=(cache+6*one_layer if validation else 0)+self.batch*132096
        extra+=len(self.layers)*self.batch*4096*2+self.batch*self.model.config.vocab_size*2+2*2**30
        if (free<extra or torch.cuda.memory_allocated()+extra>.85*total or
                torch.cuda.max_memory_allocated()>.85*total):
            raise CandidateRejected('optional attention memory preflight')

    def _prepare_sg(self,prompts,output):
        c=self.sg_control
        # Mandatory original preparation runs before any optional handler.
        native=RetainedEngine.generate(self,prompts,output)
        try:
            for _ in native:pass
        finally:
            try:native.close()
            finally:c.drain()
        self.sg_a=snapshot(self)
        self.sg_bound=self.sg_a
        accepted=False;mandatory_failed=False
        def retained_stream():
            nonlocal mandatory_failed
            mandatory=self.sg_bound is self.sg_a
            try:yield from RetainedEngine.generate(self,prompts,output)
            except Exception:
                if mandatory:mandatory_failed=True
                raise
        try:
            with torch.inference_mode():
                base=timed_complete(self,self.sg_a,retained_stream,'base')
                require_stream(base[0],self.batch,output,self.model.config.vocab_size)
                base=None
                restore(self,prompts);c.reserve();self._sg_memory_guard(validation=True)
                port=SGLangAttention(self,self.sg_a.attention)
                validate(self,self.sg_a,port,prompts)
                micro_admission(self,self.sg_a,port,prompts)
                first=restore(self,prompts)
                self.fused_cache_attention=port
                if self.sg_a.native_chunks is not None:
                    native_chunks={}
                    for size in (1,2,3,4):
                        c.live('native_capture',5.)
                        native_chunks[size]=NativeChunks(self,first,size,size)
                    chunks=self.sg_a.chunks
                else:
                    native_chunks=None
                    chunks=NativeChunks(self,first,output-1,4)
                self.sg_b=Configuration(self.shape,port,chunks,native_chunks,self.sg_a.verifier,True)
                bind(self,self.sg_a)
                whole_call_admission(self,self.sg_a,self.sg_b,prompts,output,retained_stream)
                self._sg_memory_guard();c.live();accepted=True
        except (CandidateRejected,TimeoutError,CompilationError,OutOfResources,torch.cuda.OutOfMemoryError):
            if mandatory_failed:raise
            accepted=False
        finally:
            c.drain()
            bind(self,self.sg_b if accepted else self.sg_a)
            if not accepted:
                self.sg_b=None;c.owners.clear()
            with torch.inference_mode():restore(self,prompts)
            self.sg_decided=True

    def generate(self,input_ids,max_new_tokens):
        if max_new_tokens<=0:return
        self.sg_control.healthy()
        try:
            batch,prompt=len(input_ids),len(input_ids[0])
            if not(1<=batch<=32 and prompt>=1 and max_new_tokens>=8):
                yield from RetainedEngine.generate(self,input_ids,max_new_tokens)
                return
            if self.shape!=(batch,prompt,max_new_tokens) or not self.sg_decided:
                self._prepare_sg(input_ids,max_new_tokens)
            yield from RetainedEngine.generate(self,input_ids,max_new_tokens)
        finally:self.sg_control.drain()
