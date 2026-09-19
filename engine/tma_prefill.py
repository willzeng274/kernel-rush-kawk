import triton
import triton.language as tl

@triton.jit
def tma_prefill_kernel(X_DESC,W_DESC,OUT_DESC,
                       M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                       BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,
                       GROUP:tl.constexpr=8):
    tile=tl.program_id(0)
    mt,nt=tl.cdiv(M,BM),tl.cdiv(N,BN)
    group_width=GROUP*nt
    first_m=(tile//group_width)*GROUP
    group_m=tl.minimum(mt-first_m,GROUP)
    local=tile%group_width
    im=first_m+local%group_m
    jn=local//group_m
    acc=tl.full((BM,BN),0,tl.float32)
    for block in range(tl.cdiv(K,BK)):
        k=block*BK
        x=tl._experimental_descriptor_load(X_DESC,[im*BM,k],[BM,BK],tl.bfloat16)
        w=tl._experimental_descriptor_load(W_DESC,[jn*BN,k],[BN,BK],tl.bfloat16)
        acc=tl.dot(x,tl.trans(w),acc)
    tl._experimental_descriptor_store(OUT_DESC,acc.to(tl.bfloat16),[im*BM,jn*BN])

import math
import re
import sys
import time
import numpy as np
import torch
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources
from dense_prefill import DensePrefill


def _matrix_ok(t):
    return (t.is_cuda and t.dtype == torch.bfloat16 and len(t.shape) == 2
            and t.is_contiguous() and tuple(t.stride()) == (t.shape[1], 1)
            and all(isinstance(v, int) and 0 < v < 2**32 for v in t.shape)
            and t.data_ptr() > 0 and t.data_ptr() % 16 == 0
            and t.shape[1] * 2 % 16 == 0 and t.shape[1] * 2 < 2**40)


class TmaCompileFailure(Exception):
    pass


def _is_ptxas_failure(message):
    # Only messages emitted by pinned Triton3.1 make_cubin. Runtime/device
    # failures and exceptions from the accepted baseline must still escape.
    return (message.startswith("Internal Triton PTX codegen error: \n")
            or re.match(r"\APlease run `ptxas [^`\n]+` to confirm that this is a bug in `ptxas`\n",message) is not None
            or re.match(r"\A`ptxas` failed with error code -?\d+: \n",message) is not None)


class Descriptor:
    """Immutable descriptor and its original tensor live as long as the plan."""
    def __init__(self, tensor, box):
        if (not _matrix_ok(tensor) or len(box) != 2
                or any(not isinstance(v,int) or not 1 <= v <= 256 for v in box)
                or box[1] * 2 < 32
                or any(s % b for s,b in zip(tensor.shape,box))
                or torch.cuda.is_current_stream_capturing()
                or torch.cuda.get_device_capability(tensor.device)[0] != 9):
            raise ValueError("unsupported TMA descriptor shape, alignment, device or capture")
        self.tensor, self.pointer = tensor, tensor.data_ptr()
        self.shape, self.box = tuple(tensor.shape), tuple(box)
        self.storage = np.zeros(255,dtype=np.int8)
        offset = (-self.storage.ctypes.data) % 128
        self.host = self.storage[offset:offset+128]
        if (self.host.nbytes != 128 or self.host.ctypes.data % 128
                or not self.host.flags.c_contiguous or not self.host.flags.writeable):
            raise ValueError("unaligned TMA descriptor host storage")
        fill = triton.runtime.driver.active.utils.fill_2d_tma_descriptor
        fill(self.pointer,*self.shape,*self.box,2,self.host)
        self.gpu = torch.tensor(self.host,dtype=torch.int8,device=tensor.device)
        if self.gpu.data_ptr() % 128:
            raise ValueError("unaligned TMA descriptor device storage")

    def matches(self,tensor):
        return (tensor is self.tensor and tensor.data_ptr() == self.pointer
                and tuple(tensor.shape) == self.shape and _matrix_ok(tensor))


class TmaPlan:
    CONFIGS = ((128,128,64,4),(128,256,64,3))
    SHAPES = ((19456,2560),(6144,2560),(2560,9728))
    def __init__(self,x,weights,output,config,deadline):
        if not weights or config not in self.CONFIGS:
            raise ValueError("unsupported TMA configuration or empty weight family")
        if time.monotonic() >= deadline:
            raise TimeoutError("TMA tuning deadline expired")
        self.rows,self.k = x.shape
        self.n = weights[0].shape[0]
        bm,bn,bk,stages = config
        if (not _matrix_ok(x) or not _matrix_ok(output)
                or self.rows < 256 or self.rows % bm or self.n % bn or self.k % bk
                or (self.n,self.k) not in self.SHAPES
                or output.shape != (self.rows,self.n)
                or output.device != x.device
                or (self.rows//bm)*(self.n//bn) >= 2**31
                or any(not _matrix_ok(w) or w.shape != (self.n,self.k)
                       or w.device != x.device for w in weights)):
            raise ValueError("unsupported TMA matrix family")
        self.config = config
        self.name = f"tma_m{bm}_n{bn}_k{bk}_s{stages}_w8"
        self.x_desc,self.out_desc = Descriptor(x,(bm,bk)),Descriptor(output,(bm,bn))
        self.weights = {}
        if not self.extend(weights,deadline):
            raise TimeoutError("descriptor family exceeded tuning deadline")

    def extend(self,weights,deadline):
        for weight in weights:
            if time.monotonic() >= deadline:
                return False
            if (not _matrix_ok(weight) or weight.shape != (self.n,self.k)
                    or weight.device != self.x_desc.tensor.device):
                return False
            existing = self.weights.get(id(weight))
            if existing is None:
                self.weights[id(weight)] = Descriptor(weight,(self.config[1],self.config[2]))
            elif not existing.matches(weight):
                return False
        return time.monotonic() < deadline

    @property
    def extra_bytes(self):
        return (2+len(self.weights))*128

    def eligible(self,x,weight,output):
        desc = self.weights.get(id(weight))
        return (desc is not None and desc.matches(weight)
                and self.x_desc.matches(x) and self.out_desc.matches(output))

    def __call__(self,x,weight,output):
        if not self.eligible(x,weight,output):
            raise ValueError("TMA descriptors do not match current buffers")
        bm,bn,bk,stages = self.config
        try:
            tma_prefill_kernel[((self.rows//bm)*(self.n//bn),)](
                self.x_desc.gpu,self.weights[id(weight)].gpu,self.out_desc.gpu,
                self.rows,self.n,self.k,BM=bm,BN=bn,BK=bk,GROUP=8,
                num_warps=8,num_stages=stages)
        except RuntimeError as error:
            if _is_ptxas_failure(str(error)):
                raise TmaCompileFailure(str(error)) from error
            raise


class TmaPrefill(DensePrefill):
    """Static descriptors for three fixed prefill buffers; decode is untouched."""
    def __init__(self,engine,native,deadline):
        self.rows,self.device = engine.prefill_rows,engine.prefill_normalized.device
        self.native,self.plans = native,{}
        deadline = min(deadline,time.monotonic()+35.0)
        if (self.rows < 256 or self.rows % 128 or time.monotonic() >= deadline
                or torch.cuda.is_current_stream_capturing()
                or torch.cuda.get_device_capability(self.device)[0] != 9):
            return
        if not callable(getattr(triton.runtime.driver.active.utils,
                                "fill_2d_tma_descriptor",None)):
            return
        groups = (
            ("gateup",engine.prefill_normalized,engine.prefill_gateup,
             [packed[1] for packed in engine.packed]),
            ("down",engine.prefill_intermediate,engine.prefill_branch,
             [layer.mlp.down_proj.weight for layer in engine.layers]),
            ("qkv",engine.prefill_normalized,engine.prefill_qkv,
             [packed[0] for packed in engine.packed]),
        )
        generator = torch.Generator(device=self.device)
        generator.manual_seed(92579)
        try:
            for name,x,output,all_weights in groups:
                if time.monotonic() >= deadline:
                    break
                indices = ([round(i*(len(all_weights)-1)/5) for i in range(6)]
                           if len(all_weights) >= 6 else list(range(len(all_weights))))
                weights = [all_weights[i] for i in indices]
                winner = self._select(name,x,weights,output,generator,deadline)
                if winner is not None and time.monotonic() < deadline:
                    try:
                        complete = winner.extend(all_weights,deadline)
                    except (torch.cuda.OutOfMemoryError,TimeoutError):
                        complete = False
                    if complete and time.monotonic() < deadline:
                        self.plans[name] = winner
                self._log(name,"selected "+(self.plans[name].name
                           if name in self.plans else "previous"))
        finally:
            # Graphs and descriptor storage cannot be released with side-stream
            # work outstanding, including when selection raises unexpectedly.
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()

    @property
    def extra_bytes(self):
        return sum(plan.extra_bytes for plan in self.plans.values())

    def _log(self,name,message):
        print(f"[tma-prefill] M={self.rows} {name}: {message}",
              file=sys.stderr,flush=True)

    def _select(self,name,x,weights,output,generator,deadline):
        if (time.monotonic() >= deadline or not weights
                or not all(_matrix_ok(t) for t in (x,output,*weights))
                or 2*self.rows*weights[0].numel()*len(weights) > self.MAX_POOL_FLOPS):
            return None
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        if time.monotonic() >= deadline or not self._room(output):
            return None
        try:
            reference = torch.empty_like(output)
        except torch.cuda.OutOfMemoryError:
            return None
        x.normal_(generator=generator)
        native_graph = self._graph(name,None,x,weights,output,deadline)
        if native_graph is None or time.monotonic() >= deadline:
            return None
        winner,winner_ms = None,float("inf")
        for config in TmaPlan.CONFIGS:
            if time.monotonic() >= deadline:
                break
            try:
                plan = TmaPlan(x,weights,output,config,deadline)
                if not self._check(name,plan,x,weights,output,reference,generator,deadline):
                    self._log(name,f"{plan.name}: numerical check rejected")
                    continue
                graph = self._graph(name,plan,x,weights,output,deadline)
                if graph is None or time.monotonic() >= deadline:
                    break
            except (CompilationError,OutOfResources,TmaCompileFailure,torch.cuda.OutOfMemoryError,TimeoutError) as error:
                self._log(name,f"{config}: {type(error).__name__}")
                continue
            natives,customs = [],[]
            for timed,values in ((native_graph,natives),(graph,customs),
                                 (graph,customs),(native_graph,natives)):
                value = self._time(timed,len(weights),deadline)
                if value is None or not math.isfinite(value) or value <= 0:
                    break
                values.append(value)
            del graph
            if len(natives) != 2 or len(customs) != 2:
                break
            old,new = min(natives),max(customs)
            self._log(name,f"{plan.name}: previous {old*1000:.2f} us, custom {new*1000:.2f} us")
            if new < old*.95 and new < winner_ms:
                winner,winner_ms = plan,new
        return winner if time.monotonic() < deadline else None

    def _graph(self,name,plan,x,weights,output,deadline):
        if time.monotonic() >= deadline:
            return None
        current = torch.cuda.current_stream(x.device)
        stream = torch.cuda.Stream(device=x.device)
        stream.wait_stream(current)

        def launch():
            for index,weight in enumerate(weights):
                if plan is None:
                    self.native.run(name,index,x,weight,output)
                else:
                    plan(x,weight,output)

        try:
            with torch.cuda.stream(stream):
                launch()
        finally:
            current.wait_stream(stream)
        torch.cuda.synchronize(x.device)
        if time.monotonic() >= deadline:
            return None
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph,stream=stream):
                launch()
        finally:
            current.wait_stream(stream)
        if time.monotonic() >= deadline:
            return None
        graph.replay()
        torch.cuda.synchronize(x.device)
        return graph if time.monotonic() < deadline else None
