"""Captured W1/W4 full BF16 verifier and accepted-row compaction graphs."""
import torch
import triton
from custom_kernels import embedding_norm_kernel, residual_norm_kernel, swiglu_kernel
from recycled_kernels import (chain_ids_kernel, chain_qkv_kernel,
                              chain_attention_kernel, chain_merge_kernel,
                              chain_compact_kernel)
from recycled_single import (single_qkv_cache_kernel, single_fused_attention_kernel,
                             single_attention_split_kernel)


class ChainGraph:
    def __init__(self, engine, width):
        self.engine, self.width = engine, int(width)
        if self.width not in (1,4):
            raise ValueError("fixed W1/W4 only")
        self.control = engine.suffix_control
        self.control.register(self)  # Before ANY optional asynchronous operation.
        self.allocated, self.current, self.capturing = [], None, False
        self.control.live()
        self.single_fused = self.width == 1 and engine.fused_cache_attention.enabled
        b, w, device = engine.batch, self.width, engine.ids.device
        self.rows = b*w
        def empty(*shape, dtype=torch.bfloat16, **options):
            self.control.live()
            value = torch.empty(shape,dtype=dtype,**options)
            self.allocated.append(value)
            return value
        self.meta = empty(b,w+2,dtype=torch.int64,device=device)
        self.meta.zero_()
        self.host_meta = empty(b,w+2,dtype=torch.int64,pin_memory=True)
        self.counts = empty(b,dtype=torch.int64,device=device)
        self.counts.zero_()
        self.host_counts = empty(b,dtype=torch.int64,pin_memory=True)
        self.host_meta_array, self.host_counts_array = self.host_meta.numpy(), self.host_counts.numpy()
        self.inputs = empty(self.rows,dtype=torch.int64,device=device)
        self.output = empty(b,w,dtype=torch.int64,device=device)
        self.hidden = empty(self.rows,engine.h,device=device)
        self.normalized = empty(self.rows,engine.h,device=device)
        self.qkv = empty(self.rows,6144,device=device)
        self.query = empty(self.rows,32,128,device=device)
        self.attention = empty(self.rows,4096,device=device)
        self.branch = empty(self.rows,engine.h,device=device)
        self.gateup = empty(self.rows,2*engine.i,device=device)
        self.intermediate = empty(self.rows,engine.i,device=device)
        self.logits = empty(self.rows,engine.model.config.vocab_size,device=device)
        self.partial = empty(self.rows*32,engine.splits,128,dtype=torch.float32,device=device)
        self.pmax = empty(self.rows*32,engine.splits,dtype=torch.float32,device=device)
        self.psum = empty(self.rows*32,engine.splits,dtype=torch.float32,device=device)
        self.keys = [empty(b,8,w,128,device=device) for _ in engine.layers]
        self.values = [empty(b,8,w,128,device=device) for _ in engine.layers]
        self.output_flat = self.output.view(-1)
        self.meta[:,w].fill_(engine.prompt)
        self.meta[:,w+1].fill_(w)
        self.control.live()
        self.stream = torch.cuda.Stream(device=device)
        self.control.drain()
        with torch.cuda.stream(self.stream):
            self._verify()
            self._compact()
        self.control.wait(self.stream.synchronize)
        self.control.live()
        self.graph = torch.cuda.CUDAGraph()
        self.control.live()  # Constructor may be nonpreemptible.
        self.capturing = True
        try:
            with torch.cuda.graph(self.graph,stream=self.stream):
                self._verify()
        finally:
            self.capturing = False
        self.control.live()
        self.compact_graph = None
        if w == 4:
            self.control.live()
            self.compact_graph = torch.cuda.CUDAGraph()
            self.control.live()
            self.capturing = True
            try:
                with torch.cuda.graph(self.compact_graph,stream=self.stream):
                    self._compact()
            finally:
                self.capturing = False
            self.control.live()
        self.graph.replay()
        if self.compact_graph is not None:
            self.compact_graph.replay()
        self.control.drain()
        self.control.live()

    def _check(self):
        if not self.capturing:
            self.control.live()

    def _linear(self, family, index, x, weight, out):
        self._check()
        # W1 reuses accepted native projection choices at the actual batch.
        # W4 is a true M=B*4 matmul, not B unrelated one-row weight scans.
        if self.width == 1:
            self.engine.native_layout.run(family, index, x, weight, out)
        else:
            torch.mm(x, weight.t(), out=out)
        self._check()

    def _verify(self):
        self._check()
        e, rows, w = self.engine, self.rows, self.width
        chain_ids_kernel[(triton.cdiv(rows, 128),)](
            self.meta, self.inputs, w, rows, num_warps=4)
        embedding_norm_kernel[(rows,)](
            self.inputs, e.base.embed_tokens.weight, e.layers[0].input_layernorm.weight,
            self.hidden, self.normalized, e.h, e.eps, 4096,
            num_warps=4, enable_fp_fusion=False)
        for index, layer in enumerate(e.layers):
            self._check()
            a, m = layer.self_attn, layer.mlp
            qkv_weight, gu_weight = e.packed[index]
            self._linear("qkv", index, self.normalized, qkv_weight, self.qkv)
            if self.single_fused:
                single_fused_attention_kernel[(e.batch, 8)](
                    self.qkv, a.q_norm.weight, a.k_norm.weight, e.cos, e.sin,
                    self.meta, e.keys[index], e.values[index],
                    self.keys[index], self.values[index], self.attention,
                    e.capacity, e.eps, 128 ** -0.5,
                    num_warps=8, num_stages=1, enable_fp_fusion=False)
            else:
                if w == 1:
                    single_qkv_cache_kernel[(rows, 40)](
                        self.qkv, a.q_norm.weight, a.k_norm.weight, e.cos, e.sin,
                        self.meta, self.query, self.keys[index], self.values[index],
                        e.keys[index], e.values[index], e.capacity, e.eps,
                        num_warps=4, enable_fp_fusion=False)
                else:
                    chain_qkv_kernel[(rows, 40)](
                        self.qkv, a.q_norm.weight, a.k_norm.weight, e.cos, e.sin,
                        self.meta, self.query, self.keys[index], self.values[index],
                        e.capacity, w, e.eps, num_warps=4, enable_fp_fusion=False)
                if w == 1:
                    # The preceding QKV launch completed the active root write.
                    # Read the now-committed current slot directly, exactly as
                    # the retained split path, with vector lengths/active mask.
                    single_attention_split_kernel[(e.batch, 8, e.splits)](
                        self.query, e.keys[index], e.values[index], self.meta,
                        self.partial, self.pmax, self.psum,
                        e.capacity, e.splits, 128 ** -0.5, num_warps=4, num_stages=1)
                else:
                    chain_attention_kernel[(e.batch, 8, e.splits)](
                        self.query, e.keys[index], e.values[index], self.keys[index], self.values[index],
                        self.meta, self.partial, self.pmax, self.psum,
                        e.capacity, w, e.splits, 128 ** -0.5, num_warps=8, num_stages=1)
                chain_merge_kernel[(rows * 32,)](
                    self.partial, self.pmax, self.psum, self.attention,
                    e.splits, triton.next_power_of_2(e.splits), num_warps=4)
            self._linear("output", index, self.attention, a.o_proj.weight, self.branch)
            residual_norm_kernel[(rows,)](
                self.branch, self.hidden, layer.post_attention_layernorm.weight,
                self.normalized, e.h, e.eps, 4096, num_warps=4, enable_fp_fusion=False)
            self._linear("gateup", index, self.normalized, gu_weight, self.gateup)
            swiglu_kernel[(triton.cdiv(rows * e.i, 1024),)](
                self.gateup, self.intermediate, e.i, rows * e.i,
                num_warps=4, enable_fp_fusion=False)
            self._linear("down", index, self.intermediate, m.down_proj.weight, self.branch)
            weight = (e.layers[index + 1].input_layernorm.weight
                      if index + 1 < len(e.layers) else e.base.norm.weight)
            residual_norm_kernel[(rows,)](
                self.branch, self.hidden, weight, self.normalized,
                e.h, e.eps, 4096, num_warps=4, enable_fp_fusion=False)
        self._linear("head", 0, self.normalized, e.model.lm_head.weight, self.logits)
        torch.argmax(self.logits, dim=-1, out=self.output_flat)

    def _compact(self):
        self._check()
        if self.width == 1:
            return  # Every active W1 root wrote its own committed main slot.
        e, w = self.engine, self.width
        for sk, sv, k, v in zip(self.keys, self.values, e.keys, e.values):
            self._check()
            chain_compact_kernel[(e.batch, 8, w)](
                sk, sv, k, v, self.meta, self.counts, e.capacity, w, num_warps=4)

    def replay(self, inputs, lengths, active):
        self.control.healthy()
        e,w = self.engine,self.width
        if self.current is not None:
            raise ValueError("previous replay must compact or be explicitly discarded")
        if not (len(inputs) == len(lengths) == len(active) == e.batch):
            raise ValueError("metadata batch mismatch")
        for row,length,count in zip(inputs,lengths,active):
            if (len(row) != w or type(length) is not int or type(count) is not int or
                    not 0 <= length < e.capacity or not 0 <= count <= w or
                    length+count > e.capacity):
                raise ValueError("invalid active positions")
            if any(type(y) is not int or not 0 <= y < e.model.config.vocab_size for y in row):
                raise ValueError("invalid embedding ID")
        self.host_meta_array[:] = [(*row,lengths[b],active[b]) for b,row in enumerate(inputs)]
        self.meta.copy_(self.host_meta,non_blocking=False)
        self.graph.replay()
        self.current = (tuple(lengths),tuple(active))

    def compact(self, counts):
        self.control.healthy()
        if self.current is None or len(counts) != self.engine.batch:
            raise ValueError("compaction requires current replay")
        if any(type(c) is not int or not 0 <= c <= self.current[1][b]
               for b,c in enumerate(counts)):
            raise ValueError("invalid compaction count")
        if self.width == 4:
            self.host_counts_array[:] = counts
            self.counts.copy_(self.host_counts,non_blocking=False)
            self.compact_graph.replay()
        self.current = None

    def discard(self):
        # Warmup reference checks only; no scratch is committed.
        self.control.drain()
        self.current = None
