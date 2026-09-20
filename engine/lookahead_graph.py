"""Captured W1/W12 exact independent-position graphs; no optional layout tuner."""
import torch
import triton
from custom_kernels import embedding_norm_kernel, residual_norm_kernel, swiglu_kernel
from lookahead_helpers import chain_ids_kernel, chain_merge_kernel
from lookahead_kernels import lookahead_qkv_kernel, lookahead_attention_kernel, lookahead_compact_kernel
from lookahead_single import (single_qkv_cache_kernel, single_fused_attention_kernel,
                              single_attention_split_kernel)
from lookahead_host import DEPTHS



class LookaheadGraph:
    def __init__(self, engine, width):
        self.engine, self.width = engine, int(width)
        assert self.width in (1, 12)
        self.control = engine.la_control
        self.control.register(self)  # Before ANY optional asynchronous allocation.
        self.control.live()
        self.current = None
        self.allocated = []
        self.capturing = False
        self.single_fused = self.width == 1 and engine.fused_cache_attention.enabled
        b, w, device = engine.batch, self.width, engine.ids.device
        self.rows = b * w
        self.splits = triton.cdiv(engine.capacity + 11, 128) if w == 12 else engine.splits
        def empty(*shape, dtype=torch.bfloat16):
            self.control.live()
            value = torch.empty(shape, dtype=dtype, device=device)
            self.allocated.append(value)
            return value
        self.meta = torch.zeros((b, w + 2), device=device, dtype=torch.int64)
        self.host_meta = torch.empty((b, w + 2), dtype=torch.int64, pin_memory=True)
        self.paths = torch.full((b, w), -1, device=device, dtype=torch.int64)
        self.host_paths = torch.empty((b, w), dtype=torch.int64, pin_memory=True)
        # Persistent views avoid dozens of Python-to-Torch scalar dispatches
        # per round. They share the existing pinned tensor allocations.
        self.host_meta_array = self.host_meta.numpy()
        self.host_paths_array = self.host_paths.numpy()
        self.inputs = empty(self.rows, dtype=torch.int64)
        self.output = empty(b, w, dtype=torch.int64)
        self.hidden = empty(self.rows, engine.h)
        self.normalized = empty(self.rows, engine.h)
        self.qkv = empty(self.rows, 6144)
        self.query = empty(self.rows, 32, 128)
        self.attention = empty(self.rows, 4096)
        self.branch = empty(self.rows, engine.h)
        self.gateup = empty(self.rows, 2 * engine.i)
        self.intermediate = empty(self.rows, engine.i)
        self.logits = empty(self.rows, engine.model.config.vocab_size)
        self.partial = empty(self.rows * 32, self.splits, 128, dtype=torch.float32)
        self.pmax = empty(self.rows * 32, self.splits, dtype=torch.float32)
        self.psum = empty(self.rows * 32, self.splits, dtype=torch.float32)
        self.keys = [empty(b, 8, w, 128) for _ in engine.layers]
        self.values = [empty(b, 8, w, 128) for _ in engine.layers]
        self.output_flat = self.output.view(-1)
        self.meta[:, w].fill_(engine.prompt)
        self.meta[:, w + 1].fill_(1)
        self.stream = torch.cuda.Stream(device=device)
        # The engine controller retains this partially built object on any
        # construction failure; caller recovery drains before releasing it.
        self.control.drain()
        self.control.live()
        with torch.cuda.stream(self.stream):
            self._verify()
            self._compact()
        self.control.wait(self.stream.synchronize)
        self.control.live()
        self.graph = torch.cuda.CUDAGraph()
        self.control.live()
        self.capturing = True
        try:
            with torch.cuda.graph(self.graph, stream=self.stream):
                self._verify()
        finally:
            self.capturing = False
        self.control.live()
        self.compact_graph = None
        if w == 12:
            self.compact_graph = torch.cuda.CUDAGraph()
            self.control.live()
            self.capturing = True
            try:
                with torch.cuda.graph(self.compact_graph, stream=self.stream):
                    self._compact()
            finally:
                self.capturing = False
        self.control.live()
        self.graph.replay()
        if self.compact_graph is not None:
            self.compact_graph.replay()
        self.control.drain()
        self.control.live()  # Post-constructor/last-nonpreemptible-call guard.

    def _linear(self, family, index, x, weight, out):
        self._check()
        # W1 reuses accepted native projection choices at the actual batch.
        # W12 is a true M=B*12 matmul, not B unrelated one-row weight scans.
        if self.width == 1:
            self.engine.native_layout.run(family, index, x, weight, out)
        else:
            torch.mm(x, weight.t(), out=out)
        self._check()

    def _check(self):
        if not self.capturing:
            self.control.live()

    def _verify(self):
        self._check()
        e, rows, w = self.engine, self.rows, self.width
        chain_ids_kernel[(triton.cdiv(rows, 128),)](
            self.meta, self.inputs, w, rows,
            num_warps=4, num_stages=3, enable_fp_fusion=True)
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
                    lookahead_qkv_kernel[(rows, 40)](
                        self.qkv, a.q_norm.weight, a.k_norm.weight, e.cos, e.sin,
                        self.meta, self.query, self.keys[index], self.values[index],
                        e.capacity, w, e.eps, num_warps=4, num_stages=3, enable_fp_fusion=False)
                self._check()
                if w == 1:
                    # The preceding QKV launch completed the active root write.
                    # Read the now-committed current slot directly, exactly as
                    # the retained split path, with vector lengths/active mask.
                    single_attention_split_kernel[(e.batch, 8, e.splits)](
                        self.query, e.keys[index], e.values[index], self.meta,
                        self.partial, self.pmax, self.psum,
                        e.capacity, e.splits, 128 ** -0.5, num_warps=4, num_stages=1)
                else:
                    lookahead_attention_kernel[(e.batch, 16, self.splits)](
                        self.query, e.keys[index], e.values[index], self.keys[index], self.values[index],
                        self.meta, self.partial, self.pmax, self.psum,
                        e.capacity, w, self.splits, 128 ** -0.5, BLOCK_N=128,
                        num_warps=8, num_stages=1, enable_fp_fusion=True)
                self._check()
                chain_merge_kernel[(rows * 32,)](
                    self.partial, self.pmax, self.psum, self.attention,
                    self.splits, triton.next_power_of_2(self.splits),
                    num_warps=4, num_stages=3, enable_fp_fusion=True)
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
        if self.width == 1:
            return  # Every active W1 root wrote its own committed main slot.
        e, w = self.engine, self.width
        for sk, sv, k, v in zip(self.keys, self.values, e.keys, e.values):
            self._check()
            lookahead_compact_kernel[(e.batch, 8, w)](
                sk, sv, k, v, self.meta, self.paths, e.capacity, w,
                num_warps=4, num_stages=3, enable_fp_fusion=True)

    def replay(self, inputs, lengths, active):
        self.control.healthy()
        e, w = self.engine, self.width
        if not (len(inputs) == len(lengths) == len(active) == e.batch):
            raise ValueError("metadata batch mismatch")
        for nodes, length, mask in zip(inputs, lengths, active):
            if len(nodes) != w or not 0 <= mask < (1 << w) or not 0 <= length < e.capacity:
                raise ValueError("invalid verifier metadata")
            if any(type(t) is not int or not 0 <= t < e.model.config.vocab_size for t in nodes):
                raise ValueError("invalid embedding ID")
            if any(mask & (1 << row) and length + (DEPTHS[row] if w == 12 else 0) >= e.capacity
                   for row in range(w)):
                raise ValueError("active position exceeds capacity")
        self.host_meta_array[:] = [(*row, int(lengths[b]), int(active[b]))
                                   for b, row in enumerate(inputs)]
        # Blocking copy intentionally establishes pinned-host reuse safety.
        # Complete-round admission charges this synchronization.
        self.meta.copy_(self.host_meta, non_blocking=False)
        self.graph.replay()
        self.current = (tuple(lengths), tuple(active))

    def compact(self, paths):
        self.control.healthy()
        if self.current is None or len(paths) != self.engine.batch:
            raise ValueError("compaction requires the current verifier batch")
        legal = ((), (0,)) if self.width == 1 else ((), (0,), (0,8), (0,8,9), (0,10), (0,10,11))
        for b, path in enumerate(paths):
            if tuple(path) not in legal or any(not self.current[1][b] & (1 << j) for j in path):
                raise ValueError("invalid or inactive compaction path")
            if self.current[0][b] + len(path) > self.engine.capacity:
                raise ValueError("compaction exceeds capacity")
        if self.width == 12:
            self.host_paths_array[:] = [tuple(path) + (-1,) * (12 - len(path)) for path in paths]
            self.paths.copy_(self.host_paths, non_blocking=False)
            self.compact_graph.replay()
        self.current = None
