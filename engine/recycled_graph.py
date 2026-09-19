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
        assert self.width in (1, 4)
        self.single_fused = self.width == 1 and engine.fused_cache_attention.enabled
        b, w, device = engine.batch, self.width, engine.ids.device
        self.rows = b * w
        def empty(*shape, dtype=torch.bfloat16):
            return torch.empty(shape, dtype=dtype, device=device)
        self.meta = torch.zeros((b, w + 2), device=device, dtype=torch.int64)
        self.host_meta = torch.empty((b, w + 2), dtype=torch.int64, pin_memory=True)
        self.counts = torch.zeros((b,), device=device, dtype=torch.int64)
        self.host_counts = torch.empty((b,), dtype=torch.int64, pin_memory=True)
        # Persistent views avoid dozens of Python-to-Torch scalar dispatches
        # per round. They share the existing pinned tensor allocations.
        self.host_meta_array = self.host_meta.numpy()
        self.host_counts_array = self.host_counts.numpy()
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
        self.partial = empty(self.rows * 32, engine.splits, 128, dtype=torch.float32)
        self.pmax = empty(self.rows * 32, engine.splits, dtype=torch.float32)
        self.psum = empty(self.rows * 32, engine.splits, dtype=torch.float32)
        self.keys = [empty(b, 8, w, 128) for _ in engine.layers]
        self.values = [empty(b, 8, w, 128) for _ in engine.layers]
        self.output_flat = self.output.view(-1)
        self.meta[:, w].fill_(engine.prompt)
        self.meta[:, w + 1].fill_(w)
        self.stream = torch.cuda.Stream(device=device)
        # A failed constructor has no caller-owned reference yet. Pin its
        # buffers before any custom kernel can be queued on the capture stream.
        # Failed drains leave the owner quarantined until engine/process exit.
        engine.recycle_quarantine.append(self)
        try:
            torch.cuda.synchronize(device)
            with torch.cuda.stream(self.stream):
                self._verify()
                # Zero counts warm all compaction kernels without main writes.
                self._compact()
            self.stream.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=self.stream):
                self._verify()
            self.compact_graph = None
            if self.width == 4:
                self.compact_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.compact_graph, stream=self.stream):
                    self._compact()
            self.graph.replay()
            if self.compact_graph is not None:
                self.compact_graph.replay()
            torch.cuda.synchronize(device)
        except BaseException:
            # If synchronization itself fails, this RuntimeError propagates and
            # the persistent quarantine deliberately retains every allocation.
            torch.cuda.synchronize(device)
            engine.recycle_quarantine.remove(self)
            raise
        engine.recycle_quarantine.remove(self)

    def _linear(self, family, index, x, weight, out):
        # W1 reuses accepted native projection choices at the actual batch.
        # W4 is a true M=B*4 matmul, not B unrelated one-row weight scans.
        if self.width == 1:
            self.engine.native_layout.run(family, index, x, weight, out)
        else:
            torch.mm(x, weight.t(), out=out)

    def _verify(self):
        e, rows, w = self.engine, self.rows, self.width
        chain_ids_kernel[(triton.cdiv(rows, 128),)](
            self.meta, self.inputs, w, rows, num_warps=4)
        embedding_norm_kernel[(rows,)](
            self.inputs, e.base.embed_tokens.weight, e.layers[0].input_layernorm.weight,
            self.hidden, self.normalized, e.h, e.eps, 4096,
            num_warps=4, enable_fp_fusion=False)
        for index, layer in enumerate(e.layers):
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
        if self.width == 1:
            return  # Every active W1 root wrote its own committed main slot.
        e, w = self.engine, self.width
        for sk, sv, k, v in zip(self.keys, self.values, e.keys, e.values):
            chain_compact_kernel[(e.batch, 8, w)](
                sk, sv, k, v, self.meta, self.counts, e.capacity, w, num_warps=4)

    def replay(self, inputs, lengths, active):
        self.host_meta_array[:] = [(*row, int(lengths[b]), int(active[b]))
                                   for b, row in enumerate(inputs)]
        self.meta.copy_(self.host_meta, non_blocking=True)
        self.graph.replay()

    def compact(self, counts):
        if self.width == 1:
            if any(count not in (0, 1) for count in counts):
                raise ValueError("W1 can consume only one active root")
            return
        self.host_counts_array[:] = counts
        self.counts.copy_(self.host_counts, non_blocking=True)
        self.compact_graph.replay()
