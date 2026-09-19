"""Captured W8 full BF16 tree verifier and visited-path compaction graphs."""
import torch
import triton
from custom_kernels import embedding_norm_kernel, residual_norm_kernel, swiglu_kernel
from recycled_kernels import chain_ids_kernel, chain_merge_kernel
from tree_kernels import tree_qkv_kernel, tree_attention_kernel, tree_compact_kernel


class TreeGraph:
    def __init__(self, engine, width=8):
        self.engine, self.width = engine, int(width)
        assert self.width == 8
        b, w, device = engine.batch, self.width, engine.ids.device
        self.rows = b * w
        self.splits = triton.cdiv(engine.capacity + w - 1, 256)
        def empty(*shape, dtype=torch.bfloat16):
            return torch.empty(shape, dtype=dtype, device=device)
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
        # A failed constructor has no caller-owned reference yet. Pin its
        # buffers before any custom kernel can be queued on the capture stream.
        # Failed drains leave the owner quarantined until engine/process exit.
        engine.recycle_quarantine.append(self)
        try:
            torch.cuda.synchronize(device)
            with torch.cuda.stream(self.stream):
                self._verify()
                # Negative paths warm compaction kernels without main writes.
                self._compact()
            self.stream.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=self.stream):
                self._verify()
            self.compact_graph = None
            if self.width == 8:
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
        # One BF16 projection over all B*8 hypothetical rows.
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
            tree_qkv_kernel[(rows, 40)](
                self.qkv, a.q_norm.weight, a.k_norm.weight, e.cos, e.sin,
                self.meta, self.query, self.keys[index], self.values[index],
                e.capacity, w, e.eps, num_warps=4, enable_fp_fusion=False)
            tree_attention_kernel[(e.batch, 8, self.splits)](
                self.query, e.keys[index], e.values[index], self.keys[index], self.values[index],
                self.meta, self.partial, self.pmax, self.psum,
                e.capacity, w, self.splits, 128 ** -0.5, num_warps=8, num_stages=1)
            chain_merge_kernel[(rows * 32,)](
                self.partial, self.pmax, self.psum, self.attention,
                self.splits, triton.next_power_of_2(self.splits), num_warps=4)
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
        e, w = self.engine, self.width
        for sk, sv, k, v in zip(self.keys, self.values, e.keys, e.values):
            tree_compact_kernel[(e.batch, 8, w)](
                sk, sv, k, v, self.meta, self.paths, e.capacity, w, num_warps=4)

    def replay(self, inputs, lengths, active):
        self.host_meta_array[:] = [(*row, int(lengths[b]), int(active[b]))
                                   for b, row in enumerate(inputs)]
        self.meta.copy_(self.host_meta, non_blocking=True)
        self.graph.replay()

    def compact(self, paths):
        if len(paths) != self.engine.batch or any(len(path) > 5 for path in paths):
            raise ValueError("one visited tree path per sequence required")
        self.host_paths_array[:] = [tuple(path) + (-1,) * (self.width - len(path))
                                   for path in paths]
        self.paths.copy_(self.host_paths, non_blocking=True)
        self.compact_graph.replay()
