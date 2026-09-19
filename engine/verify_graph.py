"""Captured four-row exact model verification over the native B1 cache."""
import torch
import triton

from custom_kernels import (
    embedding_norm_kernel, residual_norm_kernel, attention_merge_kernel,
    swiglu_kernel,
)
from speculative_kernels import (
    verify_qkv_rope_cache_kernel, verify_attention_split_kernel,
)


class VerifyGraph:
    def __init__(self, engine, first):
        self.engine = engine
        device = engine.ids.device

        def empty(*shape, dtype=torch.bfloat16):
            return torch.empty(shape, device=device, dtype=dtype)

        self.inputs = empty(4, dtype=torch.int64)
        self.host_inputs = torch.empty((4,), dtype=torch.int64, pin_memory=True)
        self.output = empty(4, dtype=torch.int64)
        self.hidden = empty(4, engine.h)
        self.normalized = empty(4, engine.h)
        self.qkv = empty(4, 6144)
        self.query = empty(4, 32, 128)
        self.attention = empty(4, 4096)
        self.branch = empty(4, engine.h)
        self.gateup = empty(4, 2 * engine.i)
        self.intermediate = empty(4, engine.i)
        self.logits = empty(4, engine.model.config.vocab_size)
        self.partial = empty(128, engine.splits, 128, dtype=torch.float32)
        self.pmax = empty(128, engine.splits, dtype=torch.float32)
        self.psum = empty(128, engine.splits, dtype=torch.float32)
        self.capture_stream = torch.cuda.Stream(device=device)

        def reset():
            engine.position.fill_(engine.prompt)
            engine.ids.copy_(first)
            # Every warm/capture trial starts from the real prompt prefix and
            # pending first token; remaining rows are valid vocabulary IDs.
            self.inputs.copy_(first.expand(4))

        torch.cuda.synchronize(device)
        with torch.cuda.stream(self.capture_stream):
            reset()
            self._verify()
        self.capture_stream.synchronize()
        reset()
        torch.cuda.synchronize(device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.capture_stream):
            self._verify()
        reset()
        self.graph.replay()
        torch.cuda.synchronize(device)
        reset()
        torch.cuda.synchronize(device)

    def _verify(self):
        e = self.engine
        embedding_norm_kernel[(4,)](
            self.inputs, e.base.embed_tokens.weight,
            e.layers[0].input_layernorm.weight,
            self.hidden, self.normalized, e.h, e.eps, 4096,
            num_warps=4, enable_fp_fusion=False,
        )
        for index, layer in enumerate(e.layers):
            a, m = layer.self_attn, layer.mlp
            qkv_weight, gu_weight = e.packed[index]
            torch.mm(self.normalized, qkv_weight.t(), out=self.qkv)
            verify_qkv_rope_cache_kernel[(4, 40)](
                self.qkv, a.q_norm.weight, a.k_norm.weight,
                e.cos, e.sin, e.position, self.query,
                e.keys[index], e.values[index], e.capacity, e.eps,
                num_warps=4, enable_fp_fusion=False,
            )
            verify_attention_split_kernel[(8, e.splits)](
                self.query, e.keys[index], e.values[index], e.position,
                self.partial, self.pmax, self.psum,
                e.capacity, e.splits, 128 ** -0.5,
                num_warps=4, num_stages=1,
            )
            attention_merge_kernel[(128,)](
                self.partial, self.pmax, self.psum, self.attention,
                e.splits, triton.next_power_of_2(e.splits), num_warps=4,
            )
            torch.mm(self.attention, a.o_proj.weight.t(), out=self.branch)
            residual_norm_kernel[(4,)](
                self.branch, self.hidden, layer.post_attention_layernorm.weight,
                self.normalized, e.h, e.eps, 4096,
                num_warps=4, enable_fp_fusion=False,
            )
            torch.mm(self.normalized, gu_weight.t(), out=self.gateup)
            swiglu_kernel[(triton.cdiv(4 * e.i, 1024),)](
                self.gateup, self.intermediate, e.i, 4 * e.i,
                num_warps=4, enable_fp_fusion=False,
            )
            torch.mm(self.intermediate, m.down_proj.weight.t(), out=self.branch)
            next_weight = (e.layers[index + 1].input_layernorm.weight
                           if index + 1 < len(e.layers) else e.base.norm.weight)
            residual_norm_kernel[(4,)](
                self.branch, self.hidden, next_weight, self.normalized,
                e.h, e.eps, 4096,
                num_warps=4, enable_fp_fusion=False,
            )
        torch.mm(self.normalized, e.model.lm_head.weight.t(), out=self.logits)
        torch.argmax(self.logits, dim=-1, out=self.output)

    def replay(self, inputs):
        for index, token in enumerate(inputs):
            self.host_inputs[index] = token
        self.inputs.copy_(self.host_inputs, non_blocking=True)
        self.graph.replay()
