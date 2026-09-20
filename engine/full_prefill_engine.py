"""Captured fused BF16 prefill plus the passing four-step decode chunks."""

import time
import torch
import torch.nn.functional as F
import triton

from engine_base import Engine as BaseEngine
from chunk_graph import DecodeChunks
from native_layout import NativeLayout
from wide_gemv import WideGemvLayout
from hopper_gemm import HopperGemmLayout
from hopper_tiles import HopperTilesLayout
from persistent_vector import PersistentVectorLayout
from fused_cache_attention import FusedCacheAttention
from dense_prefill import DensePrefill
from custom_kernels import embedding_norm_kernel, residual_norm_kernel, swiglu_kernel
from prefill_kernels import prefill_qkv_rope_cache_kernel


class Engine(BaseEngine):
    def __init__(self, model_path: str) -> None:
        self._layout_deadline = time.monotonic() + 230.0
        self.native_layout = None
        super().__init__(model_path)

    def _allocate(self, batch, prompt, output):
        super()._allocate(batch, prompt, output)
        self.chunks = None
        self.prefill_graph = None
        self.prefill_rows = batch * prompt
        self.prefill_input = torch.empty((batch, prompt), dtype=torch.int64, device="cuda:0")
        def empty(width):
            return torch.empty((self.prefill_rows, width), dtype=torch.bfloat16, device="cuda:0")
        # These buffers are independent of the smaller persistent decode state.
        # Every prefill replay rewrites them and all prompt K/V cache entries.
        self.prefill_hidden = empty(self.h)
        self.prefill_normalized = empty(self.h)
        self.prefill_qkv = empty(6144)
        self.prefill_query = empty(4096)
        self.prefill_branch = empty(self.h)
        self.prefill_gateup = empty(2 * self.i)
        self.prefill_intermediate = empty(self.i)
        self.prefill_query_view = self.prefill_query.view(batch, prompt, 32, 128).transpose(1, 2)
        self.prefill_kv = [(k[:, :, :prompt, :], v[:, :, :prompt, :])
                           for k, v in zip(self.keys, self.values)]
        self.prefill_last_normalized = self.prefill_normalized.view(batch, prompt, self.h)[:, -1, :]
        if self.native_layout is None:
            self.native_layout = NativeLayout(self, self._layout_deadline)
            self.native_layout = WideGemvLayout(
                self, self.native_layout, self._layout_deadline)
            self.native_layout = HopperGemmLayout(
                self, self.native_layout, self._layout_deadline)
            self.native_layout = HopperTilesLayout(
                self, self.native_layout, self._layout_deadline)
            self.native_layout = PersistentVectorLayout(
                self, self.native_layout, self._layout_deadline)

        self.dense_prefill = DensePrefill(self, self._layout_deadline)

    def _prefill_eager(self):
        rows = self.prefill_rows
        embedding_norm_kernel[(rows,)](
            self.prefill_input, self.base.embed_tokens.weight,
            self.layers[0].input_layernorm.weight,
            self.prefill_hidden, self.prefill_normalized,
            self.h, self.eps, 4096,
            num_warps=4, enable_fp_fusion=False,
        )
        for idx, layer in enumerate(self.layers):
            attention, mlp = layer.self_attn, layer.mlp
            qkv_weight, gateup_weight = self.packed[idx]
            self.dense_prefill.run("qkv", idx, self.prefill_normalized, qkv_weight, self.prefill_qkv)
            prefill_qkv_rope_cache_kernel[(triton.cdiv(rows, 4), 40)](
                self.prefill_qkv, attention.q_norm.weight, attention.k_norm.weight,
                self.cos, self.sin, self.prefill_query,
                self.keys[idx], self.values[idx],
                rows, self.prompt, self.capacity, self.eps,
                R=4, num_warps=4, enable_fp_fusion=False,
            )
            key, value = self.prefill_kv[idx]
            attended = F.scaled_dot_product_attention(
                self.prefill_query_view, key, value,
                dropout_p=0.0, is_causal=True, scale=128 ** -0.5,
                enable_gqa=True,
            )
            attended_rows = attended.transpose(1, 2).reshape(rows, 4096)
            self.dense_prefill.run("output", idx, attended_rows, attention.o_proj.weight, self.prefill_branch)
            residual_norm_kernel[(rows,)](
                self.prefill_branch, self.prefill_hidden,
                layer.post_attention_layernorm.weight, self.prefill_normalized,
                self.h, self.eps, 4096,
                num_warps=4, enable_fp_fusion=False,
            )
            self.dense_prefill.run("gateup", idx, self.prefill_normalized, gateup_weight, self.prefill_gateup)
            swiglu_kernel[(triton.cdiv(rows * self.i, 1024),)](
                self.prefill_gateup, self.prefill_intermediate,
                self.i, rows * self.i,
                num_warps=4, enable_fp_fusion=False,
            )
            self.dense_prefill.run("down", idx, self.prefill_intermediate, mlp.down_proj.weight, self.prefill_branch)
            next_weight = (self.layers[idx + 1].input_layernorm.weight
                           if idx + 1 < len(self.layers) else self.base.norm.weight)
            residual_norm_kernel[(rows,)](
                self.prefill_branch, self.prefill_hidden, next_weight,
                self.prefill_normalized, self.h, self.eps, 4096,
                num_warps=4, enable_fp_fusion=False,
            )
        # Only the last prompt position contributes the first generated token.
        torch.mm(self.prefill_last_normalized, self.model.lm_head.weight.t(), out=self.logits)
        torch.argmax(self.logits, dim=-1, out=self.ids)

    def _capture_prefill(self):
        current = torch.cuda.current_stream()
        stream = torch.cuda.Stream()
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._prefill_eager()
        current.wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        # Keep this pool independent of all decode chunk pools.
        with torch.cuda.graph(graph, stream=stream):
            self._prefill_eager()
        current.wait_stream(stream)
        self.prefill_graph = graph

    def _capture_chunks(self, first, steps):
        self.fused_cache_attention = FusedCacheAttention(self, self._layout_deadline)
        def decode():
            self._step()
            return self.ids

        def reset():
            self.position.fill_(self.prompt)
            self.ids.copy_(first)

        self.chunks = DecodeChunks(decode, self.ids, steps, reset, chunk_size=4)

    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        if max_new_tokens <= 0:
            return
        batch, prompt = len(input_ids), len(input_ids[0])
        with torch.inference_mode():
            if self.shape != (batch, prompt, max_new_tokens):
                self._allocate(batch, prompt, max_new_tokens)
            current = torch.tensor(input_ids, device="cuda:0", dtype=torch.int64)
            self.prefill_input.copy_(current)
            if self.prefill_graph is None:
                self._capture_prefill()
            self.prefill_graph.replay()
            first = self.ids.clone()
            if max_new_tokens > 1 and self.chunks is None:
                self._capture_chunks(first, max_new_tokens - 1)
            self.position.fill_(prompt)
            self.ids.copy_(first)
            yield self.ids.tolist()
            if max_new_tokens > 1:
                yield from self.chunks.generate()
