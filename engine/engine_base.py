"""BF16 Qwen3: packed causal SDPA prefill and fused graph decode."""
import types
import torch
import torch.nn.functional as F
import triton
from transformers import AutoModelForCausalLM
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from custom_kernels import (
    rms_kernel, embedding_norm_kernel, residual_norm_kernel,
    qkv_rope_cache_kernel, attention_split_kernel, attention_merge_kernel,
    swiglu_kernel,
)


class FusedRMSNorm(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        self.weight = original.weight
        self.variance_epsilon = original.variance_epsilon

    def forward(self, x):
        # Qwen projections and layer residuals are contiguous in the native path.
        x = x.contiguous()
        out = torch.empty_like(x)
        width = x.shape[-1]
        rms_kernel[(x.numel() // width,)](
            x, self.weight, out, width, self.variance_epsilon,
            triton.next_power_of_2(width), num_warps=4, enable_fp_fusion=False,
        )
        return out


class PrefillCache(Cache):
    """Fresh full-prompt prefill writes static storage, returns only prompt KV.

    Returning the original prompt tensors preserves native SDPA's causal shape
    and never lets it inspect uninitialized static capacity.
    """
    def __init__(self, keys, values):
        super().__init__()
        self.keys = keys
        self.values = values
        self.length = 0

    def get_seq_length(self, layer_idx=0):
        return self.length

    def get_max_cache_shape(self):
        return None

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        length = key_states.shape[-2]
        self.keys[layer_idx][:, :, :length, :].copy_(key_states)
        self.values[layer_idx][:, :, :length, :].copy_(value_states)
        if layer_idx == 0:
            self.length = length
        return key_states, value_states


# These adapters only run on full causal prefill; custom decode bypasses them.
def packed_prefill_attention(self, hidden_states, position_embeddings,
                             attention_mask=None, past_key_value=None,
                             cache_position=None, **kwargs):
    shape = hidden_states.shape[:-1]
    projected = F.linear(hidden_states, self._packed_qkv)
    q, k, v = projected.split((4096, 1024, 1024), dim=-1)
    q = self.q_norm(q.reshape(*shape, 32, 128)).transpose(1, 2)
    k = self.k_norm(k.reshape(*shape, 8, 128)).transpose(1, 2)
    v = v.reshape(*shape, 8, 128).transpose(1, 2)
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    if past_key_value is not None:
        k, v = past_key_value.update(k, v, self.layer_idx,
                                    {"cos": cos, "sin": sin,
                                     "cache_position": cache_position})
    output = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attention_mask, dropout_p=0.0,
        is_causal=(attention_mask is None and q.shape[-2] > 1),
        scale=self.scaling, enable_gqa=True,
    )
    output = output.transpose(1, 2).reshape(*shape, 4096).contiguous()
    return self.o_proj(output), None


def packed_prefill_mlp(self, x):
    gu = F.linear(x, self._packed_gateup)
    out = torch.empty((*x.shape[:-1], 9728), device=x.device, dtype=x.dtype)
    size = out.numel()
    swiglu_kernel[(triton.cdiv(size, 1024),)](
        gu, out, 9728, size, num_warps=4, enable_fp_fusion=False,
    )
    return self.down_proj(out)


class Engine:
    def __init__(self, model_path: str) -> None:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", local_files_only=True,
        ).eval().to("cuda:0")
        self.base = self.model.model
        self.layers = list(self.base.layers)
        self.h = self.model.config.hidden_size
        self.i = self.model.config.intermediate_size
        self.eps = self.model.config.rms_norm_eps
        assert self.h == 2560 and self.i == 9728
        assert self.model.config.num_attention_heads == 32
        assert self.model.config.num_key_value_heads == 8
        assert self.layers[0].self_attn.head_dim == 128
        self.base.norm = FusedRMSNorm(self.base.norm)
        self.packed = []
        # Rebind native projections to packed slices, so packing does not retain
        # a duplicate checkpoint. These row slices remain fully contiguous.
        with torch.inference_mode():
            for layer in self.layers:
                a, m = layer.self_attn, layer.mlp
                layer.input_layernorm = FusedRMSNorm(layer.input_layernorm)
                layer.post_attention_layernorm = FusedRMSNorm(layer.post_attention_layernorm)
                a.q_norm = FusedRMSNorm(a.q_norm)
                a.k_norm = FusedRMSNorm(a.k_norm)
                qkv = torch.cat((a.q_proj.weight, a.k_proj.weight, a.v_proj.weight), dim=0)
                gu = torch.cat((m.gate_proj.weight, m.up_proj.weight), dim=0)
                a.q_proj.weight = torch.nn.Parameter(qkv[:4096], requires_grad=False)
                a.k_proj.weight = torch.nn.Parameter(qkv[4096:5120], requires_grad=False)
                a.v_proj.weight = torch.nn.Parameter(qkv[5120:], requires_grad=False)
                m.gate_proj.weight = torch.nn.Parameter(gu[:self.i], requires_grad=False)
                m.up_proj.weight = torch.nn.Parameter(gu[self.i:], requires_grad=False)
                self.packed.append((qkv, gu))
                a._packed_qkv = qkv
                m._packed_gateup = gu
                a.forward = types.MethodType(packed_prefill_attention, a)
                m.forward = types.MethodType(packed_prefill_mlp, m)
        self.shape = None
        self.graph = None

    def _allocate(self, batch, prompt, output):
        self.batch, self.prompt = batch, prompt
        self.capacity = prompt + output
        self.splits = triton.cdiv(self.capacity, 256)
        device = "cuda:0"
        def empty(*shape, dtype=torch.bfloat16):
            return torch.empty(shape, device=device, dtype=dtype)
        # Allocate each layer independently to avoid a second all-cache payload.
        self.keys = [empty(batch, 8, self.capacity, 128) for _ in self.layers]
        self.values = [empty(batch, 8, self.capacity, 128) for _ in self.layers]
        self.ids = torch.zeros((batch,), device=device, dtype=torch.int64)
        self.position = torch.full((1,), prompt, device=device, dtype=torch.int64)
        self.hidden = empty(batch, self.h)
        self.normalized = empty(batch, self.h)
        self.qkv = empty(batch, 6144)
        self.query = empty(batch, 32, 128)
        self.attention = empty(batch, 4096)
        self.branch = empty(batch, self.h)
        self.gateup = empty(batch, 2 * self.i)
        self.intermediate = empty(batch, self.i)
        self.logits = empty(batch, self.model.config.vocab_size)
        self.partial = empty(batch * 32, self.splits, 128, dtype=torch.float32)
        self.pmax = empty(batch * 32, self.splits, dtype=torch.float32)
        self.psum = empty(batch * 32, self.splits, dtype=torch.float32)
        positions = torch.arange(self.capacity, device=device).unsqueeze(0)
        self.cos, self.sin = self.base.rotary_emb(self.hidden, positions)
        self.cos, self.sin = self.cos.contiguous(), self.sin.contiguous()
        self.graph = None
        self.shape = (batch, prompt, output)

    def _step(self):
        """Consume self.ids at self.position; replace IDs and advance position."""
        b = self.batch
        embedding_norm_kernel[(b,)](
            self.ids, self.base.embed_tokens.weight,
            self.layers[0].input_layernorm.weight,
            self.hidden, self.normalized, self.h, self.eps, 4096,
            num_warps=4, enable_fp_fusion=False,
        )
        for idx, layer in enumerate(self.layers):
            a, m = layer.self_attn, layer.mlp
            qkv_w, gu_w = self.packed[idx]
            self.native_layout.run("qkv", idx, self.normalized, qkv_w, self.qkv)
            qkv_rope_cache_kernel[(b, 40)](
                self.qkv, a.q_norm.weight, a.k_norm.weight,
                self.cos, self.sin, self.position, self.query,
                self.keys[idx], self.values[idx], self.capacity, self.eps,
                num_warps=4, enable_fp_fusion=False,
            )
            attention_split_kernel[(b, 8, self.splits)](
                self.query, self.keys[idx], self.values[idx], self.position,
                self.partial, self.pmax, self.psum,
                self.capacity, self.splits, 128 ** -0.5,
                num_warps=4, num_stages=1,
            )
            attention_merge_kernel[(b * 32,)](
                self.partial, self.pmax, self.psum, self.attention,
                self.splits, triton.next_power_of_2(self.splits),
                num_warps=4,
            )
            self.native_layout.run("output", idx, self.attention,
                                   a.o_proj.weight, self.branch)
            residual_norm_kernel[(b,)](
                self.branch, self.hidden, layer.post_attention_layernorm.weight,
                self.normalized, self.h, self.eps, 4096,
                num_warps=4, enable_fp_fusion=False,
            )
            self.native_layout.run("gateup", idx, self.normalized, gu_w, self.gateup)
            swiglu_kernel[(triton.cdiv(b * self.i, 1024),)](
                self.gateup, self.intermediate, self.i, b * self.i,
                num_warps=4, enable_fp_fusion=False,
            )
            self.native_layout.run("down", idx, self.intermediate,
                                   m.down_proj.weight, self.branch)
            next_weight = (self.layers[idx + 1].input_layernorm.weight
                           if idx + 1 < len(self.layers) else self.base.norm.weight)
            residual_norm_kernel[(b,)](
                self.branch, self.hidden, next_weight, self.normalized,
                self.h, self.eps, 4096,
                num_warps=4, enable_fp_fusion=False,
            )
        self.native_layout.run("head", 0, self.normalized, self.model.lm_head.weight, self.logits)
        torch.argmax(self.logits, dim=-1, out=self.ids)
        self.position.add_(1)

    def _capture(self, first):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            # Compile Triton and initialize cuBLAS before graph capture. Every
            # trial starts at the same real prefix and only overwrites slot S.
            for _ in range(3):
                self.position.fill_(self.prompt)
                self.ids.copy_(first)
                self._step()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.position.fill_(self.prompt)
        self.ids.copy_(first)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._step()
        torch.cuda.synchronize()
        self.position.fill_(self.prompt)
        self.ids.copy_(first)

    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        if max_new_tokens <= 0:
            return
        batch, prompt = len(input_ids), len(input_ids[0])
        with torch.inference_mode():
            if self.shape != (batch, prompt, max_new_tokens):
                self._allocate(batch, prompt, max_new_tokens)
            current = torch.tensor(input_ids, device="cuda:0", dtype=torch.int64)
            cache = PrefillCache(self.keys, self.values)
            result = self.model(
                input_ids=current, past_key_values=cache, use_cache=True,
                logits_to_keep=1, return_dict=True,
            )
            first = result.logits[:, -1, :].argmax(dim=-1)
            del result
            if max_new_tokens > 1 and self.graph is None:
                self._capture(first)
            self.position.fill_(prompt)
            self.ids.copy_(first)
            yield self.ids.tolist()
            for _ in range(max_new_tokens - 1):
                self.graph.replay()
                yield self.ids.tolist()
