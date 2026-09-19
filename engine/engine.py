"""Qwen3 engine with native HF layers and a persistent single-step CUDA graph.

The first forward uses ordinary causal SDPA over the exact prompt. Decode uses
fixed-address BF16 K/V and an explicit device-position mask. Only exact
pointwise fusions replace reference model operations.
"""

from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM

from fused_ops import FusedMLP, FusedRMSNorm


class FixedKVCache:
    """Layer-local persistent [batch, kv_heads, capacity, head_dim] buffers.

    HF layer calls only require update(). During prompt prefill update returns
    the freshly initialized prefix, allowing ordinary causal Flash SDPA. During
    graph decode it returns full capacity, whose unused tail is explicitly
    masked by the engine. No host reads of position happen in update().
    """

    def __init__(self, config, batch, capacity, device):
        shape = (
            batch, config.num_key_value_heads, capacity,
            getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
        )
        self.key_cache = [
            torch.zeros(shape, dtype=torch.bfloat16, device=device)
            for _ in range(config.num_hidden_layers)
        ]
        self.value_cache = [torch.zeros_like(k) for k in self.key_cache]
        self.prefill = True

    def update(self, key, value, layer_idx, cache_kwargs=None):
        k = self.key_cache[layer_idx]
        v = self.value_cache[layer_idx]
        if self.prefill:
            length = key.shape[2]
            k[:, :, :length, :].copy_(key)
            v[:, :, :length, :].copy_(value)
            return k[:, :, :length, :], v[:, :, :length, :]
        position = cache_kwargs["cache_position"]
        k.index_copy_(2, position, key)
        v.index_copy_(2, position, value)
        return k, v


class Engine:
    def __init__(self, model_path: str) -> None:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device = torch.device("cuda:0")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        ).eval().to(self.device)
        base = self.model.model
        base.norm = FusedRMSNorm(base.norm)
        for layer in base.layers:
            layer.input_layernorm = FusedRMSNorm(layer.input_layernorm)
            layer.post_attention_layernorm = FusedRMSNorm(layer.post_attention_layernorm)
            layer.self_attn.q_norm = FusedRMSNorm(layer.self_attn.q_norm)
            layer.self_attn.k_norm = FusedRMSNorm(layer.self_attn.k_norm)
            layer.mlp = FusedMLP(layer.mlp)
        self.state = None

    def _new_state(self, batch, prompt_length, max_new_tokens):
        # The final emitted token does not need to be consumed by the model.
        capacity = prompt_length + max_new_tokens
        state = SimpleNamespace(
            shape=(batch, prompt_length, max_new_tokens),
            cache=FixedKVCache(self.model.config, batch, capacity, self.device),
            token=torch.zeros((batch, 1), dtype=torch.long, device=self.device),
            position=torch.full((1,), prompt_length, dtype=torch.long, device=self.device),
            key_positions=torch.arange(capacity, dtype=torch.long, device=self.device),
            prompt_positions=torch.arange(prompt_length, dtype=torch.long, device=self.device),
            graph=None,
        )
        # Generate the table with the pinned HF implementation so both prefill
        # and decode use precisely its frequency construction and BF16 cast.
        dummy = torch.empty((), dtype=torch.bfloat16, device=self.device)
        state.cos, state.sin = self.model.model.rotary_emb(
            dummy, state.key_positions.unsqueeze(0)
        )
        return state

    def _prefill(self, state, ids):
        base = self.model.model
        length = ids.shape[1]
        state.cache.prefill = True
        positions = state.prompt_positions
        embeddings = (state.cos[:, :length, :], state.sin[:, :length, :])
        x = base.embed_tokens(ids)
        for layer in base.layers:
            x = layer(
                x,
                attention_mask=None,
                position_ids=positions.unsqueeze(0),
                past_key_value=state.cache,
                use_cache=True,
                cache_position=positions,
                position_embeddings=embeddings,
            )[0]
        # Normalization is independent across rows; only this final position
        # can contribute to the requested last-position logits.
        x = base.norm(x[:, -1:, :].contiguous())
        logits = self.model.lm_head(x)
        state.token.copy_(logits[:, -1, :].argmax(-1, keepdim=True))
        state.position.fill_(length)
        state.cache.prefill = False

    def _decode(self, state):
        base = self.model.model
        position = state.position
        # Keep the mask entirely on-device. This is required for a fixed-size
        # cache: is_causal=False with a one-token query alone is insufficient.
        allowed = state.key_positions <= position[0]
        mask = torch.zeros_like(state.key_positions, dtype=torch.bfloat16)
        mask.masked_fill_(~allowed, float("-inf"))
        mask = mask.view(1, 1, 1, -1)
        embeddings = (
            state.cos.index_select(1, position),
            state.sin.index_select(1, position),
        )
        x = base.embed_tokens(state.token)
        for layer in base.layers:
            x = layer(
                x,
                attention_mask=mask,
                position_ids=position.unsqueeze(0),
                past_key_value=state.cache,
                use_cache=True,
                cache_position=position,
                position_embeddings=embeddings,
            )[0]
        logits = self.model.lm_head(base.norm(x))
        state.token.copy_(logits[:, -1, :].argmax(-1, keepdim=True))
        position.add_(1)
        return state.token

    def _capture(self, state):
        # Compilation, cuBLAS initialization, and allocator setup occur in the
        # untimed warmup generation. Use a non-default stream per CUDA docs.
        saved_token = state.token.clone()
        saved_position = state.position.clone()
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                state.token.copy_(saved_token)
                state.position.copy_(saved_position)
                self._decode(state)
            state.token.copy_(saved_token)
            state.position.copy_(saved_position)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            self._decode(state)
        state.graph = graph
        # Warmup/capture touch exactly the next slot, which the first real
        # replay overwrites. Prompt slots are never altered by capture.
        state.token.copy_(saved_token)
        state.position.copy_(saved_position)

    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        if max_new_tokens <= 0:
            return
        batch = len(input_ids)
        prompt_length = len(input_ids[0])
        shape = (batch, prompt_length, max_new_tokens)
        with torch.inference_mode():
            if self.state is None or self.state.shape != shape:
                self.state = self._new_state(*shape)
            state = self.state
            ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
            self._prefill(state, ids)
            yield state.token[:, 0].tolist()
            if max_new_tokens == 1:
                return
            if state.graph is None:
                self._capture(state)
            for _ in range(max_new_tokens - 1):
                state.graph.replay()
                yield state.token[:, 0].tolist()
