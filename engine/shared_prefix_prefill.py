"""Guarded request-local half-prefix prefill; ordinary decode is unchanged."""
import time
import torch
import torch.nn.functional as F
import triton
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right
from full_prefill_engine import Engine as FullPrefillEngine
from custom_kernels import embedding_norm_kernel, residual_norm_kernel, swiglu_kernel
from prefill_kernels import prefill_qkv_rope_cache_kernel
from shared_prefix_guard import guard_prefix, retire_prefix, prepare_prefix


class PrefillSpan:
    """Contiguous leading-row views into the existing sequential-use scratch."""
    def __init__(self, engine, batch, length, start, input_ids):
        self.batch, self.length, self.start = batch, length, start
        self.rows = batch * length
        self.input_ids = input_ids
        for name in ("hidden", "normalized", "qkv", "query", "branch",
                     "gateup", "intermediate"):
            setattr(self, name, getattr(engine, "prefill_" + name)[:self.rows])
        self.query_view = self.query.view(batch, length, 32, 128).transpose(1, 2)
        self.kv = [(k[:batch, :, :start + length, :],
                    v[:batch, :, :start + length, :])
                   for k, v in zip(engine.keys, engine.values)]
        self.last_normalized = self.normalized.view(batch, length, engine.h)[:, -1, :]
        self.bias = causal_lower_right(length, start + length) if start else None


class Engine(FullPrefillEngine):
    def _allocate(self, batch, prompt, output):
        guard_prefix(self)
        retire_prefix(self)
        super()._allocate(batch, prompt, output)
        self._prefix_attempted = False
        self._prefix_owner = None

    def _capture_prefill(self):
        guard_prefix(self)
        super()._capture_prefill()

    def _parent_stream(self, input_ids, output):
        # Explicit original method: this comparator never performs detection.
        yield from FullPrefillEngine.generate(self, input_ids, output)

    def _candidate_stream(self, input_ids, output, owner):
        # Same complete steady dispatch as public requests; the private owner
        # retains quarantine while admitting this already-prepared route.
        yield from __class__.generate(self, input_ids, output,
                                      _admission_owner=owner)

    def _use_shared_prefix(self, input_ids, owner):
        if owner is None or owner.graph is None:
            return False
        prefix = input_ids[0][:owner.prefix_length]
        return all(row[:owner.prefix_length] == prefix for row in input_ids[1:])

    def _route_stream(self, input_ids, output, owner):
        # Passing an owner is private admission routing, not a persistent flag.
        if not self._use_shared_prefix(input_ids, owner):
            yield from self._parent_stream(input_ids, output)
            return
        with torch.inference_mode():
            current = torch.tensor(input_ids, device="cuda:0", dtype=torch.int64)
            owner.current_input = current
            self.prefill_input.copy_(current)
            owner.graph.replay()
            first = self.ids.clone()
            self.position.fill_(self.prompt)
            self.ids.copy_(first)
            yield self.ids.tolist()
            if output > 1:
                yield from self.chunks.generate()

    def _make_prefix_spans(self, owner):
        # Owner is already engine-published; no optional view precedes it.
        owner.suffix_input = torch.empty(
            (self.batch, self.prompt-owner.prefix_length),
            device=self.prefill_input.device, dtype=torch.int64)
        owner.prefix = PrefillSpan(self, 1, owner.prefix_length, 0,
                                  self.prefill_input[0, :owner.prefix_length])
        owner.suffix = PrefillSpan(self, self.batch,
            self.prompt-owner.prefix_length, owner.prefix_length, owner.suffix_input)
        owner.copies = tuple((cache[1:, :, :owner.prefix_length, :],
                             cache[:1, :, :owner.prefix_length, :])
                            for pair in zip(self.keys, self.values) for cache in pair)

    def _shared_prefill_eager(self, owner):
        self._prefill_span(owner.prefix)
        for destination, source in owner.copies:
            owner.live()
            destination.copy_(source)
        owner.live()
        owner.suffix_input.copy_(self.prefill_input[:, owner.prefix_length:])
        self._prefill_span(owner.suffix)
        owner.live()
        torch.mm(owner.suffix.last_normalized,
                 self.model.lm_head.weight.t(), out=self.logits)
        owner.live()
        torch.argmax(self.logits, dim=-1, out=self.ids)

    def generate(self, input_ids: list[list[int]], max_new_tokens: int,
                 *, _admission_owner=None):
        if max_new_tokens <= 0:
            return
        if _admission_owner is None:
            guard_prefix(self)
        elif (getattr(self, '_prefix_pending', None) is not _admission_owner
              or getattr(self, '_prefix_owner', None) is not _admission_owner
              or _admission_owner.engine is not self
              or _admission_owner.drain_error is not None):
            raise RuntimeError('invalid shared-prefix admission owner')
        batch, prompt = len(input_ids), len(input_ids[0])
        if _admission_owner is not None and (
                batch <= 1 or prompt < 256
                or _admission_owner.output != max_new_tokens
                or self.shape != (batch, prompt, max_new_tokens)
                or not self._prefix_attempted or self.prefill_graph is None
                or _admission_owner.graph is None
                or (max_new_tokens > 1 and self.chunks is None)):
            raise RuntimeError('unprepared shared-prefix admission route')
        if batch <= 1 or prompt < 256:
            yield from super().generate(input_ids, max_new_tokens)
            return
        with torch.inference_mode():
            if self.shape != (batch, prompt, max_new_tokens):
                self._allocate(batch, prompt, max_new_tokens)
            if not self._prefix_attempted:
                # Preserve ALL original tuning and ordinary capture priority.
                current = torch.tensor(input_ids, device="cuda:0", dtype=torch.int64)
                self.prefill_input.copy_(current)
                started = time.monotonic()
                if self.prefill_graph is None:
                    self._capture_prefill()
                self.prefill_graph.replay()
                first = self.ids.clone()
                if max_new_tokens > 1 and self.chunks is None:
                    self._capture_chunks(first, max_new_tokens-1)
                self.position.fill_(prompt)
                self.ids.copy_(first)
                base_setup = time.monotonic()-started
                self._prefix_attempted = True
                prepare_prefix(self, input_ids, max_new_tokens, base_setup)
            owner = self._prefix_owner
            if _admission_owner is not None:
                owner = _admission_owner
            if owner is not None and not owner.enabled and _admission_owner is None:
                owner = None
            # Every path restores this call's IDs/cache before emitting tokens.
            yield from self._route_stream(input_ids, max_new_tokens, owner)

    def _prefill_span(self, span):
        rows = span.rows
        owner = getattr(self, "_prefix_pending", None)
        live = owner.live if owner is not None else lambda: None
        live()
        embedding_norm_kernel[(rows,)](
            span.input_ids, self.base.embed_tokens.weight,
            self.layers[0].input_layernorm.weight,
            span.hidden, span.normalized, self.h, self.eps, 4096,
            num_warps=4, enable_fp_fusion=False,
        )
        for idx, layer in enumerate(self.layers):
            live()
            attention, mlp = layer.self_attn, layer.mlp
            qkv_weight, gateup_weight = self.packed[idx]
            # Full-prefill plans bind B*S rows. Their actual eligibility check
            # falls back to torch.mm for these smaller spans; no extra tuning.
            self.dense_prefill.run("qkv", idx, span.normalized, qkv_weight, span.qkv)
            live()
            prefill_qkv_rope_cache_kernel[(triton.cdiv(rows, 4), 40)](
                span.qkv, attention.q_norm.weight, attention.k_norm.weight,
                self.cos, self.sin, span.query, self.keys[idx], self.values[idx],
                rows, span.length, self.capacity, self.eps, START=span.start,
                R=4, num_warps=4, enable_fp_fusion=False,
            )
            live()
            key, value = span.kv[idx]
            # Eligibility was checked for these exact tensors, with FLASH as
            # the only enabled backend. Metadata does not change during capture.
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                attended = F.scaled_dot_product_attention(
                    span.query_view, key, value, attn_mask=span.bias,
                    dropout_p=0.0, is_causal=span.start == 0,
                    scale=128 ** -0.5, enable_gqa=True,
                )
            live()
            attended_rows = attended.transpose(1, 2).reshape(rows, 4096)
            self.dense_prefill.run("output", idx, attended_rows,
                                   attention.o_proj.weight, span.branch)
            live()
            residual_norm_kernel[(rows,)](
                span.branch, span.hidden, layer.post_attention_layernorm.weight,
                span.normalized, self.h, self.eps, 4096,
                num_warps=4, enable_fp_fusion=False,
            )
            live()
            self.dense_prefill.run("gateup", idx, span.normalized, gateup_weight, span.gateup)
            live()
            swiglu_kernel[(triton.cdiv(rows * self.i, 1024),)](
                span.gateup, span.intermediate, self.i, rows * self.i,
                num_warps=4, enable_fp_fusion=False,
            )
            live()
            self.dense_prefill.run("down", idx, span.intermediate,
                                   mlp.down_proj.weight, span.branch)
            next_weight = (self.layers[idx + 1].input_layernorm.weight
                           if idx + 1 < len(self.layers) else self.base.norm.weight)
            live()
            residual_norm_kernel[(rows,)](
                span.branch, span.hidden, next_weight, span.normalized,
                self.h, self.eps, 4096, num_warps=4, enable_fp_fusion=False,
            )

