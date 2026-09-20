"""Qwen3 weights and forward passes over static buffers.

Everything here is written so that a whole prefill or decode step touches only
preallocated tensors and launches no host-synchronising op, which is what lets
``engine.py`` capture each step into a CUDA graph and replay it.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch.nn.attention import SDPBackend, sdpa_kernel

from kernels import DecodeAttention, add_rms_norm, pick_attention, pick_gateup, pick_matmul, pick_normed, qk_norm_rope_cache, rms_norm, swiglu


@dataclass(frozen=True)
class Config:
    hidden: int
    intermediate: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    eps: float
    rope_theta: float
    tie_embeddings: bool

    @classmethod
    def load(cls, model_path: Path) -> "Config":
        raw = json.loads((model_path / "config.json").read_text())
        return cls(
            hidden=raw["hidden_size"],
            intermediate=raw["intermediate_size"],
            layers=raw["num_hidden_layers"],
            heads=raw["num_attention_heads"],
            kv_heads=raw["num_key_value_heads"],
            head_dim=raw.get("head_dim") or raw["hidden_size"] // raw["num_attention_heads"],
            vocab=raw["vocab_size"],
            eps=raw["rms_norm_eps"],
            rope_theta=raw["rope_theta"],
            tie_embeddings=raw.get("tie_word_embeddings", False),
        )


@dataclass
class Layer:
    in_norm: torch.Tensor
    wqkv: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    wo: torch.Tensor
    post_norm: torch.Tensor
    wgu: torch.Tensor
    wd: torch.Tensor


def _read_all(model_path: Path, device) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for shard in sorted(model_path.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device=str(device)) as f:
            for name in f.keys():
                tensors[name] = f.get_tensor(name)
    if not tensors:
        raise FileNotFoundError(f"no *.safetensors under {model_path}; the checkpoint must be there")
    return tensors


def rope_tables(cfg: Config, max_pos: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin in bf16 exactly as ``Qwen3RotaryEmbedding.forward`` builds them.

    ``inv_freq`` is fp32, ``freqs`` is an fp32 matmul with TF32 disabled, the
    table is ``cat(freqs, freqs)`` so column ``i`` and ``i + D/2`` share an
    angle, and the cast to bf16 happens after cos/sin.
    """
    D = cfg.head_dim
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, D, 2, dtype=torch.int64, device=device).float() / D))
    positions = torch.arange(max_pos, device=device, dtype=torch.float32)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    freqs = (inv_freq[:, None].float() @ positions[None, :].float()).transpose(0, 1)
    torch.backends.cuda.matmul.allow_tf32 = tf32
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(torch.bfloat16).contiguous(), emb.sin().to(torch.bfloat16).contiguous()


class Model:
    def __init__(self, model_path: str | Path, device="cuda:0", max_pos: int = 4096):
        model_path = Path(model_path)
        self.cfg = Config.load(model_path)
        self.device = torch.device(device)
        raw = _read_all(model_path, self.device)
        self.embed = raw.pop("model.embed_tokens.weight")
        self.lm_head = raw.pop("lm_head.weight", None)
        if self.lm_head is None:
            if not self.cfg.tie_embeddings:
                raise KeyError("lm_head.weight missing and embeddings are not tied")
            self.lm_head = self.embed
        self.final_norm = raw.pop("model.norm.weight")
        self.layers: list[Layer] = []
        for i in range(self.cfg.layers):
            p = f"model.layers.{i}."
            wqkv = torch.cat(
                [raw.pop(p + "self_attn.q_proj.weight"), raw.pop(p + "self_attn.k_proj.weight"), raw.pop(p + "self_attn.v_proj.weight")],
                dim=0,
            )
            wgu = torch.cat([raw.pop(p + "mlp.gate_proj.weight"), raw.pop(p + "mlp.up_proj.weight")], dim=0)
            self.layers.append(Layer(
                in_norm=raw.pop(p + "input_layernorm.weight"),
                wqkv=wqkv,
                q_norm=raw.pop(p + "self_attn.q_norm.weight"),
                k_norm=raw.pop(p + "self_attn.k_norm.weight"),
                wo=raw.pop(p + "self_attn.o_proj.weight"),
                post_norm=raw.pop(p + "post_attention_layernorm.weight"),
                wgu=wgu,
                wd=raw.pop(p + "mlp.down_proj.weight"),
            ))
        if raw:
            raise KeyError(f"unexpected tensors in checkpoint: {sorted(raw)[:5]}")
        self.cos, self.sin = rope_tables(self.cfg, max_pos, self.device)
        self.scale = 1.0 / math.sqrt(self.cfg.head_dim)

    def ensure_rope(self, max_pos: int) -> None:
        if self.cos.shape[0] < max_pos:
            self.cos, self.sin = rope_tables(self.cfg, max_pos, self.device)


def decode_matmuls(m: Model, M: int, log) -> dict[str, object]:
    """Per-shape projection callables for M-row decode/verify steps.

    "qkv", "gu" and "lm" take ``(x, y, w_norm, xout, w)`` and fold the residual
    add and RMSNorm of their input in front of the projection; "o" and "d" are
    plain ``(a, w)``. Each is the timed winner between cuBLAS pipelines and
    Triton kernels on this device.
    """
    cfg = m.cfg
    layer = m.layers[0]
    bf16 = torch.bfloat16
    x = torch.randn((M, cfg.hidden), dtype=bf16, device=m.device)
    y = torch.randn((M, cfg.hidden), dtype=bf16, device=m.device)
    a = torch.randn((M, cfg.heads * cfg.head_dim), dtype=bf16, device=m.device)
    act = torch.randn((M, cfg.intermediate), dtype=bf16, device=m.device)
    if os.environ.get("ENGINE_FORCE_CUBLAS") == "1":
        unfused = lambda x, y, wn, xout, w: add_rms_norm(x, y, wn, cfg.eps, xout) @ w.t()
        return {
            "qkv": unfused,
            "o": lambda a, w: a @ w.t(),
            "gu": lambda x, y, wn, xout, w: swiglu(add_rms_norm(x, y, wn, cfg.eps, xout) @ w.t()),
            "d": lambda a, w: a @ w.t(),
            "lm": unfused,
        }
    if os.environ.get("ENGINE_NORM_FUSED", "1") != "1":
        L = m.layers
        qkv, gu, lm = (pick_matmul(x, layer.wqkv, log, ws=[l.wqkv for l in L]), pick_gateup(x, layer.wgu, log, ws=[l.wgu for l in L]),
                       pick_matmul(x, m.lm_head, log))
        return {
            "qkv": lambda x, y, wn, xout, w: qkv(add_rms_norm(x, y, wn, cfg.eps, xout), w),
            "o": pick_matmul(a, layer.wo, log, ws=[l.wo for l in L]),
            "gu": lambda x, y, wn, xout, w: gu(add_rms_norm(x, y, wn, cfg.eps, xout), w),
            "d": pick_matmul(act, layer.wd, log, ws=[l.wd for l in L]),
            "lm": lambda x, y, wn, xout, w: lm(add_rms_norm(x, y, wn, cfg.eps, xout), w),
        }
    L = m.layers
    return {
        "qkv": pick_normed("matmul", x, y, layer.in_norm, layer.wqkv, cfg.eps, log, ws=[l.wqkv for l in L]),
        "o": pick_matmul(a, layer.wo, log, ws=[l.wo for l in L]),
        "gu": pick_normed("gateup", x, y, layer.post_norm, layer.wgu, cfg.eps, log, ws=[l.wgu for l in L]),
        "d": pick_matmul(act, layer.wd, log, ws=[l.wd for l in L]),
        "lm": pick_normed("matmul", x, y, m.final_norm, m.lm_head, cfg.eps, log),
    }


def run_layers(plan: "Plan", mm: dict, x: torch.Tensor, q_buf: torch.Tensor, attn_out: torch.Tensor,
               attention, pos: torch.Tensor, R: int, rope_fused: bool, depth: torch.Tensor | None = None) -> torch.Tensor:
    """Decode/verify layer stack over ``M = B*R`` rows; returns the logits.

    The residual stream lives in two buffers: every fused norm-prologue GEMM
    reads one residual and writes the next, and a kernel must never write the
    buffer its other programs are still reading. QKV reads ``cur`` (the
    embedding, then always ``xb``) and writes ``xa``; gate/up reads ``xa`` and
    writes ``xb``; the LM head reads ``xb`` and writes ``xa``.
    """
    m, cfg = plan.model, plan.model.cfg
    B = plan.B
    HQ, D = cfg.heads, cfg.head_dim
    zero = plan.zero_rows.get(x.shape[0])
    if zero is None:  # allocated once per row count, outside any captured graph's per-step work
        zero = plan.zero_rows[x.shape[0]] = torch.zeros_like(x)
    xa, xb = torch.empty_like(x), torch.empty_like(x)
    cur, y = x, zero
    for i, layer in enumerate(m.layers):
        k_i, v_i = plan.k_layers[i], plan.v_layers[i]
        qkv = mm["qkv"](cur, y, layer.in_norm, xa, layer.wqkv)
        qk_norm_rope_cache(qkv, layer.q_norm, layer.k_norm, m.cos, m.sin, pos,
                           q_buf, k_i, v_i, R, cfg.eps, fused=rope_fused, depth=depth)
        attention(q_buf, k_i, v_i, pos, attn_out)
        o = mm["o"](attn_out.view(B * R, HQ * D), layer.wo)
        act = mm["gu"](xa, o, layer.post_norm, xb, layer.wgu)
        y = mm["d"](act, layer.wd)
        cur = xb
    return mm["lm"](cur, y, m.final_norm, xa, m.lm_head)


class Plan:
    """Static buffers, KV cache and step functions for one (B, T, max_new) shape."""

    def __init__(self, model: Model, B: int, T: int, max_new: int):
        self.model = model
        cfg = model.cfg
        self.B, self.T, self.max_new = B, T, max_new
        # Capacity covers the prompt, every output token, and a full draft block
        # of up to 64 rows beyond the last committed slot, plus padding so a
        # sample asking for a few more tokens than the warmup still fits.
        self.cap = ((T + max(max_new, 256) + 2 * 64 + 63) // 64) * 64
        model.ensure_rope(self.cap)
        dev = model.device
        bf16 = torch.bfloat16
        HQ, HKV, D = cfg.heads, cfg.kv_heads, cfg.head_dim
        self.ids = torch.zeros((B, T), dtype=torch.int64, device=dev)
        self.tok = torch.zeros((B,), dtype=torch.int64, device=dev)
        self.pos = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.k_cache = torch.zeros((cfg.layers, B, HKV, self.cap, D), dtype=bf16, device=dev)
        self.v_cache = torch.zeros((cfg.layers, B, HKV, self.cap, D), dtype=bf16, device=dev)
        self.k_layers = [self.k_cache[i] for i in range(cfg.layers)]
        self.v_layers = [self.v_cache[i] for i in range(cfg.layers)]
        self.zero_rows: dict[int, torch.Tensor] = {}
        self.q_prefill = torch.empty((B, HQ, T, D), dtype=bf16, device=dev)
        self.q_decode = torch.empty((B, HQ, 1, D), dtype=bf16, device=dev)
        self.attn_decode = torch.empty((B, 1, HQ, D), dtype=bf16, device=dev)
        log = lambda s: print(f"[engine] {s}", file=sys.stderr, flush=True)
        self.rope_fused = os.environ.get("ENGINE_ROPE_FUSED", "1") == "1"
        backend = os.environ.get("ENGINE_PREFILL_SDPA", "cudnn")
        self.sdpa_backends = {
            "flash": [SDPBackend.FLASH_ATTENTION],
            "cudnn": [SDPBackend.CUDNN_ATTENTION],
            "efficient": [SDPBackend.EFFICIENT_ATTENTION],
        }.get(backend)
        if self.sdpa_backends is not None:
            try:
                self._prefill_attention(self.q_prefill, self.k_cache[0, :, :, :T], self.v_cache[0, :, :, :T])
                torch.cuda.synchronize()
            except RuntimeError as exc:
                log(f"prefill SDPA backend {backend!r} unavailable here ({str(exc).splitlines()[0]}); using the default")
                self.sdpa_backends = None
        if os.environ.get("ENGINE_ATTN_DEFAULT") == "1":
            self.attention = DecodeAttention(B, HQ, HKV, D, self.cap, dev)
        else:
                self.attention = pick_attention(B, HQ, HKV, D, self.cap, T + max_new // 2, dev, log, maxlen=T + max_new + 64)
        self.mm = self._pick_decode_matmuls()
        self.recycler = None

    def _pick_decode_matmuls(self) -> dict[str, object]:
        """Time cuBLAS against the Triton skinny GEMM for every decode shape."""
        m, cfg, B = self.model, self.model.cfg, self.B
        layer = m.layers[0]
        x = torch.randn((B, cfg.hidden), dtype=torch.bfloat16, device=m.device)
        a = torch.randn((B, cfg.heads * cfg.head_dim), dtype=torch.bfloat16, device=m.device)
        act = torch.randn((B, cfg.intermediate), dtype=torch.bfloat16, device=m.device)
        log = lambda s: print(f"[engine] {s}", file=sys.stderr, flush=True)
        return decode_matmuls(m, B, log)

    def _prefill_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """The reference's own SDPA call (flash, causal) unless ENGINE_PREFILL_SDPA pins a backend."""
        scale = self.model.scale
        if self.sdpa_backends is None:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale, enable_gqa=True)
        with sdpa_kernel(self.sdpa_backends):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale, enable_gqa=True)

    def _next_norm(self, i: int) -> torch.Tensor:
        layers = self.model.layers
        return layers[i + 1].in_norm if i + 1 < len(layers) else self.model.final_norm

    @torch.inference_mode()
    def prefill(self) -> torch.Tensor:
        """Consume ``self.ids`` from position 0; returns [B, V] logits at the last position."""
        m, cfg = self.model, self.model.cfg
        B, T = self.B, self.T
        HQ, D = cfg.heads, cfg.head_dim
        self.pos.zero_()
        x = F.embedding(self.ids.view(-1), m.embed)
        h = rms_norm(x, m.layers[0].in_norm, cfg.eps)
        for i, layer in enumerate(m.layers):
            k_i, v_i = self.k_layers[i], self.v_layers[i]
            qkv = h @ layer.wqkv.t()
            qk_norm_rope_cache(qkv, layer.q_norm, layer.k_norm, m.cos, m.sin, self.pos,
                               self.q_prefill, k_i, v_i, T, cfg.eps, fused=self.rope_fused)
            a = self._prefill_attention(self.q_prefill, k_i[:, :, :T], v_i[:, :, :T])
            o = a.transpose(1, 2).reshape(B * T, HQ * D) @ layer.wo.t()
            h2 = add_rms_norm(x, o, layer.post_norm, cfg.eps)
            d = swiglu(h2 @ layer.wgu.t()) @ layer.wd.t()
            h = add_rms_norm(x, d, self._next_norm(i), cfg.eps)
        logits = h.view(B, T, cfg.hidden)[:, -1] @ m.lm_head.t()
        self.pos.fill_(T)
        return logits

    @torch.inference_mode()
    def decode(self) -> torch.Tensor:
        """Consume ``self.tok`` at ``self.pos``; returns [B, V] logits; advances ``pos``."""
        m, cfg = self.model, self.model.cfg
        B = self.B
        HQ, D = cfg.heads, cfg.head_dim
        x = F.embedding(self.tok, m.embed)
        logits = run_layers(self, self.mm, x, self.q_decode, self.attn_decode, self.attention, self.pos, 1, self.rope_fused)
        self.pos.add_(1)
        return logits


class VerifyPlan:
    """Speculative verification: R query rows per sequence in one pass.

    Row 0 holds the last accepted token, rows 1..R-1 hold drafts, either a
    chain (block-causal mask) or a tree (ancestor masks). The forward writes
    all R positions into the cache and returns the greedy token after every
    row; the host accepts the longest path the model itself predicts, exactly
    as plain greedy decode would have produced it.
    """

    def __init__(self, plan: Plan, R: int, tree: bool = False, recycler=None):
        self.plan, self.R, self.tree = plan, R, tree
        self.K = R - 1
        self.recycler = recycler
        m, cfg, B = plan.model, plan.model.cfg, plan.B
        dev, bf16 = m.device, torch.bfloat16
        HQ, HKV, D = cfg.heads, cfg.kv_heads, cfg.head_dim
        self.blk = recycler.blk if recycler is not None else torch.zeros((B, R), dtype=torch.int64, device=dev)
        self.masks = recycler.masks if recycler is not None else None
        self.pos = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.q = torch.empty((B, HQ, R, D), dtype=bf16, device=dev)
        self.attn_out = torch.empty((B, R, HQ, D), dtype=bf16, device=dev)
        log = lambda s: print(f"[engine] {s}", file=sys.stderr, flush=True)
        self.attention = pick_attention(B, HQ, HKV, D, plan.cap, plan.T + plan.max_new // 2, dev, log, R=R, tree=tree,
                                        maxlen=plan.T + plan.max_new + R + 64)
        self.mm = decode_matmuls(m, B * R, log)

    @torch.inference_mode()
    def verify(self) -> torch.Tensor:
        """Consume ``blk`` at ``pos``; returns [B, R] greedy tokens, one per row,
        and (in recycling mode) records every row's top-k in the adjacency table."""
        plan = self.plan
        B, R = plan.B, self.R
        x = F.embedding(self.blk.view(-1), plan.model.embed)
        attention = (lambda q, k, v, pos, out: self.attention(q, k, v, pos, out, self.masks)) if self.tree else self.attention
        depth = self.recycler.depth if self.recycler is not None else None
        logits = run_layers(plan, self.mm, x, self.q, self.attn_out, attention, self.pos, R, plan.rope_fused, depth=depth)
        if self.recycler is not None:
            self.recycler.update(self.blk.view(-1), logits)
        return logits.argmax(dim=-1).view(B, R)
