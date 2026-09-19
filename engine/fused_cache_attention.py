"""Bounded warmup choice of an exact fused decode norm/cache/attention path.

Contiguous BF16 QKV is [B,6144], cache is [B,8,CAP,128], output [B,4096].
One CTA owns each (batch, KV head), including its unique current cache slot.
Past loads exclude the current slot before loading: the current K/V is passed
from registers, so there is no inter-CTA producer/consumer dependency.
"""
import statistics
import sys
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources

from custom_kernels import (qkv_rope_cache_kernel, attention_split_kernel,
                            attention_merge_kernel)


@triton.jit
def fused_cache_attention_kernel(QKV, QW, KW, COS, SIN, POS, K, V, O,
                                  CAP: tl.constexpr, EPS: tl.constexpr,
                                  SCALE: tl.constexpr,
                                  BLOCK_N: tl.constexpr = 256,
                                  D: tl.constexpr = 128):
    b = tl.program_id(0)
    kh = tl.program_id(1)
    qh = tl.arange(0, 16)
    d = tl.arange(0, D)
    rd = (d + D // 2) % D
    pos = tl.load(POS)
    c = tl.load(COS + pos * D + d).to(tl.float32)
    s = tl.load(SIN + pos * D + d).to(tl.float32)

    # Every eager BF16 boundary in the accepted qkv_rope_cache_kernel remains.
    qbase = b * 6144 + (kh * 4 + qh[:, None]) * D
    qx = tl.load(QKV + qbase + d[None, :], qh[:, None] < 4, 0).to(tl.float32)
    qr = tl.load(QKV + qbase + rd[None, :], qh[:, None] < 4, 0).to(tl.float32)
    qw = tl.load(QW + d).to(tl.float32)
    qrw = tl.load(QW + rd).to(tl.float32)
    qi = tl.rsqrt(tl.sum(qx * qx, 1) / D + EPS)
    qx = ((qx * qi[:, None]).to(tl.bfloat16).to(tl.float32) * qw[None, :]).to(tl.bfloat16).to(tl.float32)
    qr = ((qr * qi[:, None]).to(tl.bfloat16).to(tl.float32) * qrw[None, :]).to(tl.bfloat16).to(tl.float32)
    qr = tl.where(d[None, :] < D // 2, -qr, qr)
    qa = (qx * c[None, :]).to(tl.bfloat16).to(tl.float32)
    qz = (qr * s[None, :]).to(tl.bfloat16).to(tl.float32)
    q = (qa + qz).to(tl.bfloat16)

    kbase = b * 6144 + 4096 + kh * D
    kx = tl.load(QKV + kbase + d).to(tl.float32)
    kr = tl.load(QKV + kbase + rd).to(tl.float32)
    kw = tl.load(KW + d).to(tl.float32)
    krw = tl.load(KW + rd).to(tl.float32)
    ki = tl.rsqrt(tl.sum(kx * kx, 0) / D + EPS)
    kx = ((kx * ki).to(tl.bfloat16).to(tl.float32) * kw).to(tl.bfloat16).to(tl.float32)
    kr = ((kr * ki).to(tl.bfloat16).to(tl.float32) * krw).to(tl.bfloat16).to(tl.float32)
    kr = tl.where(d < D // 2, -kr, kr)
    ka = (kx * c).to(tl.bfloat16).to(tl.float32)
    kz = (kr * s).to(tl.bfloat16).to(tl.float32)
    current_k = (ka + kz).to(tl.bfloat16)
    current_v = tl.load(QKV + b * 6144 + 5120 + kh * D + d)
    current_offset = ((b * 8 + kh) * CAP + pos) * D + d
    tl.store(K + current_offset, current_k)
    tl.store(V + current_offset, current_v)

    offsets = tl.arange(0, BLOCK_N)
    maximum = tl.full((16,), -1.0e30, tl.float32)
    denom = tl.zeros((16,), tl.float32)
    numerator = tl.zeros((16, D), tl.float32)
    for start in range(0, tl.minimum(pos + 1, CAP), BLOCK_N):
        t = start + offsets
        history = (t < CAP) & (t < pos)
        valid = (t < CAP) & (t <= pos)
        past_k = tl.load(K + ((b * 8 + kh) * CAP + t[None, :]) * D + d[:, None],
                         history[None, :], 0)
        k = tl.where(t[None, :] == pos, current_k[:, None], past_k)
        score = tl.dot(q, k).to(tl.float32) * SCALE
        score = tl.where(valid[None, :], score, float('-inf'))
        local_max = tl.maximum(tl.max(score, 1), -1.0e30)
        p = tl.exp(score - local_max[:, None])
        local_sum = tl.sum(p, 1)
        past_v = tl.load(V + ((b * 8 + kh) * CAP + t[:, None]) * D + d[None, :],
                         history[:, None], 0)
        v = tl.where(t[:, None] == pos, current_v[None, :], past_v)
        # Match each accepted 256-token split's probability rounding exactly.
        local_value = tl.dot(p.to(tl.bfloat16), v).to(tl.float32)
        next_max = tl.maximum(maximum, local_max)
        alpha = tl.exp(maximum - next_max)
        beta = tl.exp(local_max - next_max)
        numerator = numerator * alpha[:, None] + local_value * beta[:, None]
        denom = denom * alpha + local_sum * beta
        maximum = next_max
    h = b * 32 + kh * 4 + qh
    tl.store(O + h[:, None] * D + d[None, :],
             numerator / denom[:, None], qh[:, None] < 4)


class FusedCacheAttention:
    def __init__(self, engine, deadline):
        self.engine = engine
        self.enabled = False
        deadline = min(deadline, time.monotonic() + 12.0)
        # One additional specialization; small batches retain split parallelism.
        if not 8 <= engine.batch <= 32 or engine.capacity > 8192:
            self._log('baseline: outside bounded fusion search')
            return
        # Rotating every real layer must stream well beyond H100 L2 even at
        # the shortest timed prefix. No repeated hot single-layer benchmark.
        pool_bytes = len(engine.layers) * 2 * engine.batch * 8 * (engine.prompt + 1) * 128 * 2
        if pool_bytes < 128 * 1024**2:
            self._log('baseline: active cache pool below 128 MiB')
            return
        if time.monotonic() >= deadline:
            self._log('baseline: warmup deadline')
            return
        free, total = torch.cuda.mem_get_info()
        # B=2 numerical scratch uses at most 128 MiB at CAP=8192. Include
        # workspace/graph slack and retain a stricter guard than the 90% gate.
        if total - free + 160 * 1024**2 >= total * 0.85:
            self._log('baseline: insufficient tuning memory headroom')
            return
        try:
            self._select(deadline)
        except (CompilationError, OutOfResources, torch.cuda.OutOfMemoryError) as exc:
            self.enabled = False
            self._log(f'baseline: {type(exc).__name__}: {exc}')
        finally:
            engine.position.fill_(engine.prompt)

    def _log(self, message):
        e = self.engine
        print(f'[fused-cache-attention] B={e.batch} S={e.prompt} cap={e.capacity}: {message}',
              file=sys.stderr, flush=True)

    def _buffers(self, idx, scratch):
        e = self.engine
        if scratch is None:
            return e, e.keys[idx], e.values[idx]
        return scratch, scratch.key, scratch.value

    def _baseline(self, idx, output, scratch=None):
        e = self.engine
        s, key, value = self._buffers(idx, scratch)
        a = e.layers[idx].self_attn
        qkv_rope_cache_kernel[(s.batch, 40)](
            s.qkv, a.q_norm.weight, a.k_norm.weight,
            e.cos, e.sin, s.position, s.query, key, value,
            e.capacity, e.eps, num_warps=4, enable_fp_fusion=False,
        )
        attention_split_kernel[(s.batch, 8, e.splits)](
            s.query, key, value, s.position, s.partial, s.pmax, s.psum,
            e.capacity, e.splits, 128 ** -0.5, num_warps=4, num_stages=1,
        )
        attention_merge_kernel[(s.batch * 32,)](
            s.partial, s.pmax, s.psum, output, e.splits,
            triton.next_power_of_2(e.splits), num_warps=4,
        )

    def _fused(self, idx, output, scratch=None):
        e = self.engine
        s, key, value = self._buffers(idx, scratch)
        a = e.layers[idx].self_attn
        fused_cache_attention_kernel[(s.batch, 8)](
            s.qkv, a.q_norm.weight, a.k_norm.weight, e.cos, e.sin,
            s.position, key, value, output, e.capacity, e.eps,
            128 ** -0.5, num_warps=4, num_stages=1, enable_fp_fusion=False,
        )

    def run(self, idx):
        if self.enabled:
            self._fused(idx, self.engine.attention)
        else:
            self._baseline(idx, self.engine.attention)

    @staticmethod
    def _unchanged(s, original_key, original_value, position):
        return (torch.equal(s.key[:, :, :position], original_key[:, :, :position])
                and torch.equal(s.value[:, :, :position], original_value[:, :, :position])
                and torch.isnan(s.key[:, :, position+1:]).all().item()
                and torch.isnan(s.value[:, :, position+1:]).all().item())

    def _numerics(self, deadline):
        e = self.engine
        # Private cache: probing earlier positions cannot corrupt the prompt.
        original_key = e.keys[-1][:2].clone()
        original_value = e.values[-1][:2].clone()
        original_key[:, :, e.prompt:].copy_(original_key[:, :, e.prompt-1:e.prompt])
        original_value[:, :, e.prompt:].copy_(original_value[:, :, e.prompt-1:e.prompt])
        s = SimpleNamespace(
            batch=2, key=original_key.clone(), value=original_value.clone(),
            qkv=e.qkv[:2].clone(), query=torch.empty_like(e.query[:2]),
            position=e.position.clone(), partial=torch.empty_like(e.partial[:64]),
            pmax=torch.empty_like(e.pmax[:64]), psum=torch.empty_like(e.psum[:64]),
        )
        original_qkv = s.qkv.clone()
        reference = torch.empty_like(e.attention[:2])
        candidate = torch.empty_like(reference)
        positions = sorted({0, min(255, e.capacity-1), min(256, e.capacity-1),
                            e.prompt, e.capacity-2})
        for scale in (0.25, 1.0, 4.0):
            s.qkv.copy_((original_qkv.float() * scale).to(torch.bfloat16))
            for position in positions:
                if time.monotonic() >= deadline:
                    self._log('baseline: warmup deadline in numerical probes')
                    return False
                s.position.fill_(position)
                s.key.copy_(original_key)
                s.value.copy_(original_value)
                s.key[:, :, position:].fill_(float('nan'))
                s.value[:, :, position:].fill_(float('nan'))
                self._baseline(len(e.layers)-1, reference, s)
                expected_key = s.key[:, :, position].clone()
                expected_value = s.value[:, :, position].clone()
                if not self._unchanged(s, original_key, original_value, position):
                    self._log('baseline: reference cache probe failed')
                    return False
                if time.monotonic() >= deadline:
                    self._log('baseline: warmup deadline before fused numerical probe')
                    return False
                # Poison current again: fused attention must obtain it from
                # registers; future NaNs must never enter either dot product.
                s.key[:, :, position].fill_(float('nan'))
                s.value[:, :, position].fill_(float('nan'))
                self._fused(len(e.layers)-1, candidate, s)
                if (not torch.allclose(candidate, reference, rtol=0.0078125, atol=0.001953125)
                        or not torch.equal(s.key[:, :, position], expected_key)
                        or not torch.equal(s.value[:, :, position], expected_value)
                        or not self._unchanged(s, original_key, original_value, position)):
                    self._log(f'baseline: numerical/cache probe failed pos={position} scale={scale}')
                    return False
        if time.monotonic() >= deadline:
            self._log('baseline: warmup deadline before native probe')
            return False
        # An independent native full-prefix check, including newly written KV.
        native = F.scaled_dot_product_attention(
            s.query[:, :, None, :], s.key[:, :, :position+1],
            s.value[:, :, :position+1], dropout_p=0.0, is_causal=False,
            scale=128 ** -0.5, enable_gqa=True,
        ).reshape(2, 4096)
        if not torch.allclose(candidate, native, rtol=0.03, atol=0.004):
            self._log('baseline: native SDPA probe failed')
            return False
        return True

    def _graph(self, launch, deadline):
        e = self.engine
        current = torch.cuda.current_stream()
        stream = torch.cuda.Stream()
        stream.wait_stream(current)
        try:
            with torch.cuda.stream(stream):
                for idx in range(len(e.layers)):
                    if time.monotonic() >= deadline:
                        break
                    launch(idx, e.attention)
            current.wait_stream(stream)
            torch.cuda.synchronize()
            if time.monotonic() >= deadline:
                return None
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for idx in range(len(e.layers)):
                    launch(idx, e.attention)
            if time.monotonic() >= deadline:
                return None
            return graph
        finally:
            # Even a caught compile/resource failure must order the constructor
            # position reset after all already-enqueued side-stream work.
            current.wait_stream(stream)

    @staticmethod
    def _time(graph, deadline):
        samples = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(3):
            if time.monotonic() >= deadline:
                return None
            start.record()
            for _ in range(8):
                graph.replay()
            end.record()
            end.synchronize()
            if time.monotonic() >= deadline:
                return None
            samples.append(start.elapsed_time(end) / 8)
        return statistics.median(samples)

    def _select(self, deadline):
        e = self.engine
        e.qkv.copy_(e.prefill_qkv.view(e.batch, e.prompt, 6144)[:, -1])
        if not self._numerics(deadline):
            return
        if time.monotonic() >= deadline:
            self._log('baseline: warmup deadline after numerical probes')
            return
        # Timing touches only future slots, preserving every prompt slot. Every
        # real decode overwrites its own slot before any later step can use it.
        for key, value in zip(e.keys, e.values):
            key[:, :, e.prompt:].copy_(key[:, :, e.prompt-1:e.prompt])
            value[:, :, e.prompt:].copy_(value[:, :, e.prompt-1:e.prompt])
        e.position.fill_(e.prompt)
        baseline = self._graph(self._baseline, deadline)
        if baseline is None:
            self._log('baseline: warmup deadline before baseline capture completed')
            return
        fused = self._graph(self._fused, deadline)
        if fused is None:
            self._log('baseline: warmup deadline before fused capture completed')
            return
        timings = []
        for position in sorted({e.prompt, e.capacity-2}):
            if time.monotonic() >= deadline:
                self._log('baseline: warmup deadline before timing completed')
                return
            e.position.fill_(position)
            baseline.replay()
            fused.replay()
            torch.cuda.synchronize()
            arms = []
            for graph in (baseline, fused, fused, baseline):
                elapsed = self._time(graph, deadline)
                if elapsed is None:
                    self._log('baseline: warmup deadline in timing')
                    return
                arms.append(elapsed)
            base_a, fused_a, fused_b, base_b = arms
            timings.append((position+1, min(base_a, base_b), max(fused_a, fused_b)))
        if time.monotonic() >= deadline:
            self._log('baseline: warmup deadline after timing')
            return
        self.enabled = all(fast < base * 0.92 for _, base, fast in timings)
        detail = '; '.join(f'L={length} baseline={base:.4f}ms fused={fast:.4f}ms'
                           for length, base, fast in timings)
        self._log(f'{"fused" if self.enabled else "baseline"}; all-layer full norm/cache/attention graphs: {detail}')
