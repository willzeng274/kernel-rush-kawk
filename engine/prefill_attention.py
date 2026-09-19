"""Select native cuDNN causal GQA only after complete-operation comparison.

Inputs are the engine's real BF16 Q [B,32,S,128] BSHD backing and capacity-
strided K/V [B,8,S,128] prompt views. Both routes return [B*S,4096] BF16;
cuDNN's BHSD-to-BSHD copy is part of the operation and its measured cost.
Selection may overwrite prompt Q/K/V during allocation. Real prefill rewrites
all of them before use; no prompt, decoded token, or partial result survives.
"""

import math
import statistics
import sys
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


class PrefillAttention:
    def __init__(self, engine, deadline):
        self.cudnn = False
        self.device = engine.prefill_query.device
        self.deadline = min(deadline, time.monotonic() + 25.0)
        self.batch, self.prompt = engine.batch, engine.prompt
        if self._expired():
            return
        q = engine.prefill_query_view
        key, value = engine.prefill_kv[0]
        if not self._supported(q, key, value):
            self._log("native: cuDNN capability gate rejected shape/layout")
            return

        # Rotating actual layer prompt views must exceed any H100 L2 size.
        # Tiny shapes unable to reach this with 36 layers retain the baseline.
        pair_bytes = (key.numel() + value.numel()) * key.element_size()
        count = max(2, (64 * 2**20) // pair_bytes + 1)
        if count > len(engine.prefill_kv):
            self._log("native: prompt cache pool too small for fair timing")
            return
        # Dense noncausal QK+PV is an upper bound on useful causal work.
        # Keep even a single timing traversal bounded before native planning.
        pool_flops = 4 * self.batch * 32 * 128 * self.prompt**2 * count
        if pool_flops > 4 * 10**12:
            self._log("native: attention pool exceeds bounded search work")
            return
        pool = engine.prefill_kv[:count]
        query_bytes = q.numel() * q.element_size()
        # Covers full-row comparisons, two graph pools, output conversion,
        # cached cuDNN workspace, and allocator headroom without new KV copies.
        self.reserve = 20 * query_bytes + 256 * 2**20
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        if not self._memory_ok(self.reserve):
            self._log("native: comparison scratch exceeds memory budget")
            return
        generator = torch.Generator(device=self.device)
        generator.manual_seed(51343)
        result = self._compare(q, pool, generator)
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        if (result is not None and not self._expired()
                and self._memory_ok(0)):
            native_ms, cudnn_ms = result
            self.cudnn = (math.isfinite(native_ms) and native_ms > 0
                          and math.isfinite(cudnn_ms) and cudnn_ms > 0
                          and cudnn_ms < 0.95 * native_ms)
            self._log(f"{'cuDNN' if self.cudnn else 'native'}; "
                      f"native {native_ms * 1000:.2f} us, "
                      f"cuDNN {cudnn_ms * 1000:.2f} us; "
                      f"KV pool {count * pair_bytes / 2**20:.1f} MiB")
        else:
            self._log("native: comparison incomplete or rejected")

    def _log(self, message):
        print(f"[prefill-attention] B={self.batch} S={self.prompt}: {message}",
              file=sys.stderr, flush=True)

    def _expired(self):
        return time.monotonic() >= self.deadline

    def _memory_ok(self, extra):
        free, total = torch.cuda.mem_get_info(self.device)
        return free - extra >= total // 4

    @staticmethod
    def _supported(q, key, value):
        capability = getattr(torch._C, "_can_use_cudnn_attention", None)
        if capability is None or not torch.backends.cudnn.is_available():
            return False
        version = torch.backends.cudnn.version()
        if version is None or version < 90000:
            return False
        # Pinned 2.5.1 has this C binding, but no public Python wrapper.
        # Entering this context enables the candidate for its capability test;
        # all prior global backend flags are restored on leaving the context.
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            params = torch.backends.cuda.SDPAParams(
                q, key, value, None, 0.0, True, True)
            return capability(params, False)

    @staticmethod
    def _unsupported_plan(error):
        # Exact pre-execution failures from the frontend revision vendored by
        # Torch 2.5.1. Never turn an execution/CUDA failure into a fallback.
        message = str(error)
        prefix = "cuDNN Frontend error: "
        if not message.startswith(prefix):
            return False
        message = message[len(prefix):]
        return (message in (
            "[cudnn_frontend] Error: No execution plans support the graph.",
            "[cudnn_frontend] Error: No valid execution plans built.",
        ) or message.startswith("No valid engine configs for "))

    @staticmethod
    def _rows(q, key, value, cudnn=False):
        if cudnn:
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                attended = F.scaled_dot_product_attention(
                    q, key, value, dropout_p=0.0, is_causal=True,
                    scale=128 ** -0.5, enable_gqa=True)
        else:
            attended = F.scaled_dot_product_attention(
                q, key, value, dropout_p=0.0, is_causal=True,
                scale=128 ** -0.5, enable_gqa=True)
        return attended.transpose(1, 2).reshape(q.shape[0] * q.shape[2], 4096)

    def run(self, q, key, value):
        return self._rows(q, key, value, self.cudnn)

    def _compare(self, q, pool, generator):
        # Fill only prompt views, preserving their exact capacity strides.
        # There is no valid prompt/cache content at this allocation-time stage.
        for key, value in pool:
            if self._expired():
                return None
            key.normal_(generator=generator)
            value.normal_(generator=generator)
        for index, scale in zip((0, len(pool) // 2, len(pool) - 1),
                                (0.5, 1.0, 2.0)):
            if self._expired() or not self._memory_ok(self.reserve):
                return None
            q.normal_(std=scale, generator=generator)
            key, value = pool[index]
            key.normal_(std=scale, generator=generator)
            value.normal_(std=scale, generator=generator)
            reference = self._rows(q, key, value)
            # The first eager call builds a native cuDNN plan outside capture.
            try:
                alternate = self._rows(q, key, value, True)
            except RuntimeError as error:
                if self._unsupported_plan(error):
                    self._log("native: " + str(error))
                    return None
                raise
            if not torch.allclose(alternate, reference, rtol=0.01, atol=0.005):
                return None
            del reference, alternate
        if self._expired():
            return None
        q.normal_(generator=generator)
        native_graph = self._graph(q, pool, False)
        if native_graph is None or self._expired():
            return None
        cudnn_graph = self._graph(q, pool, True)
        if cudnn_graph is None or self._expired():
            return None
        # Use the slower candidate median against the faster baseline median.
        arms = []
        for graph in (native_graph, cudnn_graph, cudnn_graph, native_graph):
            measured = self._time(graph, len(pool))
            if measured is None:
                return None
            arms.append(measured)
        if self._expired():
            return None
        return min(arms[0], arms[3]), max(arms[1], arms[2])

    def _graph(self, q, pool, cudnn):
        if self._expired() or not self._memory_ok(self.reserve):
            return None
        current = torch.cuda.current_stream(self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for key, value in pool:
                if self._expired():
                    break
                self._rows(q, key, value, cudnn)
        current.wait_stream(stream)
        torch.cuda.synchronize(self.device)
        if self._expired() or not self._memory_ok(self.reserve):
            return None
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for key, value in pool:
                self._rows(q, key, value, cudnn)
        current.wait_stream(stream)
        graph.replay()
        torch.cuda.synchronize(self.device)
        return graph if not self._expired() else None

    def _time(self, graph, count):
        samples = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        if self._expired():
            return None
        # Calibrate one pool traversal before choosing repeats. Attention is
        # quadratic in S, so large hidden shapes must not inherit 8 replays.
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        if self._expired():
            return None
        pool_ms = start.elapsed_time(end)
        if not math.isfinite(pool_ms) or pool_ms <= 0:
            return None
        repeats = max(1, min(8, int(8.0 / pool_ms)))
        for _ in range(3):
            if self._expired():
                return None
            start.record()
            for _ in range(repeats):
                graph.replay()
            end.record()
            end.synchronize()
            if self._expired():
                return None
            samples.append(start.elapsed_time(end) / (repeats * count))
        return statistics.median(samples)
