"""Exact dense prefill GEMM with a bounded static persistent schedule.

X[M,K], original W[N,K] and caller-owned OUT[M,N] are contiguous BF16 CUDA
tensors without aliasing. Each program owns whole output tiles, accumulates
all K in FP32, then stores BF16 once. No workspace, weight copy or split-K.
Selection completes before prefill capture; ordinary torch.mm is permanent
fallback and the numerical/timing reference. Decode is unchanged.
"""

import math
import statistics
import sys
import time

import torch
import triton
import triton.language as tl
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources

from wide_gemv import WideGemvLayout


@triton.jit
def _persistent_dense(X, W, OUT, M: tl.constexpr, N: tl.constexpr,
                      K: tl.constexpr, SMS: tl.constexpr,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                      GROUP: tl.constexpr):
    mt, nt = tl.cdiv(M, BM), tl.cdiv(N, BN)
    kk = tl.arange(0, BK)
    for tile in range(tl.program_id(0), mt * nt, SMS):
        # Grouped ordering reuses a weight tile across several input tiles.
        # Use the group-relative tile id so the final short group is exact.
        group_width = GROUP * nt
        first_m = (tile // group_width) * GROUP
        group_m = tl.minimum(mt - first_m, GROUP)
        local = tile % group_width
        im = first_m + local % group_m
        jn = local // group_m
        mm = im * BM + tl.arange(0, BM)
        nn = jn * BN + tl.arange(0, BN)
        acc = tl.full((BM, BN), 0, tl.float32)
        for block in range(tl.cdiv(K, BK)):
            k = block * BK + kk
            x = tl.load(X + mm[:, None] * K + k[None, :],
                        (mm[:, None] < M) & (k[None, :] < K), 0)
            w = tl.load(W + nn[None, :] * K + k[:, None],
                        (nn[None, :] < N) & (k[:, None] < K), 0)
            acc = tl.dot(x, w, acc)
        tl.store(OUT + mm[:, None] * N + nn[None, :], acc,
                 (mm[:, None] < M) & (nn[None, :] < N))


class DensePlan:
    # Triton 3.1's persistent test records an eight-warp issue. Both choices
    # retain four warps, with equal 16,384-element FP32 accumulator tiles.
    CONFIGS = ((128, 128, 64, 4), (64, 256, 64, 4))
    SHAPES = ((19456, 2560), (2560, 9728), (6144, 2560), (2560, 4096))

    def __init__(self, x, weight, output, config, sms):
        self.rows, self.k = x.shape
        self.n = weight.shape[0]
        if (config not in self.CONFIGS or not isinstance(sms, int) or sms <= 0
                or self.rows < 256 or (self.n, self.k) not in self.SHAPES
                or not self.eligible(x, weight, output)):
            raise ValueError("unsupported dense prefill shape, layout or tile")
        self.config, self.sms = config, sms
        self.name = f"persistent_m{config[0]}_n{config[1]}_k{config[2]}_p{config[3]}"

    def eligible(self, x, weight, output):
        return (tuple(x.shape) == (self.rows, self.k)
                and tuple(weight.shape) == (self.n, self.k)
                and tuple(output.shape) == (self.rows, self.n)
                and all(t.is_cuda and t.dtype == torch.bfloat16
                        and t.device == x.device and t.is_contiguous()
                        for t in (x, weight, output)))

    def __call__(self, x, weight, output):
        bm, bn, bk, stages = self.config
        count = triton.cdiv(self.rows, bm) * triton.cdiv(self.n, bn)
        _persistent_dense[(min(self.sms, count),)](
            x, weight, output, M=self.rows, N=self.n, K=self.k,
            SMS=self.sms, BM=bm, BN=bn, BK=bk, GROUP=8,
            num_warps=4, num_stages=stages,
        )


class _NativePrefill:
    @staticmethod
    def run(name, layer, x, weight, output):
        torch.mm(x, weight.t(), out=output)


class DensePrefill(WideGemvLayout):
    """Reuse hardened memory, numerical and graph helpers from the base."""
    MAX_POOL_FLOPS = 8 * 10**12

    def __init__(self, engine, deadline):
        self.rows, self.device = engine.prefill_rows, engine.prefill_normalized.device
        self.plans, self.native = {}, _NativePrefill()
        deadline = min(deadline, time.monotonic() + 45.0)
        if self.rows < 256 or time.monotonic() >= deadline:
            return
        if torch.cuda.get_device_capability(self.device)[0] != 9:
            return
        self.sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        # Span all layers with six complete real matrices; each pool exceeds
        # H100 L2. Use actual M, never a shortened proxy for the workload.
        indices = ([round(i * (len(engine.layers) - 1) / 5) for i in range(6)]
                   if len(engine.layers) >= 6 else list(range(len(engine.layers))))
        groups = (
            ("gateup", engine.prefill_normalized, engine.prefill_gateup,
             [engine.packed[i][1] for i in indices]),
            ("down", engine.prefill_intermediate, engine.prefill_branch,
             [engine.layers[i].mlp.down_proj.weight for i in indices]),
            ("qkv", engine.prefill_normalized, engine.prefill_qkv,
             [engine.packed[i][0] for i in indices]),
            ("output", engine.prefill_query, engine.prefill_branch,
             [engine.layers[i].self_attn.o_proj.weight for i in indices]),
        )
        generator = torch.Generator(device=self.device)
        generator.manual_seed(91579)
        for name, x, output, weights in groups:
            if time.monotonic() >= deadline:
                break
            winner = self._select(name, x, weights, output, generator, deadline)
            # _select returns only fully checked, timed, and drained plans.
            if winner is not None:
                self.plans[name] = winner
            self._log(name, "selected " + (winner.name if name in self.plans else "native"))
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()

    def _log(self, name, message):
        print(f"[dense-prefill] M={self.rows} {name}: {message}",
              file=sys.stderr, flush=True)

    def _select(self, name, x, weights, output, generator, deadline):
        try:
            if (time.monotonic() >= deadline or not weights
                    or 2 * self.rows * weights[0].numel() * len(weights) > self.MAX_POOL_FLOPS):
                return None
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            if time.monotonic() >= deadline or not self._room(output):
                return None
            if time.monotonic() >= deadline:
                return None
            try:
                reference = torch.empty_like(output)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                return None
            if time.monotonic() >= deadline:
                return None
            x.normal_(generator=generator)
            native_graph = self._graph(name, None, x, weights, output, deadline)
            if native_graph is None or time.monotonic() >= deadline:
                return None
            winner, winner_ms = None, float("inf")
            for config in DensePlan.CONFIGS:
                if time.monotonic() >= deadline:
                    break
                plan = DensePlan(x, weights[0], output, config, self.sms)
                try:
                    # Three independent full-row probes, scales 1, 0.1 and 10,
                    # use the actual torch.mm output and both pool endpoints.
                    if not self._check(name, plan, x, weights, output, reference, generator, deadline):
                        self._log(name, f"{plan.name}: numerical check rejected")
                        continue
                    if time.monotonic() >= deadline:
                        break
                    graph = self._graph(name, plan, x, weights, output, deadline)
                    if graph is None or time.monotonic() >= deadline:
                        break
                except (CompilationError, OutOfResources) as error:
                    self._log(name, f"{plan.name}: {type(error).__name__}")
                    torch.cuda.synchronize(self.device)
                    continue
                natives, customs = [], []
                for timed, values in ((native_graph, natives), (graph, customs),
                                       (graph, customs), (native_graph, natives)):
                    value = self._time(timed, len(weights), deadline)
                    if value is None:
                        break
                    values.append(value)
                del graph
                if len(natives) != 2 or len(customs) != 2:
                    break
                if time.monotonic() >= deadline:
                    break
                old, new = min(natives), max(customs)
                self._log(name, f"{plan.name}: native {old * 1000:.2f} us, custom {new * 1000:.2f} us")
                if new < old * 0.95 and new < winner_ms:
                    winner, winner_ms = plan, new
            # A later incomplete trial must not erase an earlier valid winner.
            return winner
        finally:
            # Drain before local references/workspaces leave scope.
            # CUDA execution or drain errors must still propagate.
            torch.cuda.synchronize(self.device)

    @staticmethod
    def _time(graph, count, deadline):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        if time.monotonic() >= deadline:
            return None
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        elapsed = start.elapsed_time(end)
        if time.monotonic() >= deadline or not math.isfinite(elapsed) or elapsed <= 0:
            return None
        # Large GEMMs need few graph replays. Target at most ~8ms per sample,
        # with a hard cap of four replays and three medians per ABBA arm.
        repeats = max(1, min(4, int(8.0 / max(elapsed, 0.001))))
        samples = []
        for _ in range(3):
            if time.monotonic() >= deadline:
                return None
            start.record()
            for _ in range(repeats):
                if time.monotonic() >= deadline:
                    return None
                graph.replay()
            end.record()
            end.synchronize()
            elapsed = start.elapsed_time(end)
            if time.monotonic() >= deadline or not math.isfinite(elapsed) or elapsed <= 0:
                return None
            samples.append(elapsed / (repeats * count))
        return statistics.median(samples)

    def run(self, name, layer, x, weight, output):
        plan = self.plans.get(name)
        if plan is not None and plan.eligible(x, weight, output):
            plan(x, weight, output)
        else:
            self.native.run(name, layer, x, weight, output)
