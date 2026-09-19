"""Full-prefill BF16 gate/up GEMM with the existing SwiGLU epilogue.

X [M,H], W [2*I,H] and OUT [M,I] are contiguous BF16 CUDA tensors.
W retains the packed gate-then-up representation used by the native path.
No input aliases OUT; the caller owns every buffer. The fallback also uses
caller-owned GU [M,2*I]. Selection happens once per shape before graph capture.
"""

import statistics
import sys
import time

import torch
import triton
import triton.language as tl
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources

from custom_kernels import swiglu_kernel


@triton.jit
def _gateup_swiglu(
    X, W, OUT, M: tl.constexpr, H: tl.constexpr, I: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP_M: tl.constexpr = 8,
):
    # BN counts both gate and up. Adjacent logical columns form one pair;
    # each pair maps to two separated rows of the unchanged packed weight.
    pid = tl.program_id(0)
    tiles_m = tl.cdiv(M, BM)
    tiles_n = tl.cdiv(I, BN // 2)
    group_size = GROUP_M * tiles_n
    group = pid // group_size
    first_m = group * GROUP_M
    group_m = tl.minimum(tiles_m - first_m, GROUP_M)
    tile_m = first_m + (pid % group_size) % group_m
    tile_n = (pid % group_size) // group_m
    rows = tile_m * BM + tl.arange(0, BM)
    lanes = tl.arange(0, BN)
    channels = tile_n * (BN // 2) + lanes // 2
    weight_rows = channels + (lanes % 2) * I
    reduction = tl.arange(0, BK)
    xp = X + rows[:, None] * H + reduction[None, :]
    wp = W + reduction[:, None] + weight_rows[None, :] * H
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(H, BK)):
        k_valid = block * BK + reduction < H
        x = tl.load(xp, (rows[:, None] < M) & k_valid[None, :], 0)
        w = tl.load(wp, k_valid[:, None] & (channels[None, :] < I), 0)
        acc = tl.dot(x, w, acc)
        xp += BK
        wp += BK
    # Native mm stores BF16 gate/up before the pointwise kernel consumes it.
    pairs = tl.reshape(acc.to(tl.bfloat16), (BM, BN // 2, 2))
    gate, up = tl.split(pairs)
    gate = gate.to(tl.float32)
    up = up.to(tl.float32)
    # Preserve the separate BF16 SiLU result before multiplying by BF16 up.
    act = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    out_channels = tile_n * (BN // 2) + tl.arange(0, BN // 2)
    tl.store(OUT + rows[:, None] * I + out_channels[None, :], act * up,
             (rows[:, None] < M) & (out_channels[None, :] < I))


class PrefillSwiGLU:
    # Two fixed tiles only. No persistent weight copies or online autotuning.
    CONFIGS = ((64, 128, 64, 4, 4), (128, 256, 64, 8, 3))

    def __init__(self, engine, deadline):
        self.config = None
        self.rows = engine.prefill_rows
        deadline = min(deadline, time.monotonic() + 25.0)
        if time.monotonic() >= deadline:
            self._log("native: warmup tuning deadline already reached")
            return
        x = engine.prefill_normalized
        gu = engine.prefill_gateup
        out = engine.prefill_intermediate
        weights = [pair[1] for pair in engine.packed[:6]]
        # These six complete real matrices exceed H100 L2 by a wide margin.
        # Use the full workload M and both output stages in the native timing.
        generator = torch.Generator(device=x.device)
        generator.manual_seed(28571)
        torch.cuda.synchronize(x.device)
        free, total = torch.cuda.mem_get_info(x.device)
        # Reference plus comparison temporaries must fit while preserving the
        # same quarter-device free-space margin used by the layout tuner.
        if free - 4 * out.numel() * out.element_size() < total // 4:
            self._log("native: insufficient room for the numerical probe")
            return
        try:
            reference = torch.empty_like(out)
        except torch.cuda.OutOfMemoryError:
            self._log("native: numerical probe allocation exhausted memory")
            return
        x.normal_(generator=generator)
        native_graph = self._graph(x, weights, gu, out, None)
        best_ms = float("inf")
        for config in self.CONFIGS:
            if time.monotonic() >= deadline:
                break
            try:
                self._fused(x, weights[0], out, config)
                torch.cuda.synchronize(x.device)
                if time.monotonic() >= deadline:
                    self._log(f"tile {config}: compilation used remaining tuning time")
                    break
                valid = True
                for index in (0, len(weights) - 1):
                    if time.monotonic() >= deadline:
                        self._log(f"tile {config}: numerical probe deadline reached")
                        valid = False
                        break
                    x.normal_(generator=generator)
                    self._native(x, weights[index], gu, reference)
                    self._fused(x, weights[index], out, config)
                    if not torch.allclose(out, reference, rtol=0.02, atol=0.01):
                        valid = False
                        break
                if not valid:
                    self._log(f"tile {config}: numerical probe rejected")
                    continue
                if time.monotonic() >= deadline:
                    break
                candidate_graph = self._graph(x, weights, gu, out, config)
            except (CompilationError, OutOfResources) as error:
                self._log(f"tile {config}: unavailable ({type(error).__name__})")
                continue
            # Each timing is the median of three four-replay samples. ABBA
            # requires both fused medians to beat both native medians by >5%.
            native, fused = [], []
            for graph, samples in ((native_graph, native), (candidate_graph, fused),
                                   (candidate_graph, fused), (native_graph, native)):
                if time.monotonic() >= deadline:
                    break
                samples.append(self._time(graph, len(weights)))
            if len(native) != 2 or len(fused) != 2:
                self._log(f"tile {config}: timing deadline reached")
                del candidate_graph
                break
            native_ms, fused_ms = min(native), max(fused)
            accepted = fused_ms < 0.95 * native_ms
            if accepted and fused_ms < best_ms:
                self.config = config
                best_ms = fused_ms
            self._log(f"tile {config}: native {native_ms * 1000:.1f} us, "
                      f"fused {fused_ms * 1000:.1f} us; "
                      f"{'eligible' if accepted else 'native retained'}")
            del candidate_graph
        del reference, native_graph
        torch.cuda.synchronize(x.device)
        torch.cuda.empty_cache()
        self._log(f"selected {self.config if self.config is not None else 'native'}")

    def _log(self, message):
        print(f"[prefill-swiglu] M={self.rows} {message}", file=sys.stderr, flush=True)

    @staticmethod
    def _native(x, weight, gu, out):
        torch.mm(x, weight.t(), out=gu)
        width = out.shape[1]
        total = out.numel()
        swiglu_kernel[(triton.cdiv(total, 1024),)](
            gu, out, width, total, num_warps=4, enable_fp_fusion=False,
        )

    @staticmethod
    def _fused(x, weight, out, config):
        bm, bn, bk, warps, stages = config
        rows, hidden = x.shape
        width = out.shape[1]
        _gateup_swiglu[(triton.cdiv(rows, bm) * triton.cdiv(width, bn // 2),)](
            x, weight, out, rows, hidden, width, bm, bn, bk,
            num_warps=warps, num_stages=stages, enable_fp_fusion=False,
        )

    @classmethod
    def _graph(cls, x, weights, gu, out, config):
        def run():
            for weight in weights:
                if config is None:
                    cls._native(x, weight, gu, out)
                else:
                    cls._fused(x, weight, out, config)
        current = torch.cuda.current_stream(x.device)
        stream = torch.cuda.Stream(device=x.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            run()
        current.wait_stream(stream)
        torch.cuda.synchronize(x.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run()
        current.wait_stream(stream)
        graph.replay()
        torch.cuda.synchronize(x.device)
        return graph

    @staticmethod
    def _time(graph, count):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        samples = []
        for _ in range(3):
            start.record()
            for _ in range(4):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / (4 * count))
        return statistics.median(samples)

    def run(self, x, weight, gu, out):
        if self.config is None:
            self._native(x, weight, gu, out)
        else:
            self._fused(x, weight, out, self.config)
