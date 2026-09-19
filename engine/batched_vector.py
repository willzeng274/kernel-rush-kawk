"""Full-K FP32 vector reductions sharing BF16 weights across 2--4 rows.

Small-batch decode only: B=2 uses two activation rows and two output rows
per CTA. B=3..8 uses four activation rows and one output row, with masked
tails. K=9728 remains native to avoid a large live full-K product. B=1 is
owned by the unchanged WideGemvLayout; B>=9 stays native.

The full product tile has at most 16384 FP32 elements. Each CTA loads a
weight tile once and broadcasts it across activation rows; there is no K
loop, partial-sum workspace, additional weight copy, or fused BF16 boundary.
For two batch tiles, neighboring CTAs consume the same output-weight tile.
"""

import sys
import time

import torch
import triton
import triton.language as tl
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources

from wide_gemv import WideGemvLayout


@triton.jit
def _batched_dot(X, W, OUT, B, N, K: tl.constexpr, SX, SW, SO,
                 BM: tl.constexpr, ROWS: tl.constexpr, WIDTH: tl.constexpr):
    pid = tl.program_id(0)
    batch_tiles = tl.cdiv(B, BM)
    # Batch tile varies fastest: B5..8 reuses a weight tile in adjacent CTAs.
    m = (pid % batch_tiles) * BM + tl.arange(0, BM)
    n = (pid // batch_tiles) * ROWS + tl.arange(0, ROWS)
    k = tl.arange(0, WIDTH)
    x = tl.load(X + m[:, None] * SX + k[None, :],
                (m[:, None] < B) & (k[None, :] < K), 0).to(tl.float32)
    w = tl.load(W + n[:, None] * SW + k[None, :],
                (n[:, None] < N) & (k[None, :] < K), 0).to(tl.float32)
    result = tl.sum(x[:, None, :] * w[None, :, :], axis=2)
    # The only output cast is the projection's usual final BF16 store.
    tl.store(OUT + m[:, None] * SO + n[None, :], result,
             (m[:, None] < B) & (n[None, :] < N))


class BatchedPlan:
    """Nonaliasing BF16 [B,K], [N,K] -> [B,N], unit inner strides.

    Row strides are passed explicitly, allowing padded rows and row slices.
    Only caller-owned output is written; replay allocates no tensors.
    """

    def __init__(self, batch, k, warps):
        if not 2 <= batch <= 8 or k not in (2560, 4096):
            raise ValueError("batched full-K plan is limited to B2..8 and K2560/4096")
        self.batch, self.k, self.warps = batch, k, warps
        self.width = triton.next_power_of_2(k)
        self.bm = 2 if batch == 2 else 4
        self.rows = 16384 // (self.bm * self.width)
        self.name = f"batchK_m{self.bm}_r{self.rows}_k{self.width}_w{warps}"

    def __call__(self, x, weight, output):
        if (x.shape != (self.batch, self.k)
                or weight.shape[1] != self.k
                or output.shape != (self.batch, weight.shape[0])
                or x.stride(1) != 1 or weight.stride(1) != 1
                or output.stride(1) != 1):
            raise ValueError("incompatible shape or inner stride for batched full-K plan")
        grid = (triton.cdiv(weight.shape[0], self.rows)
                * triton.cdiv(self.batch, self.bm),)
        _batched_dot[grid](
            x, weight, output, self.batch, weight.shape[0], self.k,
            x.stride(0), weight.stride(0), output.stride(0),
            BM=self.bm, ROWS=self.rows, WIDTH=self.width,
            num_warps=self.warps, num_stages=1, enable_fp_fusion=False,
        )


class BatchedVectorLayout(WideGemvLayout):
    """Use guarded numerical/graph utilities from the B1 selector unchanged."""

    def __init__(self, engine, native, deadline):
        self.native, self.batch = native, engine.batch
        self.device, self.plans = engine.normalized.device, {}
        if not 2 <= self.batch <= 8:
            return
        # Two K values by two warp choices: at most four kernel variants for
        # this workload. N and row strides remain runtime scalar arguments.
        deadline = min(deadline, time.monotonic() + 45.0)
        groups = (
            ("gateup", engine.normalized, engine.gateup,
             [pair[1] for pair in engine.packed[:6]]),
            ("qkv", engine.normalized, engine.qkv,
             [pair[0] for pair in engine.packed[:6]]),
            ("output", engine.attention, engine.branch,
             [layer.self_attn.o_proj.weight for layer in engine.layers[:6]]),
            # Canonical group name; head is one 742 MiB tied dense matrix.
            ("head", engine.normalized, engine.logits,
             [engine.model.lm_head.weight]),
        )
        generator = torch.Generator(device=self.device)
        generator.manual_seed(20573)
        for name, template, output, weights in groups:
            if time.monotonic() >= deadline:
                break
            winner = self._select(name, template, weights, output, generator, deadline)
            if winner is not None and time.monotonic() < deadline:
                self.plans[name] = winner
            chosen = self.plans.get(name)
            self._log(name, "selected " + (chosen.name if chosen else "native layout"))
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()

    def _log(self, name, message):
        print(f"[batched-vector] B={self.batch} {name}: {message}",
              file=sys.stderr, flush=True)

    def _select(self, name, template, weights, output, generator, deadline):
        if time.monotonic() >= deadline:
            return None
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        if time.monotonic() >= deadline or not self._room(output):
            self._log(name, "native: deadline or numerical memory guard")
            return None
        try:
            reference = torch.empty_like(output)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self._log(name, "native: numerical reference allocation failed")
            return None
        x = template
        x.normal_(generator=generator)
        if time.monotonic() >= deadline:
            return None
        native_graph = self._graph(name, None, x, weights, output, deadline)
        if native_graph is None:
            return None
        winner, winner_ms = None, float("inf")
        for warps in (4, 8):
            if time.monotonic() >= deadline:
                break
            plan = BatchedPlan(self.batch, x.shape[1], warps)
            try:
                if not self._check(name, plan, x, weights, output, reference,
                                   generator, deadline):
                    self._log(name, f"{plan.name}: numerical check rejected")
                    continue
                if time.monotonic() >= deadline:
                    break
                candidate_graph = self._graph(name, plan, x, weights, output, deadline)
                if candidate_graph is None:
                    break
            except (CompilationError, OutOfResources) as error:
                self._log(name, f"{plan.name}: {type(error).__name__}")
                continue
            # Same six real layer weights and ABBA >5% gate as the B1 plan.
            native_times, candidate_times = [], []
            for graph, samples in ((native_graph, native_times),
                                   (candidate_graph, candidate_times),
                                   (candidate_graph, candidate_times),
                                   (native_graph, native_times)):
                if time.monotonic() >= deadline:
                    break
                samples.append(self._time(graph, len(weights)))
            if len(native_times) != 2 or len(candidate_times) != 2:
                del candidate_graph
                break
            native_ms, custom_ms = min(native_times), max(candidate_times)
            self._log(name, f"{plan.name}: native {native_ms * 1000:.2f} us, "
                      f"custom {custom_ms * 1000:.2f} us")
            if custom_ms < native_ms * 0.95 and custom_ms < winner_ms:
                winner, winner_ms = plan, custom_ms
            del candidate_graph
        return winner if time.monotonic() < deadline else None

    def run(self, name, layer, x, weight, output):
        plan = self.plans.get(name) if x.shape[0] == self.batch else None
        if plan is None:
            self.native.run(name, layer, x, weight, output)
        else:
            plan(x, weight, output)
