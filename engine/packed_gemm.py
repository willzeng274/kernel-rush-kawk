"""Dense small-batch BF16 tensor-core GEMM with exact weight reconstruction.

Only B=2..32 is eligible. The original B=1 codec/GEMV module is unchanged.
Integer reconstruction precedes BF16 dot operands, FP32 accumulation, and one
BF16 output rounding. No partial sums or activation/weight quantization.
"""

import sys
import time

import torch
import triton
import triton.language as tl
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources

from lossless_packing import PackedGemvLayout, SCRATCH_RESERVE, _unpack_bits, packed_bytes


@triton.jit
def _packed_gemm(X, W, SIGNIFICAND, EXPONENTS, OUT, N, BASE,
                 M: tl.constexpr, K: tl.constexpr,
                 SXM: tl.constexpr, SXK: tl.constexpr,
                 SOM: tl.constexpr, SON: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # Exactly one M tile avoids rereading the weights for a second row tile.
    m = tl.arange(0, BM)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    lane_k = tl.arange(0, BK)
    accumulator = tl.zeros((BM, BN), dtype=tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + lane_k
        x = tl.load(X + m[:, None] * SXM + k[None, :] * SXK,
                    (m[:, None] < M) & (k[None, :] < K), 0)
        # Original W is row-major [N,K]; adjacent K values share nibbles.
        index = n[:, None] * K + k[None, :]
        valid = (n[:, None] < N) & (k[None, :] < K)
        bits = _unpack_bits(SIGNIFICAND, EXPONENTS, W, index, valid, BASE)
        weights = bits.to(tl.bfloat16, bitcast=True)
        accumulator = tl.dot(x, tl.trans(weights), accumulator,
                             out_dtype=tl.float32)
    result = accumulator.to(tl.bfloat16)
    tl.store(OUT + m[:, None] * SOM + n[None, :] * SON,
             result, (m[:, None] < M) & (n[None, :] < N))


class PackedGemmPlan:
    def __init__(self, batch, bn, bk):
        self.batch = batch
        self.bm = 16 if batch <= 16 else 32
        self.bn, self.bk = bn, bk
        self.name = f"lossless12_tc_m{self.bm}_n{bn}_k{bk}_w4_s2"

    def __call__(self, x, packed, output):
        weight = packed.original
        _packed_gemm[(triton.cdiv(weight.shape[0], self.bn),)](
            x, weight, packed.significand, packed.exponents, output,
            weight.shape[0], packed.base, self.batch, weight.shape[1],
            x.stride(0), x.stride(1), output.stride(0), output.stride(1),
            BM=self.bm, BN=self.bn, BK=self.bk,
            num_warps=4, num_stages=2,
        )


class PackedGemmLayout(PackedGemvLayout):
    """Reuse the reviewed exact codec, ownership, probes, and timing helpers."""

    def __init__(self, engine, native, deadline):
        self.native = native
        self.batch = engine.batch
        self.plans, self.packed = {}, {}
        self.packed_extra_bytes = 0
        self.device = engine.normalized.device
        if not 2 <= self.batch <= 32:
            return
        deadline = min(deadline, time.monotonic() + 70.0)
        groups = (
            ("gateup", engine.normalized, engine.gateup,
             [pair[1] for pair in engine.packed]),
            ("down", engine.intermediate, engine.branch,
             [layer.mlp.down_proj.weight for layer in engine.layers]),
            ("qkv", engine.normalized, engine.qkv,
             [pair[0] for pair in engine.packed]),
            ("output", engine.attention, engine.branch,
             [layer.self_attn.o_proj.weight for layer in engine.layers]),
            ("head", engine.normalized, engine.logits, [engine.model.lm_head.weight]),
        )
        generator = torch.Generator(device=self.device)
        generator.manual_seed(17091)
        for name, template, output, weights in groups:
            if time.monotonic() >= deadline:
                break
            # Large head probes at B>=16 exceed the inherited 64 MiB reserve.
            # Reject before packing the head rather than after that expense.
            if self._scratch_bytes(template, output) > SCRATCH_RESERVE:
                self._log(name, "native: numerical scratch reserve")
                continue
            required = sum(packed_bytes(weight) for weight in weights)
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            if time.monotonic() >= deadline:
                break
            if not self._room(required):
                self._log(name, "native: full packed family exceeds memory budget")
                continue
            alternate = []
            try:
                # Six smallest packed projections occupy 90 MiB, beyond H100
                # L2. The single packed vocabulary head is 556 MiB.
                representatives = weights[:6]
                if not self._extend(representatives, alternate, deadline):
                    self._log(name, "native: packing/escape/deadline guard")
                    self._discard(alternate)
                    continue
                winner = self._select(name, template, output, representatives,
                                      alternate, generator, deadline)
                if (winner is not None and time.monotonic() < deadline
                        and self._extend(weights, alternate, deadline)
                        and time.monotonic() < deadline):
                    self.plans[name], self.packed[name] = winner, alternate
                    self.packed_extra_bytes += required
                    count = sum(weight.numel() for weight in weights)
                    escaped = sum(weight.escaped for weight in alternate)
                    sectors = sum(weight.sectors for weight in alternate)
                    sector_count = sum(triton.cdiv(weight.numel(), 16) for weight in weights)
                    self._log(name, f"selected {winner.name}; escapes {escaped/count:.6%}; "
                              f"original sectors {sectors/sector_count:.6%}; "
                              f"packed {self.packed_extra_bytes/2**30:.3f} GiB; "
                              f"all layout copies {self.extra_bytes/2**30:.3f} GiB")
                else:
                    self._discard(alternate)
                    self._log(name, "native: no complete measured winner")
            except (CompilationError, OutOfResources) as error:
                self._discard(alternate)
                self._log(name, f"native: {type(error).__name__}")
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()

    def _log(self, name, message):
        print(f"[lossless-gemm] B={self.batch} {name}: {message}",
              file=sys.stderr, flush=True)

    @staticmethod
    def _scratch_bytes(template, output):
        return template.numel() * template.element_size() + output.numel() * 32 + 65536

    def _select(self, name, template, output, weights, packed, generator, deadline):
        if time.monotonic() >= deadline:
            return None
        pending = sum(packed_bytes(weight.original) for weight in packed)
        if self._scratch_bytes(template, output) > SCRATCH_RESERVE or not self._room(0, pending):
            return None
        x, reference = torch.empty_like(template), torch.empty_like(output)
        x.normal_(generator=generator)
        if time.monotonic() >= deadline:
            return None
        native_graph = self._graph(name, None, x, weights, packed, output)
        winner, winner_ms = None, float("inf")
        # Both have 2,048 reconstructed values. N16 supplies 160 CTAs for
        # the narrowest projections; N32 uses a wider output tile and smaller
        # K64 input tiles without expanding the unpack register footprint.
        for bn, bk in ((16, 128), (32, 64)):
            if time.monotonic() >= deadline:
                break
            plan = PackedGemmPlan(self.batch, bn, bk)
            try:
                if not self._check(name, plan, x, reference, output, packed, generator, deadline):
                    self._log(name, f"{plan.name}: numerical probe rejected")
                    continue
                if time.monotonic() >= deadline:
                    break
                candidate = self._graph(name, plan, x, weights, packed, output)
            except (CompilationError, OutOfResources) as error:
                self._log(name, f"{plan.name}: {type(error).__name__}")
                continue
            native, custom = [], []
            for graph, samples in ((native_graph, native), (candidate, custom),
                                   (candidate, custom), (native_graph, native)):
                if time.monotonic() >= deadline:
                    break
                samples.append(self._time(graph, len(weights)))
            if len(native) != 2 or len(custom) != 2:
                del candidate
                break
            native_ms, custom_ms = min(native), max(custom)
            self._log(name, f"{plan.name}: native {native_ms*1000:.2f} us; "
                      f"packed {custom_ms*1000:.2f} us")
            if custom_ms < native_ms * 0.95 and custom_ms < winner_ms:
                winner, winner_ms = plan, custom_ms
            del candidate
        return winner

    def run(self, name, layer, x, weight, output):
        plan = self.plans.get(name) if x.shape[0] == self.batch else None
        if plan is None or self.packed[name][layer].original is not weight:
            self.native.run(name, layer, x, weight, output)
        else:
            plan(x, self.packed[name][layer], output)
