"""Exact packed BF16 operands in the accepted M64 transposed Hopper product.

Only decode B2..32 is eligible; B1 retains its existing path. Integer reconstruction precedes the unchanged
BF16 dot, FP32 accumulation/split merge, and final BF16 output store. Original
weights remain authoritative for escapes, prefill, and the selected fallback.
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

from hopper_gemm import HopperPlan, _hopper_merge
from hopper_tiles import HopperTilesPlan
from lossless_packing import PackedGemvLayout, BLOCK, packed_bytes


# Full B32 vocabulary probes require ~149 MiB under the conservative 32-byte
# per-output comparison bound. Reserve 192 MiB, including codec temporaries.
PROBE_RESERVE = 192 * 1024 * 1024


@triton.jit
def _unpack_word_groups(SM, EXP, ORIGINAL, group, valid, BASE):
    # group/valid have shape [ROWS, GROUPS], with eight weights per group.
    # All original row widths are multiples of eight, so groups never cross
    # a valid/padded row boundary. uint32 prevents sign-extending right shifts.
    word = tl.arange(0, 2)
    sm_words = tl.load(SM + group[:, :, None] * 2 + word[None, None, :],
                       valid[:, :, None], 0).to(tl.uint32)
    codes = tl.load(EXP + group, valid, 0).to(tl.uint32)
    half = tl.arange(0, 2)
    sm = (sm_words[:, :, :, None] >> (half[None, None, None, :] * 16)) & 65535
    sm = sm.reshape((group.shape[0], group.shape[1], 4))
    pair = tl.arange(0, 4)
    # Spread two sign/mantissa bytes into two 16-bit lanes. Exponent fields
    # are disjoint, and BASE+code<=255 prevents cross-lane carries.
    spread = (sm & 255) | ((sm & 65280) << 8)
    sign_mantissa = (spread & 0x007f007f) | ((spread & 0x00800080) << 8)
    code_pair = (codes[:, :, None] >> (pair[None, None, :] * 8)) & 255
    exponents = (code_pair & 15) | ((code_pair & 240) << 12)
    normal = sign_mantissa | ((exponents + BASE * 0x00010001) << 7)
    low_escape = (code_pair & 15) == 15
    high_escape = (code_pair & 240) == 240
    original = tl.load(ORIGINAL + group[:, :, None] * 4 + pair[None, None, :],
                       valid[:, :, None] & (low_escape | high_escape), 0).to(tl.uint32)
    escape_mask = tl.where(low_escape, 0x0000ffff, 0).to(tl.uint32)
    escape_mask = escape_mask | tl.where(high_escape, 0xffff0000, 0).to(tl.uint32)
    bits = (normal & ~escape_mask) | (original & escape_mask)
    return tl.where(valid[:, :, None], bits, 0).to(tl.uint32)


@triton.jit
def _verify_word_bits(ORIGINAL, SM, EXP, ERRORS, COUNT, BASE, BLOCK: tl.constexpr):
    group = tl.program_id(0) * (BLOCK // 8) + tl.arange(0, BLOCK // 8)
    valid = group < COUNT // 8
    reconstructed = _unpack_word_groups(SM, EXP, ORIGINAL,
                                         group[None, :], valid[None, :], BASE)
    index = group[:, None] * 4 + tl.arange(0, 4)[None, :]
    original = tl.load(ORIGINAL + index, valid[:, None], 0).to(tl.uint32)
    different = valid[:, None] & (reconstructed.reshape((BLOCK // 8, 4)) != original)
    tl.store(ERRORS + tl.program_id(0), tl.sum(different.to(tl.int32).reshape((BLOCK // 2,)), axis=0))


@triton.jit
def _word_hopper_dot(X, W, SIGNIFICAND, EXPONENTS, OUT, PART, N, BASE,
                       X_ROW, OUT_ROW, B: tl.constexpr, K: tl.constexpr,
                       BB: tl.constexpr, BK: tl.constexpr, SPLITS: tl.constexpr):
    n = tl.program_id(0) * 64 + tl.arange(0, 64)
    split = tl.program_id(1)
    b = tl.arange(0, BB)
    k = tl.arange(0, BK)
    local_group = tl.arange(0, BK // 8)
    steps = tl.cdiv(K, SPLITS * BK)
    acc = tl.full((64, BB), 0, tl.float32)
    for block in range(steps):
        kk = (split * steps + block) * BK + k
        # BK and every model K are multiples of eight, so a loaded word
        # group stays within one real/padded row and one split-K block.
        kg = (split * steps + block) * (BK // 8) + local_group
        group = n[:, None] * (K // 8) + kg[None, :]
        valid = (n[:, None] < N) & (kg[None, :] < K // 8)
        pairs = _unpack_word_groups(SIGNIFICAND, EXPONENTS, W, group, valid, BASE)
        pairs = pairs.reshape((64, BK // 2))
        half = tl.arange(0, 2)
        bits = (pairs[:, :, None] >> (half[None, None, :] * 16)) & 65535
        w = bits.to(tl.uint16).reshape((64, BK)).to(tl.bfloat16, bitcast=True)
        x = tl.load(X + b[None, :] * X_ROW + kk[:, None],
                    (b[None, :] < B) & (kk[:, None] < K), 0)
        acc = tl.dot(w, x, acc)
    if SPLITS == 1:
        tl.store(OUT + b[None, :] * OUT_ROW + n[:, None], acc,
                 (b[None, :] < B) & (n[:, None] < N))
    else:
        tl.store(PART + (split * B + b[None, :]) * N + n[:, None], acc,
                 (b[None, :] < B) & (n[:, None] < N))


class WordHopperPlan(HopperPlan):
    def __init__(self, x, weight, output, block_k):
        super().__init__(x, weight, output, block_k)
        self.name = "lossless12_word_" + self.name

    def __call__(self, x, packed, output):
        part = output if self.workspace is None else self.workspace
        _word_hopper_dot[(triton.cdiv(self.n, 64), self.splits)](
            x, packed.original_words, packed.significand_words, packed.exponent_words, output,
            part, self.n, packed.base, x.stride(0), output.stride(0),
            B=self.batch, K=self.k, BB=self.bb, BK=self.block_k, SPLITS=self.splits,
            num_warps=4, num_stages=self.stages,
        )
        if self.splits > 1:
            _hopper_merge[(triton.cdiv(self.batch * self.n, 512),)](
                self.workspace, output, self.n, output.stride(0), self.batch * self.n,
                SPLITS=self.splits, BLOCK=512, num_warps=4,
            )


class WordHopperLayout(PackedGemvLayout):
    """Reuse the exact codec, requiring a complete verified family and >5% win."""

    def __init__(self, engine, native, deadline):
        self.native, self.batch = native, engine.batch
        self.plans, self.packed = {}, {}
        self.packed_extra_bytes = self.pending_workspace_bytes = 0
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
        generator.manual_seed(53081)
        for name, template, output, weights in groups:
            if time.monotonic() >= deadline:
                break
            if output.numel() * 32 + 65536 > PROBE_RESERVE:
                self._log(name, "existing: comparison reserve exceeded")
                continue
            required = sum(packed_bytes(weight) for weight in weights)
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            if time.monotonic() >= deadline:
                break
            if not self._room(required):
                self._log(name, "existing: complete packed family memory guard")
                continue
            alternate, winner = [], None
            try:
                # Six actual packed output projections are 90 MiB; the single
                # complete packed head is 556 MiB. Both exceed H100 L2.
                representatives = weights[:6]
                if self._extend(representatives, alternate, deadline):
                    winner = self._select(name, template, output, representatives,
                                          alternate, generator, deadline)
                if (winner is not None and time.monotonic() < deadline
                        and self._extend(weights, alternate, deadline)
                        and time.monotonic() < deadline):
                    self.plans[name], self.packed[name] = winner, alternate
                    self.packed_extra_bytes += required
                    self.pending_workspace_bytes = 0
                    self._log(name, f"selected {winner.name}; "
                              f"packed {self.packed_extra_bytes/2**30:.3f} GiB; "
                              f"all layouts {self.extra_bytes/2**30:.3f} GiB")
                else:
                    self._discard(alternate)
                    self._log(name, "existing: no complete verified measured winner")
            except (CompilationError, OutOfResources, torch.cuda.OutOfMemoryError) as error:
                # Explicit compiler/resource/allocation failures may fall back;
                # unexpected CUDA execution errors remain fatal and visible.
                self._discard(alternate)
                self._log(name, f"existing: {type(error).__name__}")
            finally:
                winner = None
                self.pending_workspace_bytes = 0
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()

    def _pack(self, weight, deadline):
        if weight.numel() % 8 or weight.shape[1] % 8:
            return None
        packed = super()._pack(weight, deadline)
        if packed is None or time.monotonic() >= deadline:
            return None
        # Reinterpretation views allocate no payload or transpose. Both original
        # and packed tensors have aligned contiguous storage and divisible sizes.
        packed.original_words = weight.view(torch.int32)
        packed.significand_words = packed.significand.view(torch.int32)
        packed.exponent_words = packed.exponents.view(torch.int32)
        blocks = triton.cdiv(weight.numel(), BLOCK)
        errors = torch.empty((blocks,), dtype=torch.int32, device=self.device)
        _verify_word_bits[(blocks,)](
            packed.original_words, packed.significand_words, packed.exponent_words,
            errors, weight.numel(), packed.base, BLOCK=BLOCK, num_warps=4)
        if errors.sum().item() != 0:
            raise ValueError("word-packed BF16 bit reconstruction failed")
        return packed if time.monotonic() < deadline else None

    @property
    def extra_bytes(self):
        return (self.native.extra_bytes + self.packed_extra_bytes
                + sum(plan.extra_bytes for plan in self.plans.values()))

    def _room(self, required, pending=0):
        free, total = torch.cuda.mem_get_info(self.device)
        return (self.packed_extra_bytes + pending + required <= total // 10
                and self.extra_bytes + pending + self.pending_workspace_bytes + required <= total // 5
                and free - required - PROBE_RESERVE >= total // 4)

    def _log(self, name, message):
        print(f"[word-hopper] B={self.batch} {name}: {message}",
              file=sys.stderr, flush=True)

    def _check(self, name, plan, x, reference, output, packed, generator, deadline):
        pending = sum(packed_bytes(weight.original) for weight in packed)
        for index, scale in ((0, 1.0), (len(packed) - 1, 0.1), (0, 10.0)):
            if time.monotonic() >= deadline:
                return False
            x.normal_(generator=generator).mul_(scale)
            self.native.run(name, index, x, packed[index].original, reference)
            if time.monotonic() >= deadline:
                return False
            plan(x, packed[index], output)
            if time.monotonic() >= deadline or not self._room(0, pending):
                return False
            if not torch.allclose(output, reference, rtol=0.01, atol=0.01):
                return False
        return time.monotonic() < deadline

    def _select(self, name, template, output, weights, packed, generator, deadline):
        pending = sum(packed_bytes(weight.original) for weight in packed)
        if time.monotonic() >= deadline or not self._room(0, pending):
            return None
        reference = torch.empty_like(output)
        if time.monotonic() >= deadline:
            return None
        x = template  # Real prefill overwrites these persistent decode buffers.
        x.normal_(generator=generator)
        native_graph = self._graph(name, None, x, weights, packed, output, deadline)
        if native_graph is None:
            return None
        winner, winner_ms = None, float("inf")
        # Try the accepted Hopper tile first when this family selected one.
        # The direct fallback is the accepted Tiles wrapper, whose own
        # fallback is Hopper. A selected M128 tile prefers K128 here, while
        # an unselected Tiles family exposes the actual selected Hopper K.
        previous = self.native.plans.get(name)
        if previous is None:
            previous = self.native.native.plans.get(name)
        preferred = previous.block_k if isinstance(previous, (HopperPlan, HopperTilesPlan)) else 128
        for block_k in (preferred, 384 - preferred):
            if time.monotonic() >= deadline:
                break
            self.pending_workspace_bytes = 0 if winner is None else winner.extra_bytes
            splits = HopperPlan.split_count(weights[0].shape[0], block_k)
            workspace = output.numel() * splits * 4 if splits > 1 else 0
            if not self._room(workspace, pending):
                continue
            plan, candidate = None, None
            try:
                plan = WordHopperPlan(x, weights[0], output, block_k)
                self.pending_workspace_bytes += plan.extra_bytes
                if not self._check(name, plan, x, reference, output, packed, generator, deadline):
                    continue
                candidate = self._graph(name, plan, x, weights, packed, output, deadline)
                if candidate is None:
                    break
                native_times, custom_times = [], []
                for graph, samples in ((native_graph, native_times), (candidate, custom_times),
                                       (candidate, custom_times), (native_graph, native_times)):
                    value = self._time(graph, len(weights), deadline)
                    if value is None:
                        break
                    samples.append(value)
                del candidate
                if len(native_times) != 2 or len(custom_times) != 2:
                    break
                native_ms, custom_ms = min(native_times), max(custom_times)
                self._log(name, f"{plan.name}: existing {native_ms*1000:.2f} us; "
                          f"packed {custom_ms*1000:.2f} us")
                if (time.monotonic() < deadline
                        and all(math.isfinite(value) and value > 0
                                for value in native_times + custom_times)
                        and custom_ms < native_ms * 0.95 and custom_ms < winner_ms):
                    winner, winner_ms = plan, custom_ms
            except (CompilationError, OutOfResources, torch.cuda.OutOfMemoryError) as error:
                self._log(name, f"existing: {type(error).__name__}")
            finally:
                candidate = None
                plan = None
                self.pending_workspace_bytes = 0 if winner is None else winner.extra_bytes
        if time.monotonic() >= deadline:
            self.pending_workspace_bytes = 0
            return None
        return winner

    def _graph(self, name, plan, x, weights, packed, output, deadline):
        if time.monotonic() >= deadline:
            return None
        current = torch.cuda.current_stream(x.device)
        stream = torch.cuda.Stream(device=x.device)
        stream.wait_stream(current)

        def launch():
            for index, weight in enumerate(weights):
                if plan is None:
                    self.native.run(name, index, x, weight, output)
                else:
                    plan(x, packed[index], output)

        try:
            with torch.cuda.stream(stream):
                launch()
        finally:
            # A later launch may fail after earlier work reached this stream.
            # Order that work before the caller releases candidate buffers.
            current.wait_stream(stream)
        torch.cuda.synchronize(x.device)
        if time.monotonic() >= deadline:
            return None
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=stream):
                launch()
        finally:
            current.wait_stream(stream)
        if time.monotonic() >= deadline:
            return None
        graph.replay()
        torch.cuda.synchronize(x.device)
        return graph if time.monotonic() < deadline else None

    @staticmethod
    def _time(graph, count, deadline):
        samples = []
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(3):
            if time.monotonic() >= deadline:
                return None
            start.record()
            for _ in range(16):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / (16 * count))
        return statistics.median(samples) if time.monotonic() < deadline else None

    def run(self, name, layer, x, weight, output):
        plan = self.plans.get(name) if x.shape[0] == self.batch else None
        family = self.packed.get(name, ())
        if (plan is None or not 0 <= layer < len(family)
                or family[layer].original is not weight
                or tuple(x.shape) != (plan.batch, plan.k)
                or tuple(weight.shape) != (plan.n, plan.k)
                or tuple(output.shape) != (plan.batch, plan.n)):
            self.native.run(name, layer, x, weight, output)
        else:
            plan(x, family[layer], output)
