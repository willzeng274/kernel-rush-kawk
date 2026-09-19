"""Lossless BF16 storage with fused decode-only unpacking.

A byte stores the sign plus all seven mantissa bits; a nibble stores one of
15 consecutive normal exponents. Nibble 15 loads the original BF16 bits.
Original model weights remain authoritative for escapes, prefill, and native
fallback. No floating-point rounding occurs while packing or reconstructing.

Only B=1 uses this experimental full-K FP32 reduction. Actual-weight bit checks,
escape-sector counts, numerical probes, >L2 ABBA timings, and memory/deadline
guards all precede acceptance. There is no packing in measured generations.
"""

import statistics
import sys
import time

import torch
import triton
import triton.language as tl
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources


BLOCK = 4096
MAX_ESCAPE_SECTORS = 0.05
SCRATCH_RESERVE = 64 * 1024 * 1024


@triton.jit
def _exponent_histogram(W, HIST, COUNT, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    bits = tl.load(W + index, index < COUNT, 0).to(tl.uint16, bitcast=True)
    exponent = (bits.to(tl.int32) >> 7) & 255
    counts = tl.histogram(exponent, 256)
    tl.atomic_add(HIST + tl.arange(0, 256), counts, mask=counts != 0)


@triton.jit
def _pack_weights(W, SIGNIFICAND, EXPONENTS, COUNTS, COUNT, BASE,
                  BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    local = tl.arange(0, BLOCK)
    index = pid * BLOCK + local
    valid = index < COUNT
    bits = tl.load(W + index, valid, 0).to(tl.uint16, bitcast=True).to(tl.int32)
    exponent = (bits >> 7) & 255
    ordinary = (exponent >= BASE) & (exponent < BASE + 15)
    # BASE is in [1,240], so zero/subnormal and Inf/NaN always escape.
    code = tl.where(valid & ordinary, exponent - BASE, 15)
    significand = (bits & 127) | ((bits >> 8) & 128)
    tl.store(SIGNIFICAND + index, significand.to(tl.uint8), valid)
    pairs = (code << ((local & 1) * 4)).reshape((BLOCK // 2, 2))
    packed = tl.sum(pairs, axis=1).to(tl.uint8)
    pair_index = pid * (BLOCK // 2) + tl.arange(0, BLOCK // 2)
    tl.store(EXPONENTS + pair_index, packed, pair_index < (COUNT + 1) // 2)
    escaping = valid & ~ordinary
    # One original BF16 32-byte sector contains 16 contiguous weights.
    sectors = tl.sum(escaping.to(tl.int32).reshape((BLOCK // 16, 16)), axis=1) > 0
    tl.store(COUNTS + pid * 2, tl.sum(escaping.to(tl.int32), axis=0))
    tl.store(COUNTS + pid * 2 + 1, tl.sum(sectors.to(tl.int32), axis=0))


@triton.jit
def _unpack_bits(SIGNIFICAND, EXPONENTS, W, index, valid, BASE):
    sm = tl.load(SIGNIFICAND + index, valid, 0).to(tl.int32)
    pair = tl.load(EXPONENTS + index // 2, valid, 0).to(tl.int32)
    code = (pair >> ((index & 1) * 4)) & 15
    normal = ((sm & 128) << 8) | ((BASE + code) << 7) | (sm & 127)
    escape = tl.load(W + index, valid & (code == 15), 0)
    original = escape.to(tl.uint16, bitcast=True).to(tl.int32)
    bits = tl.where(code == 15, original, normal)
    return tl.where(valid, bits, 0).to(tl.uint16)


@triton.jit
def _verify_bits(W, SIGNIFICAND, EXPONENTS, ERRORS, COUNT, BASE,
                 BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    index = pid * BLOCK + tl.arange(0, BLOCK)
    valid = index < COUNT
    reconstructed = _unpack_bits(SIGNIFICAND, EXPONENTS, W, index, valid, BASE)
    original = tl.load(W + index, valid, 0).to(tl.uint16, bitcast=True)
    different = valid & (reconstructed != original)
    tl.store(ERRORS + pid, tl.sum(different.to(tl.int32), axis=0))


@triton.jit
def _packed_dot(X, W, SIGNIFICAND, EXPONENTS, OUT, N, BASE, K: tl.constexpr,
                ROWS: tl.constexpr, WIDTH: tl.constexpr):
    n = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    k = tl.arange(0, WIDTH)
    valid = (n[:, None] < N) & (k[None, :] < K)
    index = n[:, None] * K + k[None, :]
    bits = _unpack_bits(SIGNIFICAND, EXPONENTS, W, index, valid, BASE)
    w = bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
    x = tl.load(X + k, k < K, 0).to(tl.float32)
    result = tl.sum(w * x[None, :], axis=1)
    tl.store(OUT + n, result, n < N)


def best_exponent_base(histogram):
    """Exact deterministic count optimum; only normal exponent windows qualify."""
    if len(histogram) != 256:
        raise ValueError("BF16 has 256 exponent codes")
    # Prefer the lower base for an exact tie. Zero/subnormal/Inf/NaN escape.
    return max(range(1, 241), key=lambda base: sum(histogram[base:base + 15]))


def packed_bytes(weight):
    count = weight.numel()
    return count + (count + 1) // 2


class PackedWeight:
    """Own packed bytes; retain original model tensor for exact escapes."""

    def __init__(self, original, significand, exponents, base, escaped, sectors):
        self.original = original
        self.significand = significand
        self.exponents = exponents
        self.base = base
        self.escaped = escaped
        self.sectors = sectors


class PackedPlan:
    def __init__(self, k, warps):
        self.k = k
        self.width = triton.next_power_of_2(k)
        self.rows = max(1, 16384 // self.width)
        self.warps = warps
        self.name = f"lossless12_r{self.rows}_k{self.width}_w{warps}"

    def __call__(self, x, packed, output):
        original = packed.original
        _packed_dot[(triton.cdiv(original.shape[0], self.rows),)](
            x, original, packed.significand, packed.exponents, output,
            original.shape[0], packed.base, self.k,
            ROWS=self.rows, WIDTH=self.width,
            num_warps=self.warps, num_stages=1, enable_fp_fusion=False,
        )


class PackedGemvLayout:
    def __init__(self, engine, native, deadline):
        self.native = native
        self.batch = engine.batch
        self.plans = {}
        self.packed = {}
        self.packed_extra_bytes = 0
        self.device = engine.normalized.device
        if self.batch != 1:
            return
        # Existing overall cutoff starts before model load. Reserve the rest of
        # the 300-second platform budget for prefill/decode graph warmup.
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
            required = sum(packed_bytes(weight) for weight in weights)
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            if not self._room(required):
                self._log(name, "native: full packed family exceeds memory budget")
                continue
            alternate = []
            try:
                # Six smallest projections are 90 MiB packed, above H100 L2;
                # the tied head is 556 MiB packed by itself.
                if not self._extend(weights[:6], alternate, deadline):
                    self._log(name, "native: packing/escape/deadline guard")
                    self._discard(alternate)
                    continue
                winner = self._select(name, template, output, weights[:6],
                                      alternate, generator, deadline)
                if (winner is not None and time.monotonic() < deadline
                        and self._extend(weights, alternate, deadline)
                        and time.monotonic() < deadline):
                    self.plans[name] = winner
                    self.packed[name] = alternate
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
                # Compiler/resource rejection may safely select native. CUDA
                # execution errors and unexpected RuntimeErrors propagate.
                self._discard(alternate)
                self._log(name, f"native: {type(error).__name__}")
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()

    @property
    def weights(self):
        return self.native.weights

    @property
    def extra_bytes(self):
        return self.native.extra_bytes + self.packed_extra_bytes

    @staticmethod
    def _log(name, message):
        print(f"[lossless-packing] B=1 {name}: {message}", file=sys.stderr, flush=True)

    def _room(self, required, pending=0):
        free, total = torch.cuda.mem_get_info(self.device)
        # Physical free memory accounts for original model, KV, persistent
        # prefill/decode buffers, native transposes, and any pending candidates.
        # The explicit layout budgets also include native transpose ownership.
        return (self.packed_extra_bytes + pending + required <= total // 10
                and self.extra_bytes + pending + required <= total // 5
                and free - required - SCRATCH_RESERVE >= total // 4)

    def _extend(self, weights, alternate, deadline):
        pending = sum(packed_bytes(weight.original) for weight in alternate)
        for weight in weights[len(alternate):]:
            if time.monotonic() >= deadline or not self._room(packed_bytes(weight), pending):
                return False
            packed = self._pack(weight, deadline)
            if packed is None:
                return False
            alternate.append(packed)
            pending += packed_bytes(weight)
        return True

    def _discard(self, alternate):
        torch.cuda.synchronize(self.device)
        alternate.clear()
        torch.cuda.empty_cache()

    def _pack(self, weight, deadline):
        if (time.monotonic() >= deadline or weight.dtype != torch.bfloat16
                or not weight.is_contiguous()):
            return None
        count = weight.numel()
        blocks = triton.cdiv(count, BLOCK)
        histogram = torch.zeros((256,), dtype=torch.int32, device=self.device)
        _exponent_histogram[(blocks,)](weight, histogram, count, BLOCK=BLOCK,
                                      num_warps=4)
        counts = histogram.tolist()
        counts[0] -= blocks * BLOCK - count  # Masked padding was exponent zero.
        assert sum(counts) == count and min(counts) >= 0
        base = best_exponent_base(counts)
        escaped = count - sum(counts[base:base + 15])
        # Escape element fraction is a lower bound on sector fraction. Avoid
        # allocating a representation that cannot pass the stronger check.
        if escaped / count > MAX_ESCAPE_SECTORS or time.monotonic() >= deadline:
            return None
        significand = torch.empty((count,), dtype=torch.uint8, device=self.device)
        exponents = torch.empty(((count + 1) // 2,), dtype=torch.uint8, device=self.device)
        statistics_buffer = torch.empty((blocks, 2), dtype=torch.int32, device=self.device)
        _pack_weights[(blocks,)](weight, significand, exponents, statistics_buffer,
                                count, base, BLOCK=BLOCK, num_warps=4)
        actual_escaped, sectors = statistics_buffer.sum(dim=0).tolist()
        assert actual_escaped == escaped
        if sectors / triton.cdiv(count, 16) > MAX_ESCAPE_SECTORS:
            return None
        if time.monotonic() >= deadline:
            return None
        errors = torch.empty((blocks,), dtype=torch.int32, device=self.device)
        _verify_bits[(blocks,)](weight, significand, exponents, errors, count, base,
                               BLOCK=BLOCK, num_warps=4)
        if errors.sum().item() != 0:
            raise ValueError("lossless BF16 bit reconstruction failed")
        return PackedWeight(weight, significand, exponents, base, escaped, sectors)

    def _check(self, name, plan, x, reference, output, packed, generator, deadline):
        for index, scale in ((0, 1.0), (len(packed) - 1, 0.1), (0, 10.0)):
            if time.monotonic() >= deadline:
                return False
            x.normal_(generator=generator)
            x.mul_(scale)
            self.native.run(name, index, x, packed[index].original, reference)
            if time.monotonic() >= deadline:
                return False
            plan(x, packed[index], output)
            if not torch.allclose(output, reference, rtol=0.01, atol=0.01):
                return False
        return True

    def _select(self, name, template, output, weights, packed, generator, deadline):
        if time.monotonic() >= deadline:
            return None
        # Bound one reusable activation, one reusable native result, and a
        # conservative allowance for allclose temporaries inside the existing
        # reserve. The largest B=1 head probe is well below this 64 MiB cap.
        scratch = template.numel() * template.element_size() + output.numel() * 32 + 65536
        pending = sum(packed_bytes(weight.original) for weight in packed)
        if scratch > SCRATCH_RESERVE or not self._room(0, pending):
            return None
        x, reference = torch.empty_like(template), torch.empty_like(output)
        x.normal_(generator=generator)
        if time.monotonic() >= deadline:
            return None
        native_graph = self._graph(name, None, x, weights, packed, output)
        winner, winner_ms = None, float("inf")
        for warps in (4, 8):
            if time.monotonic() >= deadline:
                break
            plan = PackedPlan(x.shape[1], warps)
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

    def _graph(self, name, plan, x, weights, packed, output):
        current = torch.cuda.current_stream(x.device)
        stream = torch.cuda.Stream(device=x.device)
        stream.wait_stream(current)

        def launch():
            for index, weight in enumerate(weights):
                if plan is None:
                    self.native.run(name, index, x, weight, output)
                else:
                    plan(x, packed[index], output)

        with torch.cuda.stream(stream):
            launch()
        current.wait_stream(stream)
        torch.cuda.synchronize(x.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            launch()
        current.wait_stream(stream)
        graph.replay()
        torch.cuda.synchronize(x.device)
        return graph

    @staticmethod
    def _time(graph, count):
        samples = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(3):
            start.record()
            for _ in range(16):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / (16 * count))
        return statistics.median(samples)

    def run(self, name, layer, x, weight, output):
        plan = self.plans.get(name) if x.shape[0] == 1 else None
        if plan is None:
            self.native.run(name, layer, x, weight, output)
        else:
            packed = self.packed[name][layer]
            # Original weight identity is stable in this model. Keeping this
            # guard protects callers that later rebind a model parameter.
            if packed.original is not weight:
                self.native.run(name, layer, x, weight, output)
            else:
                plan(x, packed, output)
