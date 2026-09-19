"""B1 exact BF16 word-plane unpacking with packed-pair reconstruction.

Each eight-weight group loads two significand uint32 words and one exponent
uint32 word. Integer operations reconstruct four BF16 pairs before expansion
for the existing full-K FP32 product/reduction. An escaping half selects its
original BF16 bits; ordinary halves never use original weight values. The byte
codec, original tensors, projection rounding and prefill are unchanged.
"""

import math
import sys
import time

import torch
import triton
import triton.language as tl
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources

from lossless_packing import PackedGemvLayout, SCRATCH_RESERVE, BLOCK, packed_bytes


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
def _packed_word_dot(X, ORIGINAL, SM, EXP, OUT, N, BASE, K: tl.constexpr,
                     ROWS: tl.constexpr, WIDTH: tl.constexpr):
    n = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    local_group = tl.arange(0, WIDTH // 8)
    group = n[:, None] * (K // 8) + local_group[None, :]
    valid = (n[:, None] < N) & (local_group[None, :] < K // 8)
    pairs = _unpack_word_groups(SM, EXP, ORIGINAL, group, valid, BASE)
    pairs = pairs.reshape((ROWS, WIDTH // 2))
    halves = tl.arange(0, 2)
    bits = ((pairs[:, :, None] >> (halves[None, None, :] * 16)) & 65535)
    w = bits.to(tl.uint16).reshape((ROWS, WIDTH)).to(tl.bfloat16, bitcast=True).to(tl.float32)
    k = tl.arange(0, WIDTH)
    x = tl.load(X + k, k < K, 0).to(tl.float32)
    result = tl.sum(w * x[None, :], axis=1)
    tl.store(OUT + n, result, n < N)


class PackedWordPlan:
    def __init__(self, k, warps):
        if k not in (2560, 4096, 9728) or warps not in (4, 8):
            raise ValueError("word-packed plan requires K2560/4096/9728 and 4/8 warps")
        self.k, self.warps = k, warps
        self.width = triton.next_power_of_2(k)
        self.rows = 16384 // self.width
        self.name = f"lossless12_word_r{self.rows}_k{self.width}_w{warps}"

    def __call__(self, x, packed, output):
        weight = packed.original
        if (x.shape != (1, self.k) or weight.shape[1] != self.k
                or output.shape != (1, weight.shape[0])
                or x.stride(1) != 1 or output.stride(1) != 1
                or not weight.is_contiguous()
                or x.dtype != torch.bfloat16 or output.dtype != torch.bfloat16
                or weight.dtype != torch.bfloat16):
            raise ValueError("incompatible word-packed shape, stride or dtype")
        _packed_word_dot[(triton.cdiv(weight.shape[0], self.rows),)](
            x, packed.original_words, packed.significand_words, packed.exponent_words,
            output, weight.shape[0], packed.base, self.k, ROWS=self.rows,
            WIDTH=self.width, num_warps=self.warps, num_stages=1,
            enable_fp_fusion=False,
        )


class PackedWordLayout(PackedGemvLayout):
    """Reuse the bit-verified codec and memory/weight ownership accounting."""

    def __init__(self, engine, native, deadline):
        self.native, self.batch = native, engine.batch
        self.plans, self.packed = {}, {}
        self.packed_extra_bytes = 0
        self.device = engine.normalized.device
        if self.batch != 1:
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
        generator.manual_seed(31847)
        for name, template, output, weights in groups:
            if time.monotonic() >= deadline:
                break
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
                # Six smallest packed projections total 90 MiB (>H100 L2).
                # The single vocabulary head is 556 MiB packed by itself.
                representatives = weights[:6]
                if not self._extend(representatives, alternate, deadline):
                    self._discard(alternate)
                    self._log(name, "native: packing/escape/deadline guard")
                    continue
                winner = self._select(name, template, output, representatives,
                                      alternate, generator, deadline)
                if (winner is not None and time.monotonic() < deadline
                        and self._extend(weights, alternate, deadline)
                        and time.monotonic() < deadline):
                    self.plans[name], self.packed[name] = winner, alternate
                    self.packed_extra_bytes += required
                    self._log(name, f"selected {winner.name}; "
                              f"packed {self.packed_extra_bytes/2**30:.3f} GiB; "
                              f"all layout copies {self.extra_bytes/2**30:.3f} GiB")
                else:
                    self._discard(alternate)
                    self._log(name, "native: no complete measured winner")
            except (CompilationError, OutOfResources, torch.cuda.OutOfMemoryError) as error:
                self._discard(alternate)
                self._log(name, f"native: {type(error).__name__}")
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

    def _log(self, name, message):
        print(f"[packed-word] B={self.batch} {name}: {message}",
              file=sys.stderr, flush=True)

    @staticmethod
    def _scratch_bytes(template, output):
        return template.numel() * template.element_size() + output.numel() * 32 + 65536

    def _comparison_room(self, output):
        free, total = torch.cuda.mem_get_info(self.device)
        # x/reference already exist; retain the reserve for comparison temps.
        required = output.numel() * (32 - output.element_size()) + 65536
        return free - required >= total // 4

    def _check(self, name, plan, x, reference, output, packed, generator, deadline):
        for index, scale in ((0, 1.0), (len(packed) - 1, 0.1), (0, 10.0)):
            if time.monotonic() >= deadline:
                return False
            x.normal_(generator=generator).mul_(scale)
            self.native.run(name, index, x, packed[index].original, reference)
            if time.monotonic() >= deadline:
                return False
            plan(x, packed[index], output)
            if time.monotonic() >= deadline or not self._comparison_room(output):
                return False
            try:
                if not torch.allclose(output, reference, rtol=0.01, atol=0.01):
                    return False
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                return False
        return time.monotonic() < deadline

    def _select(self, name, template, output, weights, packed, generator, deadline):
        if time.monotonic() >= deadline:
            return None
        pending = sum(packed_bytes(weight.original) for weight in packed)
        if self._scratch_bytes(template, output) > SCRATCH_RESERVE or not self._room(0, pending):
            return None
        try:
            x, reference = torch.empty_like(template), torch.empty_like(output)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None
        x.normal_(generator=generator)
        if time.monotonic() >= deadline:
            return None
        native_graph = self._graph(name, None, x, weights, packed, output, deadline)
        if native_graph is None:
            return None
        winner, winner_ms = None, float("inf")
        # Three K values by two warp counts; N/BASE/strides remain runtime scalars.
        for warps in (4, 8):
            if time.monotonic() >= deadline:
                break
            plan = PackedWordPlan(x.shape[1], warps)
            try:
                if not self._check(name, plan, x, reference, output, packed, generator, deadline):
                    self._log(name, f"{plan.name}: numerical probe rejected")
                    continue
                if time.monotonic() >= deadline:
                    break
                candidate = self._graph(name, plan, x, weights, packed, output, deadline)
                if candidate is None:
                    break
            except (CompilationError, OutOfResources, torch.cuda.OutOfMemoryError) as error:
                self._log(name, f"{plan.name}: {type(error).__name__}")
                continue
            native, custom = [], []
            for graph, samples in ((native_graph, native), (candidate, custom),
                                   (candidate, custom), (native_graph, native)):
                if time.monotonic() >= deadline:
                    break
                samples.append(self._time(graph, len(weights)))
            if (len(native) != 2 or len(custom) != 2
                    or time.monotonic() >= deadline):
                del candidate
                break
            if not all(math.isfinite(value) and value > 0 for value in native + custom):
                del candidate
                continue
            native_ms, custom_ms = min(native), max(custom)
            self._log(name, f"{plan.name}: native {native_ms*1000:.2f} us; "
                      f"packed {custom_ms*1000:.2f} us")
            if custom_ms < native_ms * 0.95 and custom_ms < winner_ms:
                winner, winner_ms = plan, custom_ms
            del candidate
        return winner if time.monotonic() < deadline else None

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

        with torch.cuda.stream(stream):
            launch()
        current.wait_stream(stream)
        torch.cuda.synchronize(x.device)
        if time.monotonic() >= deadline:
            return None
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            launch()
        current.wait_stream(stream)
        if time.monotonic() >= deadline:
            return None
        graph.replay()
        torch.cuda.synchronize(x.device)
        return graph if time.monotonic() < deadline else None

    def run(self, name, layer, x, weight, output):
        plan = self.plans.get(name) if x.shape[0] == self.batch else None
        if plan is None or self.packed[name][layer].original is not weight:
            self.native.run(name, layer, x, weight, output)
        else:
            plan(x, self.packed[name][layer], output)
