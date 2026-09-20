"""Standard-library CPU controls. No Torch, Triton import, compiler, or GPU.

These check address/mask/ownership algebra, actual norm-source cast semantics,
head assembly, and host dispatch. They do not emulate GPU rsqrt/dot lowering.
"""

import ast
import copy
import importlib.util
import itertools
import math
from pathlib import Path
import random
import struct
import sys
import types
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
BASELINE = HERE.parents[1] / "candidates/krxfty_median5_on90/engine"
DEVICE = HERE / "tree_fused_attention_device.py"
HOST = HERE / "tree_fused_attention.py"


def fp32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bf16(x):
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    if bits & 0x7f800000 == 0x7f800000:
        return fp32(x)
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def tree_sum(values):
    values = list(values)
    while len(values) > 1:
        values = [fp32(values[i] + values[i + 1]) for i in range(0, len(values), 2)]
    return values[0]


class Block:
    """Tiny FP32 block model for executing the actual norm helper AST only."""

    def __init__(self, values, shape):
        self.values, self.shape = list(values), tuple(shape)
        assert len(self.values) == math.prod(shape)

    def at(self, index):
        offset = 0
        for i, size in zip(index, self.shape):
            offset = offset * size + i
        return self.values[offset]

    def binary(self, other, op):
        if not isinstance(other, Block):
            other = Block([other], ())
        rank = max(len(self.shape), len(other.shape))
        a = (1,) * (rank - len(self.shape)) + self.shape
        b = (1,) * (rank - len(other.shape)) + other.shape
        assert all(x == y or x == 1 or y == 1 for x, y in zip(a, b))
        shape = tuple(max(x, y) for x, y in zip(a, b))
        values = []
        for idx in itertools.product(*(range(s) for s in shape)):
            ia = tuple(0 if size == 1 else i for i, size in zip(idx, a))
            ib = tuple(0 if size == 1 else i for i, size in zip(idx, b))
            av = self.at(ia[rank - len(self.shape):])
            bv = other.at(ib[rank - len(other.shape):])
            values.append(fp32(op(av, bv)))
        return Block(values, shape)

    def __add__(self, other):
        return self.binary(other, lambda a, b: a + b)

    def __mul__(self, other):
        return self.binary(other, lambda a, b: a * b)

    def __truediv__(self, other):
        return self.binary(other, lambda a, b: a / b)

    def __neg__(self):
        return Block([-x for x in self.values], self.shape)

    def to(self, dtype):
        convert = bf16 if dtype == "bf16" else fp32
        return Block([convert(x) for x in self.values], self.shape)


def sum_axis_zero(block, axis):
    assert axis == 0
    tail = block.shape[1:]
    values = [tree_sum(block.at((i,) + index) for i in range(block.shape[0]))
              for index in itertools.product(*(range(s) for s in tail))]
    return Block(values, tail)


def source_norm():
    fn = copy.deepcopy(next(n for n in ast.parse(DEVICE.read_text()).body
                            if isinstance(n, ast.FunctionDef) and n.name == "_norm_rope_row"))
    fn.decorator_list = []
    for arg in fn.args.args:
        arg.annotation = None
    fake_tl = types.SimpleNamespace(
        sum=sum_axis_zero, bfloat16="bf16", float32="fp32",
        math=types.SimpleNamespace(rsqrt=lambda x: Block([fp32(1 / math.sqrt(v)) for v in x.values], x.shape)),
    )
    ns = {"tl": fake_tl}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "norm_cpu", "exec"), ns)
    return ns["_norm_rope_row"]


def scalar_norm_reference(x, w, cos, sin, *, omit=None):
    # Separate-half FP32 balanced reductions; sqrt model is CPU, not GPU rsqrt.
    ss = [tree_sum(fp32(v * v) for v in x[half:half + 64]) for half in (0, 64)]
    inverse = fp32(1 / math.sqrt(fp32(fp32(fp32(ss[0] + ss[1]) / 128) + 1e-6)))
    norm = [fp32(v * inverse) for v in x]
    if omit != "normalization":
        norm = list(map(bf16, norm))
    weighted = [fp32(a * b) for a, b in zip(norm, w)]
    if omit != "gain":
        weighted = list(map(bf16, weighted))
    result = []
    for i in range(128):
        a = fp32(weighted[i] * cos[i])
        b = fp32((-weighted[i + 64] if i < 64 else weighted[i - 64]) * sin[i])
        if omit != "rotary_products":
            a, b = bf16(a), bf16(b)
        result.append(bf16(fp32(a + b)))
    return result


def tile_schedule(pos, R, block_n, split_len, split):
    start, end = split * split_len, min((split + 1) * split_len, pos + R)
    prefix_end = min(end, pos // block_n * block_n)
    prefix = list(range(start, prefix_end, block_n))
    tail = list(range(max(start, prefix_end), end, block_n))
    return start, end, prefix, tail


def allowed(mask, n, pos, length):
    rel = n - pos
    shift = min(max(rel, 0), 63)
    return (rel < 0 or ((mask >> shift) & 1) == 1) and n < length


class FakeTensor:
    def __init__(self, shape, dtype="torch.bfloat16", row_stride=None):
        self.shape, self.dtype = tuple(shape), dtype
        self.device, self.is_cuda = "cuda:0", True
        self.row_stride = row_stride

    def is_contiguous(self):
        return self.row_stride is None

    def stride(self, axis):
        return self.row_stride if axis == 0 and self.row_stride else math.prod(self.shape[axis + 1:])


class LaunchRecorder:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def call(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
        return call


def fake_plan(R=4, config=(64, 4, 2)):
    B, HQ = 2, 32
    return types.SimpleNamespace(
        B=B, HQ=HQ, HKV=8, D=128, G=4, GP=R * 4, R=R, cap=768,
        NSPLIT=2, NSP=2, row_blocks=1, tree=True, BLOCK_N=config[0],
        SPLIT_LEN=384, num_warps=config[1], num_stages=config[2],
        scale=1 / math.sqrt(128),
        o_part=FakeTensor((B, HQ, R, 2, 128), "torch.float32"),
        m_part=FakeTensor((B, HQ, R, 2), "torch.float32"),
        l_part=FakeTensor((B, HQ, R, 2), "torch.float32"),
    )


def fake_inputs(plan):
    return [FakeTensor((plan.B * plan.R, 6144), row_stride=6208),
            FakeTensor((128,)), FakeTensor((128,)),
            FakeTensor((1024, 128)), FakeTensor((1024, 128)),
            FakeTensor((plan.B,), "torch.int32"), FakeTensor((plan.R,), "torch.int32"),
            FakeTensor((plan.B, 8, plan.cap, 128)), FakeTensor((plan.B, 8, plan.cap, 128)),
            FakeTensor((plan.B, plan.R, 32, 128)), FakeTensor((plan.R,), "torch.int64"), 1e-6]


class Controls(unittest.TestCase):
    def test_all_residues_split_boundaries_and_unique_stores(self):
        self.geometry_cases = 0
        for R, bn, split_blocks in itertools.product((4, 8), (32, 64, 128), (1, 2, 5)):
            split_len = split_blocks * bn
            for pos in range(2 * split_len - R + 1):
                owners_k, owners_v = [[] for _ in range(R)], [[] for _ in range(R)]
                all_loaded_prefix = set()
                for s in range(2):
                    start, end, prefix, tail = tile_schedule(pos, R, bn, split_len, s)
                    self.assertEqual(prefix + tail, list(range(start, end, bn)))
                    for n0 in prefix:
                        self.assertTrue(n0 + bn <= pos)
                    for n0 in prefix + tail:
                        for n in range(n0, n0 + bn):
                            if n < end and n < pos:
                                self.assertTrue(0 <= n < pos)
                                all_loaded_prefix.add(n)
                    for n0 in tail:
                        for j in range(R):
                            slot = pos + j
                            if slot >= n0 and slot < n0 + bn and slot < end:
                                owners_k[j].append((s, n0))
                        for n in range(n0, n0 + bn):
                            if n >= pos and n < end and n < pos + R:
                                owners_v[n - pos].append((s, n0))
                self.assertEqual(all_loaded_prefix, set(range(pos)))
                self.assertEqual(owners_k, owners_v)
                self.assertTrue(all(len(x) == 1 for x in owners_k))
                self.geometry_cases += 1
        self.assertEqual(self.geometry_cases, 7078)

    def test_current_cache_poison_is_never_loaded(self):
        for R, bn in itertools.product((4, 8), (32, 64, 128)):
            split_len = bn * 2
            for pos in (0, bn - 3, bn, split_len - 3, split_len, split_len + 1):
                D = 128
                # Distinct per-element sentinels expose skipped nodes, wrong
                # raw V source, accidental half interleaving, and stale reads.
                prefix_k = [[n * D + d for d in range(D)] for n in range(pos)]
                prefix_v = [[-n * D - d - 1 for d in range(D)] for n in range(pos)]
                current_k = [[1000000 + j * D + d for d in range(D)] for j in range(R)]
                current_v = [[-1000000 - j * D - d for d in range(D)] for j in range(R)]
                reference_k, reference_v = prefix_k + current_k, prefix_v + current_v
                stores = {}
                for s in range(2):
                    start, end, prefix, tail = tile_schedule(pos, R, bn, split_len, s)
                    for n0 in prefix + tail:
                        tile_k, tile_v = [], []
                        for n in range(n0, n0 + bn):
                            # A poisoned-cache accessor fails any current read.
                            def prefix_read(values):
                                self.assertLess(n, pos)
                                return values[n]
                            k = prefix_read(prefix_k) if n < end and n < pos else [0] * D
                            v = prefix_read(prefix_v) if n < end and n < pos else [0] * D
                            if n >= pos and n < end and n < pos + R:
                                k, v = current_k[n - pos], current_v[n - pos]
                                self.assertNotIn(n, stores)
                                stores[n] = (k, v)
                            tile_k.append(k)
                            tile_v.append(v)
                        expected_k = [reference_k[n] if n < end else [0] * D for n in range(n0, n0 + bn)]
                        expected_v = [reference_v[n] if n < end else [0] * D for n in range(n0, n0 + bn)]
                        self.assertEqual(tile_k, expected_k)
                        self.assertEqual(tile_v, expected_v)
                self.assertEqual(set(stores), set(range(pos, pos + R)))

    def test_tree_mask_siblings_signed_shift_and_poison_tail(self):
        for R in (4, 8):
            depth = [0, 1, 1, 2, 2, 2, 3, 3][:R]
            for pos in (0, 1, 31, 32, 33, 63, 127, 128, 129):
                self.assertEqual(pos + depth[1], pos + depth[2])
                self.assertNotEqual(pos + 1, pos + 2)
                for mask in range(1 << R):
                    for n in range(max(0, pos - 1), pos + R + 128):
                        expected = n < pos or (n < pos + R and bool(mask & (1 << (n - pos))))
                        self.assertEqual(allowed(mask, n, pos, pos + R), expected)
                # Signed-shift behavior retained for source fidelity, though
                # actual candidate scope contains only node bits 0..7.
                self.assertFalse(allowed(-(1 << 63), pos + 63, pos, pos + R))
                self.assertTrue(allowed(-(1 << 63), pos + 63, pos, pos + 64))

    def test_address_layout_padded_qkv_and_partial_bijection(self):
        for R in (4, 8):
            B, D, cap, stride = 3, 128, 768, 6208
            q_addresses, k_addresses, v_addresses = set(), set(), set()
            partial, output = set(), set()
            for b, kh, row, d in itertools.product(range(B), range(8), range(4 * R), range(D)):
                t, g = divmod(row, 4)
                h = kh * 4 + g
                q_addresses.add((b * R + t) * stride + h * D + d)
                output.add(((b * R + t) * 32 + h) * D + d)
                for s in range(2):
                    partial.add((((b * 32 + h) * R + t) * 2 + s) * D + d)
            self.assertEqual(len(q_addresses), B * R * 32 * D)
            self.assertEqual(output, set(range(B * R * 32 * D)))
            self.assertEqual(partial, set(range(B * R * 32 * 2 * D)))
            for b, kh, j, d in itertools.product(range(B), range(8), range(R), range(D)):
                k_addresses.add((b * R + j) * stride + (32 + kh) * D + d)
                v_addresses.add((b * R + j) * stride + (40 + kh) * D + d)
                p = (b * 13 + 31)
                address = ((b * 8 + kh) * cap + p + j) * D + d
                self.assertTrue(0 <= address < B * 8 * cap * D)
                self.assertEqual(address // D % cap, p + j)
            self.assertEqual(len(k_addresses), B * R * 8 * D)
            self.assertEqual(len(v_addresses), B * R * 8 * D)
            self.assertFalse(q_addresses & k_addresses or q_addresses & v_addresses or k_addresses & v_addresses)

    def test_actual_norm_source_and_ordered_halves(self):
        rng, fn = random.Random(95), source_norm()
        for GP in (16, 32):
            x = [[bf16(rng.uniform(-4, 4)) for _ in range(128)] for _ in range(GP)]
            w = [bf16(rng.uniform(-2, 2)) for _ in range(128)]
            cos = [[bf16(math.cos(rng.uniform(-4, 4))) for _ in range(128)] for _ in range(GP)]
            sin = [[bf16(math.sin(rng.uniform(-4, 4))) for _ in range(128)] for _ in range(GP)]
            trans = lambda a, start: Block([a[g][i] for i in range(start, start + 64) for g in range(GP)], (64, GP))
            actual = fn(trans(x, 0), trans(x, 64), Block(w[:64], (64, 1)), Block(w[64:], (64, 1)),
                        trans(cos, 0), trans(cos, 64), trans(sin, 0), trans(sin, 64), 1e-6, 128)
            # Logical tl.join([64,GP],[64,GP]) -> [64,GP,2], then
            # permute(1,2,0) and ordered reshape(GP,128).
            assembled = [[actual[half].at((i, g)) for half in range(2) for i in range(64)] for g in range(GP)]
            expected = [scalar_norm_reference(x[g], w, cos[g], sin[g]) for g in range(GP)]
            self.assertEqual(assembled, expected)
            wrong_interleaving = [[actual[half].at((i, g)) for i in range(64) for half in range(2)] for g in range(GP)]
            self.assertNotEqual(wrong_interleaving, expected)
            for stage in ("normalization", "gain", "rotary_products"):
                mutant = [scalar_norm_reference(x[g], w, cos[g], sin[g], omit=stage) for g in range(GP)]
                self.assertNotEqual(mutant, expected, stage)
            # Also exercise the exact source with one 128-value K head.
            for g in (0, GP - 1):
                block = lambda a, start: Block(a[start:start + 64], (64,))
                k1, k2 = fn(block(x[g], 0), block(x[g], 64), block(w, 0), block(w, 64),
                            block(cos[g], 0), block(cos[g], 64), block(sin[g], 0), block(sin[g], 64), 1e-6, 128)
                self.assertEqual(k1.values + k2.values, expected[g])

    def test_bf16_ties(self):
        self.assertEqual(bf16(1 + 1 / 256), 1.0)
        self.assertEqual(bf16(1 + 3 / 256), 1 + 2 / 128)
        self.assertEqual(bf16(-1 - 1 / 256), -1.0)
        self.assertEqual(math.copysign(1, bf16(-0.0)), -1)

    def test_empty_split_and_tail_no_visible_keys(self):
        # Scalar counterpart of the unchanged guarded max/exp recurrence.
        neginf = float("-inf")
        m, l, acc = neginf, 0.0, [0.0, 0.0]
        for scores in ([neginf] * 32, [neginf] * 32):
            m_new = max(m, max(scores))
            safe = 0 if m_new == neginf else m_new
            alpha = math.exp(m - safe)
            p = [math.exp(sc - safe) for sc in scores]
            l = l * alpha + sum(p)
            acc = [x * alpha for x in acc]
            m = m_new
        self.assertEqual((m, l, acc), (neginf, 0.0, [0.0, 0.0]))
        # Unchanged merge gives a populated split full weight and empty one 0.
        maxima, sums, accum = [2.0, neginf], [3.0, l], [[6.0, 9.0], acc]
        maximum = max(maxima)
        weights = [math.exp(x - maximum) for x in maxima]
        den = sum(w * x for w, x in zip(weights, sums))
        merged = [sum(weights[s] * accum[s][d] for s in range(2)) / den for d in range(2)]
        self.assertEqual(merged, [2.0, 3.0])

    def test_host_accepts_all_configs_without_changing_plan(self):
        spec = importlib.util.spec_from_file_location("pure_host", HOST)
        host = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(host)
        split, reduce = LaunchRecorder(), LaunchRecorder()
        modules = {"kernels.attention": types.SimpleNamespace(_reduce_kernel=reduce),
                   "kernels.tree_fused_attention_device": types.SimpleNamespace(_fused_tree_split_kernel=split)}
        with patch.dict(sys.modules, modules):
            for R, config in itertools.product((4, 8), sorted(host.SUPPORTED_CONFIGS)):
                plan = fake_plan(R, config)
                before = vars(plan).copy()
                wrapper = host.TreeFusedAttention(plan, authorized=True)
                self.assertIs(wrapper.incumbent, plan)
                args = fake_inputs(plan)
                wrapper(*args)
                self.assertEqual(vars(plan), before)
                grid, call, opts = split.calls[-1]
                self.assertEqual(grid, (plan.B, 8, 2))
                self.assertEqual((opts["BLOCK_N"], opts["num_warps"], opts["num_stages"]), config)
                self.assertEqual(opts["ROW"], 6208)
                self.assertEqual(call[13:17], (plan.cap, plan.scale, 2, plan.SPLIT_LEN))
                self.assertIs(call[10], plan.o_part)
                self.assertEqual(reduce.calls[-1][0], (plan.B, 32, R))
                self.assertIs(wrapper._reduce_kernel, reduce)
            for attr, bad in (("NSPLIT", 1), ("NSPLIT", 3), ("row_blocks", 2), ("R", 16),
                              ("GP", 32), ("tree", False), ("D", 64), ("BLOCK_N", 256), ("SPLIT_LEN", 383)):
                plan = fake_plan()
                setattr(plan, attr, bad)
                with self.assertRaises(ValueError):
                    host.TreeFusedAttention(plan, authorized=True)
            with self.assertRaises(ValueError):
                host.TreeFusedAttention(fake_plan())
            plan = fake_plan()
            wrapper = host.TreeFusedAttention(plan, authorized=True)
            plan.num_stages = 3
            with self.assertRaises(ValueError):
                wrapper(*fake_inputs(plan))

    def test_limited_source_hazards(self):
        device = ast.parse(DEVICE.read_text())
        functions = {x.name: x for x in device.body if isinstance(x, ast.FunctionDef)}
        baseline_rope = ast.parse((BASELINE / "kernels/rope.py").read_text())
        original = next(x for x in baseline_rope.body if isinstance(x, ast.FunctionDef) and x.name == "_norm_rope_row")
        self.assertEqual(ast.dump(functions["_norm_rope_row"]), ast.dump(original))
        # Actual cache loads must use the prefix-only mask, independent of
        # store scheduling. Reject accidentally introduced current cache reads.
        for name in ("_prefix_tile", "_current_tile"):
            fn = functions[name]
            mask = next(n.value for n in fn.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "kmask" for t in n.targets))
            self.assertEqual(ast.dump(mask), ast.dump(ast.parse("(n < end) & (n < pos)", mode="eval").body))
            cache_reads = []
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and ast.unparse(node.func) == "tl.load":
                    names = {n.id for n in ast.walk(node.args[0]) if isinstance(n, ast.Name)}
                    if names & {"k_ptr", "v_ptr"}:
                        cache_reads.append(node)
                        self.assertEqual(ast.unparse(next(k.value for k in node.keywords if k.arg == "mask")), "kmask[:, None]")
            self.assertEqual(len(cache_reads), 2)
        calls = [ast.unparse(x.func) for x in ast.walk(device) if isinstance(x, ast.Call)]
        self.assertNotIn("tl.gather", calls)
        self.assertNotIn("tl.cat", calls)
        self.assertNotIn("tl.debug_barrier", calls)
        self.assertFalse(any(isinstance(n, ast.BoolOp) for n in ast.walk(device)))
        split = functions["_fused_tree_split_kernel"]
        q_assignment = next(n.value for n in split.body if isinstance(n, ast.Assign)
                            and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "q")
        expected_q = ast.parse("tl.reshape(tl.permute(tl.join(q1, q2), (1, 2, 0)), (GP, D), can_reorder=False)", mode="eval").body
        self.assertEqual(ast.dump(q_assignment), ast.dump(expected_q))
        current = functions["_current_tile"]
        node_loops = [n for n in current.body if isinstance(n, ast.For)]
        self.assertEqual(len(node_loops), 1)
        self.assertEqual(ast.unparse(node_loops[0].iter), "tl.static_range(R)")
        # Reusing the incumbent merge is a host-source contract.
        host = ast.parse(HOST.read_text())
        imports = [n for n in ast.walk(host) if isinstance(n, ast.ImportFrom)]
        self.assertTrue(any(n.module == "kernels.attention" and [a.name for a in n.names] == ["_reduce_kernel"] for n in imports))
        # Recurrence statements themselves must match incumbent _tree_tile.
        baseline_attention = ast.parse((BASELINE / "kernels/attention.py").read_text())
        original_tile = next(x for x in baseline_attention.body if isinstance(x, ast.FunctionDef) and x.name == "_tree_tile")
        for variable in ("sc", "m_new", "m_safe", "alpha", "p", "l", "acc"):
            original_assignment = next(n for n in original_tile.body if isinstance(n, ast.Assign) and n.targets[0].id == variable)
            for name in ("_prefix_tile", "_current_tile"):
                assignment = next(n for n in functions[name].body if isinstance(n, ast.Assign) and n.targets[0].id == variable)
                self.assertEqual(ast.dump(assignment), ast.dump(original_assignment))


if __name__ == "__main__":
    unittest.main(verbosity=2)
