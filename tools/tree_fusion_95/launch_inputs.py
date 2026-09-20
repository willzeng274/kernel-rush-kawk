"""Execute frozen incumbent construction and wrapper launches with CPU stand-ins."""
from __future__ import annotations

import ast
import math
import sys
import types
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
CASES = ['gp16_basic', 'gp32_large', 'gp32_small_tile']
FIXTURES = [
    dict(case=CASES[0], B=16, R=4, block_n=64, num_warps=4, num_stages=2),
    dict(case=CASES[1], B=8, R=8, block_n=128, num_warps=8, num_stages=3),
    dict(case=CASES[2], B=8, R=8, block_n=32, num_warps=4, num_stages=3),
]


class Tensor:
    next_pointer = 0x100000

    def __init__(self, shape, dtype, device):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        self.is_cuda = True
        self.pointer = Tensor.next_pointer
        Tensor.next_pointer += 0x10000000

    def data_ptr(self):
        return self.pointer

    def is_contiguous(self):
        return True

    def stride(self, axis):
        return math.prod(self.shape[axis + 1:])

    def descriptor(self):
        return dict(shape=list(self.shape), dtype=self.dtype, device=self.device,
                    contiguous=True, aligned_to_16=self.pointer % 16 == 0)


class TorchStandIn:
    Tensor = Tensor
    bfloat16 = 'torch.bfloat16'
    float32 = 'torch.float32'
    int32 = 'torch.int32'
    int64 = 'torch.int64'

    @staticmethod
    def empty(shape, *, dtype, device):
        return Tensor(shape, dtype, device)


class LaunchRecorder:
    def __init__(self, name, calls):
        self.name, self.calls = name, calls

    def __getitem__(self, grid):
        def capture(*args, **kwargs):
            self.calls.append(dict(kernel=self.name, grid=list(grid), args=args, kwargs=kwargs))
        return capture


def capture_launches():
    incumbent_source = HERE / 'snapshots/engine/kernels/attention.py'
    incumbent_class = next(n for n in ast.parse(incumbent_source.read_text()).body
                           if isinstance(n, ast.ClassDef) and n.name == 'DecodeAttention')
    namespace = dict(math=math, torch=TorchStandIn,
                     triton=types.SimpleNamespace(cdiv=lambda x, y: (x + y - 1) // y,
                                                 next_power_of_2=lambda x: 1 << (x - 1).bit_length()))
    exec(compile(ast.Module(body=[incumbent_class], type_ignores=[]), str(incumbent_source), 'exec'), namespace)
    host = HERE / 'snapshots/preparation/tree_fused_attention.py'
    host_namespace = {}
    exec(compile(host.read_text(), str(host), 'exec'), host_namespace)
    calls = []
    modules = {
        'kernels': types.ModuleType('kernels'),
        'kernels.attention': types.SimpleNamespace(_reduce_kernel=LaunchRecorder('_reduce_kernel', calls)),
        'kernels.tree_fused_attention_device': types.SimpleNamespace(
            _fused_tree_split_kernel=LaunchRecorder('_fused_tree_split_kernel', calls)),
    }
    with patch.dict(sys.modules, modules):
        for fixture in FIXTURES:
            case, B, R = fixture['case'], fixture['B'], fixture['R']
            owner = namespace['DecodeAttention'](
                B, 32, 8, 128, 768, 'cuda:0', nsplit=2, R=R, tree=True,
                block_n=fixture['block_n'], num_warps=fixture['num_warps'], num_stages=fixture['num_stages'],
            )
            before = vars(owner).copy()
            assert owner.NSPLIT == 2 and owner.SPLIT_LEN == 384 and owner.row_blocks == 1
            fused = host_namespace['TreeFusedAttention'](owner, authorized=True)
            bf16 = TorchStandIn.bfloat16
            qkv = Tensor((B * R, 6144), bf16, 'cuda:0')
            qw, kw = [Tensor((128,), bf16, 'cuda:0') for _ in range(2)]
            cos, sin = [Tensor((768, 128), bf16, 'cuda:0') for _ in range(2)]
            pos = Tensor((B,), TorchStandIn.int32, 'cuda:0')
            depth = Tensor((R,), TorchStandIn.int32, 'cuda:0')
            tree = Tensor((R,), TorchStandIn.int64, 'cuda:0')
            k, v = [Tensor((B, 8, 768, 128), bf16, 'cuda:0') for _ in range(2)]
            out = Tensor((B, R, 32, 128), bf16, 'cuda:0')
            start = len(calls)
            fused(qkv, qw, kw, cos, sin, pos, depth, k, v, out, tree, 1e-6)
            assert vars(owner) == before and len(calls) == start + 2
            first, second = calls[-2:]
            assert first['kernel'] == '_fused_tree_split_kernel' and second['kernel'] == '_reduce_kernel'
            assert first['args'][10] is second['args'][0] is owner.o_part
            assert first['args'][11] is second['args'][1] is owner.m_part
            assert first['args'][12] is second['args'][2] is owner.l_part
            assert second['args'][3] is out and fused.incumbent is owner
            assert len({x.data_ptr() for x in (qkv, qw, kw, cos, sin, pos, depth, tree, k, v, out,
                                              owner.o_part, owner.m_part, owner.l_part)}) == 14
            first.update(case=case, compile_requested=True)
            second.update(case=case + '_reduce_reused', compile_requested=False)
    assert len(calls) == 6
    return calls


def serialize(value):
    return value.descriptor() if isinstance(value, Tensor) else value


def launch_receipt(calls):
    return [dict(case=c['case'], kernel=c['kernel'], compile_requested=c['compile_requested'],
                 grid=c['grid'], args=[serialize(v) for v in c['args']], kwargs=c['kwargs']) for c in calls]
