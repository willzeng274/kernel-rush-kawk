"""Pure host adapter; imports no Torch/Triton until explicitly authorized.

Proposed installation location: kernels/tree_fused_attention.py plus its device
module. This source preparation does not install or select the adapter.
"""

import math


SUPPORTED_CONFIGS = frozenset({
    (64, 4, 2), (128, 4, 2), (128, 8, 3), (128, 4, 3),
    (64, 4, 3), (64, 8, 3), (64, 8, 2), (32, 4, 3),
})


class TreeFusedAttention:
    """Borrow a previously selected DecodeAttention plan without retuning it.

    `authorized=True` is supplied by the external host selector for its bounded
    evaluation or selected installation. The caller serializes all incumbent
    and fused calls on one stream and retains both owners through graph drain.
    The owner is retained here; no tensor is allocated or reconfigured.

    All inputs reside on the same CUDA device. QKV is BF16 [B*R,6144], with
    unit column stride (row padding is allowed); weights are contiguous BF16
    [128]; cos/sin contiguous BF16 [table_length,128]. Position is contiguous
    int32 [B]; depth int32 [R]; tree int64 [R]; cache BF16 [B,8,cap,128]; output
    contiguous BF16 [B,R,32,128]. QKV/weights/tables/metadata are immutable during
    the call and do not alias cache/output/partials. All writable tensors are
    pairwise disjoint. Require 0<=pos[b], pos[b]+R<=cap, pos[b]+R<=2*SPLIT_LEN,
    and 0<=pos[b]+depth[j]<table_length; tree masks include ancestor/self bits.
    Dynamic values are caller invariants; this wrapper never copies them to CPU.

    Mutates only cache slots pos[b]+j, the incumbent partials, and output.
    No Q buffer is produced. Cache contents at >=pos are never consumed.
    """

    def __init__(self, incumbent, *, authorized=False):
        if authorized is not True:
            raise ValueError("external host selector must explicitly authorize fusion")
        required = {
            "HQ": 32, "HKV": 8, "D": 128, "G": 4, "NSPLIT": 2,
            "NSP": 2, "row_blocks": 1, "tree": True,
        }
        for name, expected in required.items():
            if getattr(incumbent, name, None) != expected:
                raise ValueError(f"unsupported incumbent {name}")
        if incumbent.R not in (4, 8) or incumbent.GP != 4 * incumbent.R:
            raise ValueError("fusion requires R4/GP16 or R8/GP32")
        config = (incumbent.BLOCK_N, incumbent.num_warps, incumbent.num_stages)
        if config not in SUPPORTED_CONFIGS:
            raise ValueError("unsupported incumbent attention config")
        if incumbent.B < 1 or incumbent.cap < incumbent.R:
            raise ValueError("invalid incumbent batch/capacity")
        if incumbent.SPLIT_LEN < incumbent.BLOCK_N or incumbent.SPLIT_LEN % incumbent.BLOCK_N:
            raise ValueError("incumbent split must preserve aligned tile intervals")
        if incumbent.scale != 1.0 / math.sqrt(128):
            raise ValueError("unsupported incumbent attention scale")
        self.incumbent = incumbent
        for name in (
            "B", "HQ", "HKV", "D", "G", "GP", "R", "cap", "NSPLIT",
            "NSP", "BLOCK_N", "SPLIT_LEN", "num_warps", "num_stages", "scale",
            "o_part", "m_part", "l_part",
        ):
            setattr(self, name, getattr(incumbent, name))
        self._geometry = self._geometry_key(incumbent)
        self._check_tensor(self.o_part, (self.B, 32, self.R, 2, 128), "torch.float32")
        self._check_tensor(self.m_part, (self.B, 32, self.R, 2), "torch.float32")
        self._check_tensor(self.l_part, (self.B, 32, self.R, 2), "torch.float32")
        # Keep the incumbent reduction kernel itself; no copied merge path.
        from kernels.attention import _reduce_kernel
        from kernels.tree_fused_attention_device import _fused_tree_split_kernel
        self._split_kernel = _fused_tree_split_kernel
        self._reduce_kernel = _reduce_kernel

    @staticmethod
    def _geometry_key(plan):
        return tuple(getattr(plan, name) for name in (
            "B", "HQ", "HKV", "D", "G", "GP", "R", "cap", "NSPLIT",
            "NSP", "BLOCK_N", "SPLIT_LEN", "num_warps", "num_stages", "scale",
            "row_blocks", "tree",
        ))

    @staticmethod
    def _check_tensor(tensor, shape, dtype, *, contiguous=True):
        if tuple(tensor.shape) != shape or str(tensor.dtype) != dtype:
            raise ValueError(f"tensor must have shape {shape}, dtype {dtype}")
        if not tensor.is_cuda or (contiguous and not tensor.is_contiguous()):
            raise ValueError("tensor must be CUDA and have the documented layout")

    def __call__(self, qkv, q_norm_w, k_norm_w, cos, sin, pos, depth,
                 k_cache, v_cache, out, tree, eps):
        if self._geometry_key(self.incumbent) != self._geometry:
            raise ValueError("incumbent geometry changed after fusion authorization")
        if (self.incumbent.o_part is not self.o_part or
                self.incumbent.m_part is not self.m_part or self.incumbent.l_part is not self.l_part):
            raise ValueError("incumbent partial-buffer ownership changed")
        self._check_tensor(qkv, (self.B * self.R, 6144), "torch.bfloat16", contiguous=False)
        if qkv.stride(1) != 1 or qkv.stride(0) < 6144:
            raise ValueError("QKV must have unit columns and nonoverlapping padded rows")
        for weight in (q_norm_w, k_norm_w):
            self._check_tensor(weight, (128,), "torch.bfloat16")
        if len(cos.shape) != 2 or cos.shape[0] < self.R:
            raise ValueError("invalid rotary tables")
        for table in (cos, sin):
            self._check_tensor(table, (cos.shape[0], 128), "torch.bfloat16")
        self._check_tensor(pos, (self.B,), "torch.int32")
        self._check_tensor(depth, (self.R,), "torch.int32")
        self._check_tensor(tree, (self.R,), "torch.int64")
        for cache in (k_cache, v_cache):
            self._check_tensor(cache, (self.B, 8, self.cap, 128), "torch.bfloat16")
        self._check_tensor(out, (self.B, self.R, 32, 128), "torch.bfloat16")
        tensors = (qkv, q_norm_w, k_norm_w, cos, sin, pos, depth, tree,
                   k_cache, v_cache, out, self.o_part, self.m_part, self.l_part)
        if any(t.device != qkv.device for t in tensors):
            raise ValueError("all tensors must share one CUDA device")
        self._split_kernel[(self.B, self.HKV, self.NSPLIT)](
            qkv, q_norm_w, k_norm_w, cos, sin, pos, depth, tree,
            k_cache, v_cache, self.o_part, self.m_part, self.l_part,
            self.cap, self.scale, self.NSPLIT, self.SPLIT_LEN, eps,
            ROW=qkv.stride(0), HQ=self.HQ, HKV=self.HKV, G=self.G,
            GP=self.GP, R=self.R, D=self.D, BLOCK_N=self.BLOCK_N,
            num_warps=self.num_warps, num_stages=self.num_stages,
        )
        self._reduce_kernel[(self.B, self.HQ, self.R)](
            self.o_part, self.m_part, self.l_part, out, self.NSPLIT,
            HQ=self.HQ, R=self.R, D=self.D, NSP=self.NSP, num_warps=1,
        )
