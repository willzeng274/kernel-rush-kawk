"""Checked CUDA 12.4 tensor maps for exact uint8 weight-plane loads.

Uses the already-open driver, without the pinned Triton helper's fatal assert.
Descriptor and plane ownership survive graph captures and quarantine together.
"""
import ctypes as C

import torch

from ilc_memory import ILCUnavailable


class PlaneDescriptor:
    def __init__(self, plane, tile_n, block_k):
        self.plane, self.owner = plane, plane.owner
        self.allocator = self.owner.allocator
        self.gpu = None
        if (plane.dtype != torch.uint8 or not plane.is_cuda
                or not plane.is_contiguous() or len(plane.shape) != 2
                or any(type(v) is not int or not 1 <= v <= 2 ** 32 for v in plane.shape)
                or tuple(plane.stride()) != (plane.shape[1], 1)
                or plane.shape[1] % 16 or plane.shape[1] >= 2 ** 40
                or plane.data_ptr() % 16
                or tile_n not in (64, 128) or block_k not in (128, 256)
                or torch.cuda.is_current_stream_capturing()
                or tuple(torch.cuda.get_device_capability(plane.device)) != (9, 0)):
            raise ILCUnavailable("unsupported byte-plane tensor-map metadata")
        context = C.c_void_p()
        self.allocator.call("cuCtxGetCurrent", C.byref(context))
        if context.value != self.allocator.context.value:
            raise ILCUnavailable("tensor-map CUDA context changed")
        try:
            encode = self.allocator.lib.cuTensorMapEncodeTiled
        except AttributeError as error:
            raise ILCUnavailable("CUDA tensor-map symbol unavailable") from error
        encode.restype = C.c_int
        encode.argtypes = [C.c_void_p, C.c_int, C.c_uint32, C.c_void_p,
                           C.POINTER(C.c_uint64), C.POINTER(C.c_uint64),
                           C.POINTER(C.c_uint32), C.POINTER(C.c_uint32),
                           C.c_int, C.c_int, C.c_int, C.c_int]
        self.storage = C.create_string_buffer(255)
        self.host_address = (C.addressof(self.storage) + 127) & ~127
        self.box = (tile_n, block_k)
        n, k = plane.shape
        dimensions = (C.c_uint64 * 2)(k, n)
        strides = (C.c_uint64 * 1)(k)
        # Triton 3.1's descriptor convention caps inner bytes at 128, including
        # BK256: its lowering emits multiple asynchronous copies for that tile.
        box = (C.c_uint32 * 2)(min(block_k, 128), tile_n)
        element_strides = (C.c_uint32 * 2)(1, 1)
        self.allocator.call("cuTensorMapEncodeTiled", C.c_void_p(self.host_address),
                            0, 2, C.c_void_p(plane.data_ptr()), dimensions, strides,
                            box, element_strides, 0, 3, 2, 0, fallback=(1, 801))
        self.plane_metadata = self.snapshot(plane)
        # Owners are explicitly held by the allocator until successful retire.
        # Pin BEFORE upload so a failed upload/global sync cannot release a
        # descriptor that work on another stream may still be using.
        if not hasattr(self.owner, "descriptor_owners"):
            self.owner.descriptor_owners = []
        self.owner.descriptor_owners.append(self)
        try:
            self.gpu = torch.tensor(list(C.string_at(self.host_address, 128)),
                                    dtype=torch.uint8, device=plane.device)
            if self.gpu.data_ptr() % 128 or self.gpu.numel() != 128:
                raise ILCUnavailable("unaligned GPU tensor-map storage")
            self.gpu_metadata = self.snapshot(self.gpu)
        finally:
            self.allocator.synchronize()

    @staticmethod
    def snapshot(tensor):
        return (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()),
                tensor.dtype, tensor.device, tensor.is_cuda)

    def matches(self, plane):
        return (plane is self.plane and self.owner.mapped
                and self.snapshot(plane) == self.plane_metadata
                and self.gpu is not None
                and self.snapshot(self.gpu) == self.gpu_metadata)


def verify_tma_compilation(compiled, tile_n, block_k, stages, steps=None):
    """Reject absent compiled transport/MMA/staged-byte evidence, not timing.

    These source/assembly structures establish code generation only. They do
    not prove overlap, unchanged register/shared usage, or a performance win.
    """
    import re
    assembly = getattr(compiled, "asm", {})
    ptx, ttgir = assembly.get("ptx", ""), assembly.get("ttgir", "")
    pattern = (rf"(?m)^\s*(%[A-Za-z0-9_.$-]+)\s*=\s*"
               rf"(?:triton_gpu|ttg)\.local_alloc[^\n]*?"
               rf"!tt\.memdesc<([0-9]+)x{tile_n}x{block_k}xi8\s*,")
    buffers = {name: int(depth) for name, depth in re.findall(pattern, ttgir)}
    # Two distinct plane allocations, not repeated references to one type.
    # With two selected stages, one byte ring buffer can be valid because
    # reconstruction consumes it before the later BF16 MMA operand is used.
    required_depth = max(1, stages - 1)
    loop_body = ""
    loop = ttgir.find("scf.for")
    opening = ttgir.find("{", loop) if loop >= 0 else -1
    if opening >= 0:
        depth = 1
        for end in range(opening + 1, len(ttgir)):
            depth += (ttgir[end] == "{") - (ttgir[end] == "}")
            if depth == 0:
                loop_body = ttgir[opening + 1:end]
                break
    form = "staged loop"
    if loop < 0 and type(steps) is int and 0 < steps <= stages:
        # Canonicalization can erase a loop whose entire trip count fits its
        # expanded prologue/epilogue. Retain every other structural requirement.
        loop_body, form = ttgir, "unrolled short pipeline"
    missing = []
    if ptx.count("cp.async.bulk.tensor.2d.shared::cluster.global") < 2:
        missing.append("two PTX TMA loads")
    if "wgmma.mma_async" not in ptx:
        missing.append("PTX WGMMA")
    if loop_body.count("async_tma_copy_global_to_local") < 2:
        missing.append("two loop/short-pipeline async TMA copies")
    if "memdesc_subview" not in loop_body or "wait_barrier" not in loop_body:
        missing.append("loop/short-pipeline byte views and barrier wait")
    if len(buffers) < 2 or min(buffers.values()) < required_depth:
        missing.append(f"two distinct byte buffers with ring depth >= {required_depth}")
    if missing:
        raise ILCUnavailable("compiled evidence absent: " + "; ".join(missing))
    metadata = getattr(compiled, "metadata", None)
    return {"registers": getattr(compiled, "n_regs", None),
            "spills": getattr(compiled, "n_spills", None),
            "shared_bytes": getattr(metadata, "shared", None),
            "byte_buffer_stages": min(buffers.values()), "pipeline_form": form}
