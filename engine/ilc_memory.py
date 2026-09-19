"""Owned CUDA 12.4 VMM mappings, using the already-current Torch context.

No new context, exported handles, native extension, or Torch storage ownership.
ILC changes physical traffic, never the allocation's charged byte count.
"""
import ctypes as C
import sys


class ILCUnavailable(Exception):
    """Expected unsupported-feature or memory-capacity fallback."""


class DriverError(RuntimeError):
    pass


class Location(C.Structure):
    _fields_ = [("type", C.c_int), ("id", C.c_int)]


class AllocationFlags(C.Structure):
    _fields_ = [("compressionType", C.c_ubyte),
                ("gpuDirectRDMACapable", C.c_ubyte),
                ("usage", C.c_ushort), ("reserved", C.c_ubyte * 4)]


class AllocationProp(C.Structure):
    _fields_ = [("type", C.c_int), ("requestedHandleTypes", C.c_int),
                ("location", Location), ("win32HandleMetaData", C.c_void_p),
                ("allocFlags", AllocationFlags)]


class AccessDesc(C.Structure):
    _fields_ = [("location", Location), ("flags", C.c_int)]


def validate_abi():
    # Verified against the actual CUDA 12.4.127 cuda.h LP64 ABI.
    expected = (8, 8, 8, 32, 8, 0, 4, 8, 16, 24, 12, 8)
    actual = (C.sizeof(C.c_void_p), C.sizeof(C.c_size_t), C.sizeof(Location),
              C.sizeof(AllocationProp), C.alignment(AllocationProp),
              AllocationProp.type.offset, AllocationProp.requestedHandleTypes.offset,
              AllocationProp.location.offset, AllocationProp.win32HandleMetaData.offset,
              AllocationProp.allocFlags.offset, C.sizeof(AccessDesc), AccessDesc.flags.offset)
    if actual != expected or C.sizeof(AllocationFlags) != 8:
        raise ILCUnavailable("unsupported CUDA VMM host ABI")


def round_up(size, granularity):
    if size <= 0 or granularity <= 0:
        raise ValueError("positive allocation and granularity required")
    return ((size + granularity - 1) // granularity) * granularity


class Pointer:
    """Triton 3.1 pointer protocol. Owner remains live independently of graphs."""
    def __init__(self, allocation, shape, dtype, device):
        self.owner = allocation
        self.shape = tuple(shape)
        self.dtype, self.device = dtype, device
        self.is_cuda = True

    def data_ptr(self):
        if not self.owner.mapped:
            raise RuntimeError("retired ILC mapping")
        return self.owner.address.value

    def stride(self, axis=None):
        strides = (self.shape[1], 1)
        return strides if axis is None else strides[axis]

    def is_contiguous(self):
        return True


class Allocation:
    # Intentionally no __del__: device graphs contain raw addresses. Accepted
    # owners live in the layout/allocator for the engine's entire process.
    # Temporary mappings are explicitly retired only after graphs are dropped
    # and all CUDA work synchronizes. On synchronization failure they stay live.
    def __init__(self, allocator, size, compressed):
        self.allocator, self.size = allocator, size
        self.compressed = compressed
        self.address = C.c_uint64(0)
        self.handle = C.c_uint64(0)
        self.reserved = self.created = self.mapped = False
        self.charged = False

    def _release_after_sync(self):
        d = self.allocator
        if self.mapped:
            d.call("cuMemUnmap", self.address, self.size)
            self.mapped = False
        if self.created:
            d.call("cuMemRelease", self.handle)
            self.created = False
        if self.reserved:
            d.call("cuMemAddressFree", self.address, self.size)
            self.reserved = False
        if self.charged:
            d.live_bytes -= self.size
            self.charged = False
        if self in d.owners:
            d.owners.remove(self)


class Allocator:
    def __init__(self, device, synchronize, mem_info, library=None):
        validate_abi()
        if library is None and not sys.platform.startswith("linux"):
            raise ILCUnavailable("CUDA driver VMM requires Linux")
        try:
            self.lib = library if library is not None else C.CDLL("libcuda.so.1")
        except OSError as error:
            raise ILCUnavailable("CUDA driver library unavailable") from error
        self.device = int(device)
        self._synchronize, self.mem_info = synchronize, mem_info
        self.quarantined = False
        self.owners, self.live_bytes, self.peak_bytes = [], 0, 0
        u64, size, integer, ptr = C.c_uint64, C.c_size_t, C.c_int, C.POINTER
        signatures = {
            "cuCtxGetCurrent": [ptr(C.c_void_p)],
            "cuCtxGetDevice": [ptr(integer)],
            "cuDeviceGetAttribute": [ptr(integer), integer, integer],
            "cuMemGetAllocationGranularity": [ptr(size), ptr(AllocationProp), integer],
            "cuMemAddressReserve": [ptr(u64), size, size, u64, u64],
            "cuMemCreate": [ptr(u64), size, ptr(AllocationProp), u64],
            "cuMemGetAllocationPropertiesFromHandle": [ptr(AllocationProp), u64],
            "cuMemMap": [u64, size, size, u64, u64],
            "cuMemSetAccess": [u64, size, ptr(AccessDesc), size],
            "cuMemUnmap": [u64, size],
            "cuMemRelease": [u64],
            "cuMemAddressFree": [u64, size],
        }
        try:
            for name, args in signatures.items():
                function = getattr(self.lib, name)
                function.restype, function.argtypes = integer, args
        except AttributeError as error:
            raise ILCUnavailable("CUDA VMM symbol unavailable") from error
        self.context = C.c_void_p()
        self.call("cuCtxGetCurrent", C.byref(self.context))
        current_device = integer()
        self.call("cuCtxGetDevice", C.byref(current_device))
        if not self.context.value or current_device.value != self.device:
            raise ILCUnavailable("expected the active Torch device/context")
        self.support = {}
        for attribute in (102, 107):
            value = integer()
            self.call("cuDeviceGetAttribute", C.byref(value), attribute, self.device,
                      fallback=(1, 801))
            self.support[attribute] = value.value
            if not value.value:
                raise ILCUnavailable(f"CUDA device attribute {attribute} unsupported")
        self.granularity = {}
        for compressed in (False, True):
            granularity = size()
            prop = self.properties(compressed)
            self.call("cuMemGetAllocationGranularity", C.byref(granularity),
                      C.byref(prop), 0, fallback=(801,))
            if not granularity.value:
                raise DriverError("driver returned zero VMM granularity")
            self.granularity[compressed] = granularity.value

    def call(self, name, *args, fallback=()):
        status = int(getattr(self.lib, name)(*args))
        if status:
            if status in fallback:
                raise ILCUnavailable(f"{name}: CUDA status {status}")
            raise DriverError(f"{name}: CUDA status {status}")

    def properties(self, compressed):
        prop = AllocationProp()  # Zero all reserved and unused fields.
        prop.type = 1  # CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location = Location(1, self.device)  # CU_MEM_LOCATION_TYPE_DEVICE
        prop.allocFlags.compressionType = int(compressed)
        return prop

    def rounded_size(self, logical_bytes, compressed):
        return round_up(logical_bytes, self.granularity[compressed])

    def has_room(self, required, scratch=0):
        # Driver free memory already includes every mapping, including those
        # invisible to Torch. live_bytes is explicit accounting, not subtracted
        # again here. Reserve a quarter of physical capacity at every extension.
        free, total = self.mem_info()
        return free - required - scratch >= total // 4

    def allocate(self, logical_bytes, compressed, scratch=0):
        context = C.c_void_p()
        self.call("cuCtxGetCurrent", C.byref(context))
        if context.value != self.context.value:
            raise ILCUnavailable("current CUDA context changed")
        size = self.rounded_size(logical_bytes, compressed)
        if not self.has_room(size, scratch):
            raise ILCUnavailable("25 percent physical reserve")
        allocation = Allocation(self, size, compressed)
        self.owners.append(allocation)
        prop = self.properties(compressed)
        try:
            self.call("cuMemAddressReserve", C.byref(allocation.address), size, 0, 0, 0,
                      fallback=(2, 801))
            allocation.reserved = True
            self.call("cuMemCreate", C.byref(allocation.handle), size,
                      C.byref(prop), 0, fallback=(2, 801))
            allocation.created = allocation.charged = True
            self.live_bytes += size
            self.peak_bytes = max(self.peak_bytes, self.live_bytes)
            granted = AllocationProp()
            self.call("cuMemGetAllocationPropertiesFromHandle", C.byref(granted),
                      allocation.handle, fallback=(801,))
            allocation.granted_compression = int(granted.allocFlags.compressionType)
            if (granted.type != 1 or granted.location.type != 1
                    or granted.location.id != self.device
                    or granted.allocFlags.compressionType != int(compressed)):
                raise ILCUnavailable("requested compression properties not granted")
            self.call("cuMemMap", allocation.address, size, 0, allocation.handle, 0,
                      fallback=(2, 801))
            allocation.mapped = True
            access = AccessDesc(Location(1, self.device), 3)  # READWRITE
            self.call("cuMemSetAccess", allocation.address, size, C.byref(access), 1,
                      fallback=(2, 801))
        except BaseException:
            # No kernel can have consumed this unpublished allocation yet.
            allocation._release_after_sync()
            raise
        return allocation

    def synchronize(self):
        if self.quarantined:
            raise DriverError("ILC owners quarantined after failed synchronization")
        try:
            self._synchronize()
        except BaseException:
            self.quarantined = True
            raise

    def retire(self, allocations):
        # Caller first destroys all temporary graphs that reference these
        # mappings. A failed sync deliberately preserves every owner/mapping.
        self.synchronize()
        for allocation in list(allocations):
            allocation._release_after_sync()
