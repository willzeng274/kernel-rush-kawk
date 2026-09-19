"""Bounded dense BF16 cuBLASLt algorithm search using the installed CUDA library.

The source-only archive contains no library, binary kernel, or generated code.
This module loads libcublasLt.so.12 already supplied by the CUDA 12.4 runtime.
All layouts compute Y = X @ W.T, with BF16 inputs/output and FP32 computation.
Only NONE or COMPUTE_TYPE split-K reductions are accepted. The explicit
32-MiB workspace permits algorithms outside Torch 2.5.1's default Lt search.

Initialization is untimed warmup. Graph execution owns stable workspace and
only calls a frozen algorithm on the current Torch capture stream. Rejected,
unavailable, changed-batch, and untested operations retain NativeLayout.
"""
import ctypes as C
from pathlib import Path
import statistics
import sys
import time
import weakref

import torch


# CUDA 12.4.5.8 public cublasLt.h / cublas_api.h / library_types.h ABI.
CUDA_R_32F = 0
CUDA_R_16BF = 14
CUBLAS_COMPUTE_32F = 68
DESC_TRANSA = 3
PREF_WORKSPACE = 1
PREF_REDUCTION = 3
ALGO_REDUCTION = 3
REDUCTION_NONE = 0
REDUCTION_COMPUTE_TYPE = 2
WORKSPACE_BYTES = 32 * 1024 * 1024
MAX_HEURISTICS = 12


class Algo(C.Structure):
    _fields_ = [("data", C.c_uint64 * 8)]


class Heuristic(C.Structure):
    _fields_ = [("algo", Algo), ("workspaceSize", C.c_size_t),
                ("state", C.c_int), ("wavesCount", C.c_float),
                ("reserved", C.c_int * 4)]


class LtUnsupported(RuntimeError):
    """A documented unsupported configuration; no CUDA error is concealed."""


def check(status, operation, unsupported=()):
    if status == 0:
        return
    # These describe unsupported descriptors/algorithms, not device execution.
    if status in unsupported:
        raise LtUnsupported(f"{operation}: cuBLAS status {status}")
    raise RuntimeError(f"{operation}: cuBLAS status {status}")


def _load_library():
    # Normal dynamic-loader lookup first. CUDA Python wheels place this public
    # library under nvidia/cublas/lib; Torch distributions may place it in lib.
    candidates = ["libcublasLt.so.12"]
    candidates.append(str(Path(torch.__file__).resolve().parent / "lib" / "libcublasLt.so.12"))
    for root in sys.path:
        if root:
            candidates.append(str(Path(root) / "nvidia" / "cublas" / "lib" / "libcublasLt.so.12"))
    error = None
    for name in dict.fromkeys(candidates):
        try:
            return C.CDLL(name)
        except OSError as exc:
            error = exc
    raise OSError("installed libcublasLt.so.12 unavailable") from error


class Library:
    def __init__(self, lib=None):
        self.lib = _load_library() if lib is None else lib
        p, i, size = C.c_void_p, C.c_int, C.c_size_t
        signatures = {
            "cublasLtCreate": [C.POINTER(p)],
            "cublasLtDestroy": [p],
            "cublasLtMatmulDescCreate": [C.POINTER(p), i, i],
            "cublasLtMatmulDescDestroy": [p],
            "cublasLtMatmulDescSetAttribute": [p, i, p, size],
            "cublasLtMatrixLayoutCreate": [C.POINTER(p), i, C.c_uint64, C.c_uint64, C.c_int64],
            "cublasLtMatrixLayoutDestroy": [p],
            "cublasLtMatmulPreferenceCreate": [C.POINTER(p)],
            "cublasLtMatmulPreferenceDestroy": [p],
            "cublasLtMatmulPreferenceSetAttribute": [p, i, p, size],
            "cublasLtMatmulAlgoConfigGetAttribute": [C.POINTER(Algo), i, p, size, C.POINTER(size)],
            "cublasLtMatmulAlgoGetHeuristic": [p, p, p, p, p, p, p, i, C.POINTER(Heuristic), C.POINTER(i)],
            "cublasLtMatmulAlgoCheck": [p, p, p, p, p, p, C.POINTER(Algo), C.POINTER(Heuristic)],
            "cublasLtMatmul": [p, p, p, p, p, p, p, p, p, p, p, p, C.POINTER(Algo), p, size, p],
        }
        for name, args in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, i
        self.handle = p()
        check(self.lib.cublasLtCreate(C.byref(self.handle)), "create handle")
        self._finalizer = weakref.finalize(self, self.lib.cublasLtDestroy, self.handle)

    def create(self, name, *args):
        value = C.c_void_p()
        check(getattr(self.lib, name)(C.byref(value), *args), name)
        return value

    def attribute(self, function, desc, key, value):
        check(getattr(self.lib, function)(desc, key, C.byref(value), C.sizeof(value)), function)


class Problem:
    """Descriptors for transpose-equivalent column-major [N,B] output.

    Original row-major W[N,K] is column-major [K,N], with op(A)=T.
    Existing contiguous W.T[K,N] is column-major [N,K], with op(A)=N.
    X[B,K] is column-major [K,B]; Y[B,N] is column-major [N,B].
    There is no weight transformation, output aliasing, or BF16 intermediate.
    """
    def __init__(self, api, batch, n, k, transposed, workspace):
        self.api, self.batch, self.n, self.k = api, batch, n, k
        self.transposed, self.workspace = transposed, workspace
        self.alpha, self.beta = C.c_float(1.0), C.c_float(0.0)
        self._resources = []
        try:
            self.desc = self._make("cublasLtMatmulDescCreate", "cublasLtMatmulDescDestroy",
                                   CUBLAS_COMPUTE_32F, CUDA_R_32F)
            api.attribute("cublasLtMatmulDescSetAttribute", self.desc,
                          DESC_TRANSA, C.c_int(0 if transposed else 1))
            ar, ac, ald = (n, k, n) if transposed else (k, n, k)
            self.a = self._layout(ar, ac, ald)
            self.b = self._layout(k, batch, k)
            self.y = self._layout(n, batch, n)
        except BaseException:
            self.close()
            raise
        self._finalizer = weakref.finalize(self, Problem._release, self._resources)

    def _make(self, create, destroy, *args):
        value = self.api.create(create, *args)
        self._resources.append((getattr(self.api.lib, destroy), value))
        return value

    def _layout(self, rows, cols, ld):
        return self._make("cublasLtMatrixLayoutCreate", "cublasLtMatrixLayoutDestroy",
                          CUDA_R_16BF, rows, cols, ld)

    @staticmethod
    def _release(resources):
        while resources:
            function, value = resources.pop()
            function(value)

    def close(self):
        # Descriptors are host metadata. Callers release only after all trial
        # graphs have completed, and accepted Problems live with the engine.
        self._release(self._resources)

    def algorithms(self):
        api = self.api
        pref = api.create("cublasLtMatmulPreferenceCreate")
        try:
            api.attribute("cublasLtMatmulPreferenceSetAttribute", pref,
                          PREF_WORKSPACE, C.c_uint64(self.workspace.numel()))
            api.attribute("cublasLtMatmulPreferenceSetAttribute", pref,
                          PREF_REDUCTION, C.c_uint32(REDUCTION_COMPUTE_TYPE))
            # Persistent decode allocations and packed/selected weights all
            # satisfy this minimum. The dispatcher checks it before use.
            for attr in (5, 6, 7, 8):
                api.attribute("cublasLtMatmulPreferenceSetAttribute", pref,
                              attr, C.c_uint32(256))
            results, count = (Heuristic * MAX_HEURISTICS)(), C.c_int()
            check(api.lib.cublasLtMatmulAlgoGetHeuristic(
                api.handle, self.desc, self.a, self.b, self.y, self.y, pref,
                MAX_HEURISTICS, results, C.byref(count)), "algorithm heuristic", (7, 8, 15))
            if not 0 <= count.value <= MAX_HEURISTICS:
                raise RuntimeError("invalid cuBLASLt heuristic result count")
            accepted, seen = [], set()
            for index in range(count.value):
                result = results[index]
                if result.state != 0 or result.workspaceSize > self.workspace.numel():
                    continue
                algo = Algo.from_buffer_copy(result.algo)
                encoded = bytes(algo)
                if encoded in seen:
                    continue
                seen.add(encoded)
                reduction, written = C.c_uint32(), C.c_size_t()
                check(api.lib.cublasLtMatmulAlgoConfigGetAttribute(
                    C.byref(algo), ALGO_REDUCTION, C.byref(reduction),
                    C.sizeof(reduction), C.byref(written)), "algorithm reduction")
                if written.value != C.sizeof(reduction):
                    raise RuntimeError("invalid cuBLASLt reduction attribute width")
                if reduction.value not in (REDUCTION_NONE, REDUCTION_COMPUTE_TYPE):
                    continue
                checked = Heuristic()
                status = api.lib.cublasLtMatmulAlgoCheck(
                    api.handle, self.desc, self.a, self.b, self.y, self.y,
                    C.byref(algo), C.byref(checked))
                if status in (7, 8, 15):
                    continue
                check(status, "algorithm check")
                if checked.state == 0 and checked.workspaceSize <= self.workspace.numel():
                    accepted.append(algo)
            return accepted
        finally:
            check(api.lib.cublasLtMatmulPreferenceDestroy(pref), "destroy preference")

    def run(self, algo, x, matrix, output):
        # Use the current capture stream, never an initialization-time stream.
        stream = torch.cuda.current_stream(x.device).cuda_stream
        check(self.api.lib.cublasLtMatmul(
            self.api.handle, self.desc, C.byref(self.alpha),
            matrix.data_ptr(), self.a, x.data_ptr(), self.b,
            C.byref(self.beta), output.data_ptr(), self.y,
            output.data_ptr(), self.y, C.byref(algo),
            self.workspace.data_ptr(), self.workspace.numel(), stream), "matmul", (8, 15))


class TunedLayout:
    def __init__(self, native, engine, deadline):
        self.native, self.batch, self.plans = native, engine.batch, {}
        self.workspace, self.api = None, None
        deadline = min(deadline, time.monotonic() + 45.0)
        if time.monotonic() >= deadline:
            return
        try:
            self.api = Library()
        except (OSError, AttributeError, LtUnsupported) as exc:
            self._log("all", f"native fallback: {exc}")
            return
        torch.cuda.synchronize(engine.normalized.device)
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(engine.normalized.device)
        if free - WORKSPACE_BYTES < total // 4:
            self._log("all", "native fallback: workspace memory guard")
            return
        try:
            self.workspace = torch.empty(WORKSPACE_BYTES, device=engine.normalized.device,
                                         dtype=torch.uint8)
        except torch.cuda.OutOfMemoryError:
            self._log("all", "native fallback: workspace allocation failed")
            return
        # The high-byte operations receive search time first. Real weights,
        # caller dtype and exact decode batch are retained in every timing.
        groups = (
            ("gateup", engine.normalized, engine.gateup, [p[1] for p in engine.packed]),
            ("down", engine.intermediate, engine.branch, [l.mlp.down_proj.weight for l in engine.layers]),
            ("qkv", engine.normalized, engine.qkv, [p[0] for p in engine.packed]),
            ("output", engine.attention, engine.branch, [l.self_attn.o_proj.weight for l in engine.layers]),
            ("head", engine.normalized, engine.logits, [engine.model.lm_head.weight]),
        )
        generator = torch.Generator(device=engine.normalized.device)
        generator.manual_seed(82251)
        for name, template, output, weights in groups:
            if time.monotonic() >= deadline:
                break
            self._select(name, template, output, weights, generator, deadline)
        torch.cuda.synchronize(engine.normalized.device)
        if not self.plans:
            self.workspace = None
        torch.cuda.empty_cache()

    @property
    def weights(self):
        return self.native.weights

    @property
    def extra_bytes(self):
        return self.native.extra_bytes

    def _log(self, name, message):
        print(f"[lt-algorithm] B={self.batch} {name}: {message}", file=sys.stderr, flush=True)

    @staticmethod
    def _eligible(x, matrix, output, batch, n, k, transposed):
        shape = (k, n) if transposed else (n, k)
        return (tuple(x.shape) == (batch, k) and tuple(matrix.shape) == shape
                and tuple(output.shape) == (batch, n)
                and x.dtype == matrix.dtype == output.dtype == torch.bfloat16
                and x.device == matrix.device == output.device and x.is_cuda
                and x.is_contiguous() and matrix.is_contiguous() and output.is_contiguous()
                and all(t.data_ptr() % 256 == 0 for t in (x, matrix, output)))

    @staticmethod
    def _graph(call, device):
        current = torch.cuda.current_stream(device)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            call()
        current.wait_stream(stream)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            call()
        current.wait_stream(stream)
        graph.replay()
        torch.cuda.synchronize(device)
        return graph

    @staticmethod
    def _time(graph, count, repeats=16):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        samples = []
        for _ in range(3):
            start.record()
            for _ in range(repeats):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / (repeats * count))
        return statistics.median(samples)

    def _select(self, name, template, output, weights, generator, deadline):
        indices = ([round(i * (len(weights) - 1) / 5) for i in range(6)]
                   if len(weights) >= 6 else list(range(len(weights))))
        if time.monotonic() >= deadline:
            return
        # These existing decode activations are scratch until real prefill.
        # Prefill and every subsequent decode overwrite them before use.
        x = template
        free, total = torch.cuda.mem_get_info(template.device)
        # One BF16 reference plus conservative headroom for allclose's temporary
        # arithmetic/bool buffers. No full-model or weight copies are created.
        required = output.numel() * output.element_size() * 10
        if free - required < total // 4:
            self._log(name, "native; numerical workspace memory guard")
            return
        try:
            reference = torch.empty_like(output)
        except torch.cuda.OutOfMemoryError:
            self._log(name, "native; numerical reference allocation failed")
            return
        x.normal_(generator=generator)
        if time.monotonic() >= deadline:
            return
        native_call = lambda: [self.native.run(name, i, x, weights[i], output) for i in indices]
        native_graph = self._graph(native_call, template.device)
        if time.monotonic() >= deadline:
            return
        baseline = self._time(native_graph, len(indices))
        layouts = [(False, weights)]
        existing = self.native.weights.get(name) if self.native.batch == self.batch else None
        if existing is not None:
            layouts.append((True, existing))
        best, best_ms, candidates_seen = None, baseline * 0.95, 0
        problems = []
        try:
            for transposed, matrices in layouts:
                if time.monotonic() >= deadline:
                    break
                n, k = weights[0].shape
                if not all(self._eligible(x, matrices[i], output, self.batch, n, k, transposed)
                           for i in range(len(weights))):
                    continue
                try:
                    problem = Problem(self.api, self.batch, n, k, transposed, self.workspace)
                    problems.append(problem)
                    algos = problem.algorithms()
                except LtUnsupported as exc:
                    self._log(name, f"layout unsupported: {exc}")
                    continue
                for algo in algos:
                    if time.monotonic() >= deadline:
                        break
                    candidates_seen += 1
                    try:
                        self.native.run(name, indices[0], x, weights[indices[0]], reference)
                        problem.run(algo, x, matrices[indices[0]], output)
                    except LtUnsupported:
                        continue
                    if not torch.allclose(output, reference, rtol=0.01, atol=0.01):
                        continue
                    if time.monotonic() >= deadline:
                        break
                    call = lambda: [problem.run(algo, x, matrices[i], output) for i in indices]
                    graph = self._graph(call, template.device)
                    if time.monotonic() >= deadline:
                        del graph
                        break
                    trial_ms = self._time(graph, len(indices), repeats=8)
                    if trial_ms < best_ms:
                        best, best_ms = (problem, algo, matrices), trial_ms
                    del graph
            if best is None or time.monotonic() >= deadline:
                self._log(name, f"native; examined {candidates_seen} explicit algorithms")
                return
            problem, algo, matrices = best
            # Fresh activations and separate first/last real layers. This is a
            # smoke check, not a substitute for native teacher-forced judging.
            for idx, scale in ((indices[0], 0.25), (indices[-1], 1.0), (indices[-1], 4.0)):
                if time.monotonic() >= deadline:
                    self._log(name, "native; numerical confirmation reached deadline")
                    return
                x.normal_(generator=generator).mul_(scale)
                self.native.run(name, idx, x, weights[idx], reference)
                problem.run(algo, x, matrices[idx], output)
                if not torch.allclose(output, reference, rtol=0.01, atol=0.01):
                    self._log(name, "native; independent numerical probe rejected winner")
                    return
            if time.monotonic() >= deadline:
                return
            custom_call = lambda: [problem.run(algo, x, matrices[i], output) for i in indices]
            custom_graph = self._graph(custom_call, template.device)
            # Fresh ABBA confirmation removes optimistic search/timing bias.
            natives, others = [], []
            for graph, samples in ((native_graph, natives), (custom_graph, others),
                                   (custom_graph, others), (native_graph, natives)):
                if time.monotonic() >= deadline:
                    self._log(name, "native; timing confirmation reached deadline")
                    del custom_graph
                    return
                samples.append(self._time(graph, len(indices)))
            old, new = min(natives), max(others)
            if new < old * 0.95:
                # The captured full model uses the same matrix objects as these
                # benchmarks. Retain descriptors, algorithm, and workspace.
                self.plans[name] = (problem, algo, matrices)
                self._log(name, f"explicit Lt ({'transpose' if problem.transposed else 'original'}); "
                          f"{old * 1000:.2f} -> {new * 1000:.2f} us; {candidates_seen} examined")
            else:
                self._log(name, f"native; confirmation {old * 1000:.2f} versus {new * 1000:.2f} us")
            del custom_graph
        finally:
            # Trial graphs complete synchronously before descriptors disappear.
            torch.cuda.synchronize(template.device)
            retained = self.plans.get(name)
            for problem in problems:
                if retained is None or retained[0] is not problem:
                    problem.close()

    def run(self, name, layer, x, weight, output):
        plan = self.plans.get(name)
        if plan is not None:
            problem, algo, matrices = plan
            matrix = matrices[layer]
            if self._eligible(x, matrix, output, problem.batch, problem.n,
                              problem.k, problem.transposed):
                problem.run(algo, x, matrix, output)
                return
        self.native.run(name, layer, x, weight, output)
