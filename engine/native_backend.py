"""Bounded warmup selection of dense BF16 cuBLAS versus cuBLASLt kernels.

Weights and BF16 cast boundaries are unchanged. PyTorch 2.5.1's Lt path uses
FP32 compute but does not consult allow_bf16_reduced_precision_reduction;
numerical probes are a smoke check, and full-model replay remains authoritative.
Only captured GPU kernels run during measured generations.
"""

import sys
import time

import torch


class NativeBackend:
    def __init__(self, engine, deadline):
        self.layout = engine.native_layout
        self.choices = {}
        self.device = engine.normalized.device
        self.batch = engine.batch
        self.rows = engine.prefill_rows
        self.deadline = min(deadline, time.monotonic() + 25.0)
        self.stream = None
        generator = torch.Generator(device=self.device)
        generator.manual_seed(240319)
        weights = {
            "qkv": [pair[0] for pair in engine.packed[:6]],
            "output": [layer.self_attn.o_proj.weight for layer in engine.layers[:6]],
            "gateup": [pair[1] for pair in engine.packed[:6]],
            "down": [layer.mlp.down_proj.weight for layer in engine.layers[:6]],
        }
        groups = [
            ("prefill", "gateup", engine.prefill_normalized, engine.prefill_gateup),
            ("prefill", "down", engine.prefill_intermediate, engine.prefill_branch),
            ("prefill", "qkv", engine.prefill_normalized, engine.prefill_qkv),
            ("prefill", "output", engine.prefill_query, engine.prefill_branch),
            ("decode", "gateup", engine.normalized, engine.gateup),
            ("decode", "down", engine.intermediate, engine.branch),
            ("decode", "qkv", engine.normalized, engine.qkv),
            ("decode", "output", engine.attention, engine.branch),
            ("decode", "head", engine.normalized, engine.logits),
            ("prefill", "head", engine.prefill_last_normalized, engine.logits),
        ]
        try:
            torch.backends.cuda.preferred_blas_library("cublas")
            for phase, name, scratch, output in groups:
                if self._expired():
                    self._log(phase, name, "cuBLAS: tuning deadline reached")
                    break
                if name == "head":
                    matrices = [engine.model.lm_head.weight.t()]
                elif (phase == "decode" and self.batch == self.layout.batch
                      and name in self.layout.weights):
                    matrices = self.layout.weights[name][:6]
                else:
                    matrices = [weight.t() for weight in weights[name]]
                result = self._compare(scratch, output, matrices, generator, phase, name)
                if result is None:
                    continue
                native, alternate = result
                selected = alternate < native * 0.95
                if selected:
                    self.choices[(phase, name)] = "cublaslt"
                label = "cuBLASLt" if selected else "cuBLAS"
                self._log(phase, name, f"{label}; native {native * 1000:.2f} us, "
                          f"Lt {alternate * 1000:.2f} us")
        finally:
            torch.backends.cuda.preferred_blas_library("cublas")
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()

    def _expired(self):
        return time.monotonic() >= self.deadline

    def _log(self, phase, name, message):
        print(f"[native-backend] B={self.batch} M={self.rows} {phase}/{name}: {message}",
              file=sys.stderr, flush=True)

    @staticmethod
    def _mm(backend, x, matrix, output):
        try:
            torch.backends.cuda.preferred_blas_library(backend)
            torch.mm(x, matrix, out=output)
        finally:
            torch.backends.cuda.preferred_blas_library("cublas")

    def _graph(self, backend, x, matrices, output):
        if self._expired():
            return None
        if self.stream is None:
            self.stream = torch.cuda.Stream(device=self.device)
        current = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(current)
        try:
            torch.backends.cuda.preferred_blas_library(backend)
            with torch.cuda.stream(self.stream):
                for matrix in matrices:
                    torch.mm(x, matrix, out=output)
            current.wait_stream(self.stream)
            torch.cuda.synchronize(self.device)
            if self._expired():
                return None
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.stream):
                for matrix in matrices:
                    torch.mm(x, matrix, out=output)
            graph.replay()
            torch.cuda.synchronize(self.device)
            return graph
        finally:
            torch.backends.cuda.preferred_blas_library("cublas")

    def _time(self, graph, count, repeats):
        if self._expired():
            return None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / (repeats * count)

    def _compare(self, scratch, output, matrices, generator, phase, name):
        # Existing persistent buffers supply the exact strides without a second
        # padded head allocation. All are overwritten by the following prefill.
        torch.cuda.synchronize(self.device)
        free, total = torch.cuda.mem_get_info(self.device)
        # Reserve for the BF16 reference plus temporary numerical-comparison
        # tensors; leave a quarter of physical memory free.
        required = output.numel() * output.element_size() * 4
        if free - required < total // 4:
            self._log(phase, name, "cuBLAS: numerical scratch exceeds memory budget")
            return None
        try:
            reference = torch.empty_like(output)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self._log(phase, name, "cuBLAS: reference allocation unavailable")
            return None
        for index in (0, len(matrices) - 1):
            if self._expired():
                return None
            scratch.normal_(generator=generator)
            self._mm("cublas", scratch, matrices[index], reference)
            self._mm("cublaslt", scratch, matrices[index], output)
            if not torch.allclose(output, reference, rtol=0.01, atol=0.01):
                self._log(phase, name, "cuBLAS: Lt numerical smoke check rejected")
                return None
        del reference
        native_graph = self._graph("cublas", scratch, matrices, output)
        if native_graph is None:
            return None
        alternate_graph = self._graph("cublaslt", scratch, matrices, output)
        if alternate_graph is None:
            return None
        repeats = 3 if phase == "prefill" else 16
        native, alternate = [], []
        for graph, samples in ((native_graph, native), (alternate_graph, alternate),
                               (alternate_graph, alternate), (native_graph, native)):
            elapsed = self._time(graph, len(matrices), repeats)
            if elapsed is None:
                return None
            samples.append(elapsed)
        # Conservative ABBA comparison: the slower Lt result must beat the
        # faster baseline by more than 5%, with identical weights held live.
        return min(native), max(alternate)

    def run_decode(self, name, layer, x, weight, output):
        matrices = self.layout.weights.get(name) if x.shape[0] == self.layout.batch else None
        matrix = matrices[layer] if matrices is not None else weight.t()
        self._mm(self.choices.get(("decode", name), "cublas"), x, matrix, output)

    def run_prefill(self, name, x, weight, output):
        self._mm(self.choices.get(("prefill", name), "cublas"), x, weight.t(), output)

    def run_head(self, phase, x, weight, output):
        self._mm(self.choices.get((phase, "head"), "cublas"), x, weight.t(), output)
