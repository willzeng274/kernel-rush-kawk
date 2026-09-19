"""Choose a native cuBLAS weight layout once, before decode graph capture.

Both choices compute BF16 X @ W.T with torch.mm and BF16 output. Original
weights stay in their model/prefill layout. Only validated, measured winners
retain a second, contiguous [K,N] representation for decode.
"""

import statistics
import sys
import time

import torch


class NativeLayout:
    def __init__(self, engine, deadline):
        self.batch = engine.batch
        self.weights = {}
        self.extra_bytes = 0
        self.device = engine.normalized.device
        # Four binary choices, with no compilation search. Stop adding work
        # after 15 seconds even when the overall warmup budget is larger.
        deadline = min(deadline, time.monotonic() + 15.0)
        groups = (
            ("gateup", engine.normalized, engine.gateup,
             [pair[1] for pair in engine.packed]),
            ("down", engine.intermediate, engine.branch,
             [layer.mlp.down_proj.weight for layer in engine.layers]),
            ("qkv", engine.normalized, engine.qkv,
             [pair[0] for pair in engine.packed]),
            ("output", engine.attention, engine.branch,
             [layer.self_attn.o_proj.weight for layer in engine.layers]),
        )
        generator = torch.Generator(device=self.device)
        generator.manual_seed(12423)
        for name, template, output, weights in groups:
            if time.monotonic() >= deadline:
                break
            # _allocate has already created this workload's full KV cache.
            # Release unused allocator blocks before reading physical free
            # memory. Live model/cache tensors and accepted layouts survive.
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info(self.device)
            required = sum(w.numel() * w.element_size() for w in weights)
            if (self.extra_bytes + required > total // 10
                    or free - required < total // 4):
                self._log(name, "native: full transpose exceeds memory budget")
                continue

            # Six actual layers exceed L2 even for the smallest projection.
            # If selected, retain these copies and add the remaining layers,
            # avoiding a temporary duplicate of the accepted weight pool.
            alternate = []
            if not self._extend(weights[:6], alternate, deadline):
                self._discard(alternate)
                self._log(name, "native: transpose allocation skipped")
                continue
            accepted, native_ms, alternate_ms = self._compare(
                template, output, weights[:6], alternate, generator)
            if accepted and time.monotonic() < deadline:
                accepted = self._extend(weights, alternate, deadline)
                torch.cuda.synchronize(self.device)
                free, total = torch.cuda.mem_get_info(self.device)
                accepted = accepted and free >= total // 4
            else:
                accepted = False
            if accepted:
                self.weights[name] = alternate
                self.extra_bytes += required
            else:
                self._discard(alternate)
            choice = "contiguous transpose" if accepted else "native"
            if native_ms is None:
                self._log(name, f"{choice}: numerical check rejected alternate")
            else:
                self._log(name, f"{choice}; native {native_ms * 1000:.2f} us, "
                          f"alternate {alternate_ms * 1000:.2f} us; "
                          f"extra {self.extra_bytes / 2**30:.2f} GiB")
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()

    def _log(self, name, message):
        print(f"[native-layout] B={self.batch} {name}: {message}",
              file=sys.stderr, flush=True)

    def _extend(self, weights, alternate, deadline):
        for weight in weights[len(alternate):]:
            if time.monotonic() >= deadline:
                return False
            free, total = torch.cuda.mem_get_info(self.device)
            size = weight.numel() * weight.element_size()
            if free - size < total // 4:
                return False
            try:
                alternate.append(weight.t().contiguous())
            except torch.cuda.OutOfMemoryError:
                return False
        return True

    def _discard(self, alternate):
        torch.cuda.synchronize(self.device)
        alternate.clear()
        torch.cuda.empty_cache()

    @staticmethod
    def _graph(x, matrices, output):
        current = torch.cuda.current_stream(x.device)
        stream = torch.cuda.Stream(device=x.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for matrix in matrices:
                torch.mm(x, matrix, out=output)
        current.wait_stream(stream)
        torch.cuda.synchronize(x.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for matrix in matrices:
                torch.mm(x, matrix, out=output)
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
            for _ in range(32):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / (32 * count))
        return statistics.median(samples)

    @classmethod
    def _compare(cls, template, output, weights, alternate, generator):
        x = torch.randn(template.shape, dtype=template.dtype,
                        device=template.device, generator=generator)
        # Check both ends of the rotated pool with independent activations.
        # This is a smoke check; full-model replay validates emitted tokens.
        for index in (0, len(weights) - 1):
            probe = torch.randn(template.shape, dtype=template.dtype,
                                device=template.device, generator=generator)
            reference = torch.mm(probe, weights[index].t())
            torch.mm(probe, alternate[index], out=output)
            if not torch.allclose(output, reference, rtol=0.01, atol=0.01):
                return False, None, None
        native_graph = cls._graph(x, [w.t() for w in weights], output)
        alternate_graph = cls._graph(x, alternate, output)
        # ABBA balances timing order. Require the slower alternate median to
        # beat the faster native median by >5%, with both layouts held live.
        native = [cls._time(native_graph, len(weights))]
        other = [cls._time(alternate_graph, len(weights))]
        other.append(cls._time(alternate_graph, len(weights)))
        native.append(cls._time(native_graph, len(weights)))
        native_ms, alternate_ms = min(native), max(other)
        return alternate_ms < native_ms * 0.95, native_ms, alternate_ms

    def run(self, name, layer, x, weight, output):
        matrices = self.weights.get(name) if x.shape[0] == self.batch else None
        matrix = matrices[layer] if matrices is not None else weight.t()
        torch.mm(x, matrix, out=output)
