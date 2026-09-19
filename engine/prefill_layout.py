"""Choose native BF16 cuBLAS weight layouts for the actual prefill row count.

Decode decisions and their retained matrices are read-only. A prefill winner
can share those matrices; otherwise only this selector owns its new copies.
All selection ends before model graph capture, and dispatch never retunes.
"""

import statistics
import sys
import time

import torch

from native_layout import NativeLayout


class PrefillNativeLayout(NativeLayout):
    def __init__(self, engine, deadline):
        self.rows = engine.prefill_rows
        self.device = engine.prefill_normalized.device
        self.weights = {}
        # Count only new prefill copies. Shared decode matrices count once.
        self.extra_bytes = 0
        self.decode_extra_bytes = engine.native_layout.extra_bytes
        deadline = min(deadline, time.monotonic() + 15.0)
        groups = (
            ("gateup", engine.prefill_normalized, engine.prefill_gateup,
             [pair[1] for pair in engine.packed]),
            ("down", engine.prefill_intermediate, engine.prefill_branch,
             [layer.mlp.down_proj.weight for layer in engine.layers]),
            ("qkv", engine.prefill_normalized, engine.prefill_qkv,
             [pair[0] for pair in engine.packed]),
            ("output", engine.prefill_query, engine.prefill_branch,
             [layer.self_attn.o_proj.weight for layer in engine.layers]),
        )
        generator = torch.Generator(device=self.device)
        generator.manual_seed(72419)
        for name, x, output, weights in groups:
            if time.monotonic() >= deadline:
                break
            shared = engine.native_layout.weights.get(name)
            required = (sum(w.numel() * w.element_size() for w in weights)
                        if shared is None else 0)
            # All persistent prompt buffers and KV cache already exist. The
            # reference is the sole additional full-M activation allocation.
            reference_bytes = output.numel() * output.element_size()
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info(self.device)
            if (self.decode_extra_bytes + self.extra_bytes + required > total // 10
                    or free - required - reference_bytes < total // 4):
                self._log(name, "native: transpose/reference exceeds memory budget")
                continue
            try:
                reference = torch.empty_like(output)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self._log(name, "native: reference allocation skipped")
                continue

            # A shared list belongs to decode and must never be extended or
            # cleared, including on numerical/timing rejection.
            alternate = shared if shared is not None else []
            if shared is None and not self._extend(weights[:6], alternate, deadline):
                del reference
                self._discard(alternate)
                self._log(name, "native: transpose allocation skipped")
                continue
            accepted, native_ms, alternate_ms, reason = self._compare_prefill(
                x, output, reference, weights[:6], alternate[:6], generator, deadline)
            del reference
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            if accepted and time.monotonic() < deadline:
                if shared is None:
                    accepted = self._extend(weights, alternate, deadline)
                torch.cuda.synchronize(self.device)
                free, total = torch.cuda.mem_get_info(self.device)
                accepted = accepted and free >= total // 4
            else:
                accepted = False
            if accepted:
                self.weights[name] = alternate
                self.extra_bytes += required
            elif shared is None:
                self._discard(alternate)
            choice = "contiguous transpose" if accepted else "native"
            timing = (f"native {native_ms * 1000:.2f} us, "
                      f"alternate {alternate_ms * 1000:.2f} us"
                      if native_ms is not None else reason)
            self._log(name, f"{choice}; {timing}; "
                      f"combined extra {(self.decode_extra_bytes + self.extra_bytes) / 2**30:.2f} GiB")
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()

    def _log(self, name, message):
        print(f"[prefill-layout] M={self.rows} {name}: {message}",
              file=sys.stderr, flush=True)

    @staticmethod
    def _time_prefill(graph, count, deadline):
        # Large prefill GEMMs need far fewer replays than small decode GEMMs.
        # Do not inherit NativeLayout._time's 32-replay timing loop.
        samples = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(3):
            if time.monotonic() >= deadline:
                return None
            start.record()
            for _ in range(4):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / (4 * count))
        return statistics.median(samples)

    @classmethod
    def _compare_prefill(cls, x, output, reference, weights, alternate,
                         generator, deadline):
        # Reuse not-yet-initialized persistent prefill scratch. Full-M GEMMs
        # ensure numerical probes use the actual cuBLAS problem dimensions.
        for index in (0, len(weights) - 1):
            if time.monotonic() >= deadline:
                return False, None, None, "search deadline reached"
            x.normal_(generator=generator)
            torch.mm(x, weights[index].t(), out=reference)
            torch.mm(x, alternate[index], out=output)
            if not torch.allclose(output, reference, rtol=0.01, atol=0.01):
                return False, None, None, "numerical check rejected alternate"
        if time.monotonic() >= deadline:
            return False, None, None, "search deadline reached"
        x.normal_(generator=generator)
        native_graph = cls._graph(x, [w.t() for w in weights], output)
        if time.monotonic() >= deadline:
            return False, None, None, "search deadline reached"
        alternate_graph = cls._graph(x, alternate, output)
        # Six real layer matrices rotate in each graph. ABBA balances order;
        # the slower alternate must beat the faster native median by >5%.
        measurements = []
        for graph in (native_graph, alternate_graph, alternate_graph, native_graph):
            value = cls._time_prefill(graph, len(weights), deadline)
            if value is None:
                return False, None, None, "search deadline reached"
            measurements.append(value)
        native_ms = min(measurements[0], measurements[3])
        alternate_ms = max(measurements[1], measurements[2])
        return alternate_ms < native_ms * 0.95, native_ms, alternate_ms, ""

    def run(self, name, layer, x, weight, output):
        matrices = self.weights.get(name) if x.shape[0] == self.rows else None
        matrix = matrices[layer] if matrices is not None else weight.t()
        torch.mm(x, matrix, out=output)
