"""Bounded native cuBLASLt search at the complete, actual prefill row count.

Uses the already validated public CUDA binding from lt_search. No weight copy,
conversion, approximation, or change to decode is introduced. The original
torch.mm is both the numerical/timing baseline and the permanent fallback.
"""
import math
import statistics
import sys
import time

import torch

from lt_search import Library, Problem, LtUnsupported, TunedLayout


WORKSPACE_BYTES = 128 * 1024 * 1024
MAX_POOL_FLOPS = 8 * 10**12


class PrefillLt:
    def __init__(self, engine, deadline):
        self.rows, self.plans = engine.prefill_rows, {}
        self.device = engine.prefill_normalized.device
        self.api, self.workspace = None, None
        deadline = min(deadline, time.monotonic() + 45.0)
        if time.monotonic() >= deadline:
            return
        try:
            self.api = Library()
        except (OSError, AttributeError, LtUnsupported) as exc:
            self._log("all", f"native: unavailable installed Lt library ({exc})")
            return
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        if time.monotonic() >= deadline or not self._room(WORKSPACE_BYTES):
            return
        try:
            self.workspace = torch.empty(WORKSPACE_BYTES, device=self.device,
                                         dtype=torch.uint8)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self._log("all", "native: workspace allocation failed")
            return
        groups = (
            ("gateup", engine.prefill_normalized, engine.prefill_gateup,
             [p[1] for p in engine.packed]),
            ("down", engine.prefill_intermediate, engine.prefill_branch,
             [l.mlp.down_proj.weight for l in engine.layers]),
            ("qkv", engine.prefill_normalized, engine.prefill_qkv,
             [p[0] for p in engine.packed]),
            ("output", engine.prefill_query, engine.prefill_branch,
             [l.self_attn.o_proj.weight for l in engine.layers]),
        )
        generator = torch.Generator(device=self.device)
        generator.manual_seed(71829)
        for name, x, output, weights in groups:
            if time.monotonic() >= deadline:
                break
            # Read-only sharing of already retained decode weight layouts. This
            # selector never allocates or changes a single model weight.
            shared = engine.native_layout.weights.get(name)
            self._select(name, x, output, weights, shared, generator, deadline)
        torch.cuda.synchronize(self.device)
        if not self.plans:
            self.workspace = None
        torch.cuda.empty_cache()

    def _log(self, name, message):
        print(f"[prefill-lt] M={self.rows} {name}: {message}",
              file=sys.stderr, flush=True)

    def _room(self, required):
        free, total = torch.cuda.mem_get_info(self.device)
        return free - required >= total // 4

    @staticmethod
    def _native(x, weight, output):
        torch.mm(x, weight.t(), out=output)

    @staticmethod
    def _indices(weights):
        return ([round(i * (len(weights) - 1) / 5) for i in range(6)]
                if len(weights) >= 6 else list(range(len(weights))))

    @staticmethod
    def _graph(call, device, deadline):
        if time.monotonic() >= deadline:
            return None
        graph = TunedLayout._graph(call, device)
        return graph if time.monotonic() < deadline else None

    @staticmethod
    def _time(graph, count, deadline):
        # Full-size prefill GEMMs can be milliseconds each. Calibrate one pool
        # traversal, then use 1--4 replays targeting at most about 8 ms/sample.
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        if time.monotonic() >= deadline:
            return None
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        elapsed = start.elapsed_time(end)
        if (time.monotonic() >= deadline or not math.isfinite(elapsed)
                or elapsed <= 0):
            return None
        repeats = max(1, min(4, int(8.0 / max(elapsed, 0.001))))
        samples = []
        for _ in range(3):
            if time.monotonic() >= deadline:
                return None
            start.record()
            for _ in range(repeats):
                graph.replay()
            end.record()
            end.synchronize()
            elapsed = start.elapsed_time(end)
            if (time.monotonic() >= deadline or not math.isfinite(elapsed)
                    or elapsed <= 0):
                return None
            samples.append(elapsed / (repeats * count))
        return statistics.median(samples) if time.monotonic() < deadline else None

    def _close(self, output, reference):
        # A BF16 reference already exists. Leave space for allclose arithmetic,
        # bool temporaries and allocator alignment before evaluating it.
        if not self._room(output.numel() * output.element_size() * 9):
            return False
        try:
            return torch.allclose(output, reference, rtol=0.01, atol=0.01)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return False

    def _select(self, name, x, output, weights, shared, generator, deadline):
        indices = self._indices(weights)
        n, k = weights[0].shape
        # Bound each timing traversal independently of undocumented shapes.
        if (time.monotonic() >= deadline
                or 2 * self.rows * n * k * len(indices) > MAX_POOL_FLOPS
                or not self._room(output.numel() * output.element_size() * 10)):
            self._log(name, "native: deadline, work, or comparison memory guard")
            return
        try:
            reference = torch.empty_like(output)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self._log(name, "native: reference allocation failed")
            return
        x.normal_(generator=generator)
        if time.monotonic() >= deadline:
            return
        native_call = lambda: [self._native(x, weights[i], output) for i in indices]
        native_graph = self._graph(native_call, self.device, deadline)
        if native_graph is None:
            return
        baseline = self._time(native_graph, len(indices), deadline)
        if baseline is None:
            return
        layouts = [(False, weights)]
        if shared is not None and len(shared) == len(weights):
            layouts.append((True, shared))
        best, best_ms, problems, examined = None, baseline * 0.95, [], 0
        try:
            for transposed, matrices in layouts:
                if time.monotonic() >= deadline:
                    break
                if not all(TunedLayout._eligible(x, matrix, output, self.rows,
                                                  n, k, transposed)
                           for matrix in matrices):
                    continue
                try:
                    problem = Problem(self.api, self.rows, n, k, transposed,
                                      self.workspace)
                    problems.append(problem)
                    algos = problem.algorithms()
                except LtUnsupported:
                    continue
                for algo in algos:
                    if time.monotonic() >= deadline:
                        break
                    examined += 1
                    self._native(x, weights[indices[0]], reference)
                    try:
                        problem.run(algo, x, matrices[indices[0]], output)
                    except LtUnsupported:
                        continue
                    if time.monotonic() >= deadline or not self._close(output, reference):
                        continue
                    call = lambda: [problem.run(algo, x, matrices[i], output) for i in indices]
                    graph = self._graph(call, self.device, deadline)
                    if graph is None:
                        return
                    trial = self._time(graph, len(indices), deadline)
                    del graph
                    if trial is None:
                        return
                    if trial < best_ms:
                        best, best_ms = (problem, algo, matrices), trial
            if best is None or time.monotonic() >= deadline:
                self._log(name, f"native: {examined} algorithms examined")
                return
            problem, algo, matrices = best
            for index, scale in ((indices[0], 0.25), (indices[-1], 1.0),
                                  (indices[-1], 4.0)):
                if time.monotonic() >= deadline:
                    return
                x.normal_(generator=generator).mul_(scale)
                self._native(x, weights[index], reference)
                problem.run(algo, x, matrices[index], output)
                if time.monotonic() >= deadline or not self._close(output, reference):
                    self._log(name, "native: independent numerical confirmation rejected")
                    return
            call = lambda: [problem.run(algo, x, matrices[i], output) for i in indices]
            custom_graph = self._graph(call, self.device, deadline)
            if custom_graph is None:
                return
            natives, customs = [], []
            for graph, timings in ((native_graph, natives), (custom_graph, customs),
                                    (custom_graph, customs), (native_graph, natives)):
                result = self._time(graph, len(indices), deadline)
                if result is None:
                    del custom_graph
                    return
                timings.append(result)
            old, new = min(natives), max(customs)
            if time.monotonic() < deadline and new < old * 0.95:
                self.plans[name] = best
                self._log(name, f"accepted {'transpose' if problem.transposed else 'original'}: "
                          f"{old * 1000:.2f} -> {new * 1000:.2f} us; {examined} examined")
            else:
                self._log(name, f"native: confirmation {old * 1000:.2f} versus {new * 1000:.2f} us")
            del custom_graph
        finally:
            # All scratch graphs have completed before host descriptors close.
            torch.cuda.synchronize(self.device)
            retained = self.plans.get(name)
            for problem in problems:
                if retained is None or retained[0] is not problem:
                    problem.close()

    def run(self, name, layer, x, weight, output):
        plan = self.plans.get(name)
        if plan is not None:
            problem, algo, matrices = plan
            if 0 <= layer < len(matrices):
                matrix = matrices[layer]
                if TunedLayout._eligible(x, matrix, output, problem.batch,
                                          problem.n, problem.k, problem.transposed):
                    problem.run(algo, x, matrix, output)
                    return
        self._native(x, weight, output)
