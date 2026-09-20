"""All-layer family selection and full-vector/cache tests on actual native graphs."""
import math
import time
import torch
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources
from gemv_lifetime import CandidateRejected, Temporaries
from gemv_layout import FAMILIES, CONFIGS, OptionalGemvLayout, family_tensors, interval
from gemv_runtime import bind


def close(a, b, label, rtol=.02, atol=.03):
    if (not torch.isfinite(a).all() or not torch.isfinite(b).all()
            or not torch.allclose(a, b, rtol=rtol, atol=atol)):
        raise CandidateRejected('numerical mismatch: ' + label)


def bits(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def restore(e, prompts):
    c = e.gv_control
    c.healthy()
    started = time.perf_counter()
    owner = Temporaries(c)
    try:
        host = owner.hold(torch.tensor(prompts, dtype=torch.int64))
        e.prefill_input.copy_(host, non_blocking=False)
        e.prefill_graph.replay()
        e.position.fill_(e.prompt)
        c.drain()
        first = owner.hold(e.ids.clone())
        c.drain()
        elapsed = time.perf_counter() - started
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise CandidateRejected('invalid prefill recovery timing')
        c.restore = max(c.restore, elapsed)
        c.phases['restore'] = max(c.phases.get('restore', 0.), elapsed)
        if c.base:
            c.reserve()
        return first
    finally:
        owner.close()


def prefix(e, a, prompts, pos):
    if not e.prompt <= pos < e.capacity:
        raise ValueError('invalid real retained prefix')
    bind(e, a)
    restore(e, prompts)
    current = e.prompt
    chunk = a.native_chunks[4] if a.native_chunks is not None else a.chunks
    while current + 4 <= pos and 4 in chunk.graphs:
        e.gv_control.live('prefix', 5.)
        chunk.graphs[4].replay()
        e.gv_control.drain()
        e.gv_control.live()
        current += 4
    while current < pos:
        e.gv_control.live('prefix', 5.)
        e._step()
        e.gv_control.drain()
        e.gv_control.live()
        current += 1


class ProjectionGraph:
    def __init__(self, e, operation, name, fixtures, weights):
        c = self.control = e.gv_control
        c.register(self)
        self.stream = self.graph = None
        self.operation, self.fixtures, self.weights = operation, fixtures, weights
        c.live('family_compare', 5.)
        e._gv_memory_guard()
        self.stream = torch.cuda.Stream(device=e.ids.device)
        def launch(check):
            for index, ((x, y), weight) in enumerate(zip(fixtures, weights)):
                if check:
                    c.live()
                operation.run(name, index, x, weight, y)
                if check:
                    c.live()
        c.drain()
        with torch.cuda.stream(self.stream):
            launch(True)
        c.wait(self.stream.synchronize)
        c.live()
        self.graph = torch.cuda.CUDAGraph()
        c.live()
        e.gv_captures += 1
        with torch.cuda.graph(self.graph, stream=self.stream):
            launch(False)
        c.live()
        self.graph.replay()
        c.drain()
        e._gv_memory_guard()


def time_graph(e, graph):
    c = e.gv_control
    c.live('family_compare', 5.)
    owner = Temporaries(c)
    try:
        start = owner.hold(torch.cuda.Event(enable_timing=True))
        end = owner.hold(torch.cuda.Event(enable_timing=True))
        start.record()
        for _ in range(8):
            graph.replay()
        end.record()
        c.wait(end.synchronize)
        elapsed = start.elapsed_time(end) / 8
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise CandidateRejected('invalid CUDA projection timing')
        c.live()
        return elapsed
    finally:
        owner.close()


def select_family(e, native, name):
    c = e.gv_control
    c.live('family_compile', 20.)
    _, _, weights = family_tensors(e, name)
    l2 = getattr(torch.cuda.get_device_properties(e.ids.device), 'L2_cache_size', 0)
    regions = sorted(interval(w) for w in weights)
    if (len(weights) != 36 or type(l2) is not int or l2 <= 0
            or any(a[1] > b[0] for a, b in zip(regions, regions[1:]))
            or sum(b - a for a, b in regions) <= l2):
        raise CandidateRejected('invalid or small actual all-layer weight pool')
    n, k = CONFIGS[name]
    e._gv_memory_guard(extra_bytes=36 * (k + 2 * n) * 2)
    owner = Temporaries(c)
    graphs = []
    try:
        pairs = [(owner.hold(torch.empty((1, k), dtype=torch.bfloat16, device=e.ids.device)),
                  owner.hold(torch.empty((1, n), dtype=torch.bfloat16, device=e.ids.device)))
                 for _ in weights]
        refs = [owner.hold(torch.empty_like(y)) for _, y in pairs]
        port = OptionalGemvLayout(e, native, (name,), private={name: pairs})
        generator = torch.Generator(device=e.ids.device)
        generator.manual_seed(73491 + FAMILIES.index(name))
        compiled = False
        for scale in (.1, 1., 10.):
            for index, ((x, y), weight, reference) in enumerate(zip(pairs, weights, refs)):
                c.live('family_compare', 5.)
                x.normal_(generator=generator)
                x.mul_(scale)
                native.run(name, index, x, weight, reference)
                if not compiled:
                    c.live('family_compile', 20.)
                    compile_started = time.perf_counter()
                port.run(name, index, x, weight, y)
                c.drain()
                if not compiled:
                    c.observed('family_compile', time.perf_counter() - compile_started)
                    compiled = True
                c.live()
                close(y, reference, 'all layer projection channels', .01, .01)
                c.live()
        started = time.perf_counter()
        graphs.append(ProjectionGraph(e, native, name, pairs, weights))
        graphs.append(ProjectionGraph(e, port, name, pairs, weights))
        timings = [time_graph(e, graphs[index].graph) for index in (0, 1, 1, 0)]
        if not max(timings[1:3]) < .95 * min(timings[0], timings[3]):
            raise CandidateRejected('selected GEMV family gain insufficient')
        e._gv_memory_guard()
        c.observed('family_compare', time.perf_counter() - started)
    finally:
        c.drain()
        for graph in graphs:
            c.release(graph)
        graphs.clear()
        owner.close()


def select_families(e, native):
    chosen = []
    for name in FAMILIES:
        before = list(e.gv_control.owners)
        try:
            select_family(e, native, name)
            chosen.append(name)
        except (CandidateRejected, CompilationError, OutOfResources, torch.cuda.OutOfMemoryError):
            e.gv_control.drain()
            # Constructors publish before allocation; also release partial owners.
            e.gv_control.owners[:] = before
        e.gv_control.live()
    if not chosen:
        raise CandidateRejected('no selected GEMV family')
    return tuple(chosen)


class CacheSnapshot:
    def __init__(self, e):
        self.e, self.control = e, e.gv_control
        e._gv_memory_guard()
        self.owner = Temporaries(self.control)
        self.keys = [self.owner.hold(k.clone()) for k in e.keys]
        self.values = [self.owner.hold(v.clone()) for v in e.values]
        self.ids = self.owner.hold(e.ids.clone())
        self.position = self.owner.hold(e.position.clone())
        self.control.drain()
        e._gv_memory_guard()

    def restore(self):
        self.control.healthy()
        for dest, src in zip(self.e.keys + self.e.values, self.keys + self.values):
            dest.copy_(src, non_blocking=False)
        self.e.ids.copy_(self.ids, non_blocking=False)
        self.e.position.copy_(self.position, non_blocking=False)
        self.control.drain()

    def offpath(self, pos, steps):
        for actual, saved in zip(self.e.keys + self.e.values, self.keys + self.values):
            if (not bits(actual[:, :, :pos], saved[:, :, :pos])
                    or not bits(actual[:, :, pos + steps:], saved[:, :, pos + steps:])):
                raise CandidateRejected('off-path cache bytes changed')
            self.control.live()

    def close(self):
        self.owner.close()


def paired(e, a, b, prompts, pos, steps=1, graph=None, output=None):
    c = e.gv_control
    c.live('full_model', 5.)
    started = time.perf_counter()
    if pos + steps > e.capacity - 1:
        raise ValueError('native validation exceeds valid output prefix')
    prefix(e, a, prompts, pos)
    saved = CacheSnapshot(e)
    owner = Temporaries(c)
    try:
        expected_ids = []
        for _ in range(steps):
            c.live()
            e._step()
            c.drain()
            c.live()
            expected_ids.append(tuple(e.ids.tolist()))
        saved.offpath(pos, steps)
        expected = owner.hold(e.logits.clone())
        written = [owner.hold(t[:, :, pos:pos + steps].clone()) for t in e.keys + e.values]
        c.drain()
        if int(e.position.item()) != pos + steps:
            raise CandidateRejected('retained native position mismatch')
        saved.restore()
        bind(e, b)
        if graph is None:
            e._step()
            c.drain()
            actual_ids = [tuple(e.ids.tolist())]
        else:
            graph.replay()
            c.drain()
            actual_ids = [tuple(row) for row in output.tolist()]
        c.live()
        if actual_ids != expected_ids or int(e.position.item()) != pos + steps:
            raise CandidateRejected('native graph token or position mismatch')
        if tuple(e.ids.tolist()) != expected_ids[-1]:
            raise CandidateRejected('native pending ID mismatch')
        close(e.logits, expected, 'all vocabulary logits')
        if not torch.equal(e.logits.argmax(-1), expected.argmax(-1)):
            raise CandidateRejected('native greedy argmax mismatch')
        for actual, reference in zip(e.keys + e.values, written):
            close(actual[:, :, pos:pos + steps], reference, 'all-layer written KV')
            c.live()
        saved.offpath(pos, steps)
        e._gv_memory_guard()
        c.observed('full_model', time.perf_counter() - started)
    finally:
        c.drain()
        bind(e, a)
        saved.restore()
        owner.close()
        saved.close()


def validate(e, a, b, prompts):
    n = e.shape[2]
    for offset in sorted({0, min(7, n - 2), n - 2}):
        paired(e, a, b, prompts, e.prompt + offset)
    for chunk, size, graph, proof in b.graphs:
        if not proof.valid(graph, b.layout):
            raise RuntimeError('native validation lacks capture proof')
        pos = e.prompt + max(0, n - 1 - size)
        paired(e, a, b, prompts, pos, size, graph, chunk.outputs[size])
    bind(e, a)
    restore(e, prompts)
