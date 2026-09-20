"""Complete prepared bindings and private actual-graph replay admission."""
from dataclasses import dataclass
import math
import time
from gemv_lifetime import CandidateRejected
from narrow_layout import signature, SOURCE


def buffer_fingerprint(e):
    tensors = list(e.keys) + list(e.values)
    tensors += [getattr(e, name) for name in (
        'ids', 'position', 'hidden', 'normalized', 'qkv', 'query', 'attention',
        'branch', 'gateup', 'intermediate', 'logits', 'partial', 'pmax', 'psum',
        'cos', 'sin', 'prefill_input', 'prefill_hidden', 'prefill_normalized',
        'prefill_qkv', 'prefill_query', 'prefill_branch', 'prefill_gateup',
        'prefill_intermediate')]
    tensors += [w for pair in e.packed for w in pair]
    tensors += [l.self_attn.o_proj.weight for l in e.layers]
    tensors += [l.mlp.down_proj.weight for l in e.layers]
    tensors += [e.model.lm_head.weight]
    return tuple(signature(t) for t in tensors)


def native_graphs(cfg):
    chunks = list(cfg.native_chunks.values()) if cfg.native_chunks is not None else [cfg.chunks]
    return [(chunk, size, graph) for chunk in chunks for size, graph in chunk.graphs.items()]


@dataclass(frozen=True)
class Configuration:
    shape: tuple
    generation: int
    layout: object
    attention: object
    chunks: object
    native_chunks: object
    verifier: object
    prefill: object
    dense_prefill: object
    buffers: tuple
    graphs: tuple
    outputs: tuple
    port: bool = False

    def ready(self, e):
        if (self.shape != e.shape or self.generation != e.gv_generation
                or self.prefill is not e.prefill_graph or self.dense_prefill is not e.dense_prefill
                or self.prefill is None or self.buffers != buffer_fingerprint(e)):
            return False
        if self.native_chunks is not None:
            if (set(self.native_chunks) != {1, 2, 3, 4} or self.verifier is None
                    or any(size not in chunk.graphs for size, chunk in self.native_chunks.items())):
                return False
        elif self.chunks is None or any(size not in self.chunks.graphs for size in self.chunks.schedule):
            return False
        current = native_graphs(self)
        if len(current) != len(self.graphs):
            return False
        if tuple(signature(chunk.outputs[size]) for chunk, size, _ in current) != self.outputs:
            return False
        for (chunk, size, graph), (old, old_size, old_graph, proof) in zip(current, self.graphs):
            if chunk is not old or size != old_size or graph is not old_graph:
                return False
            if self.port and (proof is None or chunk.proofs.get(size) is not proof
                              or not proof.valid(graph, self.layout) or proof.steps != size):
                return False
        return not self.port or bool(self.layout.families)


def snapshot(e, port=False):
    chunks = list(e.native_chunks.values()) if e.native_chunks is not None else [e.chunks]
    graphs = tuple((chunk, size, graph, chunk.proofs.get(size) if port else None)
                   for chunk in chunks for size, graph in chunk.graphs.items())
    return Configuration(e.shape, e.gv_generation, e.native_layout, e.fused_cache_attention,
        e.chunks, e.native_chunks, e.verifier, e.prefill_graph, e.dense_prefill,
        buffer_fingerprint(e), graphs,
        tuple(signature(chunk.outputs[size]) for chunk, size, _, _ in graphs), port)


def assert_bound(e, cfg):
    if (e.gv_bound is not cfg or e.native_layout is not cfg.layout
            or e.fused_cache_attention is not cfg.attention or e.chunks is not cfg.chunks
            or e.native_chunks is not cfg.native_chunks or e.verifier is not cfg.verifier
            or not cfg.ready(e)):
        raise RuntimeError('stale or incomplete prepared GEMV configuration')


def bind(e, cfg):
    e.gv_control.drain()
    if e.gv_timing or not cfg.ready(e):
        raise RuntimeError('invalid prepared GEMV configuration binding')
    e.native_layout, e.fused_cache_attention, e.chunks, e.native_chunks, e.verifier = (
        cfg.layout, cfg.attention, cfg.chunks, cfg.native_chunks, cfg.verifier)
    e.gv_bound = cfg


class ReplayCounter:
    def __init__(self, e, cfg, graph, proof, counter):
        self.e, self.cfg, self.graph, self.proof, self.counter = e, cfg, graph, proof, counter

    def replay(self):
        if (self.e.gv_bound is not self.cfg or not self.e.gv_timing
                or self.proof.graph is not self.graph or self.proof.layout is not self.cfg.layout
                or self.e.native_layout is not self.cfg.layout or self.proof.source != SOURCE
                or self.proof.families != self.cfg.layout.families):
            raise RuntimeError('foreign or unproved native graph replay')
        self.graph.replay()
        self.counter[0] += 1


def timed_complete(e, cfg, factory, phase):
    c = e.gv_control
    c.live(phase, 5.)
    bind(e, cfg)
    observed, counter, replaced = [], [0], False
    if cfg.port:
        for chunk, size, graph, proof in cfg.graphs:
            proxy = ReplayCounter(e, cfg, graph, proof, counter)
            observed.append((chunk, size, graph, proxy))
            chunk.graphs[size] = proxy
    captured = e.gv_captures
    e.gv_timing = True
    started = time.perf_counter()
    rows, first, gen = [], None, None
    try:
        gen = factory()
        for row in gen:
            rows.append(tuple(row))
            if first is None:
                first = time.perf_counter() - started
            c.live()
    finally:
        try:
            try:
                if gen is not None:
                    gen.close()
            finally:
                c.drain()
        finally:
            e.gv_timing = False
            for chunk, size, graph, proxy in observed:
                replaced = replaced or chunk.graphs.get(size) is not proxy
                chunk.graphs[size] = graph
    elapsed = time.perf_counter() - started
    tpot = (elapsed - first) / (e.shape[2] - 1) if first is not None else float('nan')
    if any(value is None or not math.isfinite(value) or value <= 0 for value in (elapsed, first, tpot)):
        raise CandidateRejected('invalid complete-call timing')
    if replaced or e.gv_captures != captured:
        raise RuntimeError('timed request changed graph captures or slots')
    assert_bound(e, cfg)
    c.observed(phase, elapsed)
    return tuple(rows), elapsed, first, tpot, counter[0]


def require_stream(rows, batch, output, vocab):
    if len(rows) != output or any(len(row) != batch for row in rows):
        raise CandidateRejected('complete stream shape mismatch')
    if any(type(y) is not int or not 0 <= y < vocab for row in rows for y in row):
        raise CandidateRejected('complete stream invalid ID')


def whole_call_admission(e, a, b, factory):
    runs = [timed_complete(e, cfg, factory, phase) for cfg, phase in
            ((a, 'base'), (b, 'candidate'), (b, 'candidate'), (a, 'base'))]
    for rows, _, _, _, _ in runs:
        require_stream(rows, e.batch, e.shape[2], e.model.config.vocab_size)
    if not all(run[0] == runs[0][0] for run in runs[1:]):
        raise CandidateRejected('complete greedy stream mismatch')
    if not (runs[1][4] > 0 and runs[2][4] > 0):
        raise CandidateRejected('no actual proved GEMV native replay')
    if not max(runs[1][1], runs[2][1]) < min(runs[0][1], runs[3][1]):
        raise CandidateRejected('no robust complete-call improvement')
    for metric in (2, 3):
        if not max(runs[1][metric], runs[2][metric]) <= 1.05 * max(runs[0][metric], runs[3][metric]):
            raise CandidateRejected('complete-call TTFT or TPOT regression')
