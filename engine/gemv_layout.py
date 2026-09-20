"""Fixed original-weight GEMV bindings and evidence from actual graph capture."""
from dataclasses import dataclass
import torch
from sglang_inline_gemv import CONFIGS, make_asm, _sglang_inline_gemv

SOURCE = 'd256f2294cee9db535a7c364aa153ec1f70f5ec447e4dcc5d5969d4c79f2086a'
FAMILIES = ('qkv', 'output', 'down')


def signature(t):
    return (id(t), t.data_ptr(), tuple(t.shape), tuple(t.stride()),
            t.dtype, t.device, t.element_size())


def interval(t):
    return (t.data_ptr(), t.data_ptr() + t.numel() * t.element_size())


def family_tensors(e, name):
    if name == 'qkv':
        return e.normalized, e.qkv, [p[0] for p in e.packed]
    if name == 'output':
        return e.attention, e.branch, [l.self_attn.o_proj.weight for l in e.layers]
    if name == 'down':
        return e.intermediate, e.branch, [l.mlp.down_proj.weight for l in e.layers]
    raise ValueError('unsupported GEMV family')


def validate_binding(name, x, weight, out):
    n, k = CONFIGS[name]
    if (tuple(x.shape) != (1, k) or tuple(weight.shape) != (n, k)
            or tuple(out.shape) != (1, n)):
        raise ValueError('unsupported GEMV tensor shape')
    for t in (x, weight, out):
        if (not t.is_cuda or t.dtype != torch.bfloat16 or t.device != x.device
                or not t.is_contiguous() or t.data_ptr() % 16):
            raise ValueError('GEMV requires aligned contiguous BF16 CUDA tensors')
    regions = sorted(interval(t) for t in (x, weight, out))
    if any(a[1] > b[0] for a, b in zip(regions, regions[1:])):
        raise ValueError('GEMV tensor storage overlaps')


@dataclass(frozen=True)
class CaptureProof:
    graph: object
    layout: object
    source: str
    families: tuple
    bindings: tuple
    steps: int

    def valid(self, graph, layout):
        return (self.graph is graph and self.layout is layout and self.source == SOURCE
                and self.families == layout.families and self.steps > 0
                and self.bindings == layout.fingerprint())


class OptionalGemvLayout:
    """One immutable family choice; private probes use separately prepared bindings."""
    def __init__(self, e, native, families, private=None):
        self.native = native
        self.families = tuple(name for name in FAMILIES if name in families)
        if not self.families or set(families) != set(self.families):
            raise ValueError('invalid selected GEMV families')
        self.bindings = {}
        self.asm = {name: make_asm(CONFIGS[name][1]) for name in self.families}
        self.observer = None
        for name in self.families:
            x, out, weights = family_tensors(e, name)
            if len(weights) != 36:
                raise ValueError('expected all 36 GEMV layer weights')
            for layer, weight in enumerate(weights):
                xx, yy = (x, out) if private is None else private[name][layer]
                validate_binding(name, xx, weight, yy)
                self.bindings[name, layer] = (xx, weight, yy,
                    (signature(xx), signature(weight), signature(yy)))

    @property
    def weights(self):
        return self.native.weights

    @property
    def extra_bytes(self):
        return self.native.extra_bytes

    def fingerprint(self):
        result = []
        for key, (x, w, y, sig) in self.bindings.items():
            current = (signature(x), signature(w), signature(y))
            if current != sig:
                raise RuntimeError('prepared GEMV tensor storage changed')
            result.append((key, sig))
        return tuple(result)

    def begin_capture(self, graph, steps):
        if self.observer is not None or steps < 1:
            raise RuntimeError('nested or invalid GEMV capture')
        self.fingerprint()
        self.observer = (graph, steps, {key: 0 for key in self.bindings})

    def finish_capture(self, graph):
        observed = self.observer
        self.observer = None
        if observed is None or observed[0] is not graph:
            raise RuntimeError('foreign GEMV capture graph')
        _, steps, counts = observed
        if any(value != steps for value in counts.values()):
            raise RuntimeError('incomplete selected-family graph capture')
        return CaptureProof(graph, self, SOURCE, self.families, self.fingerprint(), steps)

    def cancel_capture(self):
        self.observer = None

    def run(self, name, layer, x, weight, out):
        if name not in self.families or x.shape[0] != 1:
            self.native.run(name, layer, x, weight, out)
            return
        binding = self.bindings.get((name, layer))
        if binding is None or (x is not binding[0] or weight is not binding[1]
                               or out is not binding[2]):
            raise RuntimeError('foreign selected GEMV binding')
        if (signature(x), signature(weight), signature(out)) != binding[3]:
            raise RuntimeError('stale selected GEMV binding')
        observed = self.observer
        if observed is not None and not torch.cuda.is_current_stream_capturing():
            raise RuntimeError('GEMV evidence requires active CUDA capture')
        n, k = CONFIGS[name]
        _sglang_inline_gemv[(n // 8,)](x, weight, out, N=n, K=k, ASM=self.asm[name],
            num_warps=8, num_stages=1, enable_fp_fusion=False)
        if observed is not None:
            observed[2][name, layer] += 1
