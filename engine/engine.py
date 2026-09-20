"""Retained #32 with direct S2 output/down and fused residual/norm at B=1..32."""
import torch
import triton
from retained_engine import Engine as RetainedEngine
from custom_kernels import embedding_norm_kernel, swiglu_kernel
from narrow_split2 import narrow_split2_projection, _merge_residual_norm, producer_config


def signature(t):
    return (id(t), t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device)


class Engine(RetainedEngine):
    def __init__(self, model_path):
        self._split_partial = self._split_bindings = self._split_configs = None
        self._split_shared = None
        self._direct_failed = False
        super().__init__(model_path)

    def _healthy(self):
        if self._direct_failed:
            raise RuntimeError('direct split2 engine previously failed')

    def _drain(self):
        try:
            torch.cuda.synchronize()
        except BaseException:
            self._direct_failed = True
            raise

    def _allocate(self, batch, prompt, output):
        self._healthy()
        try:
            # Finish queued graphs before replacing their shared partial storage.
            self._drain()
            self._split_partial = self._split_bindings = self._split_configs = None
            self._split_shared = None
            super()._allocate(batch, prompt, output)
            if 1 <= batch <= 32:
                if self.h != 2560 or self.eps != 1e-6 or len(self.layers) != 36:
                    raise ValueError('unsupported split2 model specialization')
                self._split_partial = torch.empty((2, batch, 2560), dtype=torch.float32,
                                                  device=self.hidden.device)
                self._split_configs = {name: producer_config(batch, name)
                                       for name in ('output', 'down')}
                self._split_bindings = {}
                for name, x, weights, gains in (
                    ('output', self.attention, [l.self_attn.o_proj.weight for l in self.layers],
                     [l.post_attention_layernorm.weight for l in self.layers]),
                    ('down', self.intermediate, [l.mlp.down_proj.weight for l in self.layers],
                     [l.input_layernorm.weight for l in self.layers[1:]] + [self.base.norm.weight]),
                ):
                    k = 4096 if name == 'output' else 9728
                    for index, (weight, gain) in enumerate(zip(weights, gains)):
                        tensors = (self._split_partial, x, weight, gain, self.hidden, self.normalized)
                        shapes = ((2, batch, 2560), (batch, k), (2560, k), (2560,),
                                  (batch, 2560), (batch, 2560))
                        for j, (t, shape) in enumerate(zip(tensors, shapes)):
                            dtype = torch.float32 if j == 0 else torch.bfloat16
                            if (tuple(t.shape) != shape or t.dtype != dtype or not t.is_cuda
                                    or t.device != self.hidden.device or not t.is_contiguous()
                                    or t.data_ptr() % 16):
                                raise ValueError('invalid split2 binding')
                        regions = sorted((t.data_ptr(), t.data_ptr() + t.numel() * t.element_size())
                                         for t in tensors)
                        if any(a[1] > b[0] for a, b in zip(regions, regions[1:])):
                            raise ValueError('overlapping split2 binding')
                        self._split_bindings[name, index] = (x, weight, gain,
                                                            tuple(signature(t) for t in (x, weight, gain)))
                self._split_shared = tuple(signature(t) for t in
                                           (self._split_partial, self.hidden, self.normalized))
        except BaseException:
            # Allocation sets shape before preparation ends; retry is terminal.
            self._direct_failed = True
            raise

    def _split_boundary(self, name, index, x, weight, gain):
        expected = self._split_bindings[name, index]
        if ((x is not expected[0] or weight is not expected[1] or gain is not expected[2])
                or tuple(signature(t) for t in (x, weight, gain)) != expected[3]
                or tuple(signature(t) for t in (self._split_partial, self.hidden, self.normalized))
                   != self._split_shared):
            raise RuntimeError('stale split2 binding')
        cfg = self._split_configs[name]
        narrow_split2_projection[cfg['grid']](x, weight, self._split_partial,
            **cfg['constants'], num_warps=cfg['warps'], num_stages=cfg['stages'],
            enable_fp_fusion=cfg['fusion'])
        _merge_residual_norm[(self.batch,)](self._split_partial, self.hidden, gain, self.normalized,
            B=self.batch, H=2560, SPLITS=2, EPS=self.eps, BLOCK=4096,
            num_warps=4, num_stages=3, enable_fp_fusion=False)

    def _step(self):
        if not 1 <= self.batch <= 32:
            return super()._step()
        b = self.batch
        embedding_norm_kernel[(b,)](
            self.ids, self.base.embed_tokens.weight,
            self.layers[0].input_layernorm.weight,
            self.hidden, self.normalized, self.h, self.eps, 4096,
            num_warps=4, enable_fp_fusion=False,
        )
        for idx, layer in enumerate(self.layers):
            a, m = layer.self_attn, layer.mlp
            qkv_w, gu_w = self.packed[idx]
            self.native_layout.run("qkv", idx, self.normalized, qkv_w, self.qkv)
            self.fused_cache_attention.run(idx)
            self._split_boundary("output", idx, self.attention, a.o_proj.weight,
                                 layer.post_attention_layernorm.weight)
            self.native_layout.run("gateup", idx, self.normalized, gu_w, self.gateup)
            swiglu_kernel[(triton.cdiv(b * self.i, 1024),)](
                self.gateup, self.intermediate, self.i, b * self.i,
                num_warps=4, enable_fp_fusion=False,
            )
            next_weight = (self.layers[idx + 1].input_layernorm.weight
                           if idx + 1 < len(self.layers) else self.base.norm.weight)
            self._split_boundary("down", idx, self.intermediate, m.down_proj.weight, next_weight)
        self.native_layout.run("head", 0, self.normalized,
                               self.model.lm_head.weight, self.logits)
        torch.argmax(self.logits, dim=-1, out=self.ids)
        self.position.add_(1)

    def generate(self, input_ids, max_new_tokens):
        self._healthy()
        generator = None
        try:
            generator = super().generate(input_ids, max_new_tokens)
            yield from generator
        except Exception:
            self._direct_failed = True
            raise
        finally:
            try:
                if generator is not None:
                    generator.close()
            except Exception:
                self._direct_failed = True
                raise
            finally:
                self._drain()
