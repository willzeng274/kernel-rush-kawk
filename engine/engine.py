"""Retained #32 with fixed N16 output/down projections at B=1..32."""
import torch
from retained_engine import Engine as RetainedEngine
from narrow_layout import NarrowLayout

SELECTED_FAMILIES = ('output', 'down')


class Engine(RetainedEngine):
    def __init__(self, model_path):
        self._retained_layout = self._narrow_layout = None
        self._direct_failed = False
        super().__init__(model_path)

    def _healthy(self):
        if self._direct_failed:
            raise RuntimeError('direct narrow engine previously failed')

    def _drain(self):
        try:
            torch.cuda.synchronize()
        except BaseException:
            self._direct_failed = True
            raise

    def _allocate(self, batch, prompt, output):
        self._healthy()
        try:
            # A closed generator can have queued a following chunk. Drain
            # before replacing either old graph storage or adapter bindings.
            self._drain()
            if self._retained_layout is not None:
                self.native_layout = self._retained_layout
            self._narrow_layout = None
            super()._allocate(batch, prompt, output)
            self._retained_layout = self.native_layout
            if 1 <= batch <= 32:
                self._narrow_layout = NarrowLayout(self, self._retained_layout, SELECTED_FAMILIES)
                self.native_layout = self._narrow_layout
        except BaseException:
            # Base allocation publishes shape before all preparation finishes.
            # An error must never turn a same-shape retry into retained decode.
            self._direct_failed = True
            raise

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
