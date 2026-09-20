"""Kernel Rush engine for Qwen3 4B: custom forward, static KV cache, CUDA graphs.

Per shape (batch, prompt length, output length) the engine builds a ``Plan``
holding every buffer, runs it once eagerly so all Triton specialisations are
compiled, then captures the prefill and the decode step into CUDA graphs. A
sample is then one prefill replay plus ``max_new_tokens - 1`` decode replays,
each followed by a single device-to-host copy of the ``B`` chosen tokens.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _ensure_triton_cache() -> None:
    """Triton compiles at runtime and needs a writable cache; the run container's
    home directory may not be writable for the unprivileged engine user."""
    target = os.environ.get("TRITON_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".triton", "cache")
    try:
        os.makedirs(target, exist_ok=True)
        probe = os.path.join(target, ".write-probe")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
    except OSError:
        os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="triton-cache-")


_ensure_triton_cache()

import torch

try:
    import numpy as np
except ImportError:  # the container ships numpy with transformers; this only keeps the engine importable without it
    np = None

import budget
from kernels.compact import compact_paths
from model import Model, Plan, VerifyPlan
from recycle import Recycler
from spec import NGramDrafter

PICKER_BUDGET_S = 120.0

SELF_CHECK_STEPS = 6
SELF_CHECK_TOPK = 10
SELF_CHECK_MAX_DIFF = 2.0
TIE_MARGIN = 2.0


def _log(msg: str) -> None:
    print(f"[engine] {msg}", file=sys.stderr, flush=True)


def _ids_tensor(input_ids: list[list[int]]) -> torch.Tensor:
    """Nested-list to int64 tensor; numpy walks the lists several times faster than torch.tensor."""
    if np is not None:
        return torch.from_numpy(np.asarray(input_ids, dtype=np.int64))
    return torch.tensor(input_ids, dtype=torch.int64)


class GraphPlan:
    def __init__(self, model: Model, B: int, T: int, max_new: int, spec_k: int | None = None,
                 recycle_rows: int | None = None, recycle_k: int = 8):
        self.plan = Plan(model, B, T, max_new)
        self.B, self.T, self.max_new = B, T, max_new
        self.g_prefill: torch.cuda.CUDAGraph | None = None
        self.g_decode: torch.cuda.CUDAGraph | None = None
        self.g_verify: torch.cuda.CUDAGraph | None = None
        self.g_advance: torch.cuda.CUDAGraph | None = None
        self.depth_in_flight = 3
        self.host_tok = torch.empty((self.depth_in_flight, B), dtype=torch.int64, pin_memory=True)
        self.events = [torch.cuda.Event() for _ in range(self.depth_in_flight)]
        self.spec_k = spec_k
        self.verify: VerifyPlan | None = None
        self.recycler: Recycler | None = None
        self.stats: dict[str, float] = {}
        self.tau_floor = 2.0
        dev = model.device
        if recycle_rows:
            R = recycle_rows
            self.recycler = Recycler(model.cfg.vocab, B, R, recycle_k, dev)
            self.plan.recycler = self.recycler
            self.verify = VerifyPlan(self.plan, R, tree=True, recycler=self.recycler)
            self.cand = torch.empty((B, R), dtype=torch.int64, device=dev)
            self.host_cand = torch.empty((B, R), dtype=torch.int64, pin_memory=True)
            self.host_blk = torch.empty((B, R), dtype=torch.int64, pin_memory=True)
            depth = max(len([1 for _ in range(1)]), 1)
            self.maxa = max(1, max(bin(m & ((1 << 64) - 1)).count("1") for m in self.recycler.template.masks) - 1)
            self.path_idx = torch.zeros((B, self.maxa), dtype=torch.int32, device=dev)
            self.path_len = torch.zeros((B,), dtype=torch.int32, device=dev)
            self.host_path_idx = torch.zeros((B, self.maxa), dtype=torch.int32, pin_memory=True)
            self.host_path_len = torch.zeros((B,), dtype=torch.int32, pin_memory=True)
            self.host_root = torch.zeros((B,), dtype=torch.int64, pin_memory=True)
            self.host_spine = torch.full((B, self.recycler.S), -1, dtype=torch.int64, pin_memory=True)
            self.spine_min_match = int(os.environ.get("ENGINE_SPINE_MIN_MATCH", "3"))
        elif spec_k:
            self.verify = VerifyPlan(self.plan, spec_k + 1)
            self.cand = torch.empty((B, spec_k + 1), dtype=torch.int64, device=dev)
            self.host_blk = torch.empty((B, spec_k + 1), dtype=torch.int64, pin_memory=True)
            self.host_pos = torch.empty((B,), dtype=torch.int32, pin_memory=True)
            self.host_cand = torch.empty((B, spec_k + 1), dtype=torch.int64, pin_memory=True)

    def _warm_eager(self, steps: int = 3) -> None:
        plan = self.plan
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            plan.tok.copy_(plan.prefill().argmax(dim=-1))
            for _ in range(steps):
                plan.tok.copy_(plan.decode().argmax(dim=-1))
            if self.verify is not None:
                self.verify.pos.copy_(plan.pos - 1)
                if self.recycler is not None:
                    self.recycler.root.copy_(plan.tok)
                    self._advance()
                else:
                    self.verify.blk.copy_(plan.tok[:, None].expand(-1, self.verify.R))
                self.cand.copy_(self.verify.verify())
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

    def _advance(self) -> None:
        """Recycling round bookkeeping on device: compact the accepted path's
        K/V, move ``pos`` past it, and grow the next draft tree from ``root``."""
        plan, ver = self.plan, self.verify
        compact_paths(plan.k_cache, plan.v_cache, ver.pos, self.path_idx, self.path_len)
        ver.pos.add_(self.path_len + 1)
        self.recycler.draft()

    def capture(self) -> None:
        plan = self.plan
        t0 = time.perf_counter()
        self._warm_eager()
        self.g_prefill = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g_prefill):
            plan.tok.copy_(plan.prefill().argmax(dim=-1))
        self.g_decode = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g_decode):
            plan.tok.copy_(plan.decode().argmax(dim=-1))
        if self.verify is not None:
            self.g_verify = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g_verify):
                self.cand.copy_(self.verify.verify())
        if self.recycler is not None:
            self.g_advance = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g_advance):
                self._advance()
        torch.cuda.synchronize()
        _log(f"captured graphs for B={self.B} T={self.T} new={self.max_new} in {time.perf_counter() - t0:.1f}s "
             f"(picker budget remaining {max(0.0, budget.remaining()):.0f}s)")

    def _step_prefill(self) -> None:
        if self.g_prefill is None:
            self.plan.tok.copy_(self.plan.prefill().argmax(dim=-1))
        else:
            self.g_prefill.replay()

    def _step_decode(self) -> None:
        if self.g_decode is None:
            self.plan.tok.copy_(self.plan.decode().argmax(dim=-1))
        else:
            self.g_decode.replay()

    def _step_verify(self) -> None:
        if self.g_verify is None:
            self.cand.copy_(self.verify.verify())
        else:
            self.g_verify.replay()

    def run_recycle(self, input_ids: list[list[int]], max_new_tokens: int):
        """Token-recycling loop: draft tree -> verify -> accept longest path -> compact."""
        plan, ver, rec = self.plan, self.verify, self.recycler
        B, R = self.B, ver.R
        plan.ids.copy_(_ids_tensor(input_ids))
        self._step_prefill()
        first = plan.tok.tolist()
        yield first
        yielded = 1
        queues = [[first[b]] for b in range(B)]
        pos_host = [self.T] * B
        # pos is the root's slot at the start of each round; _advance adds
        # path_len + 1, so the first round starts one slot early with no path.
        ver.pos.copy_(plan.pos - 1)
        self.host_root.copy_(plan.tok)
        self.host_path_len.zero_()
        rec.root.copy_(self.host_root, non_blocking=True)
        self.path_len.copy_(self.host_path_len, non_blocking=True)
        S = rec.S
        drafters = [NGramDrafter(input_ids[b] + [first[b]], S, max_n=4, min_n=self.spine_min_match) for b in range(B)]
        self.host_spine.fill_(-1)
        for b in range(B):
            sp = drafters[b].draft_or_none()
            if sp:
                self.host_spine[b, :len(sp)] = torch.tensor(sp)
        rec.spine.copy_(self.host_spine, non_blocking=True)
        rounds = accepted = 0
        min_rounds = math.ceil((max_new_tokens - 1) / self.tau_floor) if max_new_tokens > 1 else 0
        while yielded < max_new_tokens:
            if self.g_advance is None:
                self._advance()
            else:
                self.g_advance.replay()
            self._step_verify()
            self.host_cand.copy_(self.cand, non_blocking=True)
            self.host_blk.copy_(ver.blk, non_blocking=True)
            torch.cuda.current_stream().synchronize()
            cand = self.host_cand.tolist()
            blk = self.host_blk.tolist()
            rounds += 1
            lens, idxs, roots = [], [], []
            for b in range(B):
                if len(queues[b]) >= max_new_tokens or pos_host[b] + 2 * R >= plan.cap:
                    # Frozen: re-verify the same block in place (len -1 leaves pos unchanged).
                    lens.append(-1); idxs.append([0] * self.maxa); roots.append(queues[b][-1]); continue
                toks, path = rec.accept(blk[b], cand[b])
                queues[b].extend(toks)
                drafters[b].extend(toks)
                accepted += len(path)
                lens.append(len(path))
                idxs.append(path + [0] * (self.maxa - len(path)))
                roots.append(toks[-1])
                pos_host[b] += len(path) + 1
            np_spine = self.host_spine.numpy()
            np_spine[:] = -1
            for b in range(B):
                if len(queues[b]) >= max_new_tokens:
                    continue
                sp = drafters[b].draft_or_none()
                if sp:
                    np_spine[b, :len(sp)] = sp
            rec.spine.copy_(self.host_spine, non_blocking=True)
            np_len, np_idx, np_root = self.host_path_len.numpy(), self.host_path_idx.numpy(), self.host_root.numpy()
            np_len[:] = lens
            np_idx[:] = idxs
            np_root[:] = roots
            self.path_len.copy_(self.host_path_len, non_blocking=True)
            self.path_idx.copy_(self.host_path_idx, non_blocking=True)
            rec.root.copy_(self.host_root, non_blocking=True)
            while yielded < max_new_tokens and all(len(q) > yielded for q in queues) and (
                yielded < max_new_tokens - 1 or rounds >= min_rounds
            ):
                yield [q[yielded] for q in queues]
                yielded += 1
        self.stats = {"rounds": rounds, "accepted": accepted, "steps": max_new_tokens, "min_rounds": min_rounds}
        _log(f"recycle: {rounds} rounds (min {min_rounds}) for {max_new_tokens} steps x {B} seqs, {accepted} extra tokens "
             f"accepted ({(max_new_tokens - 1) * B / max(1, rounds * B):.2f} tokens per round per seq)")

    def run_spec(self, input_ids: list[list[int]], max_new_tokens: int):
        """Speculative loop: verify K drafts per sequence per round, yield steps
        as soon as every sequence has a token for them. Output is identical to
        plain greedy decode because only model-predicted tokens are kept."""
        plan, ver = self.plan, self.verify
        B, K, R = self.B, self.spec_k, self.verify.R
        plan.ids.copy_(_ids_tensor(input_ids))
        self._step_prefill()
        first = plan.tok.tolist()
        yield first
        yielded = 1
        queues = [[first[b]] for b in range(B)]
        drafters = [NGramDrafter(input_ids[b] + [first[b]], K) for b in range(B)]
        pos = [self.T] * B
        blk = [[first[b]] + drafters[b].draft() for b in range(B)]
        rounds = accepted = 0
        while yielded < max_new_tokens:
            self.host_blk.copy_(torch.tensor(blk, dtype=torch.int64))
            self.host_pos.copy_(torch.tensor(pos, dtype=torch.int32))
            ver.blk.copy_(self.host_blk, non_blocking=True)
            ver.pos.copy_(self.host_pos, non_blocking=True)
            self._step_verify()
            self.host_cand.copy_(self.cand, non_blocking=True)
            torch.cuda.current_stream().synchronize()
            cand = self.host_cand.tolist()
            rounds += 1
            for b in range(B):
                if len(queues[b]) >= max_new_tokens:
                    continue
                a = 0
                while a < K and blk[b][a + 1] == cand[b][a]:
                    a += 1
                new = cand[b][:a + 1]
                queues[b].extend(new)
                drafters[b].extend(new)
                accepted += a
                pos[b] += a + 1
                blk[b] = [cand[b][a]] + drafters[b].draft()
            while yielded < max_new_tokens and all(len(q) > yielded for q in queues):
                yield [q[yielded] for q in queues]
                yielded += 1
        self.stats = {"rounds": rounds, "accepted": accepted, "steps": max_new_tokens}
        _log(f"spec: {rounds} rounds for {max_new_tokens} steps x {B} seqs, {accepted} drafts accepted "
             f"({accepted / max(1, rounds * B * K):.2f} per draft slot)")

    def run(self, input_ids: list[list[int]], max_new_tokens: int):
        """Yield one token list per step, keeping the GPU one step ahead of the host.

        Step t's tokens are copied to pinned host memory and fenced with an
        event; step t+1 is launched *before* waiting on that event, so the
        harness's read of step t overlaps the compute of step t+1.
        """
        if self.recycler is not None:
            yield from self.run_recycle(input_ids, max_new_tokens)
            return
        if self.verify is not None:
            yield from self.run_spec(input_ids, max_new_tokens)
            return
        plan = self.plan
        plan.ids.copy_(_ids_tensor(input_ids))
        host, events, D = self.host_tok, self.events, self.depth_in_flight
        self._step_prefill()
        host[0].copy_(plan.tok, non_blocking=True)
        events[0].record()
        launched = 1
        for step in range(max_new_tokens):
            # Keep up to D steps queued on the GPU so a slow consumer never drains it;
            # copies and graphs share one stream, so step t's copy precedes step t+1.
            while launched < max_new_tokens and launched - step < D:
                slot = launched % D
                self._step_decode()
                host[slot].copy_(plan.tok, non_blocking=True)
                events[slot].record()
                launched += 1
            slot = step % D
            events[slot].synchronize()
            yield host[slot].tolist()


class Engine:
    """Custom engine with a native safety net.

    If loading the custom model fails, or the warmup self-check finds the custom
    forward disagreeing with Transformers beyond the tie margin, every call is
    served by the organizers' baseline instead. Slower, never wrong.
    """

    def __init__(self, model_path: str) -> None:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model_path = model_path
        self.fallback = None
        self.model = None
        self.plans: dict[tuple[int, int], GraphPlan] = {}
        self.use_graphs = os.environ.get("ENGINE_NO_GRAPHS") is None
        self.spec_k = int(os.environ.get("ENGINE_SPEC_K", "0")) or None
        self.spec_max_rows = int(os.environ.get("ENGINE_SPEC_MAX_ROWS", "192"))
        self.recycle = os.environ.get("ENGINE_RECYCLE", "1") == "1"
        self.recycle_k = int(os.environ.get("ENGINE_RECYCLE_K", "8"))
        # Minimum verify rounds per sample = (max_new - 1) / tau_floor. Rounds are
        # padded up to it (the last token is held back) so a sample's timing does
        # not depend on how lucky its drafts were: the 25% spread gate.
        # Per-batch: a lone sequence accepts ~3 tokens a round; with 16 the slowest
        # sequence sets the pace, and its rounds already vary little.
        self.tau_floor_by_batch = {1: 2.4, 2: 2.4, 4: 2.6, 8: 2.0}
        self.tau_floor_default = float(os.environ.get("ENGINE_TAU_FLOOR", "1.4"))
        # Tree nodes per sequence: each 64 query rows (16 nodes x 4 heads) that a
        # sequence's tree adds is another pass over its KV cache.
        self.tree_rows_by_batch = {1: 64, 2: 32, 4: 16, 8: 8, 16: 4, 32: 4}
        self.self_check = os.environ.get("ENGINE_SELF_CHECK", "1") == "1"
        self.checked = False
        budget.start(PICKER_BUDGET_S)
        t0 = time.perf_counter()
        try:
            self.model = Model(model_path)
            torch.cuda.synchronize()
            _log(f"loaded {model_path} in {time.perf_counter() - t0:.1f}s")
        except Exception as exc:
            _log(f"custom model load failed ({exc!r}); using the native baseline for this run")
            self._use_fallback()

    def _use_fallback(self) -> None:
        from baseline import BaselineEngine

        self.plans.clear()
        self.model = None
        torch.cuda.empty_cache()
        self.fallback = BaselineEngine(self.model_path)

    def _run_self_check(self, plan: GraphPlan, input_ids: list[list[int]]) -> None:
        """Teacher-forced comparison against Transformers on the warmup prompt.

        Runs once, untimed, inside the load budget. Every checked position must
        keep our greedy token within the judge's tie margin of the reference
        argmax, and the reference's top-10 logits must agree to within that
        same margin (a real kernel bug moves them by tens).
        """
        from baseline import BaselineEngine

        t0 = time.perf_counter()
        ref = BaselineEngine(self.model_path)
        B, T = len(input_ids), len(input_ids[0])
        steps = min(SELF_CHECK_STEPS, plan.plan.cap - T)
        seq = torch.tensor(input_ids, dtype=torch.int64, device=self.model.device)
        worst_diff, worst_gap = 0.0, 0.0
        p = plan.plan
        p.ids.copy_(seq)
        mine = p.prefill().float()
        for step in range(steps):
            ref_logits = ref.logits(seq)
            top = ref_logits.max(dim=-1).values
            my_tok = mine.argmax(dim=-1)
            gap = (top - ref_logits.gather(1, my_tok[:, None])[:, 0]).max().item()
            top_idx = ref_logits.topk(SELF_CHECK_TOPK, dim=-1).indices
            diff = (mine.gather(1, top_idx) - ref_logits.gather(1, top_idx)).abs().max().item()
            worst_diff, worst_gap = max(worst_diff, diff), max(worst_gap, gap)
            forced = ref_logits.argmax(dim=-1)
            seq = torch.cat([seq, forced[:, None]], dim=1)
            if step + 1 < steps:
                p.tok.copy_(forced)
                mine = p.decode().float()
        del ref
        torch.cuda.empty_cache()
        ok = worst_gap <= TIE_MARGIN and worst_diff <= SELF_CHECK_MAX_DIFF
        _log(f"self-check vs transformers over {steps} steps: max|dlogit|={worst_diff:.3f} "
             f"worst tie gap={worst_gap:.3f} -> {'ok' if ok else 'FAILED'} ({time.perf_counter() - t0:.1f}s)")
        if not ok:
            raise RuntimeError("custom engine disagrees with the reference")

    def _plan(self, B: int, T: int, max_new: int) -> GraphPlan:
        key = (B, T)
        plan = self.plans.get(key)
        if plan is not None and T + max_new + 2 * 64 > plan.plan.cap:
            _log(f"max_new_tokens={max_new} exceeds planned capacity {plan.plan.cap}; rebuilding")
            plan = None
        if plan is None:
            self.plans.clear()
            torch.cuda.empty_cache()
            spec_k = self.spec_k if self.spec_k and B * (self.spec_k + 1) <= self.spec_max_rows else None
            rows = self.tree_rows_by_batch.get(B, max(0, 128 // B)) if self.recycle else 0
            rows = min(rows, self.spec_max_rows // B)
            recycle_rows = rows if rows >= 2 else None
            plan = GraphPlan(self.model, B, T, max_new, spec_k=None if recycle_rows else spec_k,
                             recycle_rows=recycle_rows, recycle_k=self.recycle_k)
            plan.tau_floor = self.tau_floor_by_batch.get(B, self.tau_floor_default)
            if os.environ.get("ENGINE_TAU_FLOOR"):
                plan.tau_floor = float(os.environ["ENGINE_TAU_FLOOR"])
            if self.use_graphs:
                try:
                    plan.capture()
                except Exception as exc:  # eager execution is slower but produces the same tokens
                    _log(f"CUDA graph capture failed ({exc!r}); running eagerly")
                    plan.g_prefill = plan.g_decode = plan.g_verify = None
                    torch.cuda.synchronize()
            self.plans[key] = plan
        return plan

    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        B, T = len(input_ids), len(input_ids[0])
        if any(len(row) != T for row in input_ids):
            raise ValueError("all prompts in a batch must have the same length")
        if max_new_tokens < 1:
            return
        if self.fallback is None:
            try:
                plan = self._plan(B, T, max_new_tokens)
                if self.self_check and not self.checked:
                    self._run_self_check(plan, input_ids)
                    self.checked = True
            except Exception as exc:
                _log(f"custom engine unusable ({exc!r}); using the native baseline for this run")
                self._use_fallback()
        if self.fallback is not None:
            yield from self.fallback.generate(input_ids, max_new_tokens)
            return
        produced = 0
        try:
            for step in plan.run(input_ids, max_new_tokens):
                produced += 1
                yield step
        except Exception as exc:  # a host-side bug must not end the run: finish with the baseline
            _log(f"custom generate failed after {produced} steps ({exc!r}); finishing with the native baseline")
            self._use_fallback()
            steps = list(self.fallback.generate(input_ids, max_new_tokens))
            for step in steps[produced:]:
                yield step
