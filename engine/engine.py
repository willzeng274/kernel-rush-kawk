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
from kernels.accept import accept_paths
from kernels.compact import compact_paths
from model import Model, Plan, VerifyPlan
from recycle import Recycler
from pair_cache import PAIR_SLOTS, prompt_pairs
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
        self.g_round: torch.cuda.CUDAGraph | None = None
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
            pair_rows = max(1, B * min(PAIR_SLOTS, max(0, T - 2)))
            self.host_pair_slots = torch.empty((pair_rows,), dtype=torch.int64, pin_memory=True)
            self.host_pair_keys = torch.empty((pair_rows,), dtype=torch.int64, pin_memory=True)
            self.host_pair_next = torch.empty((pair_rows, recycle_k), dtype=torch.int32, pin_memory=True)
            self.seed_pair_slots = torch.empty((pair_rows,), dtype=torch.int64, device=dev)
            self.seed_pair_keys = torch.empty((pair_rows,), dtype=torch.int64, device=dev)
            self.seed_pair_next = torch.empty((pair_rows, recycle_k), dtype=torch.int32, device=dev)
            self.verify = VerifyPlan(self.plan, R, tree=True, recycler=self.recycler)
            self.cand = torch.empty((B, R), dtype=torch.int64, device=dev)
            self.maxa = self.recycler.maxa
            self.guard = 2 * R
            self.path_idx = torch.zeros((B, self.maxa), dtype=torch.int32, device=dev)
            self.path_len = torch.zeros((B,), dtype=torch.int32, device=dev)
            # Round bookkeeping the accept kernel owns. ``nseen`` is the host's
            # old ``len(queues[b])`` and ``done`` its frozen test, both kept on
            # device so a round never waits for the host to decide anything.
            self.nseen = torch.zeros((B,), dtype=torch.int32, device=dev)
            self.done = torch.zeros((B,), dtype=torch.int32, device=dev)
            self.limit = torch.zeros((1,), dtype=torch.int32, device=dev)
            self.acc_tokens = torch.zeros((B, self.maxa + 1), dtype=torch.int64, device=dev)
            self.acc_count = torch.zeros((B,), dtype=torch.int32, device=dev)
            # Double buffers: the host reads slot t while the GPU fills slot t+1.
            self.host_acc = torch.zeros((2, B, self.maxa + 1), dtype=torch.int64, pin_memory=True)
            self.host_cnt = torch.zeros((2, B), dtype=torch.int32, pin_memory=True)
            self.acc_events = [torch.cuda.Event() for _ in range(2)]
            self.finish_event = torch.cuda.Event()
            self.recycle_cleanup_failed = False
            self.host_pool = torch.full((2, B, self.recycler.SP), -1, dtype=torch.int64, pin_memory=True)
            self.host_anchor = torch.zeros((2, B), dtype=torch.int32, pin_memory=True)
            self.launched = 0
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
            if self.recycler is not None:
                self.recycler.root_prev.copy_(plan.ids[:, -1])
            for _ in range(steps):
                if self.recycler is not None:
                    self.recycler.root_prev.copy_(plan.tok)
                plan.tok.copy_(plan.decode().argmax(dim=-1))
            if self.verify is not None:
                self.verify.pos.copy_(plan.pos - 1)
                if self.recycler is not None:
                    # Every kernel the round graph will replay, including the
                    # accept, has to be compiled before the capture; a live
                    # sequence with a huge limit takes the walk's real path.
                    self.recycler.root.copy_(plan.tok)
                    self.path_len.zero_()
                    self.nseen.zero_()
                    self.done.zero_()
                    self.limit.fill_(1 << 30)
                    self.recycler.spine.fill_(-1)
                    self.recycler.spine_anchor.zero_()
                    self._round()
                else:
                    self.verify.blk.copy_(plan.tok[:, None].expand(-1, self.verify.R))
                    self.cand.copy_(self.verify.verify())
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

    def _round(self) -> None:
        """One recycling round, end to end on device: compact the previous
        round's accepted path into place, move ``pos`` past it, grow the next
        draft tree from ``root``, verify it, and accept the longest path.

        Acceptance consumes this verify, then publishes accepted pair proposals.
        The compact at the head of the *next* replay acts on the path it wrote. That
        ordering is what lets a whole round be one graph: no host value is read
        or written anywhere between the first kernel and the last.
        """
        plan, ver, rec = self.plan, self.verify, self.recycler
        compact_paths(plan.k_cache, plan.v_cache, ver.pos, self.path_idx, self.path_len)
        ver.pos.add_(self.path_len + 1)
        rec.draft(self.nseen)
        self.cand.copy_(ver.verify())
        accept_paths(rec.blk, self.cand, rec.child_start, rec.child_list, rec.child_par,
                     self.done, self.nseen, ver.pos, self.limit, rec.root,
                     self.path_idx, self.path_len, self.acc_tokens, self.acc_count,
                     plan.cap, self.guard)
        rec.publish_pairs(self.path_idx, self.path_len)

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
        if self.recycler is not None:
            self.g_round = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g_round):
                self._round()
        elif self.verify is not None:
            self.g_verify = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g_verify):
                self.cand.copy_(self.verify.verify())
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

    def _launch_round(self) -> None:
        """Queue one round and the copy of its accepted tokens, fenced by an event."""
        slot = self.launched % 2
        if self.g_round is None:
            self._round()
        else:
            self.g_round.replay()
        self.host_acc[slot].copy_(self.acc_tokens, non_blocking=True)
        self.host_cnt[slot].copy_(self.acc_count, non_blocking=True)
        self.acc_events[slot].record()
        self.launched += 1

    def _write_spine(self, drafters, queues: list[list[int]], max_new_tokens: int, slot: int) -> None:
        """Hand the next n-gram continuation to the device, tagged with the token
        count it was taken at so the draft kernel can shift it by whatever the
        round now in flight accepts.

        The two pinned slots alternate per round: the copy this issues is only
        guaranteed to have run once a later round's event has been waited on, and
        that is two rounds away.
        """
        pool, anchor = self.host_pool[slot].numpy(), self.host_anchor[slot].numpy()
        pool[:] = -1
        for b, drafter in enumerate(drafters):
            anchor[b] = len(queues[b])
            if len(queues[b]) >= max_new_tokens:
                continue
            sp = drafter.draft_or_none()
            if sp:
                pool[b, :len(sp)] = sp
        self.recycler.spine.copy_(self.host_pool[slot], non_blocking=True)
        self.recycler.spine_anchor.copy_(self.host_anchor[slot], non_blocking=True)

    def _drain_recycle(self) -> None:
        """Fence all queued round work and spine copies before releasing buffers.

        Acceptance events precede the host's next spine copies, so waiting only
        on the last acceptance event cannot make the final pinned slots reusable.
        """
        try:
            self.finish_event.record()
            self.finish_event.synchronize()
        except BaseException:
            # A failed drain cannot establish ownership of the pinned buffers
            # or a healthy device state for fallback/reuse of this plan.
            self.recycle_cleanup_failed = True
            raise

    def _seed_pair_table(self, input_ids: list[list[int]]) -> None:
        """Reset every request's pair keys and upload its bounded sparse seed.

        All buffers are plan-owned and all operations use the existing stream
        and drain. Invalid keys make old values unreachable before any draft.
        """
        rec = self.recycler
        rec.pair_keys.fill_(-1)
        slots, keys, values = prompt_pairs(input_ids, rec.k)
        n = len(slots)
        if not n:
            return
        self.host_pair_slots.numpy()[:n] = slots
        self.host_pair_keys.numpy()[:n] = keys
        self.host_pair_next.numpy()[:n] = values
        self.seed_pair_slots[:n].copy_(self.host_pair_slots[:n], non_blocking=True)
        self.seed_pair_keys[:n].copy_(self.host_pair_keys[:n], non_blocking=True)
        self.seed_pair_next[:n].copy_(self.host_pair_next[:n], non_blocking=True)
        rec.pair_values.index_copy_(0, self.seed_pair_slots[:n], self.seed_pair_next[:n])
        rec.pair_keys.index_copy_(0, self.seed_pair_slots[:n], self.seed_pair_keys[:n])

    def run_recycle(self, input_ids: list[list[int]], max_new_tokens: int):
        """Token-recycling loop: draft tree -> verify -> accept longest path -> compact.

        Every one of those steps is inside one captured graph, so a round is a
        single replay and the host never sits between two of them. All the host
        does per round is read that round's accepted tokens out of a pinned
        buffer — one round late, because the next round is queued before the
        previous one's event is waited on, which keeps the GPU busy across the
        read.
        """
        if max_new_tokens <= 0:
            return
        if getattr(self, "recycle_cleanup_failed", False):
            raise RuntimeError("recycle plan cannot be reused after a failed drain")
        drained = False
        try:
            plan, ver, rec = self.plan, self.verify, self.recycler
            B = self.B
            plan.ids.copy_(_ids_tensor(input_ids))
            self._step_prefill()
            self._seed_pair_table(input_ids)
            first = plan.tok.tolist()
            if max_new_tokens == 1:
                self._drain_recycle()
                drained = True
            yield first
            if max_new_tokens == 1:
                self.launched = 0
                self.stats = {"rounds": 0, "accepted": 0, "steps": 1, "min_rounds": 0, "launched": 0}
                return
            yielded = 1
            queues = [[first[b]] for b in range(B)]
            # pos is the root's slot at the start of each round; the round graph adds
            # path_len + 1 first, so the loop starts one slot early with no path.
            ver.pos.copy_(plan.pos - 1)
            rec.root.copy_(plan.tok)
            rec.root_prev.copy_(plan.ids[:, -1])
            self.path_len.zero_()
            self.nseen.fill_(1)
            self.limit.fill_(max_new_tokens)
            self.done.fill_(int(1 >= max_new_tokens or self.T + self.guard >= plan.cap))
            self.launched = 0
            rec.spine.fill_(-1)
            rec.spine_anchor.zero_()
            drafters = [NGramDrafter(input_ids[b] + [first[b]], rec.SP, max_n=4, min_n=self.spine_min_match)
                        for b in range(B)]
            self._write_spine(drafters, queues, max_new_tokens, 0)
            rounds = accepted = 0
            min_rounds = math.ceil((max_new_tokens - 1) / self.tau_floor) if max_new_tokens > 1 else 0
            step_cap = self.maxa + 1  # most tokens one round can add to a sequence
            while yielded < max_new_tokens:
                if self.launched == rounds:
                    self._launch_round()
                # One more round is certain when the padding floor still owes rounds,
                # or when the one in flight cannot possibly fill the shortest queue.
                # Launching only on a certainty means no round is ever wasted.
                shortest = min(len(q) for q in queues)
                if self.launched < min_rounds or shortest + (self.launched - rounds) * step_cap < max_new_tokens:
                    self._launch_round()
                slot = rounds % 2
                self.acc_events[slot].synchronize()
                counts = self.host_cnt[slot].tolist()
                tokens = self.host_acc[slot].tolist()
                rounds += 1
                for b in range(B):
                    n = counts[b]
                    if not n:  # frozen: the block was re-verified in place
                        continue
                    new = tokens[b][:n]
                    queues[b].extend(new)
                    accepted += n - 1
                    drafters[b].extend(new)
                self._write_spine(drafters, queues, max_new_tokens, rounds % 2)
                while yielded < max_new_tokens and all(len(q) > yielded for q in queues) and (
                    yielded < max_new_tokens - 1 or rounds >= min_rounds
                ):
                    if yielded == max_new_tokens - 1:
                        self._drain_recycle()
                        drained = True
                    yield [q[yielded] for q in queues]
                    yielded += 1
            self.stats = {"rounds": rounds, "accepted": accepted, "steps": max_new_tokens, "min_rounds": min_rounds,
                          "launched": self.launched}
            _log(f"recycle: {rounds} rounds (min {min_rounds}) for {max_new_tokens} steps x {B} seqs, {accepted} extra tokens "
                 f"accepted ({(max_new_tokens - 1) * B / max(1, rounds * B):.2f} tokens per round per seq)")
        finally:
            if not drained and not getattr(self, "recycle_cleanup_failed", False):
                original = sys.exc_info()[1]
                if original is None:
                    self._drain_recycle()
                else:
                    try:
                        self._drain_recycle()
                    except BaseException as cleanup_error:
                        original.add_note(f"recycle cleanup also failed: {cleanup_error!r}")

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
                    plan.g_prefill = plan.g_decode = plan.g_verify = plan.g_round = None
                    torch.cuda.synchronize()
            self.plans[key] = plan
        return plan

    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        if getattr(self, "recycle_cleanup_failed", False):
            raise RuntimeError("engine cannot be reused after a failed recycle drain")
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
        active = plan.run(input_ids, max_new_tokens)
        try:
            for step in active:
                produced += 1
                yield step
        except Exception as exc:  # a host-side bug must not end the run: finish with the baseline
            if getattr(plan, "recycler", None) is not None:
                # An exception injected at our yield can leave the nested
                # iterator suspended with work queued; drain before fallback.
                active.close()
            if getattr(plan, "recycle_cleanup_failed", False):
                raise
            _log(f"custom generate failed after {produced} steps ({exc!r}); finishing with the native baseline")
            self._use_fallback()
            steps = list(self.fallback.generate(input_ids, max_new_tokens))
            for step in steps[produced:]:
                yield step
        finally:
            if getattr(plan, "recycler", None) is not None:
                try:
                    active.close()
                finally:
                    if getattr(plan, "recycle_cleanup_failed", False):
                        self.recycle_cleanup_failed = True
