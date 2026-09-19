"""Untimed setup/ranking checks for the exact W8 top-two buffers and kernels."""
import torch
from recycled_validate import CandidateRejected, _live, _same_bits
from topk_proposals import proposals_out


def check_proposal_buffers(logits, greedy, partial_values, partial_ids, out):
    rows = logits.shape[0]
    expected = ((logits, (rows, 151936), torch.bfloat16),
                (greedy, (rows,), torch.int64),
                (partial_values, (rows, 75, 1), torch.float32),
                (partial_ids, (rows, 75, 1), torch.int32),
                (out, (rows, 2), torch.int64))
    intervals = []
    for value, shape, dtype in expected:
        if (not 1 <= rows <= 64 or tuple(value.shape) != shape or value.dtype != dtype
                or not value.is_cuda or value.device != logits.device or not value.is_contiguous()):
            raise CandidateRejected("top-two buffer shape/dtype/device/contiguity mismatch")
        start = value.data_ptr()
        intervals.append((start, start + value.numel() * value.element_size()))
    for index, (start, stop) in enumerate(intervals):
        if any(start < other_stop and other_start < stop
               for other_start, other_stop in intervals[:index]):
            raise CandidateRejected("top-two buffers alias")


def check_model_top2(graph):
    """Independent full-vocabulary PyTorch reference on actual captured logits."""
    if not torch.isfinite(graph.logits).all():
        raise CandidateRejected("top-two model logits are not finite")
    greedy = graph.logits.argmax(dim=-1)
    if (not torch.equal(graph.output_flat, greedy)
            or not torch.equal(graph.proposal_ids[:, 0], greedy)):
        raise CandidateRejected("top-two changed authoritative argmax")
    reference = graph.logits.clone()
    reference.scatter_(1, greedy[:, None], -float("inf"))
    # All unmasked entries are finite, so even lowest-valued finite logits
    # beat the removed greedy. PyTorch argmax selects the lowest tied ID.
    alternative = reference.argmax(dim=-1)
    if not torch.equal(graph.proposal_ids[:, 1], alternative):
        raise CandidateRejected("top-two alternative rank/tie mismatch")


def validate_top2_helper(graph, deadline):
    """Exercise real helper launches on graph-owned buffers before admission.

    Only verifier logits/results/scratch are overwritten; no main K/V changes.
    The subsequent full tree replay overwrites every affected buffer.
    """
    x = graph.logits
    for case in range(7):
        _live(deadline)
        if case == 0:
            x.zero_()
            x[:, 0].fill_(-0.0)
            expected = 1
        elif case == 1:
            x.fill_(-1.0)
            x[:, 2047:2049].fill_(7.0)
            x[:, -1].fill_(7.0)
            expected = 2048
        elif case == 2:
            x.fill_(-2.0)
            x[:, -2].fill_(5.0)
            x[:, -1].fill_(9.0)
            expected = 151934
        elif case == 3:
            x.fill_(-float("inf"))
            expected = 1
        elif case == 4:
            x.fill_(float("nan"))
            x[:, -1].fill_(-float("inf"))
            expected = 151935
        elif case == 5:
            x.fill_(float("nan"))
            expected = -1
        else:
            x.fill_(-float("inf"))
            x[:, 2047].fill_(float("inf"))
            x[:, 151935].fill_(float("inf"))
            expected = 151935
        torch.argmax(x, dim=-1, out=graph.output_flat)
        before, before_greedy = x.clone(), graph.output_flat.clone()
        proposals_out(x, graph.output_flat, graph.proposal_values,
                      graph.proposal_partials, graph.proposal_ids, 2)
        torch.cuda.synchronize()
        if (not _same_bits(x, before) or not torch.equal(graph.output_flat, before_greedy)
                or not torch.equal(graph.proposal_ids[:, 0], before_greedy)
                or not bool((graph.proposal_ids[:, 1] == expected).all())):
            raise CandidateRejected("top-two boundary/tie/nonfinite/immutability check")
