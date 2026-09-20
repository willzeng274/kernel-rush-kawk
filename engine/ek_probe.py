"""Child-process checks: can Triton compile and launch our kernels here?

Kept out of the engine process on purpose. A Triton failure is not always an
exception: on Hopper, Triton 3.1.0 aborts the compiler for a 64-row tl.dot, and
a sandbox without a C compiler or a loadable libcuda fails before that. A
child's exit code turns all of those into a plain "no".

    ek_probe.py basic
    ek_probe.py shape <batch> <nq> <bucket> <n_kv> <n_heads> <head_dim>
"""
import sys

import torch
import triton
import triton.language as tl


# module level on purpose: a jitted kernel resolves `tl` as a module global
@triton.jit
def _bump(X, N: tl.constexpr):
    offs = tl.arange(0, N)
    tl.store(X + offs, tl.load(X + offs) + 1.0)


def basic() -> int:
    x = torch.zeros(64, device="cuda", dtype=torch.float32)
    _bump[(1,)](x, N=64)
    torch.cuda.synchronize()
    return 0 if float(x.sum().item()) == 64.0 else 3


def check_fused_verify(K, b, nq, bucket, n_kv, n_heads, d):
    """Compare one fresh unfused/fused verifier-attention pair on CUDA/BF16.

    Use a prefix one token before a real split boundary near the bucket's
    midpoint when possible. Q3 then writes one slot in the earlier split and
    two in the later split. Otherwise use one legal middle prefix.

    V and preserved prefix contents must match exactly. Transformed K and
    attention outputs use atol=rtol=1/128: one BF16 machine epsilon relative,
    plus an absolute floor around zero for cancellation/reassociation. This
    conservative operation tolerance is not the judge's 2-logit allowance.
    Exceptions intentionally propagate to the existing child-process boundary.
    """
    import torch

    if (b <= 0 or nq <= 0 or n_kv <= 0 or n_heads <= 0 or n_heads % n_kv
            or d <= 0 or d % 2 or bucket < nq + 1):
        return False
    dev, dt = "cuda", torch.bfloat16
    splits, chunk, block_n = K.plan_splits(b, n_kv, bucket)
    if splits <= 0 or chunk <= 0:
        return False
    boundaries = [x for x in range(chunk, min(bucket, splits * chunk), chunk)
                  if x - 1 + nq <= bucket] if nq > 1 else []
    length = (min(boundaries, key=lambda x: abs(x - bucket // 2)) - 1
              if boundaries else min(max(1, bucket // 2), bucket - nq))

    gen = torch.Generator(device=dev).manual_seed(87231)
    raw = torch.randn(b * nq, (n_heads + 2 * n_kv) * d,
                      device=dev, dtype=dt, generator=gen)
    reference_qkv = raw.clone()  # the unfused operation overwrites Q in place
    qn = torch.linspace(0.75, 1.25, d, device=dev).to(dt)
    kn = torch.linspace(1.25, 0.75, d, device=dev).to(dt)

    # Proper rotate-half RoPE tables: repeated halves at the same positions.
    positions = (length + torch.arange(nq, device=dev)).repeat(b).float()
    inv_freq = 5_000_000.0 ** (-torch.arange(0, d, 2, device=dev).float() / d)
    phase = positions[:, None] * inv_freq[None, :]
    phase = torch.cat((phase, phase), dim=1)
    cos, sin = phase.cos().to(dt), phase.sin().to(dt)
    len_b = torch.full((b,), length, device=dev, dtype=torch.int64)
    start = torch.zeros(b, device=dev, dtype=torch.int32)

    reference_k = torch.randn(b, n_kv, bucket, d, device=dev, dtype=dt, generator=gen)
    reference_v = torch.randn(b, n_kv, bucket, d, device=dev, dtype=dt, generator=gen)
    reference_k[:, :, length:].fill_(float("nan"))
    reference_v[:, :, length:].fill_(float("nan"))
    fused_k, fused_v = reference_k.clone(), reference_v.clone()

    sp = K._next_pow2(splits)
    gp = max(16, K._next_pow2(nq * (n_heads // n_kv)))
    acc = torch.empty(b, n_kv, sp, gp, d, device=dev, dtype=torch.float32)
    lsum = torch.empty(b, n_kv, sp, gp, device=dev, dtype=torch.float32)
    mmax = torch.empty_like(lsum)
    output = torch.empty(b * nq, n_heads, d, device=dev, dtype=dt)
    workspace = (acc, lsum, mmax, output, splits, chunk, block_n)

    def reset_workspace():
        # Padded splits must stay neutral for the existing combine kernel.
        # Real splits/output are poisoned to expose missing fused stores,
        # rather than accidentally inheriting the preceding reference result.
        acc.zero_()
        lsum.zero_()
        mmax.fill_(-1e30)
        acc[:, :, :splits].fill_(float("nan"))
        lsum[:, :, :splits].fill_(float("nan"))
        mmax[:, :, :splits].fill_(float("nan"))
        output.fill_(float("nan"))

    K.qk_norm_rope_kv(reference_qkv, qn, kn, cos, sin, reference_k, reference_v,
                      len_b, n_heads, n_kv, 1e-6, nq)
    reference_q = reference_qkv[:, :n_heads * d].view(b * nq, n_heads, d)
    reset_workspace()
    reference = K.flash_verify(reference_q, reference_k, reference_v, len_b,
                               start, workspace, d ** -0.5, nq).clone()
    # Both wrappers return workspace[3]. The clone above must precede reuse.
    reset_workspace()
    fused = K.rope_attn_verify(raw, qn, kn, cos, sin, fused_k, fused_v, len_b,
                               start, workspace, d ** -0.5, n_heads, nq, 1e-6)
    torch.cuda.synchronize()

    stop = length + nq
    ref_new_k, new_k = reference_k[:, :, length:stop], fused_k[:, :, length:stop]
    ref_new_v, new_v = reference_v[:, :, length:stop], fused_v[:, :, length:stop]
    if not all(bool(torch.isfinite(x).all())
               for x in (reference, fused, ref_new_k, new_k, ref_new_v, new_v)):
        return False
    if not torch.equal(ref_new_v, new_v):
        return False
    for ref_cache, new_cache in ((reference_k, fused_k), (reference_v, fused_v)):
        if not torch.equal(ref_cache[:, :, :length], new_cache[:, :, :length]):
            return False
        if not bool(torch.isnan(new_cache[:, :, stop:]).all()):
            return False
    return (torch.allclose(ref_new_k, new_k, atol=1 / 128, rtol=1 / 128)
            and torch.allclose(reference, fused, atol=1 / 128, rtol=1 / 128))



def shape(b, nq, bucket, n_kv, n_heads, d) -> int:
    """Compile every kernel the verify step uses, with this shape's constexprs."""
    import ek_kernels as K

    dev, dt = "cuda", torch.bfloat16
    hidden, inter = 2560, 9728
    m = b * max(nq, 1)
    x = torch.randn(m, hidden, device=dev, dtype=dt)
    w = torch.randn(hidden, device=dev, dtype=dt)
    K.rms_norm(x, w, 1e-6)
    K.add_rms_norm(x, x.clone(), w, 1e-6)
    K.silu_mul(torch.randn(m, 2 * inter, device=dev, dtype=dt))
    K.heads_to_rows(torch.randn(b, n_heads, 130, d, device=dev, dtype=dt))   # prefill attention output copy

    if m <= 32:  # the decode projections may run on the Triton GEMV at this row count
        for n_out, k_in in ((n_heads * d + 2 * n_kv * d, hidden), (hidden, n_heads * d),
                            (2 * inter, hidden), (hidden, inter)):
            K.gemv(torch.randn(m, k_in, device=dev, dtype=dt), torch.randn(n_out, k_in, device=dev, dtype=dt))
        if m <= 32:  # split-K projections and the add+norm that sums their partials
            for n_out, k_in in ((hidden, n_heads * d), (hidden, inter)):
                parts = K.gemv_parts(torch.randn(m, k_in, device=dev, dtype=dt),
                                     torch.randn(n_out, k_in, device=dev, dtype=dt))
                K.add_rms_norm_parts(parts, torch.randn(m, hidden, device=dev, dtype=dt), w, 1e-6)

    kc = torch.zeros(b, n_kv, bucket, d, device=dev, dtype=dt)
    vc = torch.zeros_like(kc)
    qkv = torch.randn(m, (n_heads + 2 * n_kv) * d, device=dev, dtype=dt)
    hn = torch.ones(d, device=dev, dtype=dt)
    cs = torch.randn(m, d, device=dev, dtype=dt)
    len_b = torch.full((b,), max(1, bucket // 2), dtype=torch.int64, device=dev)
    for rows, mpb in ((m, max(nq, 1)), (b * 64, 64)):  # decode-sized and prefill-sized tilings
        q2 = qkv if rows == m else torch.randn(rows, qkv.shape[1], device=dev, dtype=dt)
        c2 = cs if rows == m else torch.randn(rows, d, device=dev, dtype=dt)
        K.qk_norm_rope_kv(q2, hn, hn, c2, c2, kc, vc,
                          len_b if rows == m else torch.zeros(b, dtype=torch.int64, device=dev),
                          n_heads, n_kv, 1e-6, mpb)

    splits, chunk, bn = K.plan_splits(b, n_kv, bucket)
    sp = K._next_pow2(splits)
    if nq == 0:  # plain decode: one query per sequence, shared length
        gp = K.group_pad(n_heads, n_kv)
        ws = (torch.zeros(b, n_kv, sp, gp, d, dtype=torch.float32, device=dev),
              torch.zeros(b, n_kv, sp, gp, dtype=torch.float32, device=dev),
              torch.full((b, n_kv, sp, gp), -1e30, dtype=torch.float32, device=dev),
              torch.empty(b, n_heads, d, device=dev, dtype=dt), splits, chunk, bn)
        out = K.flash_decode(torch.randn(b, n_heads, d, device=dev, dtype=dt), kc, vc,
                             torch.tensor([max(1, bucket // 2)], dtype=torch.int64, device=dev),
                             torch.zeros(b, dtype=torch.int32, device=dev), ws, d ** -0.5)
        # the fused norm+rope+cache-write+attention kernel the plain decode path uses
        out2 = K.rope_attn_decode(qkv[:b].contiguous(), hn, hn, cs[:b].contiguous(), cs[:b].contiguous(),
                                  kc, vc, torch.tensor([max(2, bucket // 2)], dtype=torch.int64, device=dev),
                                  torch.zeros(b, dtype=torch.int32, device=dev), ws, d ** -0.5, n_heads, 1e-6)
        torch.cuda.synchronize()
        return 0 if torch.isfinite(out.float()).all() and torch.isfinite(out2.float()).all() else 4
    # integer kernels of the speculative step: n-gram drafting and accept bookkeeping
    hist = torch.randint(0, 1000, (b, bucket), device=dev, dtype=torch.int64)
    hlen = torch.full((b,), max(4, bucket // 2), dtype=torch.int64, device=dev)
    toks = torch.zeros((b, nq), dtype=torch.int64, device=dev)
    K.ngram_draft(hist, hlen, toks)
    K.spec_accept(toks, toks.clone(), hist, hlen, hlen.clone(), hlen.clone(),
                  torch.full((b,), 8, dtype=torch.int64, device=dev),
                  torch.zeros((4, b, nq), dtype=torch.int32, device=dev),
                  torch.zeros((4, b), dtype=torch.int32, device=dev),
                  torch.zeros(1, dtype=torch.int64, device=dev))
    gp = max(16, K._next_pow2(nq * (n_heads // n_kv)))
    ws = (torch.zeros(b, n_kv, sp, gp, d, dtype=torch.float32, device=dev),
          torch.zeros(b, n_kv, sp, gp, dtype=torch.float32, device=dev),
          torch.full((b, n_kv, sp, gp), -1e30, dtype=torch.float32, device=dev),
          torch.empty(m, n_heads, d, device=dev, dtype=dt), splits, chunk, bn)
    q = torch.randn(m, n_heads, d, device=dev, dtype=dt)
    out = K.flash_verify(q, kc, vc, len_b, torch.zeros(b, dtype=torch.int32, device=dev),
                         ws, d ** -0.5, nq)
    torch.cuda.synchronize()
    if not torch.isfinite(out.float()).all():
        return 4
    return 0 if check_fused_verify(K, b, nq, bucket, n_kv, n_heads, d) else 4


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "basic"
    sys.exit(basic() if mode == "basic" else shape(*[int(a) for a in sys.argv[2:8]]))
