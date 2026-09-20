"""Never launch with Triton's ordinary kernel[grid](...) interface."""
import triton
import triton.language as tl


@triton.jit
def cooperative_smoke(VALUES, OUT, OBSERVED, ARRIVALS, EPOCH,
                      PROGRAMS: tl.constexpr, WORDS: tl.constexpr,
                      READ_BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    # All CTAs sample this epoch before arrival. The last arrival cannot
    # advance it until every participant has already sampled it.
    phase = tl.atomic_add(EPOCH, 0, sem="acquire", scope="gpu")
    offsets = pid * WORDS + tl.arange(0, WORDS)
    total: tl.constexpr = PROGRAMS * WORDS
    values = offsets.to(tl.int64) + (phase.to(tl.int64) + 1) * total
    tl.store(VALUES + offsets, values)
    tl.debug_barrier()

    # Block barrier publishes all producer-thread stores to the CTA's scalar
    # atomic participant. Acq_rel arrivals form a transitive release sequence.
    ticket = tl.atomic_add(ARRIVALS, 1, sem="acq_rel", scope="gpu")
    if ticket == PROGRAMS - 1:
        # No next invocation can run until this kernel completes on its stream.
        # Reset count before releasing the new phase; EPOCH itself never resets.
        tl.atomic_xchg(ARRIVALS, 0, sem="relaxed", scope="gpu")
        tl.atomic_xchg(EPOCH, phase + 1, sem="release", scope="gpu")
    else:
        observed = tl.atomic_add(EPOCH, 0, sem="acquire", scope="gpu")
        while observed == phase:
            observed = tl.atomic_add(EPOCH, 0, sem="acquire", scope="gpu")
    tl.debug_barrier()

    # Different CTAs consume every producer's writes. Volatile loads prevent
    # hoisting across synchronization; PTX forbids combining volatile and .cg.
    indices = tl.arange(0, READ_BLOCK)
    data = tl.load(VALUES + indices, mask=indices < total, other=0,
                   volatile=True)
    tl.store(OUT + pid, tl.sum(data, 0))
    tl.store(OBSERVED + pid, phase + 1)
