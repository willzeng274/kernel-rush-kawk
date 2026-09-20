import triton
import triton.language as tl


@triton.jit
def _gemv_kernel(
    X, W, Y, M, K,
    stride_xm, stride_wn, stride_ym,
    N: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, MP: tl.constexpr,
):
    """y = x @ w.T for decode-shaped x. W is [N, K] row-major.

    At decode batch sizes this is a pure reduction over K per output, so
    parallelism is N/BLOCK_N; the M dimension is padded to 16 to satisfy
    tl.dot and the padded lanes cost nothing (the op is bandwidth-bound).
    """
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, MP)
    nmask = offs_n < N
    mmask = offs_m < M
    acc = tl.zeros([MP, BLOCK_N], tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        kmask = offs_k < K
        x = tl.load(X + offs_m[:, None] * stride_xm + offs_k[None, :],
                    mask=mmask[:, None] & kmask[None, :], other=0.0)
        w = tl.load(W + offs_n[:, None] * stride_wn + offs_k[None, :],
                    mask=nmask[:, None] & kmask[None, :], other=0.0)
        acc += tl.dot(x, tl.trans(w))
    tl.store(Y + offs_m[:, None] * stride_ym + offs_n[None, :], acc.to(tl.bfloat16),
             mask=mmask[:, None] & nmask[None, :])


@triton.jit
def _gemv_swiglu_kernel(
    X, W, Y, M, K,
    stride_xm, stride_wn, stride_ym,
    I: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, MP: tl.constexpr,
):
    """Fused gate/up projection with SwiGLU as its epilogue.

    W is the [2I, K] gate-over-up matrix. Each program owns a block of
    activation outputs and computes both the gate rows and the matching up
    rows, so silu(gate) * up is written directly and the separate SwiGLU
    launch disappears. Same bytes streamed, half as many programs. Rounding
    follows the reference: both projections round to bf16, silu is fp32
    rounded to bf16, and the product is a bf16 multiply.
    """
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = offs_n < I
    offs_m = tl.arange(0, MP)
    mmask = offs_m < M
    acc_g = tl.zeros([MP, BLOCK_N], tl.float32)
    acc_u = tl.zeros([MP, BLOCK_N], tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        kmask = offs_k < K
        x = tl.load(X + offs_m[:, None] * stride_xm + offs_k[None, :],
                    mask=mmask[:, None] & kmask[None, :], other=0.0)
        wg = tl.load(W + offs_n[:, None] * stride_wn + offs_k[None, :],
                     mask=nmask[:, None] & kmask[None, :], other=0.0)
        wu = tl.load(W + (offs_n + I)[:, None] * stride_wn + offs_k[None, :],
                     mask=nmask[:, None] & kmask[None, :], other=0.0)
        acc_g += tl.dot(x, tl.trans(wg))
        acc_u += tl.dot(x, tl.trans(wu))
    g = acc_g.to(tl.bfloat16).to(tl.float32)
    act = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16)
    tl.store(Y + offs_m[:, None] * stride_ym + offs_n[None, :], act * acc_u.to(tl.bfloat16),
             mask=mmask[:, None] & nmask[None, :])
