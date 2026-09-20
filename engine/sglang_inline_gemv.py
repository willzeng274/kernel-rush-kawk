"""RESEARCH ONLY: faithful one-row/warp SGLang GEMV inline-PTX probe.

Adapted from sgl-project/sglang, Apache-2.0, commit
ee5fcdf0d906020860bb5f7aa1f00a127991f605:
python/sglang/kernels/jit/csrc/gemm/hopper_bf16_gemv.cuh.
Original source and complete license are preserved in this directory's sources/.

Restricted to the three eligible Qwen3-4B B1 projections below. The compiler
must establish one side-effecting assembly instance per physical thread, legal
static shared allocation, vector loads and collective barrier execution before
this can be considered executable GPU evidence. No production integration.
"""
import triton
import triton.language as tl

CONFIGS = {"qkv": (6144, 2560), "output": (2560, 4096), "down": (2560, 9728)}


def make_asm(k):
    if k not in (2560, 4096, 9728):
        raise ValueError("probe admits only the three exact retained K values")
    # $0=float result; $1=row; $2=lane; $3=X pointer; $4=W pointer;
    # $5=one logical element per thread. Hardware IDs own row/lane indexing.
    start = """{
 .reg .u32 tid, lane, warp, block, row, si, ki, koff, sbase, sptr, logical;
 .reg .u64 xp, wp, addr, rowptr, byteoff;
 .reg .pred done;
 .reg .b32 w0,w1,w2,w3,x0,x1,x2,x3;
 .reg .b16 wl,wh,xl,xh;
 .reg .f32 wf,xf,acc,dot,tmp;
 .shared .align 16 .b8 sx[SHARED_BYTES];
 mov.u32 logical, $5;
 mov.u32 tid, %tid.x;
 mov.u32 block, %ctaid.x;
 mov.u64 xp, $3;
 mov.u64 wp, $4;
 mov.u32 sbase, sx;
 mul.lo.u32 si, tid, 16;
COPY_LOOP:
 setp.ge.u32 done, si, SHARED_BYTES;
 @done bra COPY_DONE;
 cvt.u64.u32 byteoff, si;
 add.u64 addr, xp, byteoff;
 ld.global.v4.b32 {x0,x1,x2,x3}, [addr];
 add.u32 sptr, sbase, si;
 st.shared.v4.b32 [sptr], {x0,x1,x2,x3};
 add.u32 si, si, 4096;
 bra COPY_LOOP;
COPY_DONE:
 bar.sync 0;
 and.b32 lane, tid, 31;
 shr.u32 warp, tid, 5;
 mad.lo.u32 row, block, 8, warp;
 mul.wide.u32 byteoff, row, SHARED_BYTES;
 add.u64 rowptr, wp, byteoff;
 mul.lo.u32 ki, lane, 16;
 mov.f32 acc, 0f00000000;
K_LOOP:
 setp.ge.u32 done, ki, K_SIZE;
 @done bra K_DONE;
""".replace("SHARED_BYTES", str(2 * k)).replace("K_SIZE", str(k))
    chunks = []
    for unroll in range(2):
        chunk = f"""
 add.u32 koff, ki, {8 * unroll};
 mul.lo.u32 koff, koff, 2;
 add.u32 sptr, sbase, koff;
 ld.shared.v4.b32 {{x0,x1,x2,x3}}, [sptr];
 cvt.u64.u32 byteoff, koff;
 add.u64 addr, rowptr, byteoff;
 ld.global.cs.v4.b32 {{w0,w1,w2,w3}}, [addr];
 mov.f32 dot, 0f00000000;
"""
        for pair in range(4):
            chunk += f"""
 mov.b32 {{wl,wh}}, w{pair};
 mov.b32 {{xl,xh}}, x{pair};
 cvt.f32.bf16 wf, wl;
 cvt.f32.bf16 xf, xl;
 fma.rn.f32 dot, wf, xf, dot;
 cvt.f32.bf16 wf, wh;
 cvt.f32.bf16 xf, xh;
 fma.rn.f32 dot, wf, xf, dot;
"""
        chunk += " add.rn.f32 acc, acc, dot;\n"
        chunks.append(chunk)
    finish = """
 add.u32 ki, ki, 512;
 bra K_LOOP;
K_DONE:
"""
    for offset in (16, 8, 4, 2, 1):
        finish += f" shfl.sync.down.b32 tmp, acc, {offset}, 31, -1;\n"
        finish += " add.rn.f32 acc, acc, tmp;\n"
    finish += " mov.f32 $0, acc;\n mov.u32 $1, row;\n mov.u32 $2, lane;\n}\n"
    return start + "".join(chunks) + finish


@triton.jit
def _sglang_inline_gemv(X, W, OUT, N: tl.constexpr, K: tl.constexpr, ASM: tl.constexpr):
    logical = tl.arange(0, 256)
    value, row, lane = tl.inline_asm_elementwise(
        ASM, constraints="=f,=r,=r,l,l,r", args=[X, W, logical],
        dtype=(tl.float32, tl.int32, tl.int32), is_pure=False, pack=1)
    tl.store(OUT + row, value, (lane == 0) & (row < N))


def compile_configuration(name):
    """Descriptive compile-only arguments; not a GPU launch wrapper."""
    n, k = CONFIGS[name]
    return {"N": n, "K": k, "ASM": make_asm(k), "grid": (n // 8,),
            "num_warps": 8, "num_stages": 1, "enable_fp_fusion": False}
