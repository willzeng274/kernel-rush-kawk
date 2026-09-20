"""RESEARCH ONLY: faithful two-row/warp SGLang packed gate/up GEMV probe.

Modified adaptation of sgl-project/sglang, Apache-2.0, commit
 ee5fcdf0d906020860bb5f7aa1f00a127991f605,
 python/sglang/kernels/jit/csrc/gemm/hopper_bf16_gemv.cuh.
Original source and full license are preserved under sources/ and below.

Only N19456/K2560, rows2/unroll2/warps8 is in scope. Upstream excludes this
N band for empirical H200/cuBLAS performance reasons. This is a separate
mechanics/compile probe, not a production guard change or GPU speed claim.
"""

import triton
import triton.language as tl

CONFIGS = {"gateup": (19456, 2560)}


def make_asm(k=2560):
    if type(k) is not int or k != 2560:
        raise ValueError("GU2 probe admits only exact K2560")
    # $0/$1 are independent FP32 row results; $2=row0; $3=lane;
    # $4=X, $5=W, $6=one logical element per physical thread.
    start = """{
 .reg .u32 tid,lane,warp,block,row,si,ki,koff,sbase,sptr,logical;
 .reg .u64 xp,wp,addr,rowptr0,rowptr1,byteoff;
 .reg .pred done;
 .reg .b32 w0,w1,w2,w3,w4,w5,w6,w7,x0,x1,x2,x3,x4,x5,x6,x7;
 .reg .b16 wl,wh,xl,xh;
 .reg .f32 wf,xf,acc0,acc1,dot,tmp;
 .shared .align 16 .b8 sx[5120];
 mov.u32 logical, $6;
 mov.u32 tid, %tid.x;
 mov.u32 block, %ctaid.x;
 mov.u64 xp, $4;
 mov.u64 wp, $5;
 mov.u32 sbase, sx;
 mul.lo.u32 si, tid, 16;
COPY_LOOP:
 setp.ge.u32 done, si, 5120;
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
 mul.lo.u32 row, row, 2;
 mul.wide.u32 byteoff, row, 5120;
 add.u64 rowptr0, wp, byteoff;
 add.u64 rowptr1, rowptr0, 5120;
 mul.lo.u32 ki, lane, 16;
 mov.f32 acc0, 0f00000000;
 mov.f32 acc1, 0f00000000;
K_LOOP:
 setp.ge.u32 done, ki, 2560;
 @done bra K_DONE;
"""
    # Original CUDA loads both X vectors once per K iteration. The same
    # packed X registers serve row0 and row1 without a second shared load.
    body = []
    for u in range(2):
        registers = ",".join(f"x{i}" for i in range(4*u, 4*u+4))
        body.append(f"""
 add.u32 koff, ki, {8*u};
 mul.lo.u32 koff, koff, 2;
 add.u32 sptr, sbase, koff;
 ld.shared.v4.b32 {{{registers}}}, [sptr];
""")
    # Source row loop: two W preloads for one row before either dot8;
    # finish that row's dot8s before loading the next row's W vectors.
    for row in range(2):
        for u in range(2):
            registers = ",".join(f"w{i}" for i in range(4*u, 4*u+4))
            body.append(f"""
 add.u32 koff, ki, {8*u};
 mul.lo.u32 koff, koff, 2;
 cvt.u64.u32 byteoff, koff;
 add.u64 addr, rowptr{row}, byteoff;
 ld.global.cs.v4.b32 {{{registers}}}, [addr];
""")
        for u in range(2):
            chunk = " mov.f32 dot, 0f00000000;\n"
            for pair in range(4):
                register = 4*u+pair
                chunk += f"""
 mov.b32 {{wl,wh}}, w{register};
 mov.b32 {{xl,xh}}, x{register};
 cvt.f32.bf16 wf, wl;
 cvt.f32.bf16 xf, xl;
 fma.rn.f32 dot, wf, xf, dot;
 cvt.f32.bf16 wf, wh;
 cvt.f32.bf16 xf, xh;
 fma.rn.f32 dot, wf, xf, dot;
"""
            chunk += f" add.rn.f32 acc{row}, acc{row}, dot;\n"
            body.append(chunk)
    finish = """
 add.u32 ki, ki, 512;
 bra K_LOOP;
K_DONE:
"""
    for row in range(2):
        for offset in (16,8,4,2,1):
            finish += f" shfl.sync.down.b32 tmp, acc{row}, {offset}, 31, -1;\n"
            finish += f" add.rn.f32 acc{row}, acc{row}, tmp;\n"
    finish += " mov.f32 $0, acc0;\n mov.f32 $1, acc1;\n mov.u32 $2, row;\n mov.u32 $3, lane;\n}\n"
    return start+"".join(body)+finish


@triton.jit
def _sglang_inline_gemv_gu2(X, W, OUT, N: tl.constexpr, K: tl.constexpr, ASM: tl.constexpr):
    logical = tl.arange(0, 256)
    value0, value1, row, lane = tl.inline_asm_elementwise(
        ASM, constraints="=f,=f,=r,=r,l,l,r", args=[X,W,logical],
        dtype=(tl.float32,tl.float32,tl.int32,tl.int32), is_pure=False, pack=1)
    tl.store(OUT+row, value0, (lane == 0) & (row < N))
    tl.store(OUT+row+1, value1, (lane == 0) & (row+1 < N))


def compile_configuration(name="gateup"):
    """Fixed compile arguments only; no eligibility or GPU launch wrapper."""
    n,k = CONFIGS[name]
    return {"N":n,"K":k,"ASM":make_asm(k),"grid":(n//16,),
            "num_warps":8,"num_stages":1,"enable_fp_fusion":False}
_SGLANG_APACHE2_LICENSE = """
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright 2023-2024 SGLang Team

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
