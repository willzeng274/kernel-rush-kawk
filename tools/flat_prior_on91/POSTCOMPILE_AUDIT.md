No runtime release follows compiler success. Root must independently review the
complete frozen source/IR/PTX audit and artifacts before any GPU/official action.

Scope: exactly56 new keys from actual baseline/candidate B1..64 wrapper captures:
18 draft,18 accepted-pair publication,18 acceptance,2 compact. The eight unchanged
production keys are excluded. No archived exact compiler receipt reuse is claimed.

Before first build, compare every field in all512 expected native records, both
baseline and candidate: source/dependencies, arguments/shapes/dtypes, signature,
constexpr and equal-to-one constants, pointer/scalar divisibility, all parsed
backend options (including debug=None), target/compiler, grid and full keys.
Verify exactly64 candidate keys; set difference against baseline must equal the
56 frozen schedule keys and intersection must equal the eight excluded keys.

For every successful case preserve TTIR,TTGIR,LLVM IR,PTX,cubin,verbose ptxas cubin,
resource log, native config, per-artifact SHA256, compile/assembler/total timing,
compiler stdout/stderr, attempt marker and final summary. Require exact-source
hashes before and after, one attempt per case, no spill/local/atomic/MMA surprise,
zero shared bytes, and recorded registers/stack. A timeout, mismatch, assembler
error or unexpected resource result stops the sole worker without retry.

Draft (R3..64,S1..5,SP3..11): inspect each unrolled dependency in current pair-aware
body. Root and each node must be stored before all consuming loads; reconcile
scalar/redundant-lane stores and cross-warp loads from actual PTX, not the CPU
sequential model. Prove parent[i]<i and valid token/rank table address bounds;
spine slot reads clamp into SP, fresh predicate checks off>=0,off+S<=SP and the
previous delayed token, overrides only rank0 slots with nonnegative proposals.
Verify maximum delay MAXA+1 consumes at most SP-1. Pair hash is uint64, b*4096
isolates sequences, keys must match before reading values, negative pair values
fall back, root/node key packing uses the consumed predecessor correctly.

Publication: for plen=-1 no stores; for live sequences consume root plus exactly
plen accepted rows, within path_idx[B,MAXA],node_keys[B,R],top[B*R,8]. Prove step
order and masked bounds, values-before-key publication, latest accepted row wins
same-slot collisions, and root_prev is the consumed endpoint rather than bonus.
No next draft or other sequence may read/write these slots concurrently.

Acceptance: inspect full current longest-path body, including flat R4 where the
branch is newly active. Prove unique child bits, signed int64 masks interpreted
as uint64 including bit63, padded lanes cannot poison bit0, invalid ancestor
excludes every descendant, endpoint tie chooses lowest node, and replacement
requires strictly greater clipped progress. Verify exact first-match retention
on ties/clipping, MAXA-bounded reverse reconstruction, accepted tokens plus one
bonus, full nseen count vs host clipping, frozen/root behavior, and CAP/GUARD
bounds. Reconcile each load/store width, predicate and offset with HOST_CHECKS.

Compact: inspect both new native groups MAXA5/B==1 and MAXA2/B-divisible-by16;
B remains runtime i32 in the latter, HKV runtime i32=8, CAP runtime i32 divisible
by16, D128 constexpr. Prove K/V address formula for all layer/batch/head programs,
masked frozen/root-only cases, cache bounds, and ordered in-place copies: path
indices strictly increase and path[j]>=j+1, preventing an earlier destination
from corrupting a later source. Verify all eight heads and vector lanes.

CPU preparation demonstrates source geometry/semantics only. It is not proof of
compiled lane ordering, GPU replay correctness, timing spread or improvement.
