# Unchanged Krxfty public baseline

Source repository: https://github.com/IshaanBansal2006/starter

Copied revision: `7f0f98c6d7a12ab973e499f58646f251b9fb9b49`.
Engine Git tree: `e4698fc2cab13943e0e646ba133e5dea10a2acb8`.
Observed clone HEAD: `eab02a957713518baa5e1c91d79e84bff592c2a2`.

The repository log associates run `823d4533-3e4a-45a0-8e6a-e20add798c26`
with score `1109.2767773288615` and note `commit 7f0f98c`. Its heading
`@ 8db3965` names the logging checkout: `agent/dryft_cli.py` constructs the
heading with `git_rev()`, while `watch()` supplies the note from the run's
`commitSha`. The entry was committed in descendant `c0d8851`.

This establishes the author's source association, not an independently
authenticated server archive or score attestation. `8db3965` is absent from
the fetched commit objects; no equivalence with its tree is claimed.

Every file under `engine/` was copied from the recorded Git blob, with its
bytes and file set verified. No defaults, adapters, kernels or dependencies
were changed. All 14 Python files are needed by the static import closure.
`MANIFEST.json` records Git blob IDs and SHA256 hashes. Only `engine/` is the
submission source root; this provenance and manifest are not runtime files.

Known risks, including the original-prompt fallback splice fixed later in
`c0d8851`, are intentionally retained. See
`work/research/krxfty_public_baseline/REPORT.md` for the bounded source audit.
No downloaded code was executed locally. This commit prepares the first official unchanged comparison against retained #75 (1013.8 tok/s).
