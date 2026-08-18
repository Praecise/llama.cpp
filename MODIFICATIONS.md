# Modifications to llama.cpp

This is **not** stock upstream llama.cpp. The copy vendored here carries local
changes, and anything built against it should assume divergence rather than
parity with an upstream release.

Upstream: https://github.com/ggml-org/llama.cpp — Copyright (c) 2023-2026 The
ggml authors, MIT License. Those terms continue to govern this directory; see
`LICENSE` here and `LICENSE-MIT` at the repository root. The Apache-2.0 licence
applied to Praecise Engine's own crates does not relicense this code.

## What was changed

- **Architecture support** — model families the vendored revision carries that a
  matching upstream release may not: `QWEN35` (8 files), `MUSE_GLIMMER` /
  `muse_glimmer` (4 files, back-ported from the upstream PR and adapted to this
  API), `KIMI_K3` (4 files).
- **NVFP4 quantisation** (55 files) — including the block-scaled FP4 MMA path
  used on Blackwell.
- **DGX Spark / GB10 targeting** (2 files) — `GGML_CUDA_CC_DGX_SPARK` and the
  sm_121 handling around it.
- **Blackwell MMVQ occupancy** (1 file) — `MMVQ_PARAMETERS_BLACKWELL` halves the
  warps per block for the decode-shaped end of the range. Upstream has no
  Blackwell MMVQ tuning at all, so this is local work rather than a back-port.
  Measured on sm_120, +2.10% on Q3_K and nothing on Q4_K; the numbers and the
  method are in the comment beside the code.
- **Grammar robustness** — the lazy-grammar paths were made non-fatal: a grammar
  that cannot accept a piece, or that reaches end-of-generation mid-rule, now
  stops constraining instead of calling `abort()`. Aborting on malformed *model
  output* is unusable in a server, where it takes every other in-flight request
  down with the turn that upset it.

## Why this file exists

Two reasons, one legal and one practical. Redistributing modified third-party
code without saying so is poor practice regardless of what the licence strictly
compels, and Apache-2.0 §4(b) makes carrying such notices an explicit obligation
for material under that licence. And practically: this fork cannot simply be
replaced with an upstream tarball. Doing so drops architectures the models we
serve depend on.
