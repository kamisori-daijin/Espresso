# TODO

- [x] Read the minimal repo context for optimization surfaces beyond decode dispatch: `README.md`, key docs, and the smallest set of source files covering prefill, training, recurrent/speculative decode, memory bandwidth, compilation caching, batching, quantization, and benchmark metrics.
- [x] Extract the highest-leverage optimization surfaces that are not already the current decode-dispatch focus, and rank them by likely impact and implementation feasibility.
- [x] Return a concise prioritized list with direct file/doc references and a short rationale for each area.

- [x] Promote `ebd3c38` from an internal milestone to a public-release surface without changing the measured claim.
- [x] Rewrite the top-level README so the non-echo exact decode result is the first public benchmark story, with explicit caveats and one-command repro.
- [x] Add a checked-in benchmark artifact for the non-echo release claim that is stable enough to link publicly.
- [x] Add a release note document tied to the exact claim, exact caveats, repro command, and reference commit.
- [x] Create a local release tag for the public packaging milestone and leave the worktree clean apart from untracked raw result bundles.
- [x] Update lessons, Wax notes, handoff, and review with the release-packaging outcome.

# Review

- Current code/control milestone is `ebd3c38`:
  - exact two-step `1.0806302083333332 ms/token`
  - exact one-token ANE control `1.0957500000000002 ms/token`
  - matched zero-weight `6`-layer CoreML `5.085307291666668 ms/token`
  - exact two-step speedup vs CoreML `4.7583224488025415x`
  - exact one-token ANE control speedup vs CoreML `4.640428016426192x`
  - parity `match`
  - committed exact tokens/pass `2`
  - accepted future tokens/pass `1`
- The code/result is frozen enough for recovery, but the repo is not yet public-release quality:
  - README now leads with the new non-echo decode claim
  - checked-in benchmark artifacts now exist under `artifacts/benchmarks/exact-decode-non-echo/`
  - release notes now exist under `docs/releases/2026-03-11-non-echo-exact-decode.md`
  - the remaining packaging step is to tag and push the milestone, not to invent more prose
- The public claim must stay constrained:
  - non-echo local artifact family
  - exact parity preserved
  - explicit `identity-zero-trunk` backend
  - not a pretrained production checkpoint claim
- README hardening pass:
  - lead now scopes the performance number to the reproducible non-echo local-artifact benchmark
  - repro notes now state that first-run `coremltools` bootstrap may occur
  - public copy now avoids broader "CoreML in general" wording

## Current Analysis Task

- [x] Review the public benchmark framing in `README.md`, the release note, the checked-in benchmark artifact, and the reproduction script.
- [x] Review the lab notebook for prior benchmark credibility regressions, blocked avenues, and already-proven architectural seams.
- [x] Synthesize YC-differentiation bets focused on stronger benchmark credibility, productizable advantages, and demo-worthy capabilities.

## Analysis Review

- The current public claim is credible but intentionally narrow: non-echo local artifact, `identity-zero-trunk`, `6` layers, `32` sequence tokens, same-session CoreML `.cpuAndNeuralEngine`, and exact committed-token accounting.
- The repo already contains stronger latent assets than the headline suggests: recurrent checkpoint export + offline acceptance gates, exact two-token branch-state promotion, multi-layer fused decode scaffolding, serving/concurrency benchmarks, and a training/backprop runtime that CoreML cannot offer.
- The strongest next differentiation stories are the ones that either widen the benchmark contract without losing reproducibility, or demonstrate capabilities CoreML structurally cannot match even if raw latency converges.
- Additional repo-analysis review:
  - the next biggest levers are algorithmic/runtime-level, not more single-stream dispatch shaving
  - the codebase already contains prefill, speculative, recurrent fusion, batched verification, training, and benchmark infrastructure, but several surfaces remain partially connected rather than productionized
  - there is no live quantized path; the current stack remains FP16 on-device with FP32 host storage and fixed-shape model assumptions

## Post-Decode Optimization Roadmap

- [x] Create a ranked roadmap document covering value, difficulty, and YC points for the next optimization programs after decode dispatch.
- [x] Write the evergreen roadmap in `docs/optimization-roadmap.md`.
- [x] Write the implementation plan in `docs/plans/2026-03-11-post-decode-optimization-roadmap.md`.
- [ ] Phase 0: harden the shared benchmark scorecard across token latency, TTFT, compile/init, multistream throughput, and exact accepted work per pass.
- [ ] Phase 1: implement recurrent exact verification and scratch-cursor state replay.
- [ ] Phase 2: optimize concurrent serving and multi-stream scheduling.
- [ ] Phase 3: reduce cold-start compile/init latency and formalize model residency behavior.
- [ ] Phase 4: optimize prefill, batched verification, and TTFT.
- [ ] Phase 5: benchmark and optimize training/personalization throughput.
- [ ] Phase 6: iterate on ANE-native architecture co-design.
- [ ] Phase 7: evaluate hybrid ANE + Metal overlap only after the higher-value phases are complete.

## Roadmap Review

- The stable entry point now lives in `docs/optimization-roadmap.md`, with the dated execution plan in `docs/plans/2026-03-11-post-decode-optimization-roadmap.md`.
- The implementation order stays value-first, but still respects dependency structure:
  - first make exact multi-token generation cheaper
  - then make serving and startup behavior product-grade
  - then turn direct ANE training/adaptation into a visible moat
- Quantization is explicitly deferred until the current exactness, serving, TTFT, and personalization surfaces are productized.
