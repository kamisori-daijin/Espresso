# Espresso Optimization Roadmap

**Status:** Active
**Last updated:** 2026-03-11
**Primary metric:** product-relevant local generation advantage over matched CoreML
**Scope:** post-decode optimization priorities, ranking, and execution order

## Executive Summary

Espresso's current public benchmark is strong but narrow:

- exact two-step ANE decode: `1.08 ms/token`
- matched CoreML `.cpuAndNeuralEngine`: `5.09 ms/token`
- measured speedup: `4.76x`

That result proves direct ANE access is real. It does not yet prove Espresso is an exceptional product.

The next material wins are not more single-stream dispatch shaving. They are:

1. exact multi-token verification and state replay
2. concurrent serving across multiple local streams
3. cold-start, residency, prefill, and TTFT improvements

Those three bets are the shortest path from a benchmark story to a company story.

## Scoring Model

All roadmap bets are scored on three axes from 1 to 10:

- `Value`: user-facing and company-building leverage
- `Difficulty`: research plus implementation cost
- `YC points`: benchmark, demo, moat, and product differentiation value

Tie-breakers:

- evidence already present in the repo
- blocked vs unblocked execution path
- credibility of same-session benchmarking
- product/demo leverage over narrow lab wins

## Current Baseline and Constraints

### Baseline

- decode dispatch overhead has already been materially reduced
- the remaining wall is now more algorithmic and runtime-structural than dispatch-bound
- the repo already contains working seams for:
  - exact two-token generation
  - fused recurrent trunks
  - speculative generation harnesses
  - concurrent multistream benchmarks
  - direct ANE training

### Hard constraints

- several private API shortcuts are blocked or entitlement-gated
- the runtime is currently FP16 on-device with FP32 host storage
- many wins are present as partial seams, not yet as product-ready flows

### Proven dead ends

See:

- `docs/fused-decode-and-next-steps.md`
- `docs/vc-probe-results.md`

Do not spend roadmap time retrying blocked VirtualClient, SharedEvent, multi-layer fused decode, or similar routes without new evidence.

Treat those docs as lab notebooks, not a flat current-priority list:

- latest outcome/status sections and matching hardware tests are authoritative
- older "recommended execution order" notes inside the notebook may be historical and superseded

## Ranked Optimization Bets

| Rank | Bet | Category | Value | Difficulty | YC points | Confidence | Decision |
|------|-----|----------|:-----:|:----------:|:---------:|:----------:|----------|
| 1 | Exact multi-token verification and state replay | Algorithmic/runtime | 10 | 9 | 10 | High | Execute first |
| 2 | Concurrent serving and multi-stream scheduling | Product/runtime | 9 | 6 | 9 | High | Execute second |
| 3 | Cold-start compile/init and model residency | Product/runtime | 9 | 7 | 8 | High | Execute third |
| 4 | Prefill, batched verification, and TTFT | Product/runtime | 8 | 6 | 8 | High | Execute fourth |
| 5 | Training and personalization throughput | Capability/moat | 9 | 10 | 10 | Medium | Execute after serving/startup wins |
| 6 | ANE-native architecture co-design | Architecture/moat | 8 | 10 | 9 | Medium | Pursue after core product wins |
| 7 | Hybrid ANE + Metal overlap | Kernel/runtime | 6 | 9 | 6 | Low to medium | Only keep if it wins end to end |

## Why The Top 3 Win

### 1. Exact Multi-Token Verification and State Replay

This is the highest-value route because it reduces total work per output token while preserving exactness. The key blocker is already explicit in `Sources/Espresso/GenerationHarness.swift`: recurrent verify is not safe without scratch-cursor state replay.

### 2. Concurrent Serving

A YC-quality product demo is not "one fast token stream." It is "one local machine can run several live sessions well." The repo already has benchmark seams for this in `Tests/EspressoTests/GenerationHarnessHardwareTests.swift`.

### 3. Cold-Start and Residency

A product loses trust fast if compile/init time dominates the first interaction. Espresso already has useful runtime knobs in `Sources/ANEInterop/ane_interop.m`, but they need to be measured and turned into an explicit residency strategy.

## Priority Buckets

### P0

- exact multi-token verification and state replay
- concurrent serving and multi-stream scheduling
- cold-start compile/init and model residency
- prefill, batched verification, and TTFT

### P1

- training and personalization throughput
- ANE-native architecture co-design

### P2

- benchmark hardening and artifact generation to support all phases

### P3

- hybrid ANE + Metal overlap unless benchmarks clearly force it
- quantization, which remains explicitly deferred for now

## Implementation Plan

The detailed execution plan lives in:

- `docs/plans/2026-03-11-post-decode-optimization-roadmap.md`

Implementation order:

1. Measurement hardening
2. Exact multi-token verification and state replay
3. Concurrent serving and multi-stream scheduling
4. Cold-start compile/init and residency
5. Prefill, batched verification, and TTFT
6. Training and personalization throughput
7. ANE-native architecture co-design
8. Hybrid ANE + Metal overlap only if still justified

## Measurement Protocol

Every phase should report:

- median and p95 latency
- compile/init time
- warm vs cold deltas
- aggregate throughput where relevant
- exactness or parity status where relevant
- same-session matched CoreML comparison where available
- machine-readable JSON artifacts for benchmark metrics
- timestamp + git SHA (or `no git repo`) in every artifact format (`json`, `csv`, `md`)
- explicit required-vs-optional probe gates
- `N/A` plus blocker evidence when performance cannot be measured
- at least one nonzero-token hardware correctness seam against an independent CPU teacher before trusting off-echo ANE throughput claims

Use the existing benchmark/test harnesses as the canonical measurement seams before adding new ad hoc probes.

## Decision Rules

Continue a bet when:

- the benchmark is reproducible
- the win is user-facing or materially compounds later phases
- the route remains unblocked by the known private API constraints

Pause and re-rank when:

- the phase hits an entitlement wall or proven dead-end pattern
- the measured gain is too small to matter at the product level
- a lower-ranked bet now has clearer leverage or lower execution risk

## Roadmap Snapshot

### If the only goal is the next 2 weeks

- implement recurrent verify scratch-cursor state replay
- formalize the benchmark scorecard
- rerun same-session exact multi-token vs CoreML artifacts

### If the goal is maximum demo value

- multistream serving
- warm-start and TTFT improvement
- exact multi-token generation with a broader public benchmark contract

### If the goal is maximum moat

- training and personalization throughput
- ANE-native model/runtime co-design

## Linked Source Docs

- `docs/fused-decode-and-next-steps.md`
- `docs/vc-probe-results.md`
- `docs/architecture-initial-state.md`
- `docs/plans/2026-03-11-post-decode-optimization-roadmap.md`
