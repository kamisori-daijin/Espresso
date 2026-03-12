# Espresso Post-Decode Optimization Roadmap

**Date:** 2026-03-11
**Status:** Approved for execution
**Purpose:** Turn Espresso from a narrow decode benchmark into an exceptional local AI runtime and product story by implementing the remaining high-leverage optimization programs in ranked order.

## Mission

Espresso already proved a strong but narrow result: exact local ANE generation can beat a matched CoreML baseline on a constrained decode contract. That is not enough for a company-defining product.

The next phase must widen the advantage from:

1. one fast benchmark
2. to a repeatable systems advantage
3. to a product capability CoreML structurally cannot match

The five target outcomes are:

1. exact multi-token generation with materially lower verifier cost
2. strong concurrent local serving across multiple live chats
3. low cold-start latency and low time-to-first-token
4. useful on-device personalization or adaptation
5. ANE-native architecture/runtime design that is not just "transformers, but faster"

## Decision Rubric

Scores are from 1 to 10.

- `Value`: user-facing and company-building leverage
- `Difficulty`: combined research, systems, and implementation cost
- `YC points`: how strongly the result improves the demo, story, and moat

## Strategic Ranking

| Rank | Workstream | Value | Difficulty | YC points | Why it matters |
|------|------------|:-----:|:----------:|:---------:|----------------|
| 1 | Exact multi-token verification and state replay | 10 | 9 | 10 | Converts the current decode lead into fewer full passes per output token while preserving exactness. |
| 2 | Concurrent serving and multi-stream scheduling | 9 | 6 | 9 | Strongest product-facing proof that Espresso can power many local conversations at once. |
| 3 | Cold-start compile/init and model residency | 9 | 7 | 8 | Warm benchmarks do not matter if the product still feels slow to start. |
| 4 | Prefill, batched verification, and TTFT | 8 | 6 | 8 | Improves the first user-visible latency and compounds well with multi-token exact generation. |
| 5 | Training and personalization throughput | 9 | 10 | 10 | Potentially the strongest moat because CoreML does not offer direct ANE training. |
| 6 | ANE-native architecture co-design | 8 | 10 | 9 | Long-term moat: stop forcing a GPU/transformer worldview onto ANE-first hardware. |
| 7 | Hybrid ANE + Metal overlap | 6 | 9 | 6 | Worth exploring only if it proves a clear end-to-end win rather than engineering novelty. |

## Repo Seams That Matter Now

- `Sources/Espresso/GenerationHarness.swift`
  - exact two-token branch-state promotion
  - speculative generation harness
  - recurrent verify blocker: live state cannot be replayed safely without a scratch cursor
- `Tests/EspressoTests/GenerationHarnessHardwareTests.swift`
  - exact two-token benchmark seams
  - speculative benchmarks
  - concurrent multistream ANE vs CoreML benchmark seams
- `Sources/ANEInterop/ane_interop.m`
  - compile cache policy
  - model wiring and residency-related options
  - performance stats hooks
- `Sources/Espresso/DecodeForwardPass.swift`
  - hybrid ANE + Metal decode path
- `Tests/ANERuntimeTests/FusedDecodeKernelTests.swift`
  - hardware-validated compile-failure seam for fused two-layer decode
  - use only as negative evidence unless a new implementation changes the failure mode
- `docs/fused-decode-and-next-steps.md`
  - measured optimization history
  - dead ends, blocked private API routes, and already validated wins
  - older execution-order notes may be historical and superseded by later entries
- `docs/vc-probe-results.md`
  - blocked VirtualClient and SharedEvent avenues
- `docs/architecture-initial-state.md`
  - direct ANE training architecture
  - recompile and I/O costs

## Execution Principles

1. Keep the current benchmark claim intact while widening the contract.
2. Benchmark before and after every major change in the same session where possible.
3. Prefer product-relevant metrics over isolated microbenchmarks.
4. Do not retry proven dead ends unless new evidence appears.
5. Every phase needs a success metric, a kill criterion, and a demo artifact.

## Common Benchmark Contract

Every workstream below must report:

- median and p95 latency
- aggregate throughput where relevant
- compile/init time
- warm vs cold behavior where relevant
- exactness/parity status where relevant
- same-session matched CoreML comparison when a CoreML baseline exists

Minimum benchmark families:

1. single-stream decode
2. multi-stream serving
3. TTFT and prompt-length scaling
4. exact multi-token accepted work per pass
5. compile/init cold vs warm
6. training/adaptation end-to-end step time

## Execution Order

The implementation order is value-first, but still respects dependency structure.

### Phase 0: Measurement Hardening

**Priority:** Mandatory substrate before more optimization work

**Objective**

Make every upcoming improvement legible in one scoreboard instead of scattered notes.

**Implementation**

- unify benchmark output for:
  - single-stream token latency
  - TTFT
  - compile/init
  - exact accepted tokens per pass
  - multi-stream aggregate TPS
  - warm vs cold deltas
- emit benchmark metrics in machine-readable JSON
- include timestamp + git SHA (or `no git repo`) in every artifact format:
  - `json`
  - `csv`
  - `md`
- formalize required-vs-optional probe gates so optional probes cannot mask required artifact failures
- require `N/A` plus exact blocker evidence whenever a metric cannot be measured
- extend checked-in artifact generation for the new scorecard
- keep same-session CoreML comparisons whenever available

**Primary files**

- `Tests/EspressoTests/GenerationHarnessHardwareTests.swift`
- `Sources/EspressoBench/*`
- `artifacts/benchmarks/`

**Success criteria**

- one reproducible report can show progress across all later phases
- all future wins can be compared against the same baseline set
- the benchmark substrate cannot be marked done until JSON artifacts, timestamp/SHA provenance, explicit gate semantics, and `N/A` blocker reporting are all present

### Phase 1: Exact Multi-Token Verification and State Replay

**Strategic rank:** 1

**Objective**

Remove the current recurrent verify bottleneck and make exact multi-token generation the primary engine of speedup.

**Why now**

This is the cleanest path from "fast decode" to "less work per generated token." It also directly compounds with the already-built exact two-token path, fused recurrent trunks, and batched heads.

**Known blocker**

- `GenerationHarness.swift` currently throws on recurrent `verify()` because replay would mutate live recurrent state without a scratch cursor.

**Implementation**

1. Design a scratch-cursor state model for recurrent verification.
2. Implement replay-safe recurrent verify for exact candidate checking.
3. Reuse branch-state promotion for accepted work instead of recomputing live state.
4. Extend exact generation beyond the current two-token path where acceptance makes sense.
5. Add an explicit nonzero-token hardware correctness seam that compares the new replay/verify path against an independent CPU teacher before trusting off-echo throughput.
6. Benchmark accepted exact tokens per pass, verifier trunk cost, verifier logits cost, and state advance cost.

**Primary files**

- `Sources/Espresso/GenerationHarness.swift`
- `Sources/Espresso/RWKVStyleTwoStepRecurrentDecode.swift`
- `Sources/Espresso/RWKVStyleFusedTwoLayerTwoStepRecurrentDecode.swift`
- `Tests/EspressoTests/GenerationHarnessHardwareTests.swift`

**Success criteria**

- recurrent verify works without parity regressions
- at least one nonzero-token hardware seam proves the ANE replay/verify path matches an independent CPU teacher
- the 6-layer non-echo exact path materially beats exact one-token control
- accepted exact tokens per pass rises without inflating verifier cost
- same-session benchmark artifact is reproducible

**Kill criteria**

- if scratch-state replay materially corrupts parity or doubles memory cost without throughput win, isolate the code path and stop scaling candidate count until a smaller state model exists

### Phase 2: Concurrent Serving and Scheduler Design

**Strategic rank:** 2

**Objective**

Prove Espresso can serve multiple local streams efficiently, not just one benchmark stream.

**Why now**

This is the strongest immediate product proof. A YC demo where one Mac runs several live chats is more convincing than a narrow single-stream chart.

**Implementation**

1. Baseline 1 to 6 stream scaling on current best ANE path and matched CoreML.
2. Build a scheduler around compile reuse, model residency, and per-stream state pools.
3. Reduce contention on shared surfaces and model instances.
4. Report aggregate TPS, per-stream TPS, median latency, and p95 latency by stream count.

**Primary files**

- `Tests/EspressoTests/GenerationHarnessHardwareTests.swift`
- `Sources/Espresso/GenerationHarness.swift`
- `Sources/ANEInterop/ane_interop.m`

**Success criteria**

- aggregate throughput scales materially with stream count
- Espresso beats matched CoreML at every tested stream count
- p95 degradation remains acceptable as streams rise

**Kill criteria**

- if shared-resource contention collapses per-stream latency beyond usable product bounds, split the work into residency management and stream scheduler subproblems before continuing

### Phase 3: Cold-Start Compile/Init and Model Residency

**Strategic rank:** 3

**Objective**

Shrink cold-start and warm-start latency until Espresso feels product-ready.

**Why now**

A fast steady-state path with a bad init wall is not a viable product loop.

**Implementation**

1. Measure cold compile, warm compile, model load, and first-eval latency separately.
2. Formalize model residency and compile caching policy in the runtime.
3. Evaluate `compiledModelExists`, `purgeCompiledModel`, memory wiring, and option combinations that affect load behavior.
4. Add a benchmark mode that distinguishes:
   - first launch
   - warm launch
   - already-resident model

**Primary files**

- `Sources/ANEInterop/ane_interop.m`
- `Sources/EspressoBench/ANEDirectBench.swift`
- `Sources/EspressoBench/CoreMLBench.swift`
- `Tests/EspressoTests/GenerationHarnessHardwareTests.swift`

**Success criteria**

- cold-start time is materially reduced
- warm init becomes predictable and fast
- compile/init cost is explicit in all public benchmark artifacts

### Phase 4: Prefill, Batched Verification, and TTFT

**Strategic rank:** 4

**Objective**

Reduce the latency users feel before the model starts responding and improve prompt-length scaling.

**Why now**

TTFT and prompt ingest matter at least as much as steady-state tok/s for interactive use.

**Implementation**

1. Benchmark prompt-length scaling for prefill and TTFT.
2. Batch prepared activations anywhere verifier-side head work is still being done one activation at a time.
3. Extend the exact path scorecard to include TTFT and prompt-length bins.
4. Reuse the best batched-head path across exact multi-token generation and recurrent prompt ingest.

**Primary files**

- `Sources/Espresso/GenerationHarness.swift`
- `Tests/EspressoTests/GenerationHarnessHardwareTests.swift`
- `docs/fused-decode-and-next-steps.md`

**Success criteria**

- TTFT drops materially on realistic prompt lengths
- batched verification reduces verifier-side head cost
- the prompt-length curve remains favorable against matched CoreML

### Phase 5: Training and Personalization Throughput

**Strategic rank:** 5

**Objective**

Turn Espresso's direct ANE training path into a product capability, not just a research artifact.

**Why now**

If Espresso can adapt models locally on Apple hardware, the company story becomes much stronger than inference acceleration alone.

**Implementation**

1. Build a hard benchmark suite for the current training loop:
   - forward
   - backward
   - optimizer
   - recompile overhead
   - checkpoint/restart overhead
2. Reduce obvious I/O and copy tax in the training path where measurements justify it.
3. Add a bounded personalization mode before chasing full general fine-tuning:
   - small adapter or sidecar-style update path
   - measurable quality delta on a fixed local task
4. Convert checkpoint/restart from a recovery hack into a product primitive for long local adaptation jobs.

**Primary files**

- `Sources/Espresso/ForwardPass.swift`
- `Sources/Espresso/BackwardPass.swift`
- `Sources/Espresso/GradientAccumulator.swift`
- `Sources/Espresso/Checkpoint.swift`
- `docs/architecture-initial-state.md`

**Success criteria**

- one end-to-end local adaptation demo exists
- adaptation quality moves on a held-out evaluation set
- training throughput and restart overhead are benchmarked and documented

**Kill criteria**

- if full-weight training remains dominated by compile cost, narrow scope to small mutable adaptation surfaces instead of forcing full-model tuning

### Phase 6: ANE-Native Architecture Co-Design

**Strategic rank:** 6

**Objective**

Design models and generation contracts that fit ANE constraints instead of inheriting GPU-first assumptions.

**Why now**

This is likely the deepest long-term moat, but it only matters after Espresso can already win on exactness, concurrency, and usability.

**Implementation**

1. Treat the current recurrent and fused pair/triplet paths as the beginning of an ANE-native family, not a side branch.
2. Compare architecture families on:
   - exact multi-token acceptance
   - concurrent throughput
   - TTFT
   - adaptation friendliness
3. Use the benchmark suite to decide whether recurrent or hybrid contracts should replace the current headline path.

**Primary files**

- `Sources/Espresso/GenerationHarness.swift`
- `Sources/ANERuntime/RWKVStyle*.swift`
- `Sources/MILGenerator/RWKVStyle*.swift`

**Success criteria**

- at least one ANE-native architecture wins on product-relevant metrics, not only on a synthetic narrow benchmark

### Phase 7: Hybrid ANE + Metal Overlap

**Strategic rank:** 7

**Objective**

Only pursue this if the first six phases leave clear measured value on the table.

**Why last**

This avenue is expensive and easy to romanticize. It should survive only if it wins real end-to-end latency or aggregate throughput.

**Implementation**

1. Use the existing hybrid path as a measurement seam.
2. Measure overlap, not just stage timings in isolation.
3. Keep the route only if it improves complete user-facing runs.

**Primary files**

- `Sources/Espresso/DecodeForwardPass.swift`
- `Tests/EspressoTests/HybridDecodeForwardPassTests.swift`

**Success criteria**

- at least a material end-to-end win over the current best path

**Kill criteria**

- if overlap does not produce a durable real-world gain, retire the avenue and stop investing in it

## Explicitly Deferred

### Quantization

Quantization is worth revisiting later, but it is not a top-phase bet right now.

Reasons:

- the live runtime is still fundamentally FP16 on-device with FP32 host storage
- exact multi-token, serving, TTFT, and personalization are higher-leverage for both product and YC
- a half-built quantized path would increase complexity before the current system-level advantages are fully harvested

## YC Milestone Ladder

### Milestone A: "Fast Exact"

- recurrent exact verify works
- the best exact multi-token path clearly beats exact one-token control
- non-echo same-session benchmark artifact is checked in

### Milestone B: "Fast for Real Users"

- strong TTFT and prompt-length results
- warm-start behavior is product-acceptable
- public benchmark copy can describe more than one narrow decode contract

### Milestone C: "Fast at Product Scale"

- 1 to 6 stream multistream chart beats matched CoreML
- aggregate throughput story is demo-ready

### Milestone D: "Unique Capability"

- one local adaptation or personalization workflow works end to end
- Espresso demonstrates a capability CoreML does not expose directly

## Definition of Done

This roadmap is successful when Espresso can credibly claim:

1. exact local generation that beats matched CoreML on more than one narrow benchmark
2. strong multi-stream local serving
3. acceptable cold-start and TTFT behavior
4. at least one useful local personalization or adaptation demo
5. a visible technical direction that is ANE-native rather than benchmark-native
