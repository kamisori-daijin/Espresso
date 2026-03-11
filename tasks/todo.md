# TODO

## 2026-03-12 — EspressoBench SwiftUI macOS app

- [x] Add a dedicated `EspressoBenchApp` SwiftUI macOS target and executable product.
- [x] Build an app model that launches `espresso-bench`, streams live logs, and loads generated result artifacts.
- [x] Add a SwiftUI dashboard with run configuration, run history, summary cards, charts, and artifact browsing.
- [x] Package the app into a runnable `.app` bundle with the CLI embedded alongside it.
- [x] Verify the new app target builds cleanly and document the app work in review notes.

## 2026-03-12 — EspressoBench review follow-up

- [x] Restore `espresso-bench` CLI compatibility for decode, inference-only, kernel profiling, and chaining probe flows.
- [x] Reinstate benchmark artifact outputs (`summary.json`, legacy CSV filenames, kernel profile CSVs) consumed by tests and automation.
- [x] Restore `scripts/generate_coreml_model.py` support for multi-layer and zero-weight generation.
- [x] Re-align default benchmark layer/model behavior so ANE/CoreML comparisons are valid by default.
- [x] Verify with targeted build/test/smoke coverage for the restored bench contracts.

## 2026-03-12 — EspressoBenchApp production review

- [x] Inspect the SwiftUI app files under `Sources/EspressoBenchApp`.
- [x] Trace the CLI launch path, artifact loading, and data flow boundaries.
- [x] Review SwiftUI/Observation/accessibility/app-architecture quality gaps after first build.
- [x] Return prioritized findings and highest-value follow-ups without editing app files.

## 2026-03-12 — EspressoBench ANE vs Core ML executable

- [x] Align `Package.swift` `EspressoBench` product/target surface with the requested dependencies and linker settings.
- [x] Implement `Sources/EspressoBench/BenchmarkRunner.swift` with sorted statistics, percentile interpolation, signposts, and stderr progress.
- [x] Implement `Sources/EspressoBench/FLOPCalculator.swift` with the full 7-component forward-pass accounting and utilization helpers.
- [x] Implement `Sources/EspressoBench/ResultsFormatter.swift` with locale-stable report formatting and CSV export.
- [x] Implement `Sources/EspressoBench/ThermalMonitor.swift` with sampled sustained-run tracking.
- [x] Implement `Sources/EspressoBench/ANEDirectBench.swift` against the real move-only `ForwardPass.runTimed` API.
- [x] Implement `Sources/EspressoBench/CoreMLBench.swift` with graceful missing-model handling and three compute-unit baselines.
- [x] Implement `Sources/EspressoBench/main.swift` CLI flow for ANE/Core ML/thermal runs and result export.
- [x] Implement `scripts/generate_coreml_model.py` for a channel-first fp16 transformer layer `.mlpackage`.
- [x] Implement `scripts/run_power_benchmark.sh` for powermetrics-wrapped sustained runs.
- [x] Keep `.gitignore` aligned for benchmark results and generated Core ML packages.
- [x] Verify sequential build gates while editing and finish with `swift build -c release --target EspressoBench`.
- [x] Commit each logical benchmark task with `feat(bench): ...` messages.

- [x] Promote `ebd3c38` from an internal milestone to a public-release surface without changing the measured claim.
- [x] Rewrite the top-level README so the non-echo exact decode result is the first public benchmark story, with explicit caveats and one-command repro.
- [x] Add a checked-in benchmark artifact for the non-echo release claim that is stable enough to link publicly.
- [x] Add a release note document tied to the exact claim, exact caveats, repro command, and reference commit.
- [x] Create a local release tag for the public packaging milestone and leave the worktree clean apart from untracked raw result bundles.
- [x] Update lessons, Wax notes, handoff, and review with the release-packaging outcome.

# Review

## 2026-03-12 — EspressoBench rewrite

- Status: complete.
- Baseline:
  - `swift build --target EspressoBench` passes before the rewrite.
  - `Sources/EspressoBench/` already exists, but the current executable includes extra probe/introspection features and dependency drift beyond the requested ANE-vs-CoreML benchmark contract.
- Key constraints:
  - `TensorBuffer`, `LayerWeights`, and `LayerStorage` are `~Copyable`; ANE direct measurement must avoid capturing them in escaping/generic closures.
  - `ForwardPass.runTimed(...)` is the training-forward timed API available for direct ANE benchmarking.
  - Locale-stable output and attosecond-to-millisecond conversion are explicit verification gates for this task.
- Verification:
  - `swift build -c release --target EspressoBench` succeeded on 2026-03-12.
  - `Package.swift` now exposes `EspressoBench` with the requested direct dependencies.
  - `BenchmarkRunner`, `FLOPCalculator`, `ResultsFormatter`, `ThermalMonitor`, `ANEDirectBench`, `CoreMLBench`, `main.swift`, and the benchmark scripts were updated in this pass.

## 2026-03-12 — Review follow-up fixes

- Status: complete.
- Fix scope:
  - Restored the legacy `espresso-bench` CLI paths for decode, inference-only, profiling, and chaining probe modes.
  - Restored `summary.json`, legacy latency/profile CSV filenames, and the richer kernel profile CSV schema.
  - Restored multi-layer and zero-weight support in `scripts/generate_coreml_model.py`.
  - Re-aligned default layer behavior to the single-layer default Core ML package, and made the power script pass an explicit `LAYERS` value.
- Verification:
  - `swift build -c release --target EspressoBench` succeeded.
  - `./.build/arm64-apple-macosx/debug/espresso-bench --help` shows the restored decode/inference/probe flags.
  - `python3 -m py_compile scripts/generate_coreml_model.py scripts/ane_bench_sweep.py` succeeded.
  - `ANE_HARDWARE_TESTS=1 swift test --filter DecodeChainingInteropTests/test_external_prepare_probe_isolated_from_test_harness` passed.

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

## 2026-03-12 — EspressoBench SwiftUI macOS app

- Status: complete.
- Implementation scope:
  - Added `EspressoBenchApp` as a native SwiftUI macOS executable target and product.
  - Wrapped the existing `espresso-bench` CLI instead of reimplementing benchmark logic.
  - Added live process streaming, typed `summary.json` loading, CSV latency plotting, artifact browsing, and direct bundle packaging with the CLI embedded.
  - Made launch provenance explicit in the UI, stabilized history identity on output-directory paths, batched log updates, and added accessibility labels for cards/charts.
- Verification:
  - `swift build --target EspressoBenchApp` succeeded.
  - `swift build -c release --target EspressoBenchApp` succeeded.
  - `swift build -c release --target EspressoBench` succeeded.
  - `./scripts/package_espresso_bench_app.sh` succeeded and produced `.build/apps/EspressoBench.app`.
