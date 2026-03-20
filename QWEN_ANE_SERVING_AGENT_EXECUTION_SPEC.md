# Qwen ANE Serving Agent Execution Spec

Status: canonical tracked execution plan
Audience: coding agent working in a separate worktree
Primary repo: `/Users/chriskarani/CodingProjects/Espresso`
Sibling repo: `/Users/chriskarani/CodingProjects/Edgerunner`

## Mission

Complete the Qwen ANE serving-runtime rewrite end to end in a separate worktree, with small PR-quality commits that preserve the exact-CPU Qwen oracle and culminate in a new ANE-first serving backend.

The agent must execute this plan in order. Do not skip gates. Do not improvise a different architecture unless a gate fails and the fallback rules below are triggered and logged.

## Done Definition

The work is complete only when all of the following are true:

1. A dedicated serving backend exists, separate from the verifier backend.
2. The serving backend uses serving-native artifacts, not verifier float32 sidecars, for its hot path.
3. `0.6B` serving parity passes:
   - cold-start
   - late-prefix token
   - `Hello` 8-token continuation
4. `1.7B` targeted parity passes:
   - late-prefix token
   - `Hello` 8-token continuation
5. `4B` targeted parity passes:
   - late-prefix token
   - `Hello` 8-token continuation
   - if full `Hello` parity is still not practical, one exact remaining blocker is isolated with hard evidence
6. Warm decode tok/s on the new serving backend materially exceeds the current Espresso Qwen path on the same machine.
7. A reproducible benchmark command exists for:
   - warm decode
   - first-token latency
   - build/compile time
   - prepare time
8. The persistent log and `tasks/todo.md` are updated with every meaningful experiment and the final grounded outcome.

## Hard Constraints

- Never delete these protected models:
  - `/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf`
  - `/tmp/edgerunner-models/Qwen3-1.7B-Q8_0.gguf`
  - `/tmp/edgerunner-models/Qwen3-4B-Q8_0.gguf`
- Never delete any `.artifacts` directory.
- Only disposable `/var/folders/.../T/espresso_gguf_*` temp conversion directories may be removed.
- Keep the current exact-CPU Qwen path intact as the correctness oracle until the serving backend fully replaces the old serving path.
- Do not reopen known dead ends without new hard evidence:
  - generic tokenizer mismatch
  - converter mismatch for already-compared tensor surfaces
  - ANE/Metal as the primary remaining `0.6B` correctness issue
  - plain FP16 sidecar storage as the remaining semantic explanation

## Research Basis

This plan is grounded in:

- Apple Machine Learning Research, "Deploying Transformers on the Apple Neural Engine"
- Apple Machine Learning Research, "Deploying Attention-Based Vision Transformers to Apple Neural Engine"
- Apple Core ML stateful model guidance
- Apple Core ML optimization and quantization guidance
- `llama.cpp` Apple-silicon / Metal architecture and performance notes
- MLX / `mlx-lm` unified-memory, lazy execution, rotating KV cache, and prompt-cache design

Primary references:

- https://machinelearning.apple.com/research/neural-engine-transformers
- https://machinelearning.apple.com/research/vision-transformers
- https://apple.github.io/coremltools/docs-guides/source/stateful-models.html
- https://apple.github.io/coremltools/docs-guides/source/opt-workflow.html
- https://apple.github.io/coremltools/source/coremltools.optimize.coreml.quantization.html
- https://github.com/ggml-org/llama.cpp/blob/master/README.md
- https://github.com/ggml-org/llama.cpp/blob/master/docs/development/token_generation_performance_tips.md
- https://github.com/ml-explore/mlx/blob/main/README.md
- https://github.com/ml-explore/mlx-lm/blob/main/README.md

## Worktree Setup

Create separate worktrees for both repos under a shared parent directory. This is required because `Package.swift` depends on `../Edgerunner`; the Espresso worktree must therefore have a sibling directory literally named `Edgerunner`.

### Shared worktree root

```bash
WORKTREE_ROOT=/Users/chriskarani/CodingProjects/worktrees/qwen-ane-serving
mkdir -p "$WORKTREE_ROOT"
```

### Espresso worktree

```bash
cd /Users/chriskarani/CodingProjects/Espresso
BASE_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
git worktree add "$WORKTREE_ROOT/Espresso" -b qwen-ane-serving-runtime "$BASE_BRANCH"
```

### Edgerunner worktree

```bash
cd /Users/chriskarani/CodingProjects/Edgerunner
BASE_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
git worktree add "$WORKTREE_ROOT/Edgerunner" -b qwen-ane-serving-runtime "$BASE_BRANCH"
```

### Active working directories

- Espresso: `/Users/chriskarani/CodingProjects/worktrees/qwen-ane-serving/Espresso`
- Edgerunner: `/Users/chriskarani/CodingProjects/worktrees/qwen-ane-serving/Edgerunner`

All relative file references below assume those worktree roots.

## Baseline Capture

Before touching code:

1. Build the runner:

```bash
cd /Users/chriskarani/CodingProjects/worktrees/qwen-ane-serving/Espresso
swift build --product EspressoGGUFRunner
swift build --product espresso-generate
```

2. Re-run the protected `0.6B` correctness gate:

```bash
./.build/debug/EspressoGGUFRunner verify-qwen \
  /tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf \
  /Users/chriskarani/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-8bit/snapshots/11de96878523501bcaa86104e3c186de07ff9068 \
  --fresh --keep-weight-dir
```

3. Re-run focused regression tests:

```bash
swift test --filter 'QwenGGUF(Regression|SidecarComparison)Tests'
swift test --filter GGUFModelLoaderTests
swift test --filter EspressoGenerateTests
```

4. Append the baseline results to:

- `tasks/qwen_gguf_remaining_correctness_log.md`
- `tasks/todo.md`

If any baseline check fails, stop and fix that failure before continuing.

## Package Changes

Add a new target and test target in `Package.swift`:

```swift
.target(
    name: "EspressoQwenServing",
    dependencies: [
        "RealModelInference",
        "ModelSupport",
        "ANERuntime",
        "ANETypes",
        "ANEInterop",
        "CPUOps",
        "Espresso",
    ],
    path: "Sources/EspressoQwenServing",
    swiftSettings: [.swiftLanguageMode(.v6)],
    linkerSettings: [
        .linkedFramework("Accelerate"),
        .linkedFramework("IOSurface"),
    ]
),
.testTarget(
    name: "EspressoQwenServingTests",
    dependencies: [
        "EspressoQwenServing",
        "RealModelInference",
        "ModelSupport",
        "ANETypes",
    ],
    path: "Tests/EspressoQwenServingTests",
    swiftSettings: [.swiftLanguageMode(.v6)]
),
```

## Canonical Files to Create

### Core backend boundary

- `Sources/EspressoQwenServing/ServingBackend.swift`
- `Sources/EspressoQwenServing/ServingSession.swift`
- `Sources/EspressoQwenServing/ServingBackendKind.swift`

### Artifact + planning

- `Sources/EspressoQwenServing/QwenServingArtifact.swift`
- `Sources/EspressoQwenServing/QwenBucketDescriptor.swift`
- `Sources/EspressoQwenServing/QwenBucketPlanner.swift`
- `Sources/EspressoQwenServing/QwenTensorIndex.swift`

### Session + metrics

- `Sources/EspressoQwenServing/QwenServingSession.swift`
- `Sources/EspressoQwenServing/QwenKVState.swift`
- `Sources/EspressoQwenServing/QwenServingMetrics.swift`

### Execution backend

- `Sources/EspressoQwenServing/QwenServingBackend.swift`
- `Sources/EspressoQwenServing/QwenANEDecodeCell.swift`
- `Sources/EspressoQwenServing/QwenANELayouts.swift`
- `Sources/EspressoQwenServing/QwenLMHeadReducer.swift`
- `Sources/EspressoQwenServing/QwenSampler.swift`

### Tests

- `Tests/EspressoQwenServingTests/ServingBackendSelectionTests.swift`
- `Tests/EspressoQwenServingTests/QwenBucketPlannerTests.swift`
- `Tests/EspressoQwenServingTests/QwenServingArtifactTests.swift`
- `Tests/EspressoQwenServingTests/QwenSessionStateTests.swift`
- `Tests/EspressoQwenServingTests/QwenSingleLayerParityTests.swift`
- `Tests/EspressoQwenServingTests/QwenServingParityTests.swift`
- `Tests/EspressoQwenServingTests/QwenLMHeadReducerTests.swift`

## Required Type Signatures

Use these signatures unless the compiler forces a small local adjustment.

### `ServingBackend.swift`

```swift
import Foundation
import ANETypes

public enum ServingBackendKind: String, Sendable, CaseIterable {
    case verifierExactCPU
    case qwenANEExperimental
}

public struct ServingSessionOptions: Sendable, Equatable {
    public var maxSequenceLength: Int
    public var temperature: Float
    public var topK: Int?
    public var topP: Float?
    public var seed: UInt64?
}

public protocol ServingSessionProtocol: Sendable {
    mutating func prefill(tokens: [TokenID]) async throws -> Void
    mutating func decodeOne() async throws -> TokenID
    mutating func decode(count: Int) async throws -> [TokenID]
    mutating func reset() async throws
}

public protocol ServingBackendProtocol: Sendable {
    associatedtype Session: ServingSessionProtocol

    func makeSession(
        modelDirectory: URL,
        tokenizerDirectory: URL,
        options: ServingSessionOptions
    ) async throws -> Session
}
```

### `QwenBucketDescriptor.swift`

```swift
import Foundation

public struct QwenBucketDescriptor: Sendable, Codable, Hashable {
    public let sequenceLength: Int
    public let kvCapacity: Int
    public let decodeBatchWidth: Int
    public let compileKey: String
}
```

### `QwenServingArtifact.swift`

```swift
import Foundation

public struct QwenServingArtifactManifest: Sendable, Codable, Equatable {
    public let version: Int
    public let architecture: String
    public let modelName: String
    public let nLayer: Int
    public let nHead: Int
    public let nKVHead: Int
    public let dModel: Int
    public let headDim: Int
    public let hiddenDim: Int
    public let vocab: Int
    public let maxSeq: Int
    public let ropeTheta: Float
    public let eosToken: Int?
    public let quantization: String
    public let bucketDescriptors: [QwenBucketDescriptor]
    public let tensorIndexRelativePath: String
}
```

### `QwenKVState.swift`

```swift
import Foundation
import IOSurface

public struct QwenKVLayerState: Sendable {
    public let keySurface: IOSurfaceRef
    public let valueSurface: IOSurfaceRef
    public var writtenTokenCount: Int
}

public struct QwenKVState: Sendable {
    public var layers: [QwenKVLayerState]
    public let capacity: Int

    public mutating func reset() {
        for index in layers.indices {
            layers[index].writtenTokenCount = 0
        }
    }
}
```

## Execution Sequence

### Phase 1: Land the Boundary

Goal:

- Create the new target and protocols.
- Do not change serving semantics yet.

Steps:

1. Update `Package.swift` with `EspressoQwenServing` and `EspressoQwenServingTests`.
2. Create:
   - `ServingBackend.swift`
   - `ServingSession.swift`
   - `ServingBackendKind.swift`
3. Add a small adapter around the current exact-CPU Qwen path:
   - name it `VerifierExactCPUBackend`
4. Add tests:
   - backend selection returns verifier backend by default
   - verifier backend outputs match existing exact-CPU outputs on `0.6B`

Commands:

```bash
swift build
swift test --filter EspressoQwenServingTests
swift test --filter 'QwenGGUF(Regression|SidecarComparison)Tests'
```

Commit message:

```text
qwen-serving: add serving backend boundary
```

Acceptance gate:

- zero semantic regression
- tests pass

### Phase 2: Add Artifact and Bucket Types

Goal:

- Make serving-artifact planning explicit before implementing ANE execution.

Steps:

1. Create:
   - `QwenServingArtifact.swift`
   - `QwenBucketDescriptor.swift`
   - `QwenBucketPlanner.swift`
   - `QwenTensorIndex.swift`
2. Implement fixed bucket table:
   - 128, 256, 512, 1024, 2048, 4096
3. Implement deterministic bucket selection:
   - exact bucket if possible
   - next-largest bucket otherwise
   - throw on overflow
4. Add serialization round-trip tests.
5. Do not connect this to `GGUFModelLoader` yet.

Commands:

```bash
swift test --filter EspressoQwenServingTests
```

Commit message:

```text
qwen-serving: add serving artifact and bucket planning types
```

Acceptance gate:

- all new tests pass
- no runtime behavior change

### Phase 3: Wire Serving Artifact Emission

Goal:

- Extend `GGUFModelLoader` to emit a second artifact family for serving.

Steps:

1. Add a `ServingArtifactMode` option to `GGUFModelLoader.PrepareOptions`.
2. Keep the current verifier artifact path untouched.
3. Add a new output root under the prepared model directory:

```text
serving/
  manifest.json
  tensors.index.json
  ...
```

4. Start simple:
   - metadata only
   - tensor index only
   - no execution dependence yet
5. Add tests:
   - serving artifact metadata exists when enabled
   - verifier artifact path still matches old expectations

Commands:

```bash
swift test --filter GGUFModelLoaderTests
swift test --filter EspressoQwenServingTests
```

Commit message:

```text
qwen-serving: emit serving artifact metadata
```

Acceptance gate:

- loader tests pass
- existing Qwen parity tests remain green

### Phase 4: Session Skeleton

Goal:

- Add session state before executing ANE cells.

Steps:

1. Implement:
   - `QwenServingSession`
   - `QwenKVState`
   - `QwenServingMetrics`
2. Support:
   - `prefill(tokens:)` as a state update stub
   - `reset()`
   - bucket capacity checks
3. Add tests:
   - reset clears positions
   - capacity guard fires
   - token count accounting is deterministic

Commands:

```bash
swift test --filter EspressoQwenServingTests
```

Commit message:

```text
qwen-serving: add session and kv state scaffolding
```

Acceptance gate:

- state management tests pass
- no execution path switched yet

### Phase 5: Single-Layer ANE Decode Cell

Goal:

- Prove the ANE-native architecture on one layer before stacking the full model.

Steps:

1. Implement:
   - `QwenANEDecodeCell.swift`
   - `QwenANELayouts.swift`
2. Start with a single-layer path for `0.6B`.
3. Use Apple ANE transformer constraints:
   - `BC1S`
   - 1x1-conv style projections
   - per-head or per-KV-group chunking
4. Feed the output into a comparison test against the exact-CPU layer output.
5. Keep tolerances explicit in the test:
   - max abs diff threshold
   - mean abs diff threshold

Commands:

```bash
swift test --filter QwenSingleLayerParityTests
```

Commit message:

```text
qwen-serving: add single-layer ane decode prototype
```

Acceptance gate:

- single-layer parity test passes on `0.6B`

### Phase 6: Full 0.6B ANE Serving Loop

Goal:

- Replace the single-layer prototype with a real multi-layer decode session.

Steps:

1. Stack the decode cell across all layers.
2. Implement `decodeOne()` on the serving session.
3. Implement `decode(count:)`.
4. Add a minimal backend:
   - `QwenANEServingBackend`
5. Add a temporary feature flag:
   - `ESPRESSO_QWEN_SERVING_BACKEND=ane`
6. Keep verifier backend available.

Commands:

```bash
ESPRESSO_QWEN_SERVING_BACKEND=ane swift test --filter QwenServingParityTests
```

Commit message:

```text
qwen-serving: add full 0.6b ane decode loop
```

Acceptance gate:

- `0.6B` parity passes on serving backend

### Phase 7: Add Warm Decode Benchmark Command

Goal:

- Measure the new runtime correctly.

Steps:

1. Extend `EspressoGGUFRunner` or `espresso-generate` with a warm decode benchmark mode.
2. Required output fields:
   - model
   - backend
   - prompt
   - bucket
   - prepare_ms
   - build_ms
   - first_token_latency_ms
   - steady_state_tok_s
3. Support:
   - warm-only benchmark
   - optional repeated runs

Commands:

```bash
swift build --product EspressoGGUFRunner
./.build/debug/EspressoGGUFRunner benchmark-qwen ...
```

Commit message:

```text
qwen-serving: add warm decode benchmark command
```

Acceptance gate:

- benchmark output is machine-readable and stable

### Phase 8: LM Head Reduction Path

Goal:

- Prevent the Qwen vocab head from dominating decode cost.

Steps:

1. Implement `QwenLMHeadReducer.swift`.
2. Shard vocab into fixed blocks.
3. Compute block-local maxima on device.
4. Reduce block maxima into final top-1 or top-k.
5. Keep full logits off the CPU in normal mode.

Commands:

```bash
swift test --filter QwenLMHeadReducerTests
ESPRESSO_QWEN_SERVING_BACKEND=ane swift test --filter QwenServingParityTests
```

Commit message:

```text
qwen-serving: optimize qwen lm head reduction
```

Acceptance gate:

- parity unchanged
- benchmark improves or at least does not regress

### Phase 9: 1.7B Bring-Up

Goal:

- Prove scale-up beyond `0.6B`.

Steps:

1. Reuse the same serving backend.
2. Run targeted `1.7B` parity checks.
3. Tune bucket sizing or memory layout only if needed.

Commands:

```bash
ESPRESSO_QWEN_MANUAL_MODEL_PATH=/tmp/edgerunner-models/Qwen3-1.7B-Q8_0.gguf \
ESPRESSO_QWEN_MANUAL_EXPECTED_LATE_PREFIX_TOKEN=21340 \
ESPRESSO_QWEN_MANUAL_EXPECTED_HELLO_TOKENS=25,358,2776,4460,311,3535,279,7286 \
swift test --filter qwenManual
```

Commit message:

```text
qwen-serving: bring up 1.7b on ane backend
```

Acceptance gate:

- `1.7B` targeted parity passes

### Phase 10: 4B Bring-Up

Goal:

- Determine the real scaling ceiling.

Steps:

1. Run `4B` late-prefix first.
2. Then run `4B` `Hello` 8-token check.
3. If full parity is still too expensive:
   - isolate the blocker
   - record it in the persistent log
   - do not guess

Commands:

```bash
ESPRESSO_QWEN_MANUAL_MODEL_PATH=/tmp/edgerunner-models/Qwen3-4B-Q8_0.gguf \
ESPRESSO_QWEN_MANUAL_EXPECTED_LATE_PREFIX_TOKEN=60009 \
ESPRESSO_QWEN_MANUAL_EXPECTED_HELLO_TOKENS=27,18,198,9707,27,18,198,9707 \
swift test --filter qwenManual
```

Commit message:

```text
qwen-serving: bring up 4b on ane backend
```

Acceptance gate:

- `4B` parity passes, or exact remaining blocker is isolated

## Logging Rules

After every meaningful experiment, append to:

- `tasks/qwen_gguf_remaining_correctness_log.md`

Each entry must include:

- hypothesis
- exact commands run
- exact result
- whether token behavior changed
- what invariant was confirmed or ruled out
- whether the path is now a dead end

Also update:

- `tasks/todo.md`

## Fallback Rules

Fallbacks are allowed only after their trigger condition is met and the failure is logged.

### Fallback A: ANE trunk + Metal LM head

Allowed only if:

- serving parity is already good through the trunk
- profiling shows the LM head dominates token time

### Fallback B: ANE prefill + Metal decode

Allowed only if:

- the ANE decode loop cannot be made coherent without reintroducing host round-trips
- the serving parity path still holds under the mixed backend

### Fallback C: Edgerunner Metal serving backend

Allowed only if:

- the ANE runtime stalls before `1.7B`
- the exact blocker is proven

Do not take a fallback just because it is easier.

## Immediate First Action

The first coding action after reading this spec must be:

1. create the Espresso and Edgerunner worktrees
2. run the baseline commands
3. land PR 1 only:
   - new target
   - backend protocols
   - verifier adapter
   - tests

Do not start with ANE decode implementation before the boundary exists.
