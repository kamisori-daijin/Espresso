Paste the prompt below into Claude as a single message.

```text
You are the execution owner for a full Qwen ANE serving-runtime rewrite across two local Swift repositories. This is an implementation task, not a planning task. Work autonomously, write code, run builds and tests, benchmark, commit in small PR-quality increments, and keep the logs current. You are allowed to take significant implementation risks because all work must happen inside isolated worktrees, not the original checkouts.

Mission:
Complete the Qwen ANE serving-runtime rewrite end to end in paired worktrees, preserving the exact-CPU Qwen path as the correctness oracle while delivering a new ANE-first serving backend that uses serving-native artifacts in its hot path and beats the current Espresso Qwen runtime on warm decode throughput.

Authoritative source of truth:
1. `/Users/chriskarani/CodingProjects/Espresso/QWEN_ANE_SERVING_AGENT_EXECUTION_SPEC.md`
2. `/Users/chriskarani/CodingProjects/Espresso/tasks/qwen_gguf_remaining_correctness_log.md`
3. `/Users/chriskarani/CodingProjects/Espresso/tasks/todo.md`

Important repo note:
- The source sibling repo on disk is `/Users/chriskarani/CodingProjects/EdgeRunner`
- Espresso’s SwiftPM dependency is `../Edgerunner`
- Therefore the worktree for the sibling repo must be created from `/Users/chriskarani/CodingProjects/EdgeRunner` but placed at a path literally named `Edgerunner`
- The `tasks/` and `docs/` draft spec files are gitignored; do not treat them as canonical. Use only `/Users/chriskarani/CodingProjects/Espresso/QWEN_ANE_SERVING_AGENT_EXECUTION_SPEC.md` as the execution authority.

Done definition:
1. A dedicated serving backend exists, separate from the verifier backend.
2. The serving backend uses serving-native artifacts, not verifier float32 sidecars, for its hot path.
3. `0.6B` serving parity passes on the serving backend:
   - cold-start
   - late-prefix token
   - `Hello` 8-token continuation
4. `1.7B` targeted parity passes on the serving backend:
   - late-prefix token
   - `Hello` 8-token continuation
5. `4B` targeted parity passes on the serving backend:
   - late-prefix token
   - `Hello` 8-token continuation
   - if full `Hello` parity is still impractical, isolate one exact remaining blocker with hard evidence
6. Warm decode tok/s on the new serving backend materially exceeds the current Espresso Qwen path on the same machine.
7. A reproducible benchmark command exists for:
   - warm decode
   - first-token latency
   - build/compile time
   - prepare time
8. `/Users/chriskarani/CodingProjects/Espresso/tasks/qwen_gguf_remaining_correctness_log.md` and `/Users/chriskarani/CodingProjects/Espresso/tasks/todo.md` are updated after every meaningful experiment and reflect the final grounded outcome.

Hard constraints:
- Never delete or mutate these protected models:
  - `/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf`
  - `/tmp/edgerunner-models/Qwen3-1.7B-Q8_0.gguf`
  - `/tmp/edgerunner-models/Qwen3-4B-Q8_0.gguf`
- Never delete any `.artifacts` directory.
- Only disposable `/var/folders/.../T/espresso_gguf_*` temp conversion directories may be removed.
- Preserve the current exact-CPU Qwen path as the correctness oracle until the new serving backend is proven.
- Do not change expected tokens, tests, or benchmark expectations just to make results pass.
- Do not use `git reset --hard`, `git checkout --`, force-push, or any destructive git operation.
- Ask only before irreversible actions. Otherwise proceed autonomously.
- Do not reopen known dead ends without new hard evidence:
  - generic tokenizer mismatch
  - converter mismatch for already-compared tensor surfaces
  - ANE/Metal as the primary remaining `0.6B` correctness issue
  - plain FP16 sidecar storage as the remaining semantic explanation

Architecture guardrails:
- Follow `/Users/chriskarani/CodingProjects/Espresso/QWEN_ANE_SERVING_AGENT_EXECUTION_SPEC.md` phase by phase. Do not skip gates. Do not collapse later phases into earlier ones.
- Do not drift into a Metal-first rewrite, Core ML wrapper path, or Edgerunner-serving replacement unless a documented fallback trigger is met and logged with hard evidence.
- Every serving milestone must be proven with the serving backend enabled. Exact-CPU oracle tests do not count as serving proof.
- The serving hot path must not read verifier `.float32.bin` sidecars, run CPU exact decode, expose CPU-visible intermediate tensors in the token loop, or materialize full logits on CPU in normal serving mode.
- Avoid a SwiftPM target cycle. If `EspressoQwenServing` depends on `RealModelInference`, then `RealModelInference` must not become the selector or owner of that backend. Put shared protocols in a lower target or keep backend selection at the CLI or app layer.
- Keep new serving complexity out of `/Users/chriskarani/CodingProjects/Espresso/Sources/RealModelInference/RealModelInferenceEngine.swift` except for oracle-preserving adapters.
- Use the existing ANE stack in Espresso: `ANERuntime`, `MILGenerator`, `ANEInterop`, `ANETypes`, `CPUOps`, and `Espresso`. Do not silently pivot to a different serving stack.

Serving artifact contract:
- Define and implement a serving-native artifact family under the prepared model directory.
- At minimum it must have:
  - manifest schema
  - tensor index
  - bucket descriptors
  - cache key or versioning
  - explicit writer ownership
  - explicit reader ownership
- `GGUFModelLoader` should own emitting the serving artifact family.
- The serving backend should own consuming the serving artifact family.
- Early phases may emit metadata and index only, but by the time the serving backend is used for parity, the hot path must consume the serving artifact family rather than verifier sidecars.
- Keep the fixed bucket table from the execution spec:
  - `128`
  - `256`
  - `512`
  - `1024`
  - `2048`
  - `4096`
- Bucket selection must be deterministic:
  - exact bucket if possible
  - next-largest bucket otherwise
  - fail on overflow

Benchmark contract:
- Capture a baseline before edits on the same machine.
- Benchmark using the same model, same prompt, same bucket, and same backend selection rules for before and after comparisons.
- Separate:
  - prepare time
  - build time
  - first-token latency
  - warm steady-state tok/s
- Use warm-only repeated runs and report the median.
- Output benchmark results in a machine-readable format.
- Throughput success means the new serving backend exceeds the captured baseline warm steady-state tok/s by at least `20%` on `0.6B` under the same measurement method. If that threshold is not met, isolate one exact blocker with profiling or command evidence before claiming completion.

Known grounded Qwen facts you must preserve:
- `0.6B` correctness is already fixed on the exact-CPU oracle path.
- The proven root cause of the old late-token bug was repeated FP16 rounding inside Espresso’s exact-CPU Qwen decode.
- Fresh `0.6B` post-fix verification already passes:
  - cold-start: `Hello Answer`
  - late prefix `[9707, 21806, 11, 358, 2776, 14589, 369, 279] -> 3681`
  - full `Hello` continuation `[21806, 11, 358, 2776, 14589, 369, 279, 3681]`
- `1.7B` exact-CPU oracle targets are:
  - late-prefix token `21340`
  - `Hello` continuation `[25, 358, 2776, 4460, 311, 3535, 279, 7286]`
- `4B` oracle targets are:
  - late-prefix token `60009`
  - `Hello` continuation `[27, 18, 198, 9707, 27, 18, 198, 9707]`
- Current larger-model blocker is runtime and infrastructure, not the original `0.6B` semantic bug.

Execution order:
1. Read `/Users/chriskarani/CodingProjects/Espresso/QWEN_ANE_SERVING_AGENT_EXECUTION_SPEC.md` completely before editing anything.
2. Create paired worktrees under `/Users/chriskarani/CodingProjects/worktrees/qwen-ane-serving`.
3. Re-run the baseline build, verification, and test gates before edits.
4. Land the phases from the execution spec in order.
5. After each completed phase:
   - run its acceptance gate
   - update the persistent log
   - update `tasks/todo.md`
   - create a small PR-quality commit
6. If a gate fails, stop, debug, and recover before proceeding.
7. Only use a fallback if its trigger condition is satisfied, evidenced, and logged.

Exact worktree bootstrap:
```bash
WORKTREE_ROOT=/Users/chriskarani/CodingProjects/worktrees/qwen-ane-serving
mkdir -p "$WORKTREE_ROOT"

cd /Users/chriskarani/CodingProjects/Espresso
BASE_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ -d "$WORKTREE_ROOT/Espresso/.git" ] || [ -f "$WORKTREE_ROOT/Espresso/.git" ]; then
  echo "Espresso worktree already exists: $WORKTREE_ROOT/Espresso"
else
  git worktree add "$WORKTREE_ROOT/Espresso" -b qwen-ane-serving-runtime "$BASE_BRANCH"
fi

cd /Users/chriskarani/CodingProjects/EdgeRunner
BASE_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ -d "$WORKTREE_ROOT/Edgerunner/.git" ] || [ -f "$WORKTREE_ROOT/Edgerunner/.git" ]; then
  echo "Edgerunner worktree already exists: $WORKTREE_ROOT/Edgerunner"
else
  git worktree add "$WORKTREE_ROOT/Edgerunner" -b qwen-ane-serving-runtime "$BASE_BRANCH"
fi
```

After creating or reusing the worktrees, the active working directories must be:
- Espresso: `/Users/chriskarani/CodingProjects/worktrees/qwen-ane-serving/Espresso`
- Edgerunner: `/Users/chriskarani/CodingProjects/worktrees/qwen-ane-serving/Edgerunner`

Baseline gate. Run these before any edits:
```bash
cd /Users/chriskarani/CodingProjects/worktrees/qwen-ane-serving/Espresso
swift build --product EspressoGGUFRunner
swift build --product espresso-generate
swift test --filter 'QwenGGUF(Regression|SidecarComparison)Tests'
swift test --filter GGUFModelLoaderTests
swift test --filter EspressoGenerateTests
./.build/debug/EspressoGGUFRunner verify-qwen /tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf /Users/chriskarani/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-8bit/snapshots/11de96878523501bcaa86104e3c186de07ff9068 --fresh --keep-weight-dir
```

Sibling repo sanity checks. Run these whenever you change Edgerunner code:
```bash
cd /Users/chriskarani/CodingProjects/worktrees/qwen-ane-serving/Edgerunner
swift build
swift test --filter WeightConverterTests
swift test --filter EspressoTensorNameMapperTests
swift test --filter QwenGGUFTensorComparisonTests
swift test --filter "QwenBenchmark/decodeBenchmark"
```

Phase execution rules:

Phase 1. Land the boundary only.
- Add the new package target and test target.
- Add backend and session protocols.
- Add a verifier exact-CPU adapter.
- Do not change serving semantics yet.
- Acceptance gate:
  ```bash
  cd /Users/chriskarani/CodingProjects/worktrees/qwen-ane-serving/Espresso
  swift build
  swift test --filter EspressoQwenServingTests
  swift test --filter 'QwenGGUF(Regression|SidecarComparison)Tests'
  ```
- Commit after the gate passes.

Phase 2. Add artifact and bucket types only.
- Add serving artifact, bucket descriptor, bucket planner, and tensor index types.
- Add deterministic bucket selection and serialization tests.
- Do not wire execution yet.
- Acceptance gate:
  ```bash
  cd /Users/chriskarani/CodingProjects/worktrees/qwen-ane-serving/Espresso
  swift test --filter EspressoQwenServingTests
  ```
- Commit after the gate passes.

Phase 3. Wire serving artifact emission.
- Extend `GGUFModelLoader` to emit a serving artifact family under the prepared model directory.
- Keep the verifier artifact path intact.
- Add tests for serving artifact metadata and index emission.
- Acceptance gate:
  ```bash
  cd /Users/chriskarani/CodingProjects/worktrees/qwen-ane-serving/Espresso
  swift test --filter GGUFModelLoaderTests
  swift test --filter EspressoQwenServingTests
  swift test --filter 'QwenGGUF(Regression|SidecarComparison)Tests'
  ```
- Commit after the gate passes.

Phase 4. Add session skeleton.
- Implement serving session state, KV state, metrics, reset behavior, and capacity guards.
- Do not switch the runtime yet.
- Acceptance gate:
  ```bash
  cd /Users/chriskarani/CodingProjects/worktrees/qwen-ane-serving/Espresso
  swift test --filter EspressoQwenServingTests
  ```
- Commit after the gate passes.

Phase 5. Land a single-layer ANE decode cell.
- Implement ANE-native single-layer decode using the existing ANE stack.
- Use BC1S layouts and 1x1-conv style projections where appropriate.
- Add explicit parity tolerances against the exact-CPU layer output.
- Acceptance gate:
  ```bash
  cd /Users/chriskarani/CodingProjects/worktrees/qwen-ane-serving/Espresso
  swift test --filter QwenSingleLayerParityTests
  ```
- Commit after the gate passes.

Phase 6. Land the full `0.6B` ANE serving loop.
- Stack the decode cell across all layers.
- Implement `decodeOne()` and `decode(count:)`.
- Add a serving backend implementation and feature-flagged backend selection.
- Serving proof must run through the serving backend, not the exact-CPU oracle.
- Acceptance gate:
  ```bash
  cd /Users/chriskarani/CodingProjects/worktrees/qwen-ane-serving/Espresso
  ESPRESSO_QWEN_SERVING_BACKEND=ane swift test --filter QwenServingParityTests
  ```
- After the tests pass, run a backend-specific `0.6B` verification command that proves:
  - cold-start
  - late-prefix token `3681`
  - `Hello` continuation `[21806, 11, 358, 2776, 14589, 369, 279, 3681]`
- Commit after the gate passes.

Phase 7. Add a machine-readable benchmark command.
- Extend `EspressoGGUFRunner` or `espresso-generate`.
- Benchmark output must include:
  - model
  - backend
  - prompt
  - bucket
  - prepare_ms
  - build_ms
  - first_token_latency_ms
  - steady_state_tok_s
- It must support repeated warm runs and median reporting.
- Acceptance gate:
  ```bash
  cd /Users/chriskarani/CodingProjects/worktrees/qwen-ane-serving/Espresso
  swift build --product EspressoGGUFRunner
  ```
- Then run the benchmark command and compare it to the captured baseline.
- Commit after the gate passes.

Phase 8. Land the LM head reduction path.
- Implement on-device LM head reduction.
- Keep full logits off CPU in normal serving mode.
- Acceptance gate:
  ```bash
  cd /Users/chriskarani/CodingProjects/worktrees/qwen-ane-serving/Espresso
  swift test --filter QwenLMHeadReducerTests
  ESPRESSO_QWEN_SERVING_BACKEND=ane swift test --filter QwenServingParityTests
  ```
- Re-run the benchmark after this phase.
- Commit after the gate passes.

Phase 9. Bring up `1.7B` on the serving backend.
- Do not count exact-CPU `qwenManual` tests as serving proof.
- Add or use serving-backend-specific parity coverage for `1.7B`.
- Required serving-backend targets:
  - late-prefix token `21340`
  - `Hello` continuation `[25, 358, 2776, 4460, 311, 3535, 279, 7286]`
- Commit after the serving-backend proof passes.

Phase 10. Bring up `4B` on the serving backend.
- Do not count exact-CPU `qwenManual` tests as serving proof.
- Add or use serving-backend-specific parity coverage for `4B`.
- Required serving-backend targets:
  - late-prefix token `60009`
  - `Hello` continuation `[27, 18, 198, 9707, 27, 18, 198, 9707]`
- If full `Hello` parity is still impractical, isolate one exact blocker with hard evidence and log it. Do not guess.
- Commit after the serving-backend proof or blocker isolation is complete.

Fallback rules:
- Fallback A: ANE trunk plus Metal LM head
  - allowed only if parity already holds through the ANE trunk and profiling shows the LM head dominates token time
- Fallback B: ANE prefill plus Metal decode
  - allowed only if the ANE decode loop cannot be made coherent without reintroducing host round-trips and mixed-backend parity still holds
- Fallback C: Edgerunner Metal serving backend
  - allowed only if the ANE runtime stalls before `1.7B` and the exact blocker is proven
- Do not take a fallback because it is easier.
- If you trigger a fallback:
  1. quote the trigger condition
  2. provide exact evidence
  3. log it
  4. continue execution

Operational cautions:
- ANE compile-budget exhaustion is a real possibility. If compiles start failing due to repeated compile attempts, log it as infrastructure pressure rather than immediately misclassifying it as an architecture failure.
- Large-model fresh artifacts and sidecar-heavy paths can consume substantial disk. If cleanup is necessary, remove only disposable `/var/folders/.../T/espresso_gguf_*` directories and log exactly what you removed.
- Keep benchmark evidence and parity evidence tied to the same artifact family and same bucket.

Logging rules:
- After every meaningful experiment, append to `/Users/chriskarani/CodingProjects/Espresso/tasks/qwen_gguf_remaining_correctness_log.md`:
  - hypothesis
  - exact commands run
  - exact result
  - whether token behavior changed
  - what invariant was confirmed or ruled out
  - whether the path is now a dead end
- Also update `/Users/chriskarani/CodingProjects/Espresso/tasks/todo.md` with grounded progress and review notes.

Commit rules:
- Make small PR-quality commits aligned to phases.
- Use the commit message style from the canonical execution spec unless a more precise message is warranted.
- Do not batch multiple unfinished phases into one commit.

First response format:
- Name the authoritative spec file you are following.
- State whether the paired worktrees already exist or need to be created.
- State the active phase.
- List the exact first commands you will run.
- Then begin execution immediately.

Final response format:
Return only:
1. Final architectural outcome
2. Files changed
3. Exact verification commands run and the key result of each
4. Benchmark results
5. Commit list
6. Final status
   - complete
   - complete with isolated blocker
7. Residual blocker, only if not fully complete

Do not give me a plan back. Execute the work.
```
