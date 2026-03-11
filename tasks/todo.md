# TODO

- [x] Preserve the current non-echo `identity-zero-trunk` public claim as the control and do not weaken its exactness, artifact, or reporting contract.
- [x] Add TDD seams for ANE-backed future-head selection so proposer behavior can be checked without rerunning the full harness.
- [x] Replace the CPU future proposer on the exact two-token non-echo path with an ANE head that uses the saved `futureRMS` and `futureClassifier`.
- [x] Re-verify unit parity, focused hardware parity, and proposer metrics after the ANE proposer is wired in.
- [x] Rerun the matched recurrent-checkpoint/CoreML harness and keep the change because the exact two-token path beats the previous CPU-proposer two-step median and the one-token ANE identity control.
- [x] Update docs, lessons, Wax notes, handoff, and review with the stronger non-echo throughput result and the failed smaller-lane probes.

# Review

- Previous non-echo control on this branch was the exact local bigram artifact with explicit `identity-zero-trunk` backends and a CPU proposer:
  - exact two-step `1.2012578125 ms/token`
  - exact one-token ANE control `1.0598854166666667 ms/token`
  - matched zero-weight `6`-layer CoreML `4.7705807291666664 ms/token`
  - exact two-step speedup vs CoreML `3.9541428963040195x`
  - parity `match`
  - committed exact tokens/pass `2`
  - accepted future tokens/pass `1`
- Current winning non-echo result on this branch uses an ANE future proposer on the same exact `identity-zero-trunk` artifact contract:
  - exact two-step `1.0806302083333332 ms/token`
  - exact one-token ANE control `1.0957500000000002 ms/token`
  - matched zero-weight `6`-layer CoreML `5.085307291666668 ms/token`
  - exact two-step speedup vs CoreML `4.7583224488025415x`
  - exact one-token ANE control speedup vs CoreML `4.640428016426192x`
  - parity `match`
  - committed exact tokens/pass `2`
  - accepted future tokens/pass `1`
- Sample same-session run (`run-3.json`) shows the mechanism clearly:
  - exact two-step `1.0607916666666668 ms/token`
  - exact one-token ANE control `1.0760208333333334 ms/token`
  - proposer `0.9317604166666666 ms/pass`
  - verifier logits `0.9893697916666667 ms/pass`
  - verifier trunk `0.000010416666666666666 ms/pass`
- Failed bounded follow-ups were reverted:
  - proposer-only `laneSpatial=1` hit ANE `statusType=0x9`
  - proposer-only `laneSpatial=8` hit ANE `statusType=0x9`
- The generic RWKV-style recurrent ANE cell remains a separate negative result on non-echo one-hot seams and should not be reopened during this proposer phase.
