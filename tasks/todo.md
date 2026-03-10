# TODO

- [x] Keep `3e6cced` as the reference architecture checkpoint and `2e49cab` as the hardware compile/init gate result; do not restart runtime future-head work on this checkpoint family while the compile gate is blocked.
- [x] Add failing tests for a recoverable two-step student sidecar contract: exact contract metadata, round-trip serialization, teacher-seeded RMS/classifier copying, and config-mismatch rejection.
- [x] Implement a separate two-step student sidecar format for the exact branch-state-promotion contract without changing the base checkpoint version.
- [x] Add a narrow `espresso-train` export seam that writes the two-step student sidecar from loaded or resumed teacher weights.
- [ ] Append student-pivot findings to `docs/fused-decode-and-next-steps.md`, capture review notes below, update Wax memory, and commit the checkpoint.

# Review

- `swift test --filter TwoStepStudentCheckpointTests` passed `4/4`.
- `swift build --build-tests` passed after adding the new student sidecar source and test file.
- `swift run espresso-train --model /does/not/exist --export-two-step-student /tmp/espresso-two-step-student-sidecar.bin` succeeded, printed `[exported two-step student sidecar: /tmp/espresso-two-step-student-sidecar.bin]`, and exited before dataset/training work.
- The sidecar was intentionally implemented as a separate artifact family instead of extending the base checkpoint format, so training resume compatibility in `Checkpoint.save/load` stays unchanged while the future-head contract can version independently.
- The seeded sidecar copies the teacher `rmsFinal` and the teacher classifier seed exactly; on the current shared-classifier training path that means seeding from `embed`.
- The exported sidecar size for the full `dim=768`, `vocab=32000` contract is `98307104` bytes, which is consistent with one `dim` RMS vector plus one full `vocab x dim` float classifier matrix and a small header.
