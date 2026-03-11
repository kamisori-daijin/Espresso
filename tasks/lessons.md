# Lessons

## 2026-03-12

- When cancelling Swift stream-reading tasks for `FileHandle.bytes.lines`, treat `CancellationError` as expected shutdown and never surface it in user-visible logs as a runtime failure.
