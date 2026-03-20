# HEARTBEAT.md -- Founding Engineer

## Every Heartbeat

1. Check assignments via Paperclip inbox
2. Checkout assigned task before working
3. Read task context and parent/goal info
4. Do the work: code, test, review
5. Run `swift build` and `swift test` before marking done
6. Comment progress on the issue
7. Update status (done/blocked) before exiting

## Engineering Standards

- All code changes must compile (`swift build`)
- All existing tests must pass (`swift test`)
- New features require new tests (TDD)
- Performance-sensitive changes require benchmarks
- Document public APIs with DocC comments

## Escalation

- Blocked on architecture decisions: escalate to CEO
- ANE compiler failures (`InvalidMILProgram`, `statusType=0x9`): document in issue, try workaround, escalate if stuck
- CI failures: fix before moving to next task
