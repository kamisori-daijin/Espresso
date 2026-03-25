import Foundation

public struct ANECompileStatsSnapshot: Sendable, Equatable {
    public let attemptCount: Int
    public let successCount: Int
    public let failureCount: Int
    public let retryCount: Int

    public init(
        attemptCount: Int,
        successCount: Int,
        failureCount: Int,
        retryCount: Int
    ) {
        self.attemptCount = attemptCount
        self.successCount = successCount
        self.failureCount = failureCount
        self.retryCount = retryCount
    }
}

public enum ANECompileStats {
    private enum State {
        static let lock = NSLock()
        nonisolated(unsafe) static var attemptCount = 0
        nonisolated(unsafe) static var successCount = 0
        nonisolated(unsafe) static var failureCount = 0
        nonisolated(unsafe) static var retryCount = 0
    }

    public static func snapshot() -> ANECompileStatsSnapshot {
        State.lock.lock()
        defer { State.lock.unlock() }
        return ANECompileStatsSnapshot(
            attemptCount: State.attemptCount,
            successCount: State.successCount,
            failureCount: State.failureCount,
            retryCount: State.retryCount
        )
    }

    public static func reset() {
        State.lock.lock()
        defer { State.lock.unlock() }
        State.attemptCount = 0
        State.successCount = 0
        State.failureCount = 0
        State.retryCount = 0
    }

    static func recordAttempt() {
        State.lock.lock()
        defer { State.lock.unlock() }
        State.attemptCount += 1
    }

    static func recordSuccess() {
        State.lock.lock()
        defer { State.lock.unlock() }
        State.successCount += 1
    }

    static func recordFailure(willRetry: Bool) {
        State.lock.lock()
        defer { State.lock.unlock() }
        State.failureCount += 1
        if willRetry {
            State.retryCount += 1
        }
    }
}
