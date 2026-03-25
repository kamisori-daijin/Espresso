import Foundation

public struct ANECompileLabelStatsSnapshot: Sendable, Equatable {
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

    func subtracting(_ other: ANECompileLabelStatsSnapshot) -> ANECompileLabelStatsSnapshot {
        ANECompileLabelStatsSnapshot(
            attemptCount: attemptCount - other.attemptCount,
            successCount: successCount - other.successCount,
            failureCount: failureCount - other.failureCount,
            retryCount: retryCount - other.retryCount
        )
    }
}

public struct ANECompileStatsSnapshot: Sendable, Equatable {
    public let attemptCount: Int
    public let successCount: Int
    public let failureCount: Int
    public let retryCount: Int
    public let labelStats: [String: ANECompileLabelStatsSnapshot]

    public init(
        attemptCount: Int,
        successCount: Int,
        failureCount: Int,
        retryCount: Int,
        labelStats: [String: ANECompileLabelStatsSnapshot] = [:]
    ) {
        self.attemptCount = attemptCount
        self.successCount = successCount
        self.failureCount = failureCount
        self.retryCount = retryCount
        self.labelStats = labelStats
    }

    public func subtracting(_ other: ANECompileStatsSnapshot) -> ANECompileStatsSnapshot {
        let allKeys = Set(labelStats.keys).union(other.labelStats.keys)
        var deltaLabelStats: [String: ANECompileLabelStatsSnapshot] = [:]
        for key in allKeys {
            let lhs = labelStats[key] ?? ANECompileLabelStatsSnapshot(
                attemptCount: 0,
                successCount: 0,
                failureCount: 0,
                retryCount: 0
            )
            let rhs = other.labelStats[key] ?? ANECompileLabelStatsSnapshot(
                attemptCount: 0,
                successCount: 0,
                failureCount: 0,
                retryCount: 0
            )
            let delta = lhs.subtracting(rhs)
            if delta.attemptCount != 0 || delta.successCount != 0 || delta.failureCount != 0 || delta.retryCount != 0 {
                deltaLabelStats[key] = delta
            }
        }
        return ANECompileStatsSnapshot(
            attemptCount: attemptCount - other.attemptCount,
            successCount: successCount - other.successCount,
            failureCount: failureCount - other.failureCount,
            retryCount: retryCount - other.retryCount,
            labelStats: deltaLabelStats
        )
    }
}

public enum ANECompileStats {
    private struct Counter {
        var attemptCount = 0
        var successCount = 0
        var failureCount = 0
        var retryCount = 0

        func snapshot() -> ANECompileLabelStatsSnapshot {
            ANECompileLabelStatsSnapshot(
                attemptCount: attemptCount,
                successCount: successCount,
                failureCount: failureCount,
                retryCount: retryCount
            )
        }
    }

    private enum State {
        static let lock = NSLock()
        nonisolated(unsafe) static var attemptCount = 0
        nonisolated(unsafe) static var successCount = 0
        nonisolated(unsafe) static var failureCount = 0
        nonisolated(unsafe) static var retryCount = 0
        nonisolated(unsafe) static var countersByLabel: [String: Counter] = [:]
    }

    private static func normalizedLabel(_ label: String?) -> String {
        guard let trimmed = label?.trimmingCharacters(in: .whitespacesAndNewlines), !trimmed.isEmpty else {
            return "_unlabeled"
        }
        return trimmed
    }

    public static func snapshot() -> ANECompileStatsSnapshot {
        State.lock.lock()
        defer { State.lock.unlock() }
        return ANECompileStatsSnapshot(
            attemptCount: State.attemptCount,
            successCount: State.successCount,
            failureCount: State.failureCount,
            retryCount: State.retryCount,
            labelStats: State.countersByLabel.mapValues { $0.snapshot() }
        )
    }

    public static func reset() {
        State.lock.lock()
        defer { State.lock.unlock() }
        State.attemptCount = 0
        State.successCount = 0
        State.failureCount = 0
        State.retryCount = 0
        State.countersByLabel.removeAll()
    }

    static func recordAttempt(label: String?) {
        State.lock.lock()
        defer { State.lock.unlock() }
        State.attemptCount += 1
        let labelKey = normalizedLabel(label)
        State.countersByLabel[labelKey, default: Counter()].attemptCount += 1
    }

    static func recordSuccess(label: String?) {
        State.lock.lock()
        defer { State.lock.unlock() }
        State.successCount += 1
        let labelKey = normalizedLabel(label)
        State.countersByLabel[labelKey, default: Counter()].successCount += 1
    }

    static func recordFailure(label: String?, willRetry: Bool) {
        State.lock.lock()
        defer { State.lock.unlock() }
        State.failureCount += 1
        let labelKey = normalizedLabel(label)
        State.countersByLabel[labelKey, default: Counter()].failureCount += 1
        if willRetry {
            State.retryCount += 1
            State.countersByLabel[labelKey, default: Counter()].retryCount += 1
        }
    }
}
