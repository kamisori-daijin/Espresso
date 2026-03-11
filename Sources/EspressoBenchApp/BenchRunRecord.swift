import Foundation

enum BenchRunStatus: String, Sendable {
    case running
    case succeeded
    case failed
    case cancelled

    var title: String { rawValue.capitalized }
}

struct BenchLatencyPoint: Identifiable, Hashable, Sendable {
    let iteration: Int
    let latencyMs: Double

    var id: Int { iteration }
}

struct BenchLatencyStats: Hashable, Sendable {
    let sampleCount: Int
    let mean: Double
    let median: Double
    let p95: Double
    let p99: Double
    let min: Double
    let max: Double

    init(latencies: [Double]) {
        let sorted = latencies.sorted()
        sampleCount = latencies.count
        mean = latencies.isEmpty ? 0 : latencies.reduce(0, +) / Double(latencies.count)
        median = BenchLatencyStats.percentile(0.5, sorted: sorted)
        p95 = BenchLatencyStats.percentile(0.95, sorted: sorted)
        p99 = BenchLatencyStats.percentile(0.99, sorted: sorted)
        min = sorted.first ?? 0
        max = sorted.last ?? 0
    }

    private static func percentile(_ fraction: Double, sorted: [Double]) -> Double {
        guard !sorted.isEmpty else { return 0 }
        let clamped = Swift.min(Swift.max(fraction, 0), 1)
        let index = clamped * Double(sorted.count - 1)
        let lowerIndex = Int(index)
        let upperIndex = Swift.min(lowerIndex + 1, sorted.count - 1)
        if lowerIndex == upperIndex {
            return sorted[lowerIndex]
        }
        let remainder = index - Double(lowerIndex)
        return sorted[lowerIndex] + ((sorted[upperIndex] - sorted[lowerIndex]) * remainder)
    }
}

struct BenchLatencySeries: Identifiable, Hashable, Sendable {
    let name: String
    let fileName: String
    let points: [BenchLatencyPoint]
    let stats: BenchLatencyStats

    var id: String { fileName }
}

struct BenchRunRecord: Identifiable, Sendable {
    let id: String
    var title: String
    let launchedAt: Date
    let outputDirectory: URL
    var status: BenchRunStatus
    var terminationStatus: Int32?
    var commandDescription: String
    var log: String
    var summaryText: String
    var summarySnapshot: BenchSummarySnapshot?
    var latencySeries: [BenchLatencySeries]
    var artifactFiles: [URL]

    var primarySeries: BenchLatencySeries? {
        latencySeries.first
    }

    var primaryEntry: BenchSummaryEntry? {
        summarySnapshot?.primaryEntry
    }
}
