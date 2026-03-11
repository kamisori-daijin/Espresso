import Foundation

enum BenchSummaryEntryKind: String, Sendable, Hashable {
    case aneDirect
    case aneInference
    case aneDecode
    case coreML
    case coreMLDecode

    var isANE: Bool {
        switch self {
        case .aneDirect, .aneInference, .aneDecode:
            return true
        case .coreML, .coreMLDecode:
            return false
        }
    }
}

struct BenchTimingBreakdown: Sendable, Hashable {
    let aneMs: Double
    let ioMs: Double
    let elemMs: Double

    var totalMs: Double {
        aneMs + ioMs + elemMs
    }
}

struct BenchSummaryMetrics: Sendable, Hashable {
    let meanMs: Double
    let medianMs: Double
    let p95Ms: Double
    let p99Ms: Double
    let minMs: Double
    let maxMs: Double
    let stddevMs: Double
    let warmupCount: Int
    let iterationCount: Int
    let throughputPerSecond: Double
}

struct BenchSummaryEntry: Identifiable, Sendable, Hashable {
    let label: String
    let kind: BenchSummaryEntryKind
    let compileTimeMs: Double?
    let modelLoadTimeMs: Double?
    let timingBreakdown: BenchTimingBreakdown?
    let metrics: BenchSummaryMetrics

    var id: String {
        "\(kind.rawValue):\(label)"
    }
}

struct BenchSummarySnapshot: Sendable, Hashable {
    let mode: String
    let chipName: String?
    let artifactPaths: [String]
    let entries: [BenchSummaryEntry]

    var primaryEntry: BenchSummaryEntry? {
        let rankedKinds: [BenchSummaryEntryKind] = [.aneDirect, .aneInference, .aneDecode, .coreML, .coreMLDecode]
        for kind in rankedKinds {
            if let match = entries.first(where: { $0.kind == kind }) {
                return match
            }
        }
        return entries.first
    }
}
