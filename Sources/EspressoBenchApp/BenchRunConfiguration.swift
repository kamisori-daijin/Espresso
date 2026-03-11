import Foundation

enum BenchRunMode: String, CaseIterable, Identifiable, Sendable {
    case direct
    case inferenceOnly
    case decode

    var id: String { rawValue }

    var title: String {
        switch self {
        case .direct:
            return "Full"
        case .inferenceOnly:
            return "Inference Only"
        case .decode:
            return "Decode"
        }
    }

    var slug: String {
        switch self {
        case .direct:
            return "full"
        case .inferenceOnly:
            return "inference"
        case .decode:
            return "decode"
        }
    }
}

struct BenchRunConfiguration: Sendable {
    var workspacePath: String = Self.defaultWorkspacePath()
    var resultsRootRelativePath: String = "benchmarks/results"
    var modelPath: String = "benchmarks/models/transformer_layer.mlpackage"
    var mode: BenchRunMode = .direct
    var includeInferenceComparison: Bool = false
    var aneOnly: Bool = false
    var sustained: Bool = false
    var profileKernels: Bool = false
    var inferenceFP16Handoff: Bool = false
    var warmup: Int = 50
    var iterations: Int = 1_000
    var layers: Int = 1
    var decodeSteps: Int = 32
    var decodeMaxSeq: Int = 32
    var additionalFlags: String = ""

    var workspaceURL: URL {
        URL(fileURLWithPath: workspacePath, isDirectory: true)
    }

    var resultsRootURL: URL {
        if resultsRootRelativePath.hasPrefix("/") {
            return URL(fileURLWithPath: resultsRootRelativePath, isDirectory: true)
        }
        return workspaceURL.appendingPathComponent(resultsRootRelativePath, isDirectory: true)
    }

    var supportsInferencePath: Bool {
        mode != .decode
    }

    var allowsSustainedRun: Bool {
        mode == .direct
    }

    func makeOutputDirectory() -> URL {
        let formatter = DateFormatter()
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone.current
        formatter.dateFormat = "yyyy-MM-dd-HHmmss"
        let stamp = formatter.string(from: Date())
        let baseName = "app-\(mode.slug)-\(stamp)"
        var candidate = resultsRootURL.appendingPathComponent(baseName, isDirectory: true)
        var suffix = 1
        while FileManager.default.fileExists(atPath: candidate.path) {
            candidate = resultsRootURL.appendingPathComponent("\(baseName)-\(suffix)", isDirectory: true)
            suffix += 1
        }
        return candidate
    }

    func commandArguments(outputDirectory: URL) -> [String] {
        var arguments: [String] = [
            "--warmup", String(warmup),
            "--iterations", String(iterations),
            "--layers", String(layers),
            "--model", resolvedModelPath(),
            "--output", outputDirectory.path,
        ]

        switch mode {
        case .direct:
            if includeInferenceComparison {
                arguments.append("--inference")
            }
        case .inferenceOnly:
            arguments.append("--inference-only")
        case .decode:
            arguments.append(contentsOf: [
                "--decode",
                "--decode-steps", String(decodeSteps),
                "--decode-max-seq", String(decodeMaxSeq),
            ])
        }

        if aneOnly {
            arguments.append("--ane-only")
        }
        if allowsSustainedRun && sustained {
            arguments.append("--sustained")
        }
        if profileKernels {
            arguments.append("--profile-kernels")
        }
        if supportsInferencePath && inferenceFP16Handoff {
            arguments.append("--inference-fp16-handoff")
        }

        arguments.append(contentsOf: Self.splitAdditionalFlags(additionalFlags))
        return arguments
    }

    func resultsExist() -> Bool {
        FileManager.default.fileExists(atPath: resultsRootURL.path)
    }

    func resolvedModelPath() -> String {
        if modelPath.hasPrefix("/") {
            return modelPath
        }
        return workspaceURL.appendingPathComponent(modelPath).path
    }

    private static func defaultWorkspacePath() -> String {
        let fm = FileManager.default
        let env = ProcessInfo.processInfo.environment
        var candidates: [String] = []
        if let workspace = env["ESPRESSO_WORKSPACE"] {
            candidates.append(workspace)
        }
        candidates.append(fm.currentDirectoryPath)
        candidates.append(URL(fileURLWithPath: fm.currentDirectoryPath).deletingLastPathComponent().path)

        for candidate in candidates {
            let packagePath = URL(fileURLWithPath: candidate).appendingPathComponent("Package.swift").path
            if fm.fileExists(atPath: packagePath) {
                return candidate
            }
        }

        return fm.currentDirectoryPath
    }

    private static func splitAdditionalFlags(_ raw: String) -> [String] {
        raw
            .split(whereSeparator: \.isWhitespace)
            .map(String.init)
    }
}
