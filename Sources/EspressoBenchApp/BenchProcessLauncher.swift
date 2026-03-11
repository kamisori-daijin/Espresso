import Foundation

enum BenchLaunchSource: String, Sendable {
    case bundledCLI
    case workspaceRelease
    case workspaceDebug
    case swiftRunFallback

    var title: String {
        switch self {
        case .bundledCLI:
            return "Bundled CLI"
        case .workspaceRelease:
            return "Workspace Release Build"
        case .workspaceDebug:
            return "Workspace Debug Build"
        case .swiftRunFallback:
            return "swift run Fallback"
        }
    }

    var detail: String {
        switch self {
        case .bundledCLI:
            return "Runs the `espresso-bench` binary embedded inside the packaged app bundle."
        case .workspaceRelease:
            return "Runs the workspace's prebuilt release binary."
        case .workspaceDebug:
            return "Runs the workspace's prebuilt debug binary."
        case .swiftRunFallback:
            return "Falls back to `swift run espresso-bench`, which may trigger a local build before benchmarking."
        }
    }
}

struct BenchLaunchCommand: Sendable {
    let executablePath: String
    let arguments: [String]
    let displayCommand: String
    let source: BenchLaunchSource
}

enum BenchProcessLauncher {
    static func makeCommand(
        configuration: BenchRunConfiguration,
        outputDirectory: URL
    ) -> BenchLaunchCommand {
        let cliArguments = configuration.commandArguments(outputDirectory: outputDirectory)

        if let bundledPath = bundledCLIPath(), FileManager.default.fileExists(atPath: bundledPath) {
            return BenchLaunchCommand(
                executablePath: bundledPath,
                arguments: cliArguments,
                displayCommand: ([bundledPath] + cliArguments).joined(separator: " "),
                source: .bundledCLI
            )
        }

        let releasePath = configuration.workspaceURL
            .appendingPathComponent(".build/release/espresso-bench")
            .path
        if FileManager.default.fileExists(atPath: releasePath) {
            return BenchLaunchCommand(
                executablePath: releasePath,
                arguments: cliArguments,
                displayCommand: ([releasePath] + cliArguments).joined(separator: " "),
                source: .workspaceRelease
            )
        }

        let debugPath = configuration.workspaceURL
            .appendingPathComponent(".build/debug/espresso-bench")
            .path
        if FileManager.default.fileExists(atPath: debugPath) {
            return BenchLaunchCommand(
                executablePath: debugPath,
                arguments: cliArguments,
                displayCommand: ([debugPath] + cliArguments).joined(separator: " "),
                source: .workspaceDebug
            )
        }

        let swiftRunArguments = ["swift", "run", "espresso-bench"] + cliArguments
        return BenchLaunchCommand(
            executablePath: "/usr/bin/env",
            arguments: swiftRunArguments,
            displayCommand: (["/usr/bin/env"] + swiftRunArguments).joined(separator: " "),
            source: .swiftRunFallback
        )
    }

    static func previewCommand(configuration: BenchRunConfiguration) -> BenchLaunchCommand {
        makeCommand(
            configuration: configuration,
            outputDirectory: configuration.resultsRootURL.appendingPathComponent("preview-run", isDirectory: true)
        )
    }

    private static func bundledCLIPath() -> String? {
        guard let bundleURL = Bundle.main.bundleURL as URL? else {
            return nil
        }
        return bundleURL
            .appendingPathComponent("Contents/MacOS/espresso-bench")
            .path
    }
}
