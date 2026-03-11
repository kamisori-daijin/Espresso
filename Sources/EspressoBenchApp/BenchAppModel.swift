import AppKit
import Foundation
import Observation

@MainActor
@Observable
final class BenchAppModel {
    var configuration = BenchRunConfiguration()
    var runs: [BenchRunRecord] = []
    var selectedRunID: String?
    var statusMessage = "Ready"
    var isRunning = false

    private var activeProcess: Process?
    private var stdoutTask: Task<Void, Never>?
    private var stderrTask: Task<Void, Never>?
    private var logFlushTask: Task<Void, Never>?
    private var pendingLogs: [String: String] = [:]

    init() {
        reloadHistory()
    }

    var selectedRun: BenchRunRecord? {
        get { runs.first(where: { $0.id == selectedRunID }) }
        set { selectedRunID = newValue?.id }
    }

    func startRun() {
        guard !isRunning else { return }

        let outputDirectory = configuration.makeOutputDirectory()
        let command = BenchProcessLauncher.makeCommand(configuration: configuration, outputDirectory: outputDirectory)
        let runID = outputDirectory.standardizedFileURL.path
        let launchedAt = Date()

        do {
            let process = Process()
            process.currentDirectoryURL = configuration.workspaceURL
            process.executableURL = URL(fileURLWithPath: command.executablePath)
            process.arguments = command.arguments

            let stdoutPipe = Pipe()
            let stderrPipe = Pipe()
            process.standardOutput = stdoutPipe
            process.standardError = stderrPipe

            process.terminationHandler = { [weak self] terminatedProcess in
                Task { @MainActor [weak self] in
                    self?.finishRun(
                        runID: runID,
                        outputDirectory: outputDirectory,
                        launchedAt: launchedAt,
                        commandDescription: command.displayCommand,
                        terminationStatus: terminatedProcess.terminationStatus
                    )
                }
            }

            try process.run()

            let initialRun = BenchRunRecord(
                id: runID,
                title: outputDirectory.lastPathComponent,
                launchedAt: launchedAt,
                outputDirectory: outputDirectory,
                status: .running,
                terminationStatus: nil,
                commandDescription: command.displayCommand,
                log: "",
                summaryText: "Waiting for benchmark output...",
                summarySnapshot: nil,
                latencySeries: [],
                artifactFiles: []
            )
            runs.insert(initialRun, at: 0)
            selectedRunID = runID
            activeProcess = process
            isRunning = true
            statusMessage = "Running via \(command.source.title)"
            stdoutTask = makeStreamTask(
                handle: stdoutPipe.fileHandleForReading,
                runID: runID,
                prefix: ""
            )
            stderrTask = makeStreamTask(
                handle: stderrPipe.fileHandleForReading,
                runID: runID,
                prefix: "[stderr] "
            )
        } catch {
            statusMessage = "Launch failed: \(error.localizedDescription)"
            replaceRun(
                id: runID,
                with: BenchRunRecord(
                    id: runID,
                    title: outputDirectory.lastPathComponent,
                    launchedAt: launchedAt,
                    outputDirectory: outputDirectory,
                    status: .failed,
                    terminationStatus: nil,
                    commandDescription: command.displayCommand,
                    log: "Launch failed: \(error.localizedDescription)\n",
                    summaryText: "Benchmark launch failed before the CLI could start.",
                    summarySnapshot: nil,
                    latencySeries: [],
                    artifactFiles: []
                )
            )
        }
    }

    func stopRun() {
        guard let activeProcess else { return }
        activeProcess.terminate()
        statusMessage = "Stopping benchmark…"
    }

    func reloadHistory() {
        guard configuration.resultsExist() else {
            return
        }

        let previousSelection = selectedRunID
        let history = BenchResultLoader.loadHistory(from: configuration.resultsRootURL)
        let running = runs.filter { $0.status == .running }
        runs = running + history
        if let previousSelection, runs.contains(where: { $0.id == previousSelection }) {
            selectedRunID = previousSelection
        } else {
            selectedRunID = runs.first?.id
        }
    }

    func openSelectedRunDirectory() {
        guard let run = selectedRun else { return }
        NSWorkspace.shared.activateFileViewerSelecting([run.outputDirectory])
    }

    func openArtifact(_ url: URL) {
        NSWorkspace.shared.open(url)
    }

    func chooseWorkspace() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = false
        panel.allowsMultipleSelection = false
        panel.directoryURL = configuration.workspaceURL
        if panel.runModal() == .OK, let url = panel.url {
            configuration.workspacePath = url.path
            reloadHistory()
        }
    }

    func chooseModel() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsOtherFileTypes = true
        panel.directoryURL = configuration.workspaceURL
        if panel.runModal() == .OK, let url = panel.url {
            configuration.modelPath = url.path
        }
    }

    private func makeStreamTask(handle: FileHandle, runID: String, prefix: String) -> Task<Void, Never> {
        Task { [weak self] in
            do {
                for try await line in handle.bytes.lines {
                    await MainActor.run {
                        self?.enqueueLog("\(prefix)\(line)\n", to: runID)
                    }
                }
            } catch is CancellationError {
                return
            } catch {
                guard !Task.isCancelled else {
                    return
                }
                await MainActor.run {
                    self?.enqueueLog("\(prefix)stream error: \(error.localizedDescription)\n", to: runID)
                }
            }
        }
    }

    private func finishRun(
        runID: String,
        outputDirectory: URL,
        launchedAt: Date,
        commandDescription: String,
        terminationStatus: Int32
    ) {
        stdoutTask?.cancel()
        stderrTask?.cancel()
        stdoutTask = nil
        stderrTask = nil
        flushPendingLogs()
        logFlushTask?.cancel()
        logFlushTask = nil
        activeProcess = nil
        isRunning = false

        let status: BenchRunStatus
        switch terminationStatus {
        case 0:
            status = .succeeded
            statusMessage = "Benchmark completed"
        case SIGTERM:
            status = .cancelled
            statusMessage = "Benchmark cancelled"
        default:
            status = .failed
            statusMessage = "Benchmark failed (\(terminationStatus))"
        }

        let existingLog = runs.first(where: { $0.id == runID })?.log ?? ""
        let loaded = BenchResultLoader.loadRun(
            from: outputDirectory,
            fallbackLog: existingLog,
            commandDescription: commandDescription,
            launchedAt: launchedAt,
            status: status,
            terminationStatus: terminationStatus
        )
        replaceRun(id: runID, with: loaded)
        selectedRunID = loaded.id
    }

    private func appendLog(_ text: String, to runID: String) {
        guard let index = runs.firstIndex(where: { $0.id == runID }) else { return }
        runs[index].log.append(text)
    }

    private func enqueueLog(_ text: String, to runID: String) {
        pendingLogs[runID, default: ""].append(text)
        guard logFlushTask == nil else {
            return
        }

        logFlushTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(150))
            await MainActor.run {
                self?.flushPendingLogs()
            }
        }
    }

    private func flushPendingLogs() {
        guard !pendingLogs.isEmpty else {
            return
        }

        for (runID, chunk) in pendingLogs {
            appendLog(chunk, to: runID)
        }
        pendingLogs.removeAll(keepingCapacity: true)
        logFlushTask = nil
    }

    private func replaceRun(id: String, with replacement: BenchRunRecord) {
        if let index = runs.firstIndex(where: { $0.id == id }) {
            runs[index] = replacement
        } else {
            runs.insert(replacement, at: 0)
        }
    }
}
