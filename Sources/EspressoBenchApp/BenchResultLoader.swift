import Foundation

enum BenchResultLoader {
    static func loadRun(
        from outputDirectory: URL,
        fallbackLog: String,
        commandDescription: String,
        launchedAt: Date,
        status: BenchRunStatus,
        terminationStatus: Int32?
    ) -> BenchRunRecord {
        let files = (try? FileManager.default.contentsOfDirectory(
            at: outputDirectory,
            includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles]
        )) ?? []

        let summaryURL = outputDirectory.appendingPathComponent("summary.txt")
        let summaryText = (try? String(contentsOf: summaryURL, encoding: .utf8)) ?? "No summary.txt found."
        let summarySnapshot = loadSummarySnapshot(from: outputDirectory.appendingPathComponent("summary.json"))
        let latencySeries = files
            .filter { $0.pathExtension == "csv" }
            .compactMap(loadLatencySeries(from:))
            .sorted(by: latencySeriesSort)

        return BenchRunRecord(
            id: outputDirectory.standardizedFileURL.path,
            title: outputDirectory.lastPathComponent,
            launchedAt: launchedAt,
            outputDirectory: outputDirectory,
            status: status,
            terminationStatus: terminationStatus,
            commandDescription: commandDescription,
            log: fallbackLog,
            summaryText: summaryText,
            summarySnapshot: summarySnapshot,
            latencySeries: latencySeries,
            artifactFiles: files.sorted { $0.lastPathComponent < $1.lastPathComponent }
        )
    }

    static func loadHistory(from resultsRoot: URL) -> [BenchRunRecord] {
        let directories = (try? FileManager.default.contentsOfDirectory(
            at: resultsRoot,
            includingPropertiesForKeys: [.isDirectoryKey, .contentModificationDateKey],
            options: [.skipsHiddenFiles]
        )) ?? []

        return directories
            .filter { url in
                var isDirectory: ObjCBool = false
                return FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) && isDirectory.boolValue
            }
            .compactMap { directory in
                let launchedAt = (try? directory.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                let summaryExists = FileManager.default.fileExists(atPath: directory.appendingPathComponent("summary.json").path)
                    || FileManager.default.fileExists(atPath: directory.appendingPathComponent("summary.txt").path)
                return loadRun(
                    from: directory,
                    fallbackLog: "",
                    commandDescription: "",
                    launchedAt: launchedAt,
                    status: summaryExists ? .succeeded : .failed,
                    terminationStatus: summaryExists ? 0 : nil
                )
            }
            .sorted { $0.launchedAt > $1.launchedAt }
    }

    private static func loadSummarySnapshot(from url: URL) -> BenchSummarySnapshot? {
        guard let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return nil
        }

        var entries: [BenchSummaryEntry] = []
        if let entry = parseSummaryEntry(json["ane_direct"], fallbackLabel: "ANE Direct", kind: .aneDirect) {
            entries.append(entry)
        }
        if let entry = parseSummaryEntry(json["ane_inference"], fallbackLabel: "ANE Inference", kind: .aneInference) {
            entries.append(entry)
        }
        if let entry = parseSummaryEntry(json["ane_decode"], fallbackLabel: "ANE Decode", kind: .aneDecode) {
            entries.append(entry)
        }

        if let coreML = json["coreml"] as? [String: Any] {
            let modelLoadTimeMs = doubleValue(coreML["model_load_time_ms"])
            let results = coreML["results"] as? [[String: Any]] ?? []
            entries.append(contentsOf: results.compactMap { parseResultEntry($0, kind: .coreML, modelLoadTimeMs: modelLoadTimeMs) })
        }

        if let coreMLDecode = json["coreml_decode"] as? [String: Any] {
            let modelLoadTimeMs = doubleValue(coreMLDecode["model_load_time_ms"])
            let results = coreMLDecode["results"] as? [[String: Any]] ?? []
            entries.append(contentsOf: results.compactMap { parseResultEntry($0, kind: .coreMLDecode, modelLoadTimeMs: modelLoadTimeMs) })
        }

        let artifactPaths = (json["artifacts"] as? [[String: Any]] ?? [])
            .compactMap { $0["path"] as? String }

        let chipName = (json["device"] as? [String: Any]).flatMap { $0["chip"] as? String }
        let mode = (json["mode"] as? String) ?? "unknown"

        return BenchSummarySnapshot(
            mode: mode,
            chipName: chipName,
            artifactPaths: artifactPaths,
            entries: entries
        )
    }

    private static func loadLatencySeries(from url: URL) -> BenchLatencySeries? {
        guard let csv = try? String(contentsOf: url, encoding: .utf8) else {
            return nil
        }

        let rows = csv
            .split(whereSeparator: \.isNewline)
            .map(String.init)
        guard rows.first == "iteration,latency_ms" else {
            return nil
        }

        let points = rows.dropFirst().compactMap { row -> BenchLatencyPoint? in
            let parts = row.split(separator: ",", omittingEmptySubsequences: false)
            guard parts.count == 2,
                  let iteration = Int(parts[0]),
                  let latency = Double(parts[1])
            else {
                return nil
            }
            return BenchLatencyPoint(iteration: iteration, latencyMs: latency)
        }
        guard !points.isEmpty else {
            return nil
        }

        let latencies = points.map(\.latencyMs)
        return BenchLatencySeries(
            name: prettySeriesName(fileName: url.lastPathComponent),
            fileName: url.lastPathComponent,
            points: points,
            stats: BenchLatencyStats(latencies: latencies)
        )
    }

    private static func latencySeriesSort(lhs: BenchLatencySeries, rhs: BenchLatencySeries) -> Bool {
        let rank: (BenchLatencySeries) -> Int = { series in
            if series.fileName.hasPrefix("ane_") { return 0 }
            if series.fileName.contains("coreml") { return 1 }
            return 2
        }
        if rank(lhs) != rank(rhs) {
            return rank(lhs) < rank(rhs)
        }
        return lhs.fileName < rhs.fileName
    }

    private static func prettySeriesName(fileName: String) -> String {
        fileName
            .replacingOccurrences(of: ".csv", with: "")
            .replacingOccurrences(of: "_", with: " ")
            .capitalized
    }

    private static func parseSummaryEntry(
        _ value: Any?,
        fallbackLabel: String,
        kind: BenchSummaryEntryKind
    ) -> BenchSummaryEntry? {
        guard let dict = value as? [String: Any],
              let metrics = parseMetrics(dict["metrics"])
        else {
            return nil
        }

        return BenchSummaryEntry(
            label: fallbackLabel,
            kind: kind,
            compileTimeMs: doubleValue(dict["compile_time_ms"]),
            modelLoadTimeMs: nil,
            timingBreakdown: parseTimingBreakdown(dict["timing_breakdown_ms"]),
            metrics: metrics
        )
    }

    private static func parseResultEntry(
        _ dict: [String: Any],
        kind: BenchSummaryEntryKind,
        modelLoadTimeMs: Double?
    ) -> BenchSummaryEntry? {
        guard let metrics = parseMetrics(dict["metrics"]) else {
            return nil
        }

        return BenchSummaryEntry(
            label: (dict["label"] as? String) ?? "Core ML",
            kind: kind,
            compileTimeMs: nil,
            modelLoadTimeMs: modelLoadTimeMs,
            timingBreakdown: nil,
            metrics: metrics
        )
    }

    private static func parseMetrics(_ value: Any?) -> BenchSummaryMetrics? {
        guard let dict = value as? [String: Any] else {
            return nil
        }

        return BenchSummaryMetrics(
            meanMs: doubleValue(dict["mean_ms"]),
            medianMs: doubleValue(dict["median_ms"]),
            p95Ms: doubleValue(dict["p95_ms"]),
            p99Ms: doubleValue(dict["p99_ms"]),
            minMs: doubleValue(dict["min_ms"]),
            maxMs: doubleValue(dict["max_ms"]),
            stddevMs: doubleValue(dict["stddev_ms"]),
            warmupCount: intValue(dict["warmup_count"]),
            iterationCount: intValue(dict["iteration_count"]),
            throughputPerSecond: doubleValue(dict["tokens_per_second"])
        )
    }

    private static func parseTimingBreakdown(_ value: Any?) -> BenchTimingBreakdown? {
        guard let dict = value as? [String: Any] else {
            return nil
        }

        return BenchTimingBreakdown(
            aneMs: doubleValue(dict["ane"]),
            ioMs: doubleValue(dict["io"]),
            elemMs: doubleValue(dict["elem"])
        )
    }

    private static func doubleValue(_ value: Any?) -> Double {
        switch value {
        case let number as NSNumber:
            return number.doubleValue
        case let string as String:
            return Double(string) ?? 0
        default:
            return 0
        }
    }

    private static func intValue(_ value: Any?) -> Int {
        switch value {
        case let number as NSNumber:
            return number.intValue
        case let string as String:
            return Int(string) ?? 0
        default:
            return 0
        }
    }
}
