import Foundation

struct PromptSuiteEntry: Sendable, Equatable {
    let id: String
    let text: String
}

struct PromptSuiteConfig: Sendable, Equatable, Encodable {
    let runs: Int
    let warmup: Int
    let iterations: Int
    let maxTokens: Int

    enum CodingKeys: String, CodingKey {
        case runs
        case warmup
        case iterations
        case maxTokens = "max_tokens"
    }
}

struct PromptSuiteMetadata: Sendable, Equatable, Encodable {
    let commit: String
    let timestamp: String
    let promptsFile: String
    let promptIDs: [String]
    let runs: Int
    let warmup: Int
    let iterations: Int
    let maxTokens: Int
    let coldRun: Bool

    enum CodingKeys: String, CodingKey {
        case commit
        case timestamp
        case promptsFile = "prompts_file"
        case promptIDs = "prompt_ids"
        case runs
        case warmup
        case iterations
        case maxTokens = "max_tokens"
        case coldRun = "cold_run"
    }
}

struct PromptSuiteMetricSummary: Sendable, Equatable, Encodable {
    let values: [Double]
    let median: Double
    let min: Double
    let max: Double
}

struct PromptSuitePromptSummary: Sendable, Equatable, Encodable {
    let promptID: String
    let nRuns: Int
    let espressoTokS: PromptSuiteMetricSummary
    let coreMLTokS: PromptSuiteMetricSummary
    let speedup: PromptSuiteMetricSummary
    let allTokenMatch: Bool
    let allTextMatch: Bool

    enum CodingKeys: String, CodingKey {
        case promptID = "prompt_id"
        case nRuns = "n_runs"
        case espressoTokS = "espresso_tok_s"
        case coreMLTokS = "coreml_tok_s"
        case speedup
        case allTokenMatch = "all_token_match"
        case allTextMatch = "all_text_match"
    }
}

struct PromptSuiteAggregate: Sendable, Equatable, Encodable {
    let nPrompts: Int
    let totalRuns: Int
    let espressoTokSMedian: Double
    let coreMLTokSMedian: Double
    let speedupMedian: Double
    let allTokenMatch: Bool
    let allTextMatch: Bool

    enum CodingKeys: String, CodingKey {
        case nPrompts = "n_prompts"
        case totalRuns = "total_runs"
        case espressoTokSMedian = "espresso_tok_s_median"
        case coreMLTokSMedian = "coreml_tok_s_median"
        case speedupMedian = "speedup_median"
        case allTokenMatch = "all_token_match"
        case allTextMatch = "all_text_match"
    }
}

struct PromptSuiteVerdict: Sendable, Equatable, Encodable {
    let allCorrectnessGatesPass: Bool

    enum CodingKeys: String, CodingKey {
        case allCorrectnessGatesPass = "all_correctness_gates_pass"
    }
}

struct PromptSuiteSummary: Sendable, Equatable, Encodable {
    let commit: String
    let timestamp: String
    let config: PromptSuiteConfig
    let perPrompt: [PromptSuitePromptSummary]
    let aggregate: PromptSuiteAggregate
    let verdict: PromptSuiteVerdict

    enum CodingKeys: String, CodingKey {
        case commit
        case timestamp
        case config
        case perPrompt = "per_prompt"
        case aggregate
        case verdict
    }
}

struct PromptSuiteRunRecord: Sendable {
    let promptID: String
    let report: CompareReport
}

func loadPromptSuite(from path: String) throws -> [PromptSuiteEntry] {
    let expanded = NSString(string: path).expandingTildeInPath
    let contents = try String(contentsOfFile: expanded, encoding: .utf8)
    var prompts: [PromptSuiteEntry] = []
    var seenIDs: Set<String> = []
    for (lineIndex, rawLine) in contents.split(whereSeparator: \.isNewline).enumerated() {
        let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
        if line.isEmpty || line.hasPrefix("#") {
            continue
        }
        guard let separator = line.firstIndex(of: ":") else {
            throw CLIError.usage("Invalid prompt suite line \(lineIndex + 1) in \(expanded). Expected id:prompt_text")
        }
        let id = String(line[..<separator]).trimmingCharacters(in: .whitespaces)
        let text = String(line[line.index(after: separator)...]).trimmingCharacters(in: .whitespaces)
        guard !id.isEmpty, !text.isEmpty else {
            throw CLIError.usage("Invalid prompt suite line \(lineIndex + 1) in \(expanded). Expected non-empty id and prompt text")
        }
        guard seenIDs.insert(id).inserted else {
            throw CLIError.usage("Duplicate prompt id '\(id)' in \(expanded)")
        }
        prompts.append(PromptSuiteEntry(id: id, text: text))
    }
    guard !prompts.isEmpty else {
        throw CLIError.usage("No prompts found in \(expanded)")
    }
    return prompts
}

func resolvedSuiteCoreMLSequenceLength(
    explicitSequenceLength: Int?,
    promptTokenCounts: [Int],
    maxTokens: Int,
    maxModelSequenceLength: Int
) throws -> Int {
    guard !promptTokenCounts.isEmpty else {
        throw CLIError.runtime("Prompt suite is empty.")
    }
    let requiredSequenceLength = promptTokenCounts
        .map { $0 + max(maxTokens, 1) }
        .max() ?? max(maxTokens, 1)
    guard requiredSequenceLength <= maxModelSequenceLength else {
        throw CLIError.usage(
            "Prompt suite requires Core ML sequence length \(requiredSequenceLength), exceeding model capacity \(maxModelSequenceLength)"
        )
    }
    if let explicitSequenceLength {
        guard explicitSequenceLength >= requiredSequenceLength else {
            throw CLIError.usage(
                "Core ML sequence length \(explicitSequenceLength) is too small for the suite requirement \(requiredSequenceLength)"
            )
        }
        return explicitSequenceLength
    }
    let roundedSequenceLength = nextPowerOfTwo(requiredSequenceLength)
    return roundedSequenceLength <= maxModelSequenceLength ? roundedSequenceLength : requiredSequenceLength
}

func makePromptSuiteSummary(
    promptOrder: [PromptSuiteEntry],
    reports: [PromptSuiteRunRecord],
    commit: String,
    timestamp: String,
    config: PromptSuiteConfig
) -> PromptSuiteSummary {
    let perPrompt = promptOrder.compactMap { prompt -> PromptSuitePromptSummary? in
        let promptReports = reports.filter { $0.promptID == prompt.id }
        guard !promptReports.isEmpty else {
            return nil
        }
        let espressoValues = promptReports.map(\.report.espresso.tokensPerSecond)
        let coreMLValues = promptReports.map(\.report.coreML.tokensPerSecond)
        let speedupValues = promptReports.map {
            $0.report.espresso.tokensPerSecond / max($0.report.coreML.tokensPerSecond, 1e-9)
        }
        return PromptSuitePromptSummary(
            promptID: prompt.id,
            nRuns: promptReports.count,
            espressoTokS: metricSummary(espressoValues),
            coreMLTokS: metricSummary(coreMLValues),
            speedup: metricSummary(speedupValues),
            allTokenMatch: promptReports.allSatisfy(\.report.tokenMatch),
            allTextMatch: promptReports.allSatisfy(\.report.textMatch)
        )
    }

    let aggregate = PromptSuiteAggregate(
        nPrompts: perPrompt.count,
        totalRuns: perPrompt.map(\.nRuns).reduce(0, +),
        espressoTokSMedian: median(perPrompt.map(\.espressoTokS.median)),
        coreMLTokSMedian: median(perPrompt.map(\.coreMLTokS.median)),
        speedupMedian: median(perPrompt.map(\.speedup.median)),
        allTokenMatch: perPrompt.allSatisfy(\.allTokenMatch),
        allTextMatch: perPrompt.allSatisfy(\.allTextMatch)
    )
    let verdict = PromptSuiteVerdict(
        allCorrectnessGatesPass: aggregate.allTokenMatch && aggregate.allTextMatch
    )
    return PromptSuiteSummary(
        commit: commit,
        timestamp: timestamp,
        config: config,
        perPrompt: perPrompt,
        aggregate: aggregate,
        verdict: verdict
    )
}

func promptSuiteSummaryLines(_ summary: PromptSuiteSummary) -> [String] {
    var lines: [String] = [
        "Commit: \(summary.commit)",
        "Runs: \(summary.config.runs) x \(summary.perPrompt.count) prompts",
        "",
        "Per-prompt results:",
        "  Prompt ID       | Espresso tok/s (median) | CoreML tok/s (median) | Speedup (median) | Token Match | Text Match",
        "  --------------- | ----------------------- | --------------------- | ---------------- | ----------- | ----------",
    ]
    for prompt in summary.perPrompt {
        lines.append(
            "  \(pad(prompt.promptID, to: 15)) | \(pad(formatMetric(prompt.espressoTokS.median), to: 23)) | \(pad(formatMetric(prompt.coreMLTokS.median), to: 21)) | \(pad(formatSpeedup(prompt.speedup.median), to: 16)) | \(pad(prompt.allTokenMatch ? "PASS" : "FAIL", to: 11)) | \(prompt.allTextMatch ? "PASS" : "FAIL")"
        )
    }
    lines.append("")
    lines.append("Aggregate:")
    lines.append("  Espresso median: \(formatMetric(summary.aggregate.espressoTokSMedian)) tok/s")
    lines.append("  CoreML median:   \(formatMetric(summary.aggregate.coreMLTokSMedian)) tok/s")
    lines.append("  Speedup median:  \(formatSpeedup(summary.aggregate.speedupMedian))")
    lines.append("  Correctness:     \(summary.verdict.allCorrectnessGatesPass ? "ALL PASS" : "FAIL")")
    return lines
}

func promptSuiteResultsTSVHeader() -> String {
    "timestamp\tcommit\tstatus\tprimary_metric\tespresso_tokens_per_second\tcoreml_tokens_per_second\tspeedup_vs_coreml\ttoken_match\ttext_match\tespresso_first_token_ms\tcoreml_first_token_ms\tespresso_median_token_ms\tcoreml_median_token_ms\tespresso_p95_token_ms\tcoreml_p95_token_ms\tespresso_compile_ms\tcoreml_compile_ms\tespresso_compile_retry_count\tespresso_compile_failure_count\tespresso_exact_head_backend\tespresso_cached_bindings_enabled\toutput_dir\tprompt_id\tchange_summary"
}

func promptSuiteResultsTSVRow(
    timestamp: String,
    commit: String,
    promptID: String,
    outputDirectory: String,
    report: CompareReport
) -> String {
    [
        timestamp,
        commit,
        "measured",
        "espresso_tokens_per_second",
        String(report.espresso.tokensPerSecond),
        String(report.coreML.tokensPerSecond),
        String(report.espresso.tokensPerSecond / max(report.coreML.tokensPerSecond, 1e-9)),
        String(report.tokenMatch),
        String(report.textMatch),
        String(report.espresso.firstTokenLatencyMs),
        String(report.coreML.firstTokenLatencyMs),
        String(report.espresso.medianTokenMs),
        String(report.coreML.medianTokenMs),
        String(report.espresso.p95TokenMs),
        String(report.coreML.p95TokenMs),
        String(report.espresso.compileTimeMs),
        String(report.coreML.compileTimeMs),
        String(report.espresso.compileRetryCount),
        String(report.espresso.compileFailureCount),
        report.espresso.exactHeadBackend ?? "",
        report.espresso.cachedBindingsEnabled.map(String.init(describing:)) ?? "",
        outputDirectory,
        promptID,
        "",
    ].joined(separator: "\t")
}

private func metricSummary(_ values: [Double]) -> PromptSuiteMetricSummary {
    let sorted = values.sorted()
    return PromptSuiteMetricSummary(
        values: sorted,
        median: median(sorted),
        min: sorted.min() ?? 0,
        max: sorted.max() ?? 0
    )
}

private func median(_ values: [Double]) -> Double {
    guard !values.isEmpty else { return 0 }
    let sorted = values.sorted()
    let midpoint = sorted.count / 2
    if sorted.count.isMultiple(of: 2) {
        return (sorted[midpoint - 1] + sorted[midpoint]) / 2
    }
    return sorted[midpoint]
}

private func formatMetric(_ value: Double) -> String {
    String(format: "%.2f", locale: Locale(identifier: "en_US_POSIX"), value)
}

private func formatSpeedup(_ value: Double) -> String {
    String(format: "%.2fx", locale: Locale(identifier: "en_US_POSIX"), value)
}

private func pad(_ value: String, to width: Int) -> String {
    if value.count >= width {
        return String(value.prefix(width))
    }
    return value + String(repeating: " ", count: width - value.count)
}
