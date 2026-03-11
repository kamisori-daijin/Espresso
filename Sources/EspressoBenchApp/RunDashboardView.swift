import Charts
import SwiftUI

struct RunDashboardView: View {
    @Bindable var model: BenchAppModel

    var body: some View {
        if let run = model.selectedRun {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    header(run: run)
                    metrics(run: run)
                    benchmarkEntriesSection(run: run)
                    timingBreakdownSection(run: run)
                    if !run.latencySeries.isEmpty {
                        chartSection(run: run)
                    }
                    detailSection(run: run)
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 16)
            }
            .scrollIndicators(.hidden)
        } else {
            ContentUnavailableView(
                "No Benchmark Selected",
                systemImage: "waveform.path.ecg.rectangle",
                description: Text("Run a benchmark from the sidebar or select an existing result directory.")
            )
        }
    }

    private func header(run: BenchRunRecord) -> some View {
        GlassPanel(contentPadding: 22) {
            HStack(alignment: .top, spacing: 18) {
                VStack(alignment: .leading, spacing: 12) {
                    Text(run.title)
                        .font(.system(size: 28, weight: .bold, design: .rounded))

                    Text(run.outputDirectory.path)
                        .font(.footnote.monospaced())
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .textSelection(.enabled)

                    ViewThatFits(in: .horizontal) {
                        HStack(spacing: 10) {
                            metaChips(run: run)
                        }
                        VStack(alignment: .leading, spacing: 8) {
                            metaChips(run: run)
                        }
                    }
                }

                Spacer(minLength: 18)

                VStack(alignment: .trailing, spacing: 8) {
                    statusPill(run.status)
                    if let primary = run.primaryEntry {
                        Text(formatMilliseconds(primary.metrics.medianMs))
                            .font(.system(size: 24, weight: .semibold, design: .rounded))
                            .foregroundStyle(.primary)
                            .monospacedDigit()
                        Text(primary.label)
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
    }

    private func metrics(run: BenchRunRecord) -> some View {
        let summaryEntry = run.primaryEntry
        let series = run.primarySeries
        return VStack(spacing: 14) {
            primaryMetricRow(
                median: formatMilliseconds(summaryEntry?.metrics.medianMs ?? series?.stats.median ?? 0),
                p95: formatMilliseconds(summaryEntry?.metrics.p95Ms ?? series?.stats.p95 ?? 0),
                p99: formatMilliseconds(summaryEntry?.metrics.p99Ms ?? series?.stats.p99 ?? 0),
                throughput: formatRate(summaryEntry?.metrics.throughputPerSecond ?? 0)
            )

            secondaryMetricRow(
                samples: "\(summaryEntry?.metrics.iterationCount ?? series?.stats.sampleCount ?? 0)",
                setupLabel: summaryEntry?.compileTimeMs != nil ? "Compile" : "Load",
                setupValue: formatOptionalMilliseconds(summaryEntry?.compileTimeMs ?? summaryEntry?.modelLoadTimeMs),
                artifacts: "\(run.artifactFiles.count)"
            )
        }
    }

    @ViewBuilder
    private func benchmarkEntriesSection(run: BenchRunRecord) -> some View {
        if let entries = run.summarySnapshot?.entries, !entries.isEmpty {
            let baselineMedian = run.primaryEntry?.metrics.medianMs ?? 0

            GlassPanel(contentPadding: 20) {
                VStack(alignment: .leading, spacing: 16) {
                    Text("Benchmarks")
                        .font(.title3.weight(.semibold))

                    Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 10) {
                        GridRow {
                            headerText("Label")
                            headerText("Median")
                            headerText("P95")
                            headerText("Throughput")
                            headerText("Setup")
                            headerText("Relative")
                        }

                        Divider()
                            .gridCellColumns(6)

                        ForEach(entries) { entry in
                            GridRow {
                                Text(entry.label)
                                    .font(.headline)
                                valueText(formatMilliseconds(entry.metrics.medianMs))
                                valueText(formatMilliseconds(entry.metrics.p95Ms))
                                valueText(formatRate(entry.metrics.throughputPerSecond))
                                valueText(formatOptionalMilliseconds(entry.compileTimeMs ?? entry.modelLoadTimeMs))
                                valueText(relativeLabel(for: entry, baselineMedian: baselineMedian, primaryID: run.primaryEntry?.id))
                            }
                        }
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func timingBreakdownSection(run: BenchRunRecord) -> some View {
        if let breakdown = run.primaryEntry?.timingBreakdown {
            let total = breakdown.totalMs

            GlassPanel(contentPadding: 20) {
                VStack(alignment: .leading, spacing: 16) {
                    Text("Timing Breakdown")
                        .font(.title3.weight(.semibold))

                    HStack(spacing: 14) {
                        MetricCardView(
                            title: "ANE",
                            value: formatMilliseconds(breakdown.aneMs),
                            detail: formatPercent(component: breakdown.aneMs, total: total),
                            tint: .cyan
                        )
                        MetricCardView(
                            title: "IO",
                            value: formatMilliseconds(breakdown.ioMs),
                            detail: formatPercent(component: breakdown.ioMs, total: total),
                            tint: .yellow
                        )
                        MetricCardView(
                            title: "CPU",
                            value: formatMilliseconds(breakdown.elemMs),
                            detail: formatPercent(component: breakdown.elemMs, total: total),
                            tint: .green
                        )
                    }
                }
            }
        }
    }

    private func chartSection(run: BenchRunRecord) -> some View {
        GlassPanel(contentPadding: 20) {
            VStack(alignment: .leading, spacing: 16) {
                Text("Latency Curves")
                    .font(.title3.weight(.semibold))

                Chart {
                    ForEach(run.latencySeries) { series in
                        ForEach(series.points) { point in
                            LineMark(
                                x: .value("Iteration", point.iteration),
                                y: .value("Latency (ms)", point.latencyMs)
                            )
                            .interpolationMethod(.catmullRom)
                            .foregroundStyle(by: .value("Series", series.name))
                        }
                    }
                }
                .frame(minHeight: 280)
                .accessibilityLabel("Latency chart")
                .accessibilityValue(Text(accessibilityChartSummary(for: run.latencySeries)))

                VStack(alignment: .leading, spacing: 8) {
                    ForEach(run.latencySeries) { series in
                        HStack {
                            Text(series.name)
                                .font(.headline)
                            Spacer()
                            Text("median \(formatMilliseconds(series.stats.median)) · p95 \(formatMilliseconds(series.stats.p95))")
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
        }
    }

    private func detailSection(run: BenchRunRecord) -> some View {
        ViewThatFits(in: .horizontal) {
            HStack(alignment: .top, spacing: 16) {
                summaryCard(run: run)
                artifactsCard(run: run)
                    .frame(width: 320)
            }

            VStack(alignment: .leading, spacing: 16) {
                summaryCard(run: run)
                artifactsCard(run: run)
            }
        }
    }

    private func summaryCard(run: BenchRunRecord) -> some View {
        GlassPanel(contentPadding: 20) {
            VStack(alignment: .leading, spacing: 12) {
                Text("Summary")
                    .font(.title3.weight(.semibold))
                InsetSurface {
                    ScrollView {
                        Text(run.summaryText)
                            .font(.system(.body, design: .monospaced))
                            .lineSpacing(2)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .scrollIndicators(.hidden)
                    .frame(minHeight: 132, idealHeight: 160)
                }
            }
        }
    }

    private func artifactsCard(run: BenchRunRecord) -> some View {
        GlassPanel(contentPadding: 20) {
            VStack(alignment: .leading, spacing: 12) {
                Text("Artifacts")
                    .font(.title3.weight(.semibold))
                InsetSurface {
                    VStack(alignment: .leading, spacing: 10) {
                        ForEach(run.artifactFiles, id: \.path) { url in
                            HStack(spacing: 12) {
                                Text(url.lastPathComponent)
                                    .font(.body.monospaced())
                                    .textSelection(.enabled)
                                    .lineLimit(1)
                                Spacer()
                                Button("Open") {
                                    model.openArtifact(url)
                                }
                                .buttonStyle(.bordered)
                            }
                        }
                    }
                }
            }
        }
    }

    private func primaryMetricRow(
        median: String,
        p95: String,
        p99: String,
        throughput: String
    ) -> some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 14) {
                MetricCardView(title: "Median", value: median, tint: .teal, emphasis: .primary)
                MetricCardView(title: "P95", value: p95, tint: .orange, emphasis: .primary)
                MetricCardView(title: "P99", value: p99, tint: .pink, emphasis: .primary)
                MetricCardView(title: "Throughput", value: throughput, tint: .blue, emphasis: .primary)
            }

            VStack(spacing: 14) {
                HStack(spacing: 14) {
                    MetricCardView(title: "Median", value: median, tint: .teal, emphasis: .primary)
                    MetricCardView(title: "P95", value: p95, tint: .orange, emphasis: .primary)
                }
                HStack(spacing: 14) {
                    MetricCardView(title: "P99", value: p99, tint: .pink, emphasis: .primary)
                    MetricCardView(title: "Throughput", value: throughput, tint: .blue, emphasis: .primary)
                }
            }
        }
    }

    private func secondaryMetricRow(
        samples: String,
        setupLabel: String,
        setupValue: String,
        artifacts: String
    ) -> some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 14) {
                MetricCardView(title: "Samples", value: samples, tint: .mint)
                MetricCardView(title: setupLabel, value: setupValue, tint: .purple)
                MetricCardView(title: "Artifacts", value: artifacts, tint: .indigo)
            }

            VStack(spacing: 14) {
                HStack(spacing: 14) {
                    MetricCardView(title: "Samples", value: samples, tint: .mint)
                    MetricCardView(title: setupLabel, value: setupValue, tint: .purple)
                }
                MetricCardView(title: "Artifacts", value: artifacts, tint: .indigo)
            }
        }
    }

    @ViewBuilder
    private func metaChips(run: BenchRunRecord) -> some View {
        infoChip(systemImage: "clock", text: run.launchedAt.formatted(date: .abbreviated, time: .standard))
        if let summary = run.summarySnapshot {
            infoChip(systemImage: "dial.low", text: summary.mode.capitalized)
        }
        if let chipName = run.summarySnapshot?.chipName {
            infoChip(systemImage: "cpu", text: chipName)
        }
        if let code = run.terminationStatus {
            infoChip(systemImage: "terminal", text: "exit \(code)")
        }
    }

    private func statusPill(_ status: BenchRunStatus) -> some View {
        Label(status.title, systemImage: "bolt.fill")
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(statusColor(status))
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(
                Capsule(style: .continuous)
                    .fill(statusColor(status).opacity(0.12))
            )
    }

    private func infoChip(systemImage: String, text: String) -> some View {
        Label(text, systemImage: systemImage)
            .font(.caption.weight(.medium))
            .foregroundStyle(.secondary)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(
                Capsule(style: .continuous)
                    .fill(.white.opacity(0.34))
            )
    }

    private func headerText(_ title: String) -> some View {
        Text(title)
            .font(.caption.weight(.semibold))
            .foregroundStyle(.secondary)
    }

    private func valueText(_ value: String) -> some View {
        Text(value)
            .font(.body.monospacedDigit())
    }

    private func formatMilliseconds(_ value: Double) -> String {
        String(format: "%.3f ms", locale: Locale(identifier: "en_US_POSIX"), value)
    }

    private func formatOptionalMilliseconds(_ value: Double?) -> String {
        guard let value else {
            return "n/a"
        }
        return formatMilliseconds(value)
    }

    private func formatRate(_ value: Double) -> String {
        guard value > 0 else {
            return "n/a"
        }
        return String(format: "%.2f/s", locale: Locale(identifier: "en_US_POSIX"), value)
    }

    private func formatPercent(component: Double, total: Double) -> String {
        guard total > 0 else {
            return "0.0%"
        }
        return String(format: "%.1f%%", locale: Locale(identifier: "en_US_POSIX"), (component / total) * 100)
    }

    private func relativeLabel(for entry: BenchSummaryEntry, baselineMedian: Double, primaryID: String?) -> String {
        guard baselineMedian > 0, entry.metrics.medianMs > 0 else {
            return "baseline"
        }
        if entry.id == primaryID {
            return "baseline"
        }

        let ratio = entry.metrics.medianMs / baselineMedian
        return String(format: "%.2fx", locale: Locale(identifier: "en_US_POSIX"), ratio)
    }

    private func accessibilityChartSummary(for series: [BenchLatencySeries]) -> String {
        guard !series.isEmpty else {
            return "No latency samples."
        }

        return series
            .map { item in
                "\(item.name), median \(formatMilliseconds(item.stats.median)), p95 \(formatMilliseconds(item.stats.p95))"
            }
            .joined(separator: "; ")
    }

    private func statusColor(_ status: BenchRunStatus) -> Color {
        switch status {
        case .running:
            return .orange
        case .succeeded:
            return .green
        case .failed:
            return .red
        case .cancelled:
            return .gray
        }
    }
}
