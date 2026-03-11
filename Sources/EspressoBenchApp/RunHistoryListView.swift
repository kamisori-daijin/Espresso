import SwiftUI

struct RunHistoryListView: View {
    @Bindable var model: BenchAppModel

    var body: some View {
        GlassPanel(contentPadding: 16) {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text("Recent Runs")
                        .font(.title3.weight(.semibold))
                    Spacer()
                    Button("Refresh") {
                        model.reloadHistory()
                    }
                    .buttonStyle(.bordered)
                }

                List(selection: $model.selectedRunID) {
                    ForEach(model.runs) { run in
                        HStack(spacing: 12) {
                            Circle()
                                .fill(statusColor(run.status))
                                .frame(width: 10, height: 10)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(run.title)
                                    .font(.headline)
                                    .lineLimit(1)
                                Text(historySubtitle(for: run))
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                    .lineLimit(1)
                            }
                            Spacer()
                            Text(run.launchedAt, style: .time)
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(.secondary)
                        }
                        .tag(run.id)
                    }
                }
                .listStyle(.plain)
                .scrollContentBackground(.hidden)
                .frame(minHeight: 220)
            }
        }
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

    private func historySubtitle(for run: BenchRunRecord) -> String {
        if let primary = run.primaryEntry {
            return "\(run.status.title) • \(primary.label) • \(formatMilliseconds(primary.metrics.medianMs))"
        }
        if let primary = run.primarySeries {
            return "\(run.status.title) • \(formatMilliseconds(primary.stats.median))"
        }
        return run.status.title
    }

    private func formatMilliseconds(_ value: Double) -> String {
        String(format: "%.3f ms", locale: Locale(identifier: "en_US_POSIX"), value)
    }
}
