import SwiftUI

struct LogConsoleView: View {
    let run: BenchRunRecord?

    var body: some View {
        GlassPanel(contentPadding: 18) {
            VStack(alignment: .leading, spacing: 14) {
                Text("Live Logs")
                    .font(.title3.weight(.semibold))

                if let run {
                    InsetSurface {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Command")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.secondary)
                            Text(run.commandDescription.isEmpty ? "No command recorded." : run.commandDescription)
                                .font(.footnote.monospaced())
                                .foregroundStyle(.secondary)
                                .lineLimit(3)
                                .textSelection(.enabled)
                        }
                    }

                    InsetSurface {
                        ScrollView {
                            Text(run.log.isEmpty ? "No log output yet." : run.log)
                                .font(.system(.body, design: .monospaced))
                                .lineSpacing(2)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .scrollIndicators(.hidden)
                        .frame(minHeight: 190)
                    }
                } else {
                    ContentUnavailableView(
                        "No Logs",
                        systemImage: "terminal",
                        description: Text("Start a benchmark to stream stdout and stderr here.")
                    )
                }
            }
        }
    }
}
