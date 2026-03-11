import SwiftUI

struct ContentView: View {
    @Bindable var model: BenchAppModel

    var body: some View {
        ZStack {
            LinearGradient(
                colors: [
                    Color(red: 0.97, green: 0.98, blue: 0.995),
                    Color(red: 0.94, green: 0.96, blue: 0.985),
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()

            HSplitView {
                sidebar

                VSplitView {
                    RunDashboardView(model: model)
                        .frame(minHeight: 520, idealHeight: 760)
                    LogConsoleView(run: model.selectedRun)
                        .frame(minHeight: 220, idealHeight: 280)
                        .padding(.horizontal, 18)
                        .padding(.top, 8)
                        .padding(.bottom, 18)
                }
                .background(
                    LinearGradient(
                        colors: [Color(red: 0.98, green: 0.99, blue: 1.0), Color(red: 0.95, green: 0.97, blue: 0.99)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
            }
        }
        .toolbar {
            ToolbarItem(placement: .navigation) {
                Text("Espresso Bench")
                    .font(.headline.weight(.semibold))
            }

            ToolbarItem(placement: .status) {
                if let run = model.selectedRun {
                    Label(run.status.title, systemImage: "bolt.fill")
                        .foregroundStyle(statusColor(run.status))
                } else {
                    Text("Ready")
                        .foregroundStyle(.secondary)
                }
            }

            ToolbarItemGroup(placement: .primaryAction) {
                Button {
                    model.reloadHistory()
                } label: {
                    Label("Refresh", systemImage: "arrow.clockwise")
                }

                if model.selectedRun != nil {
                    Button {
                        model.openSelectedRunDirectory()
                    } label: {
                        Label("Reveal", systemImage: "folder")
                    }

                    Button {
                        if let run = model.selectedRun {
                            model.openArtifact(run.outputDirectory.appendingPathComponent("summary.txt"))
                        }
                    } label: {
                        Label("Summary", systemImage: "doc.text")
                    }
                }

                Button {
                    model.startRun()
                } label: {
                    Label("Run", systemImage: "play.fill")
                }
                .disabled(model.isRunning)

                Button {
                    model.stopRun()
                } label: {
                    Label("Stop", systemImage: "stop.fill")
                }
                .disabled(!model.isRunning)
            }
        }
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 14) {
            ScrollView {
                RunConfigurationPanel(model: model)
                    .padding(.horizontal, 16)
                    .padding(.top, 16)
                    .padding(.bottom, 8)
            }
            .scrollIndicators(.hidden)

            RunHistoryListView(model: model)
                .padding(.horizontal, 16)
                .padding(.bottom, 16)
                .frame(maxHeight: .infinity)
        }
        .frame(minWidth: 308, idealWidth: 328, maxWidth: 352)
        .background(
            LinearGradient(
                colors: [Color(red: 0.94, green: 0.97, blue: 0.99), Color(red: 0.92, green: 0.94, blue: 0.97)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        )
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
