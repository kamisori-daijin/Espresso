import SwiftUI

struct RunConfigurationPanel: View {
    @Bindable var model: BenchAppModel

    var body: some View {
        let launchPreview = BenchProcessLauncher.previewCommand(configuration: model.configuration)

        GlassPanel(contentPadding: 18) {
            VStack(alignment: .leading, spacing: 16) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Benchmark Controls")
                        .font(.title2.weight(.semibold))
                    Text("Wrap the existing `espresso-bench` CLI, preserve its output directories, and surface live progress while it runs.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }

                Divider()

                VStack(alignment: .leading, spacing: 12) {
                    sectionHeader("Workspace & Model")
                    pathRow("Workspace", path: model.configuration.workspacePath, buttonLabel: "Choose") {
                        model.chooseWorkspace()
                    }
                    pathRow("Model", path: model.configuration.modelPath, buttonLabel: "Choose") {
                        model.chooseModel()
                    }
                    labeledField("Results Root", text: $model.configuration.resultsRootRelativePath)
                }

                Divider()

                VStack(alignment: .leading, spacing: 12) {
                    sectionHeader("Run Shape")
                    Picker("", selection: $model.configuration.mode) {
                        ForEach(BenchRunMode.allCases) { mode in
                            Text(mode.title).tag(mode)
                        }
                    }
                    .labelsHidden()
                    .pickerStyle(.segmented)

                    Grid(alignment: .leading, horizontalSpacing: 10, verticalSpacing: 10) {
                        GridRow {
                            numberField("Warmup", value: $model.configuration.warmup)
                            numberField("Iterations", value: $model.configuration.iterations)
                        }
                        GridRow {
                            numberField("Layers", value: $model.configuration.layers)
                            if model.configuration.mode == .decode {
                                numberField("Decode Steps", value: $model.configuration.decodeSteps)
                            } else {
                                placeholderField("Decode Steps")
                            }
                        }
                    }

                    if model.configuration.mode == .decode {
                        numberField("Decode Max Seq", value: $model.configuration.decodeMaxSeq)
                    }
                }

                Divider()

                VStack(alignment: .leading, spacing: 10) {
                    sectionHeader("Runtime Options")
                    Toggle("ANE only", isOn: $model.configuration.aneOnly)
                }
                .toggleStyle(.switch)

                DisclosureGroup {
                    VStack(alignment: .leading, spacing: 12) {
                        labeledField("Extra Flags", text: $model.configuration.additionalFlags)

                        VStack(alignment: .leading, spacing: 10) {
                            Toggle("Record kernel profile CSVs", isOn: $model.configuration.profileKernels)

                            if model.configuration.allowsSustainedRun {
                                Toggle("Sustained thermal run", isOn: $model.configuration.sustained)
                            }

                            if model.configuration.supportsInferencePath {
                                Toggle("Use FP16 inference handoff", isOn: $model.configuration.inferenceFP16Handoff)
                            }

                            if model.configuration.mode == .direct {
                                Toggle("Include inference comparison", isOn: $model.configuration.includeInferenceComparison)
                            }
                        }
                        .toggleStyle(.switch)

                        InsetSurface {
                            VStack(alignment: .leading, spacing: 8) {
                                sectionHeader("CLI Backend")
                                Text(launchPreview.source.title)
                                    .font(.headline.weight(.semibold))
                                Text(launchPreview.source.detail)
                                    .font(.footnote)
                                    .foregroundStyle(.secondary)
                                    .fixedSize(horizontal: false, vertical: true)
                                Text(launchPreview.executablePath)
                                    .font(.caption.monospaced())
                                    .foregroundStyle(.secondary)
                                    .lineLimit(2)
                                    .textSelection(.enabled)
                            }
                        }
                    }
                    .padding(.top, 8)
                } label: {
                    Text("Advanced")
                        .font(.subheadline.weight(.semibold))
                }

                HStack(spacing: 10) {
                    Button {
                        model.startRun()
                    } label: {
                        Label(model.isRunning ? "Running…" : "Run Benchmark", systemImage: "play.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .disabled(model.isRunning)

                    Button {
                        model.stopRun()
                    } label: {
                        Label("Stop", systemImage: "stop.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.large)
                    .disabled(!model.isRunning)
                }

                Text(model.statusMessage)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func sectionHeader(_ title: String) -> some View {
        Text(title)
            .font(.caption.weight(.semibold))
            .textCase(.uppercase)
            .foregroundStyle(.secondary)
            .tracking(0.8)
    }

    private func pathRow(
        _ title: String,
        path: String,
        buttonLabel: String,
        action: @escaping () -> Void
    ) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            InsetSurface {
                HStack(spacing: 8) {
                    Text(path)
                        .font(.footnote.monospaced())
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .textSelection(.enabled)
                    Spacer(minLength: 8)
                    Button(buttonLabel, action: action)
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                }
            }
        }
    }

    private func labeledField(_ title: String, text: Binding<String>) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            TextField(title, text: text)
                .textFieldStyle(.roundedBorder)
        }
    }

    private func numberField(_ title: String, value: Binding<Int>) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            TextField(title, value: value, format: .number)
                .textFieldStyle(.roundedBorder)
                .frame(maxWidth: .infinity)
        }
    }

    private func placeholderField(_ title: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(.quaternary.opacity(0.18))
                .frame(height: 28)
                .overlay(alignment: .leading) {
                    Text("Decode-only")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                        .padding(.horizontal, 10)
                }
        }
        .frame(maxWidth: .infinity)
    }
}
