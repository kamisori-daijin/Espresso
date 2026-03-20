import Foundation
import ANETypes
import ModelSupport
import RealModelInference

/// Standalone entry point for testing GGUF → ANE inference.
public enum RunGGUF {
    public struct BuildResult: ~Copyable {
        public let engine: RealModelInferenceEngine
        public let buildMs: Double
    }

    public struct QwenVerificationRequest: Sendable {
        public let ggufPath: String
        public let tokenizerDir: String
        public let freshPrepare: Bool
        public let exactFloat32SidecarMode: GGUFModelLoader.PrepareOptions.ExactFloat32SidecarMode
        public let selectedExactFloat32Sidecars: Set<String>
        public let keepWeightDir: Bool
        public let helloPrompt: String
        public let latePrefixPrompt: String
        public let latePrefixTokens: [TokenID]
        public let latePrefixExpectedToken: TokenID
        public let helloExpectedTokens: [TokenID]
        public let coldStartExpectedText: String

        public init(
            ggufPath: String,
            tokenizerDir: String,
            freshPrepare: Bool = false,
            exactFloat32SidecarMode: GGUFModelLoader.PrepareOptions.ExactFloat32SidecarMode = .full,
            selectedExactFloat32Sidecars: Set<String> = [],
            keepWeightDir: Bool = false,
            helloPrompt: String = "Hello",
            latePrefixPrompt: String = "Hello Answer, I'm sorry for the",
            latePrefixTokens: [TokenID] = [9707, 21806, 11, 358, 2776, 14589, 369, 279],
            latePrefixExpectedToken: TokenID = 3681,
            helloExpectedTokens: [TokenID] = [21806, 11, 358, 2776, 14589, 369, 279, 3681],
            coldStartExpectedText: String = "Hello Answer"
        ) {
            self.ggufPath = ggufPath
            self.tokenizerDir = tokenizerDir
            self.freshPrepare = freshPrepare
            self.exactFloat32SidecarMode = exactFloat32SidecarMode
            self.selectedExactFloat32Sidecars = selectedExactFloat32Sidecars
            self.keepWeightDir = keepWeightDir
            self.helloPrompt = helloPrompt
            self.latePrefixPrompt = latePrefixPrompt
            self.latePrefixTokens = latePrefixTokens
            self.latePrefixExpectedToken = latePrefixExpectedToken
            self.helloExpectedTokens = helloExpectedTokens
            self.coldStartExpectedText = coldStartExpectedText
        }
    }

    public struct QwenVerificationResult: Sendable {
        public let prepared: GGUFModelLoader.PreparedModel
        public let prepareMs: Double
        public let buildMs: Double
        public let coldStartText: String
        public let coldStartTokens: [TokenID]
        public let latePrefixPromptTokens: [TokenID]
        public let latePrefixToken: TokenID
        public let helloTokens: [TokenID]
        public let helloText: String
    }

    public static func main(
        ggufPath: String,
        tokenizerDir: String,
        prompt: String,
        maxTokens: Int
    ) async throws {
        let prepareOptions = GGUFModelLoader.PrepareOptions.environment()
        let (prepared, prepareMs) = try await prepareModel(
            ggufPath: ggufPath,
            options: prepareOptions
        )
        fputs(
            "Prepared: \(prepared.config.name), \(prepared.config.nLayer) layers, d=\(prepared.config.dModel), vocab=\(prepared.config.vocab), \(prepared.tensorCount) tensors in \(String(format: "%.0f", prepareMs))ms\n",
            stderr
        )
        fputs("Architecture: \(prepared.config.architecture == .gpt2 ? "gpt2" : "llama")\n", stderr)
        fputs("Weight dir: \(prepared.weightDir)\n", stderr)
        fputs("Building engine...\n", stderr)

        let buildResult = try buildEngine(
            prepared: prepared,
            tokenizerDir: tokenizerDir
        )
        var mutableEngine = buildResult.engine
        fputs("Engine built in \(String(format: "%.0f", buildResult.buildMs))ms\n", stderr)

        fputs("Generating from prompt: \"\(prompt)\"\n", stderr)
        let result = try mutableEngine.generate(
            prompt: prompt,
            maxTokens: maxTokens,
            temperature: 0.0
        )

        print(result.text)
        fputs("\n--- Stats ---\n", stderr)
        fputs("Tokens generated: \(result.tokens.count)\n", stderr)
        fputs("Tokens/sec: \(String(format: "%.2f", result.tokensPerSecond))\n", stderr)
        fputs("Compile time: \(String(format: "%.0f", result.compileTimeMs))ms\n", stderr)
        fputs("First token latency: \(String(format: "%.2f", result.firstTokenLatencyMs))ms\n", stderr)

        if ProcessInfo.processInfo.environment["ESPRESSO_GGUF_KEEP_WEIGHT_DIR"] == "1" {
            fputs("Keeping weight dir: \(prepared.weightDir)\n", stderr)
        } else {
            GGUFModelLoader.cleanup(weightDir: prepared.weightDir)
        }
    }

    public static func verifyQwen(
        request: QwenVerificationRequest
    ) async throws -> QwenVerificationResult {
        let verificationStart = DispatchTime.now().uptimeNanoseconds
        var prepareOptions = GGUFModelLoader.PrepareOptions.environment()
        prepareOptions.artifactCacheMode = request.freshPrepare ? .disabled : prepareOptions.artifactCacheMode
        prepareOptions.exactFloat32SidecarMode = switch request.exactFloat32SidecarMode {
        case .selected:
            .selected(request.selectedExactFloat32Sidecars)
        default:
            request.exactFloat32SidecarMode
        }

        let (prepared, prepareMs) = try await prepareModel(
            ggufPath: request.ggufPath,
            options: prepareOptions
        )
        fputs("Prepared weight dir: \(prepared.weightDir)\n", stderr)
        fputs("Preparing tokenizer/engine...\n", stderr)
        let buildResult = try buildEngine(
            prepared: prepared,
            tokenizerDir: request.tokenizerDir
        )
        var mutableEngine = buildResult.engine
        fputs("Running cold-start check...\n", stderr)

        let coldStart = try mutableEngine.generate(
            prompt: request.helloPrompt,
            maxTokens: 1,
            temperature: 0
        )
        guard coldStart.text == request.coldStartExpectedText else {
            throw NSError(
                domain: "RunGGUF",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Cold-start mismatch: expected '\(request.coldStartExpectedText)', got '\(coldStart.text)'"]
            )
        }

        fputs("Running late-prefix check...\n", stderr)
        let latePrefix = try mutableEngine.generate(
            prompt: request.latePrefixPrompt,
            maxTokens: 1,
            temperature: 0
        )
        guard latePrefix.promptTokens == request.latePrefixTokens else {
            throw NSError(
                domain: "RunGGUF",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Late-prefix tokenization mismatch: expected \(request.latePrefixTokens), got \(latePrefix.promptTokens)"]
            )
        }
        guard let latePrefixToken = latePrefix.tokens.first else {
            throw NSError(
                domain: "RunGGUF",
                code: 3,
                userInfo: [NSLocalizedDescriptionKey: "Late-prefix check did not emit a token"]
            )
        }
        guard latePrefixToken == request.latePrefixExpectedToken else {
            throw NSError(
                domain: "RunGGUF",
                code: 4,
                userInfo: [NSLocalizedDescriptionKey: "Late-prefix token mismatch: expected \(request.latePrefixExpectedToken), got \(latePrefixToken)"]
            )
        }

        fputs("Running Hello continuation check...\n", stderr)
        let hello = try mutableEngine.generate(
            prompt: request.helloPrompt,
            maxTokens: request.helloExpectedTokens.count,
            temperature: 0
        )
        guard hello.tokens == request.helloExpectedTokens else {
            throw NSError(
                domain: "RunGGUF",
                code: 5,
                userInfo: [NSLocalizedDescriptionKey: "Hello continuation mismatch: expected \(request.helloExpectedTokens), got \(hello.tokens)"]
            )
        }

        let verificationMs = Double(DispatchTime.now().uptimeNanoseconds - verificationStart) / 1_000_000
        fputs("Verification total time: \(String(format: "%.0f", verificationMs))ms\n", stderr)

        return QwenVerificationResult(
            prepared: prepared,
            prepareMs: prepareMs,
            buildMs: buildResult.buildMs,
            coldStartText: coldStart.text,
            coldStartTokens: coldStart.tokens,
            latePrefixPromptTokens: latePrefix.promptTokens,
            latePrefixToken: latePrefixToken,
            helloTokens: hello.tokens,
            helloText: hello.text
        )
    }

    public static func prepareModel(
        ggufPath: String,
        options: GGUFModelLoader.PrepareOptions = .environment()
    ) async throws -> (GGUFModelLoader.PreparedModel, Double) {
        let url = URL(fileURLWithPath: ggufPath)
        fputs("Loading GGUF: \(ggufPath)\n", stderr)
        let prepareStart = DispatchTime.now().uptimeNanoseconds
        let prepared = try await GGUFModelLoader.prepare(
            ggufURL: url,
            options: options
        )
        let prepareMs = Double(DispatchTime.now().uptimeNanoseconds - prepareStart) / 1_000_000
        return (prepared, prepareMs)
    }

    public static func buildEngine(
        prepared: GGUFModelLoader.PreparedModel,
        tokenizerDir: String
    ) throws -> BuildResult {
        let buildStart = DispatchTime.now().uptimeNanoseconds
        let engine = try RealModelInferenceEngine.build(
            config: prepared.config,
            weightDir: prepared.weightDir,
            tokenizerDir: tokenizerDir
        )
        let buildMs = Double(DispatchTime.now().uptimeNanoseconds - buildStart) / 1_000_000
        return BuildResult(engine: engine, buildMs: buildMs)
    }
}
