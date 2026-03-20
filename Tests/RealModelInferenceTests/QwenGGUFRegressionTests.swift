import ANETypes
import EspressoGGUF
import Foundation
import Testing
@testable import RealModelInference

@Suite("QwenGGUFRegression", .serialized)
struct QwenGGUFRegressionTests {
    private func manualFixture(
        env: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> QwenGGUFRegressionSupport.Fixture {
        guard let modelPath = env["ESPRESSO_QWEN_MANUAL_MODEL_PATH"], !modelPath.isEmpty else {
            throw NSError(domain: "QwenGGUFRegressionTests", code: 3, userInfo: [
                NSLocalizedDescriptionKey: "ESPRESSO_QWEN_MANUAL_MODEL_PATH is required"
            ])
        }

        let tokenizerDir = env["QWEN_TOKENIZER_DIR"] ?? defaultQwenTokenizerDir
        let modelURL = URL(fileURLWithPath: modelPath)
        let tokenizerURL = URL(fileURLWithPath: tokenizerDir, isDirectory: true)
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw NSError(domain: "QwenGGUFRegressionTests", code: 4, userInfo: [
                NSLocalizedDescriptionKey: "Missing manual Qwen GGUF model at \(modelURL.path)"
            ])
        }
        guard FileManager.default.fileExists(atPath: tokenizerURL.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            throw NSError(domain: "QwenGGUFRegressionTests", code: 5, userInfo: [
                NSLocalizedDescriptionKey: "Missing tokenizer directory at \(tokenizerURL.path)"
            ])
        }

        return .init(modelURL: modelURL, tokenizerDir: tokenizerURL)
    }

    private func parseExpectedTokens(
        envKey: String,
        env: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> [TokenID] {
        guard let raw = env[envKey], !raw.isEmpty else {
            throw NSError(domain: "QwenGGUFRegressionTests", code: 6, userInfo: [
                NSLocalizedDescriptionKey: "\(envKey) is required"
            ])
        }
        let tokens = raw
            .split(separator: ",")
            .compactMap { TokenID($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
        guard !tokens.isEmpty else {
            throw NSError(domain: "QwenGGUFRegressionTests", code: 7, userInfo: [
                NSLocalizedDescriptionKey: "\(envKey) must contain at least one token"
            ])
        }
        return tokens
    }

    private func preparedModel(
        modelEnvKey: String,
        defaultModelPath: String
    ) async throws -> (QwenGGUFRegressionSupport.Fixture, GGUFModelLoader.PreparedModel) {
        let fixture = try QwenGGUFRegressionSupport.fixture(
            modelEnvKey: modelEnvKey,
            defaultModelPath: defaultModelPath
        )
        let prepared = try await QwenGGUFPreparedModelCache.shared.preparedModel(for: fixture)
        return (fixture, prepared)
    }

    private func assertFreshArtifactHelloContinuation(
        modelEnvKey: String,
        defaultModelPath: String
    ) async throws {
        let (fixture, prepared) = try await preparedModel(
            modelEnvKey: modelEnvKey,
            defaultModelPath: defaultModelPath
        )

        #expect(FileManager.default.fileExists(atPath: prepared.weightDir))
        #expect(prepared.weightDir.contains("espresso_gguf_"))

        var engine = try RealModelInferenceEngine.build(
            config: prepared.config,
            weightDir: prepared.weightDir,
            tokenizerDir: fixture.tokenizerDir.path
        )
        let result = try engine.generate(
            prompt: QwenGGUFRegressionSupport.helloPrompt,
            maxTokens: QwenGGUFRegressionSupport.helloContinuationTokens.count,
            temperature: 0
        )

        #expect(result.tokens == QwenGGUFRegressionSupport.helloContinuationTokens)
        #expect(result.tokens.first == QwenGGUFRegressionSupport.helloFirstToken)
        #expect(result.text == QwenGGUFRegressionSupport.helloContinuationText)
    }

    private func assertLatePrefixNextToken(
        modelEnvKey: String,
        defaultModelPath: String
    ) async throws {
        let (_, prepared) = try await preparedModel(
            modelEnvKey: modelEnvKey,
            defaultModelPath: defaultModelPath
        )

        let token = try RealModelInferenceEngine.generateNextTokenForTesting(
            config: prepared.config,
            weightDir: prepared.weightDir,
            promptTokens: QwenGGUFRegressionSupport.latePrefixTokens
        )

        #expect(token == QwenGGUFRegressionSupport.latePrefixExpectedToken)
    }

    private func assertExactCPULatePrefixNextToken(
        modelEnvKey: String,
        defaultModelPath: String
    ) async throws {
        let (_, prepared) = try await preparedModel(
            modelEnvKey: modelEnvKey,
            defaultModelPath: defaultModelPath
        )

        let token = try RealModelInferenceEngine.generateNextTokenExactCPUForTesting(
            config: prepared.config,
            weightDir: prepared.weightDir,
            promptTokens: QwenGGUFRegressionSupport.latePrefixTokens
        )

        #expect(!RealModelInferenceEngine.shouldRoundCPUExactDecodeIntermediatesToFP16(env: [:]))
        #expect(token == QwenGGUFRegressionSupport.latePrefixExpectedToken)
    }

    @Test(
        "Qwen 0.6B fresh artifact Hello continuation matches raw GGUF golden",
        .enabled(if: qwen060BRegressionEnabled)
    )
    func qwen060BFreshArtifactHelloContinuationMatchesGolden() async throws {
        try await assertFreshArtifactHelloContinuation(
            modelEnvKey: "QWEN_0_6B_GGUF_MODEL_PATH",
            defaultModelPath: defaultQwen060BModelPath
        )
    }

    @Test(
        "Qwen 0.6B fresh artifact late-prefix token matches raw GGUF golden",
        .enabled(if: qwen060BRegressionEnabled)
    )
    func qwen060BFreshArtifactLatePrefixTokenMatchesGolden() async throws {
        try await assertLatePrefixNextToken(
            modelEnvKey: "QWEN_0_6B_GGUF_MODEL_PATH",
            defaultModelPath: defaultQwen060BModelPath
        )
    }

    @Test(
        "Qwen 0.6B exact CPU late-prefix token keeps FP32 intermediates by default",
        .enabled(if: qwen060BRegressionEnabled)
    )
    func qwen060BExactCPULatePrefixTokenMatchesGolden() async throws {
        try await assertExactCPULatePrefixNextToken(
            modelEnvKey: "QWEN_0_6B_GGUF_MODEL_PATH",
            defaultModelPath: defaultQwen060BModelPath
        )
    }

    @Test(
        "Manual Qwen exact CPU late-prefix token matches expected",
        .enabled(if: ProcessInfo.processInfo.environment["ESPRESSO_QWEN_MANUAL_MODEL_PATH"] != nil
            && ProcessInfo.processInfo.environment["ESPRESSO_QWEN_MANUAL_EXPECTED_LATE_PREFIX_TOKEN"] != nil)
    )
    func qwenManualExactCPULatePrefixMatchesExpected() async throws {
        let env = ProcessInfo.processInfo.environment
        let fixture = try manualFixture(env: env)
        let prepared = try await QwenGGUFPreparedModelCache.shared.preparedModel(for: fixture)
        let expected = try parseExpectedTokens(
            envKey: "ESPRESSO_QWEN_MANUAL_EXPECTED_LATE_PREFIX_TOKEN",
            env: env
        )
        #expect(expected.count == 1)

        let token = try RealModelInferenceEngine.generateNextTokenExactCPUForTesting(
            config: prepared.config,
            weightDir: prepared.weightDir,
            promptTokens: QwenGGUFRegressionSupport.latePrefixTokens
        )

        #expect(token == expected[0])
    }

    @Test(
        "Manual Qwen Hello continuation matches expected tokens",
        .enabled(if: ProcessInfo.processInfo.environment["ESPRESSO_QWEN_MANUAL_MODEL_PATH"] != nil
            && ProcessInfo.processInfo.environment["ESPRESSO_QWEN_MANUAL_EXPECTED_HELLO_TOKENS"] != nil)
    )
    func qwenManualHelloContinuationMatchesExpected() async throws {
        let env = ProcessInfo.processInfo.environment
        let fixture = try manualFixture(env: env)
        let prepared = try await QwenGGUFPreparedModelCache.shared.preparedModel(for: fixture)
        let expectedTokens = try parseExpectedTokens(
            envKey: "ESPRESSO_QWEN_MANUAL_EXPECTED_HELLO_TOKENS",
            env: env
        )
        let tokens = try RealModelInferenceEngine.generateTokensExactCPUForTesting(
            config: prepared.config,
            weightDir: prepared.weightDir,
            promptTokens: QwenGGUFRegressionSupport.helloPromptTokens,
            maxTokens: expectedTokens.count
        )

        #expect(tokens == expectedTokens)
    }

}
