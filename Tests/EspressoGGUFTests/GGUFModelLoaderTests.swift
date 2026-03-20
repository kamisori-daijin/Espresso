import Testing
import Foundation
@testable import EspressoGGUF
import ModelSupport

@Suite("GGUFModelLoader")
struct GGUFModelLoaderTests {

    @Test("PreparedModel has expected fields")
    func preparedModelFields() {
        let config = MultiModelConfig(
            name: "test", nLayer: 12, nHead: 12, nKVHead: 12,
            dModel: 768, headDim: 64, hiddenDim: 3072,
            vocab: 50257, maxSeq: 1024, normEps: 1e-5,
            architecture: .gpt2
        )
        let prepared = GGUFModelLoader.PreparedModel(
            config: config, weightDir: "/tmp/test", tensorCount: 42
        )
        #expect(prepared.config.architecture == .gpt2)
        #expect(prepared.weightDir == "/tmp/test")
        #expect(prepared.tensorCount == 42)
    }

    @Test("Metadata uses runtime architecture label")
    func metadataUsesRuntimeArchitectureLabel() {
        let config = MultiModelConfig(
            name: "qwen3",
            nLayer: 28,
            nHead: 16,
            nKVHead: 8,
            dModel: 1024,
            headDim: 128,
            hiddenDim: 3072,
            vocab: 151936,
            maxSeq: 40960,
            normEps: 1e-6,
            ropeTheta: 1_000_000,
            eosToken: 151645,
            architecture: .llama
        )

        let metadata = GGUFModelLoader.metadataDictionary(for: config)
        #expect(metadata["name"] as? String == "qwen3")
        #expect(metadata["architecture"] as? String == "llama")
        #expect(metadata["ropeTheta"] as? Float == 1_000_000)
        #expect(metadata["eosToken"] as? Int == 151645)
    }

    @Test("Tied embedding models materialize lm head")
    func tiedEmbeddingModelsMaterializeLMHead() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("espresso_gguf_tied_head_\(UUID().uuidString)")
        let embeddingDir = dir.appendingPathComponent("embeddings", isDirectory: true)
        try FileManager.default.createDirectory(at: embeddingDir, withIntermediateDirectories: true)
        let embedding = embeddingDir.appendingPathComponent("token.bin")
        let expected = Data([0x01, 0x02, 0x03])
        try expected.write(to: embedding)
        let exactEmbedding = embeddingDir.appendingPathComponent("token.float32.bin")
        let exactExpected = Data([0x04, 0x05, 0x06, 0x07])
        try exactExpected.write(to: exactEmbedding)

        defer { GGUFModelLoader.cleanup(weightDir: dir.path) }

        try GGUFModelLoader.materializeTiedLMHeadIfNeeded(
            tensorNames: ["token_embd.weight"],
            outputDirectory: dir
        )

        let lmHead = dir.appendingPathComponent("lm_head.bin")
        let exactLMHead = dir.appendingPathComponent("lm_head.float32.bin")
        #expect(FileManager.default.fileExists(atPath: lmHead.path))
        #expect(FileManager.default.fileExists(atPath: exactLMHead.path))
        #expect(try Data(contentsOf: lmHead) == expected)
        #expect(try Data(contentsOf: exactLMHead) == exactExpected)
    }

    @Test("Tied embedding models can synthesize exact lm head without embedding sidecar")
    func tiedEmbeddingModelsMaterializeExactLMHeadFromProvidedValues() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("espresso_gguf_tied_exact_head_\(UUID().uuidString)")
        let embeddingDir = dir.appendingPathComponent("embeddings", isDirectory: true)
        try FileManager.default.createDirectory(at: embeddingDir, withIntermediateDirectories: true)
        let embedding = embeddingDir.appendingPathComponent("token.bin")
        try Data([0x01, 0x02, 0x03]).write(to: embedding)

        defer { GGUFModelLoader.cleanup(weightDir: dir.path) }

        let exactValues: [Float] = [1.0003, -2.0007, 3.14159]
        try GGUFModelLoader.materializeTiedLMHeadIfNeeded(
            tensorNames: ["token_embd.weight"],
            outputDirectory: dir,
            exactTiedLMHeadValues: exactValues
        )

        let exactLMHead = dir.appendingPathComponent("lm_head.float32.bin")
        #expect(FileManager.default.fileExists(atPath: exactLMHead.path))

        let data = try Data(contentsOf: exactLMHead)
        let scalarSize = MemoryLayout<UInt32>.stride
        let recovered = data.withUnsafeBytes { raw in
            stride(from: 0, to: data.count, by: scalarSize).map { index in
                let bits = raw.loadUnaligned(fromByteOffset: index, as: UInt32.self)
                return Float(bitPattern: UInt32(littleEndian: bits))
            }
        }
        #expect(recovered == exactValues)
    }

    @Test("Prepare options parse cache and sidecar controls from the environment")
    func prepareOptionsParseEnvironmentControls() {
        let options = GGUFModelLoader.PrepareOptions.environment([
            GGUFModelLoader.prepareCacheEnvKey: "0",
            GGUFModelLoader.prepareCacheRootEnvKey: "/tmp/espresso-cache",
            GGUFModelLoader.sidecarPolicyEnvKey: "selected",
            GGUFModelLoader.selectedSidecarsEnvKey: "blk.27.attn_q.weight, blk.27.attn_k.weight",
        ])

        #expect(options.artifactCacheMode == .disabled)
        #expect(options.cacheRoot?.path == "/tmp/espresso-cache")
        #expect(
            options.exactFloat32SidecarMode
                == .selected(["blk.27.attn_q.weight", "blk.27.attn_k.weight"])
        )
    }

    @Test("Artifact cache key changes when sidecar mode changes")
    func artifactCacheKeyTracksPrepareOptions() throws {
        let ggufURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("espresso_gguf_cache_key_\(UUID().uuidString).gguf")
        try Data([0x51, 0x47, 0x55, 0x46]).write(to: ggufURL)
        defer { try? FileManager.default.removeItem(at: ggufURL) }

        let essentialKey = try GGUFModelLoader.artifactCacheKey(
            ggufURL: ggufURL,
            options: .init(exactFloat32SidecarMode: .essential)
        )
        let repeatedEssentialKey = try GGUFModelLoader.artifactCacheKey(
            ggufURL: ggufURL,
            options: .init(exactFloat32SidecarMode: .essential)
        )
        let selectedKey = try GGUFModelLoader.artifactCacheKey(
            ggufURL: ggufURL,
            options: .init(exactFloat32SidecarMode: .selected(["blk.27.attn_q.weight"]))
        )

        #expect(essentialKey == repeatedEssentialKey)
        #expect(essentialKey != selectedKey)
    }

    @Test("Cleanup removes directory")
    func cleanupRemovesDir() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("espresso_gguf_test_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let marker = dir.appendingPathComponent("test.bin")
        try Data([0x01]).write(to: marker)

        #expect(FileManager.default.fileExists(atPath: dir.path))
        GGUFModelLoader.cleanup(weightDir: dir.path)
        #expect(!FileManager.default.fileExists(atPath: dir.path))
    }

    @Test("Cleanup preserves managed cached artifacts")
    func cleanupPreservesManagedCacheDir() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("espresso_gguf_cached_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let marker = dir.appendingPathComponent(".espresso-gguf-cache-artifact")
        try Data().write(to: marker)

        GGUFModelLoader.cleanup(weightDir: dir.path)
        #expect(FileManager.default.fileExists(atPath: dir.path))

        try? FileManager.default.removeItem(at: dir)
    }

    @Test("Full GGUF prepare pipeline",
          .enabled(if: ProcessInfo.processInfo.environment["GGUF_MODEL_PATH"] != nil))
    func fullPipelineWithRealModel() async throws {
        let modelPath = ProcessInfo.processInfo.environment["GGUF_MODEL_PATH"]!
        let prepared = try await GGUFModelLoader.prepare(
            ggufURL: URL(fileURLWithPath: modelPath)
        )
        #expect(prepared.tensorCount > 0)
        #expect(prepared.config.nLayer > 0)
        #expect(prepared.config.dModel > 0)
        #expect(FileManager.default.fileExists(atPath: prepared.weightDir))
        GGUFModelLoader.cleanup(weightDir: prepared.weightDir)
    }
}
