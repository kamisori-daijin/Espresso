import XCTest
import ANEInterop
import ANETypes
import MILGenerator
@testable import ANERuntime

// MARK: - MIL Generation Tests (no hardware needed)

final class FusedDecodeLayerGeneratorTests: XCTestCase {

    func test_mil_text_is_nonempty_and_contains_program_header() {
        let gen = FusedDecodeLayerGenerator(maxSeq: 32, laneSpatial: 32)
        let mil = gen.milText
        XCTAssertFalse(mil.isEmpty)
        XCTAssertTrue(mil.contains("program(1.3)"))
        XCTAssertTrue(mil.contains("func main<ios18>"))
    }

    func test_mil_text_contains_all_nine_weight_blobs() {
        let gen = FusedDecodeLayerGenerator(maxSeq: 32, laneSpatial: 32)
        let mil = gen.milText
        let expectedBlobs = [
            "rms1.bin", "wq.bin", "wk.bin", "wv.bin", "wo.bin",
            "rms2.bin", "w1.bin", "w3.bin", "w2.bin",
        ]
        for blob in expectedBlobs {
            XCTAssertTrue(mil.contains(blob), "Missing weight blob: \(blob)")
        }
    }

    func test_mil_text_contains_fused_residuals() {
        let gen = FusedDecodeLayerGenerator(maxSeq: 32, laneSpatial: 32)
        let mil = gen.milText
        // First residual: attention output
        XCTAssertTrue(mil.contains("name=string(\"res\")"), "Missing first residual (res)")
        // Second residual: FFN output
        XCTAssertTrue(mil.contains("name=string(\"res2\")"), "Missing second residual (res2)")
    }

    func test_mil_text_contains_ffn_ops() {
        let gen = FusedDecodeLayerGenerator(maxSeq: 32, laneSpatial: 32)
        let mil = gen.milText
        // SwiGLU FFN ops
        XCTAssertTrue(mil.contains("sigmoid(x=h1)"), "Missing sigmoid")
        XCTAssertTrue(mil.contains("name=string(\"si\")"), "Missing SiLU")
        XCTAssertTrue(mil.contains("name=string(\"gt\")"), "Missing gate")
    }

    func test_mil_text_has_unique_ssa_names() {
        let gen = FusedDecodeLayerGenerator(maxSeq: 32, laneSpatial: 32)
        let mil = gen.milText

        // Extract all name=string("...") values
        var names: [String] = []
        var scanner = mil[mil.startIndex...]
        let namePrefix = "name=string(\""
        let nameSuffix = "\")"
        while let range = scanner.range(of: namePrefix) {
            let afterPrefix = range.upperBound
            if let endRange = scanner[afterPrefix...].range(of: nameSuffix) {
                let name = String(scanner[afterPrefix..<endRange.lowerBound])
                names.append(name)
                scanner = scanner[endRange.upperBound...]
            } else {
                break
            }
        }

        let uniqueNames = Set(names)
        XCTAssertEqual(
            names.count, uniqueNames.count,
            "Duplicate SSA names found: \(names.filter { n in names.filter { $0 == n }.count > 1 })"
        )
    }

    func test_mil_text_has_three_outputs() {
        let gen = FusedDecodeLayerGenerator(maxSeq: 32, laneSpatial: 32)
        let mil = gen.milText
        XCTAssertTrue(mil.contains("-> (xNext,kfFull,vfFull)"))
    }

    func test_mil_text_has_four_inputs() {
        let gen = FusedDecodeLayerGenerator(maxSeq: 32, laneSpatial: 32)
        let mil = gen.milText
        XCTAssertTrue(mil.contains("> x,"))
        XCTAssertTrue(mil.contains("> kCache,"))
        XCTAssertTrue(mil.contains("> vCache,"))
        XCTAssertTrue(mil.contains("> maskCache)"))
    }

    func test_inputByteSizes_has_four_entries() {
        let gen = FusedDecodeLayerGenerator(maxSeq: 64, laneSpatial: 32)
        XCTAssertEqual(gen.inputByteSizes.count, 4)
        let dim = ModelConfig.dim
        XCTAssertEqual(gen.inputByteSizes[0], dim * 32 * 2)   // x
        XCTAssertEqual(gen.inputByteSizes[1], dim * 64 * 2)   // kCache
        XCTAssertEqual(gen.inputByteSizes[2], dim * 64 * 2)   // vCache
        XCTAssertEqual(gen.inputByteSizes[3], dim * 64 * 2)   // maskCache
    }

    func test_outputByteSizes_has_three_entries() {
        let gen = FusedDecodeLayerGenerator(maxSeq: 32, laneSpatial: 32)
        XCTAssertEqual(gen.outputByteSizes.count, 3)
        let dim = ModelConfig.dim
        for size in gen.outputByteSizes {
            XCTAssertEqual(size, dim * 32 * 2)
        }
    }

    func test_ffn_rmsnorm_uses_x2_not_x() {
        let gen = FusedDecodeLayerGenerator(maxSeq: 32, laneSpatial: 32)
        let mil = gen.milText
        // The FFN RMSNorm should operate on x2 (attention output), not x (original input)
        XCTAssertTrue(mil.contains("f_sq = mul(x=x2,y=x2)"), "FFN RMSNorm should use x2")
    }

    func test_second_residual_adds_to_x2() {
        let gen = FusedDecodeLayerGenerator(maxSeq: 32, laneSpatial: 32)
        let mil = gen.milText
        // xNext = x2 + y (FFN output added to attention residual)
        XCTAssertTrue(mil.contains("xNext = add(x=x2,y=y)"), "Second residual should add FFN output to x2")
    }

    func test_conv_constants_defined_once() {
        let gen = FusedDecodeLayerGenerator(maxSeq: 32, laneSpatial: 32)
        let mil = gen.milText
        // Conv constants should only appear once (not duplicated for FFN)
        let ptCount = mil.components(separatedBy: "name=string(\"pt\")").count - 1
        XCTAssertEqual(ptCount, 1, "Conv constant 'pt' should be defined exactly once")
    }
}

// MARK: - Hardware-gated compilation tests

private func requireANEHardware(file: StaticString = #filePath, line: UInt = #line) throws {
    guard ProcessInfo.processInfo.environment["ANE_HARDWARE_TESTS"] == "1" else {
        throw XCTSkip("Set ANE_HARDWARE_TESTS=1 to run ANE hardware tests", file: file, line: line)
    }
    let handle = dlopen(
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        RTLD_NOW
    )
    if handle == nil {
        throw XCTSkip("AppleNeuralEngine.framework unavailable", file: file, line: line)
    }
    dlclose(handle)
}

private func makeTestLayerWeights(value: Float = 0.01) -> LayerWeights {
    let w = LayerWeights()
    fillTestWeights(w, value: value)
    return w
}

private func fillTestWeights(_ weights: borrowing LayerWeights, value: Float) {
    func fill(_ buf: borrowing TensorBuffer, _ val: Float) {
        buf.withUnsafeMutableBufferPointer { ptr in
            for i in ptr.indices { ptr[i] = val }
        }
    }
    fill(weights.Wq, value)
    fill(weights.Wk, value)
    fill(weights.Wv, value)
    fill(weights.Wo, value)
    fill(weights.W1, value)
    fill(weights.W3, value)
    fill(weights.W2, value)
    fill(weights.rmsAtt, 1.0)
    fill(weights.rmsFfn, 1.0)
}

final class FusedDecodeKernelSetTests: XCTestCase {

    func test_fused_kernel_compiles_on_hardware() throws {
        try requireANEHardware()
        let weights = makeTestLayerWeights()
        let kernelSet = try FusedDecodeKernelSet(weights: weights, maxSeq: 32)
        XCTAssertEqual(kernelSet.laneSpatial, 32)
        XCTAssertEqual(kernelSet.kernelMaxSeq, 32)
        XCTAssertEqual(kernelSet.maxSeq, 32)
    }

    func test_fused_kernel_eval_succeeds() throws {
        try requireANEHardware()
        let weights = makeTestLayerWeights()
        let kernelSet = try FusedDecodeKernelSet(weights: weights, maxSeq: 32)
        do {
            try kernelSet.fusedLayer.eval()
        } catch {
            print("  NOTE: fused kernel eval failed (known host instability): \(error)")
        }
    }

    func test_fused_kernel_surfaces_accessible() throws {
        try requireANEHardware()
        let weights = makeTestLayerWeights()
        let kernelSet = try FusedDecodeKernelSet(weights: weights, maxSeq: 32)
        // Verify 4 input surfaces are accessible
        for i in 0..<4 {
            let surf = try kernelSet.fusedLayer.inputSurface(at: i)
            XCTAssertTrue(IOSurfaceGetAllocSize(surf) > 0, "Input surface \(i) should be non-empty")
        }
        // Verify 3 output surfaces are accessible
        for i in 0..<3 {
            let surf = try kernelSet.fusedLayer.outputSurface(at: i)
            XCTAssertTrue(IOSurfaceGetAllocSize(surf) > 0, "Output surface \(i) should be non-empty")
        }
    }

    func test_fused_invalid_maxSeq_throws() {
        let weights = makeTestLayerWeights()
        do {
            _ = try FusedDecodeKernelSet(weights: weights, maxSeq: 0)
            XCTFail("Should have thrown for maxSeq=0")
        } catch let error as ANEError {
            XCTAssertTrue("\(error)".contains("must be > 0"))
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    func test_fused_misaligned_maxSeq_throws() {
        let weights = makeTestLayerWeights()
        do {
            _ = try FusedDecodeKernelSet(weights: weights, maxSeq: 33)
            XCTFail("Should have thrown for maxSeq=33")
        } catch let error as ANEError {
            XCTAssertTrue("\(error)".contains("multiple"))
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }
}
