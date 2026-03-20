import XCTest
@testable import CPUOps

final class RMSNormPerHeadDecodeTests: XCTestCase {
    func test_applyPerHeadSingleTokenInPlace_matches_manual_reference() {
        var values: [Float] = [
            1.0, -2.0, 0.5, 1.5,
            2.0, 0.0, -1.0, 3.0,
        ]
        let weights: [Float] = [1.0, 0.5, 1.5, 2.0]
        let epsilon: Float = 1e-6

        values.withUnsafeMutableBufferPointer { buffer in
            weights.withUnsafeBufferPointer { norm in
                RMSNorm.applyPerHeadSingleTokenInPlace(
                    values: buffer.baseAddress!,
                    headCount: 2,
                    headDim: 4,
                    weights: norm.baseAddress!,
                    epsilon: epsilon
                )
            }
        }

        let expected = [
            manualPerHeadRMSNorm(
                values: [1.0, -2.0, 0.5, 1.5],
                weights: weights,
                epsilon: epsilon
            ),
            manualPerHeadRMSNorm(
                values: [2.0, 0.0, -1.0, 3.0],
                weights: weights,
                epsilon: epsilon
            ),
        ].flatMap { $0 }

        XCTAssertEqual(values.count, expected.count)
        for (actual, reference) in zip(values, expected) {
            XCTAssertEqual(actual, reference, accuracy: 1e-6)
        }
    }

    private func manualPerHeadRMSNorm(values: [Float], weights: [Float], epsilon: Float) -> [Float] {
        let sumSquares = values.reduce(into: Float(0)) { partialResult, value in
            partialResult += value * value
        }
        let invRms = 1.0 / sqrtf(sumSquares / Float(values.count) + epsilon)
        return zip(values, weights).map { value, weight in
            value * invRms * weight
        }
    }
}
