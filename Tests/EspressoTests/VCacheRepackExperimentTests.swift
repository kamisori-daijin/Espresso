import XCTest
@testable import Espresso

final class VCacheRepackExperimentTests: XCTestCase {
    func test_vcache_repack_environment_flag_defaults_off() {
        XCTAssertFalse(DecodeRuntimeOptions.repackVOutHeadMajor(env: [:]))
        XCTAssertTrue(
            DecodeRuntimeOptions.repackVOutHeadMajor(
                env: ["ESPRESSO_REPACK_VOUT_HEAD_MAJOR": "1"]
            )
        )
    }

    func test_repack_vout_for_single_token_experiment_reorders_dim_major_head_interleaved_source() {
        let source: [Float] = [
            10, 20,
            11, 21,
            12, 22,
        ]

        let repacked = source.withUnsafeBufferPointer { buffer in
            ForwardPass.repackVOutForSingleTokenExperiment(
                source: buffer,
                kvHeads: 2,
                headDim: 3,
                laneSpatial: 1,
                sourceSpatialIndex: 0
            )
        }

        XCTAssertEqual(repacked, [10, 11, 12, 20, 21, 22])
    }

    func test_repack_vout_for_single_token_experiment_selects_requested_lane_column() {
        let source: [Float] = [
            10, 100,
            20, 200,
            11, 101,
            21, 201,
            12, 102,
            22, 202,
        ]

        let repacked = source.withUnsafeBufferPointer { buffer in
            ForwardPass.repackVOutForSingleTokenExperiment(
                source: buffer,
                kvHeads: 2,
                headDim: 3,
                laneSpatial: 2,
                sourceSpatialIndex: 1
            )
        }

        XCTAssertEqual(repacked, [100, 101, 102, 200, 201, 202])
    }

    func test_repack_vout_for_single_token_experiment_can_fill_one_cache_column_without_touching_others() {
        let source: [Float] = [
            10, 20,
            11, 21,
            12, 22,
        ]

        let repacked = source.withUnsafeBufferPointer { buffer in
            ForwardPass.repackVOutForSingleTokenExperiment(
                source: buffer,
                kvHeads: 2,
                headDim: 3,
                laneSpatial: 1,
                sourceSpatialIndex: 0
            )
        }

        var cache = [Float](repeating: -1, count: 2 * 3 * 3)
        let cacheSpatial = 3
        let tokenIndex = 1
        for channel in 0..<repacked.count {
            cache[channel * cacheSpatial + tokenIndex] = repacked[channel]
        }

        XCTAssertEqual(cache[0], -1)
        XCTAssertEqual(cache[1], 10)
        XCTAssertEqual(cache[2], -1)
        XCTAssertEqual(cache[3], -1)
        XCTAssertEqual(cache[4], 11)
        XCTAssertEqual(cache[5], -1)
        XCTAssertEqual(cache[6], -1)
        XCTAssertEqual(cache[7], 12)
        XCTAssertEqual(cache[8], -1)
        XCTAssertEqual(cache[9], -1)
        XCTAssertEqual(cache[10], 20)
        XCTAssertEqual(cache[11], -1)
        XCTAssertEqual(cache[12], -1)
        XCTAssertEqual(cache[13], 21)
        XCTAssertEqual(cache[14], -1)
        XCTAssertEqual(cache[15], -1)
        XCTAssertEqual(cache[16], 22)
        XCTAssertEqual(cache[17], -1)
    }
}
