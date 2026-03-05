import XCTest
@testable import Espresso

final class DecodeStateTests: XCTestCase {
    func test_decode_state_rejects_non_positive_max_seq() {
        XCTAssertThrowsError(try DecodeState(maxSeq: 0))
        XCTAssertThrowsError(try DecodeState(maxSeq: -4))
    }

    func test_decode_state_step_visibility_and_commit_order() throws {
        var state = try DecodeState(maxSeq: 4)
        XCTAssertEqual(state.step, 0)
        XCTAssertEqual(state.visibleTokenCount, 0)

        let t0 = try state.beginTokenStep()
        XCTAssertEqual(t0, 0)
        XCTAssertEqual(state.visibleTokenCount, 0)
        try state.commitTokenStep(expectedIndex: t0)
        XCTAssertEqual(state.step, 1)
        XCTAssertEqual(state.visibleTokenCount, 1)

        let t1 = try state.beginTokenStep()
        XCTAssertEqual(t1, 1)
        XCTAssertEqual(state.visibleTokenCount, 1)
        try state.commitTokenStep(expectedIndex: t1)
        XCTAssertEqual(state.step, 2)
        XCTAssertEqual(state.visibleTokenCount, 2)
    }

    func test_decode_state_detects_overflow_and_mismatched_commit() throws {
        var overflow = try DecodeState(maxSeq: 1)
        let t0 = try overflow.beginTokenStep()
        try overflow.commitTokenStep(expectedIndex: t0)
        XCTAssertEqual(overflow.step, 1)
        XCTAssertThrowsError(try overflow.beginTokenStep())

        var mismatch = try DecodeState(maxSeq: 2)
        XCTAssertThrowsError(try mismatch.commitTokenStep(expectedIndex: 1))
    }
}
