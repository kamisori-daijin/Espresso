import Testing
@testable import ANEGraphIR

@Test func elementCount() throws {
    let shape = try ANEShape(batch: 1, channels: 768, height: 1, spatial: 256)
    #expect(shape.elementCount == 196_608)
}

@Test func byteSizeFP16() throws {
    let shape = try ANEShape(channels: 768, spatial: 256)
    #expect(shape.byteSize(for: .fp16) == 393_216) // 196608 * 2
}

@Test func byteSizeFP32() throws {
    let shape = try ANEShape(channels: 768, spatial: 256)
    #expect(shape.byteSize(for: .fp32) == 786_432) // 196608 * 4
}

@Test func dimensionsArray() throws {
    let shape = try ANEShape(batch: 1, channels: 768, height: 1, spatial: 256)
    #expect(shape.dimensions == [1, 768, 1, 256])
}

@Test func defaultBatchAndHeight() throws {
    let shape = try ANEShape(channels: 512, spatial: 64)
    #expect(shape.batch == 1)
    #expect(shape.height == 1)
    #expect(shape.channels == 512)
    #expect(shape.spatial == 64)
}

@Test func meetsMinimumIOSurfaceSize() throws {
    // 768 * 32 * 2 = 49,152 bytes — exactly at minimum
    let atMinimum = try ANEShape(channels: 768, spatial: 32)
    #expect(atMinimum.meetsMinimumIOSurfaceSize(for: .fp16))

    // 768 * 16 * 2 = 24,576 bytes — below minimum
    let belowMinimum = try ANEShape(channels: 768, spatial: 16)
    #expect(!belowMinimum.meetsMinimumIOSurfaceSize(for: .fp16))

    // 4 * 4 * 2 = 32 bytes — way below
    let tiny = try ANEShape(channels: 4, spatial: 4)
    #expect(!tiny.meetsMinimumIOSurfaceSize(for: .fp16))
}

@Test func exceedsSRAMBudget() throws {
    // 32768 * 1024 * 2 = 67,108,864 bytes = 64MB > 32MB
    let large = try ANEShape(channels: 32768, spatial: 1024)
    #expect(large.exceedsSRAMBudget(for: .fp16))

    // 768 * 256 * 2 = 393,216 bytes << 32MB
    let normal = try ANEShape(channels: 768, spatial: 256)
    #expect(!normal.exceedsSRAMBudget(for: .fp16))
}

@Test func equalityCheck() throws {
    let a = try ANEShape(channels: 768, spatial: 256)
    let b = try ANEShape(channels: 768, spatial: 256)
    let c = try ANEShape(channels: 768, spatial: 128)
    #expect(a == b)
    #expect(a != c)
}

@Test func rejectsNonPositiveDimensions() {
    do {
        _ = try ANEShape(channels: 0, spatial: 64)
        #expect(Bool(false), "Expected ANEShape init to reject zero channels")
    } catch let error as ANEGraphValidationError {
        #expect(error == .nonPositiveDimension(name: "channels", value: 0))
    } catch {
        #expect(Bool(false), "Unexpected error: \(error)")
    }
}
