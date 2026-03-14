/// 4D tensor shape for ANE: [batch, channels, height, spatial].
///
/// ANE always operates on 4D tensors. For typical LLM workloads:
/// - batch = 1 (always)
/// - channels = model dimension, head count, vocab size, etc.
/// - height = 1 for decode, head_dim for multi-head attention reshapes
/// - spatial = sequence length or lane width
public struct ANEShape: Sendable, Equatable {
    public var batch: Int
    public var channels: Int
    public var height: Int
    public var spatial: Int

    public init(batch: Int = 1, channels: Int, height: Int = 1, spatial: Int) throws {
        try Self.validateDimension(batch, name: "batch")
        try Self.validateDimension(channels, name: "channels")
        try Self.validateDimension(height, name: "height")
        try Self.validateDimension(spatial, name: "spatial")
        guard Self.fitsSupportedByteRange(
            batch: batch,
            channels: channels,
            height: height,
            spatial: spatial
        ) else {
            throw ANEGraphValidationError.byteSizeOverflow(
                dimensions: [batch, channels, height, spatial]
            )
        }
        self.batch = batch
        self.channels = channels
        self.height = height
        self.spatial = spatial
    }

    /// Total number of elements in the tensor.
    public var elementCount: Int {
        batch * channels * height * spatial
    }

    /// Total byte size for a given data type.
    public func byteSize(for dtype: ANEDType) -> Int {
        elementCount * dtype.byteWidth
    }

    /// Shape as a 4-element array [batch, channels, height, spatial].
    public var dimensions: [Int] {
        [batch, channels, height, spatial]
    }

    /// Whether this shape meets the ANE minimum IOSurface size (49,152 bytes / ~48KB).
    /// Tensors smaller than this fail ANE eval with status 0x1d.
    public func meetsMinimumIOSurfaceSize(for dtype: ANEDType) -> Bool {
        byteSize(for: dtype) >= 49_152
    }

    /// Whether the byte size exceeds ANE's 32MB on-chip SRAM budget.
    /// Exceeding this causes ~30% throughput drop due to DRAM spill.
    public func exceedsSRAMBudget(for dtype: ANEDType) -> Bool {
        byteSize(for: dtype) > 32 * 1024 * 1024
    }

    private static func validateDimension(_ value: Int, name: String) throws {
        guard value > 0 else {
            throw ANEGraphValidationError.nonPositiveDimension(name: name, value: value)
        }
    }

    private static func fitsSupportedByteRange(
        batch: Int,
        channels: Int,
        height: Int,
        spatial: Int
    ) -> Bool {
        let maxByteWidth = ANEDType.fp32.byteWidth
        let dims = [batch, channels, height, spatial, maxByteWidth]
        var runningProduct = 1

        for value in dims {
            let (nextProduct, overflow) = runningProduct.multipliedReportingOverflow(by: value)
            guard !overflow else { return false }
            runningProduct = nextProduct
        }

        return true
    }
}
