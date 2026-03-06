import Foundation
import ANEInterop
import IOSurface

public struct ANEChainingProbe: Sendable {
    public let hasChainingRequestClass: Bool
    public let hasPrepareSelector: Bool
    public let hasOutputSetsClass: Bool
    public let hasOutputSetsFactory: Bool
    public let hasOutputSetEnqueueClass: Bool
    public let hasInputBuffersReadyClass: Bool
    public let hasSharedSignalEventClass: Bool
    public let builtOutputSet: Bool
    public let builtOutputSetEnqueue: Bool
    public let builtInputBuffersReady: Bool
    public let builtSharedSignalEvent: Bool
    public let builtRequest: Bool
    public let usedArrayLoopbackSymbolIndices: Bool
    public let usedRealStatsSurface: Bool
    public let requestValidated: Bool
    public let requestValid: Bool
    public let requestValidationFailed: Bool
    public let inputBuffersReadyValidationFailed: Bool
    public let calledEnqueueSets: Bool
    public let enqueueSetsSucceeded: Bool
    public let calledBuffersReady: Bool
    public let buffersReadySucceeded: Bool
    public let prepared: Bool
    public let stage: Int
}

public struct ANEKernel: ~Copyable {
    private enum CompileGate {
        static let lock = NSLock()
    }

    private let handle: OpaquePointer

    @inline(__always)
    private static func checkedSurfaceIndex(_ index: Int) throws(ANEError) -> Int32 {
        guard index >= 0 && index <= Int(Int32.max) else {
            throw .invalidSurfaceIndex(index)
        }
        return Int32(index)
    }

    @inline(__always)
    private static func mapInteropCompileError() -> ANEError {
        switch ane_interop_last_compile_error() {
        case ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS:
            return .invalidArguments("ANE interop compile contract rejected arguments")
        case ANE_INTEROP_COMPILE_ERROR_DUPLICATE_WEIGHT_PATH:
            return .invalidArguments("Duplicate weight path detected")
        case ANE_INTEROP_COMPILE_ERROR_SURFACE_ALLOCATION_FAILED:
            return .surfaceAllocationFailed
        default:
            return .compilationFailed
        }
    }

    /// Compile a MIL program with optional weight blobs into an ANE kernel.
    ///
    /// - Parameters:
    ///   - milText: MIL program text (will be UTF-8 encoded)
    ///   - weights: Array of (path, data) pairs. Paths follow "@model_path/weights/<name>.bin" convention.
    ///   - inputSizes: Byte sizes for each input IOSurface (typically 1 element for single-input kernels)
    ///   - outputSizes: Byte sizes for each output IOSurface (typically 1 element)
    ///   - checkBudget: If true, checks CompileBudget before compiling. Default true.
    ///
    /// - Throws: `ANEError.compileBudgetExhausted` if budget exceeded (and checkBudget is true).
    ///           `ANEError.compilationFailed` if ANE compilation returns NULL.
    public init(
        milText: String,
        weights: [(path: String, data: Data)],
        inputSizes: [Int],
        outputSizes: [Int],
        checkBudget: Bool = true
    ) throws(ANEError) {
        guard weights.count <= Int(Int32.max),
              inputSizes.count <= Int(Int32.max),
              outputSizes.count <= Int(Int32.max),
              inputSizes.allSatisfy({ $0 >= 0 }),
              outputSizes.allSatisfy({ $0 >= 0 }) else {
            throw .invalidArguments("Counts must fit Int32 and byte sizes must be non-negative")
        }

        guard let milData = milText.data(using: .utf8), !milData.isEmpty else {
            throw .invalidArguments("MIL text must be valid, non-empty UTF-8")
        }

        let weightPaths = weights.map(\.path)
        let weightDatas = weights.map(\.data)
        var pathPointers = [UnsafePointer<CChar>?](repeating: nil, count: weights.count)
        var dataPointers = [UnsafePointer<UInt8>?](repeating: nil, count: weights.count)
        let dataLengths = weightDatas.map(\.count)

        func compileHandle() -> OpaquePointer? {
            ane_interop_init()

            return milData.withUnsafeBytes { milRaw in
                let milBuffer = milRaw.bindMemory(to: UInt8.self)
                guard let milBase = milBuffer.baseAddress else {
                    return nil
                }

                func withWeightPathPointers<R>(_ index: Int, _ body: () -> R) -> R {
                    guard index < weightPaths.count else {
                        return body()
                    }
                    return weightPaths[index].withCString { cPath in
                        pathPointers[index] = cPath
                        return withWeightPathPointers(index + 1, body)
                    }
                }

                func withWeightDataPointers<R>(_ index: Int, _ body: () -> R) -> R {
                    guard index < weightDatas.count else {
                        return body()
                    }
                    if weightDatas[index].isEmpty {
                        dataPointers[index] = nil
                        return withWeightDataPointers(index + 1, body)
                    }
                    return weightDatas[index].withUnsafeBytes { raw in
                        dataPointers[index] = raw.bindMemory(to: UInt8.self).baseAddress
                        return withWeightDataPointers(index + 1, body)
                    }
                }

                return withWeightPathPointers(0) {
                    withWeightDataPointers(0) {
                        pathPointers.withUnsafeMutableBufferPointer { pathBuf in
                            dataPointers.withUnsafeMutableBufferPointer { dataBuf in
                                dataLengths.withUnsafeBufferPointer { lenBuf in
                                    inputSizes.withUnsafeBufferPointer { inputBuf in
                                        outputSizes.withUnsafeBufferPointer { outputBuf in
                                            ane_interop_compile(
                                                milBase,
                                                milBuffer.count,
                                                weights.isEmpty ? nil : pathBuf.baseAddress,
                                                weights.isEmpty ? nil : dataBuf.baseAddress,
                                                weights.isEmpty ? nil : lenBuf.baseAddress,
                                                Int32(weights.count),
                                                Int32(inputSizes.count),
                                                inputSizes.isEmpty ? nil : inputBuf.baseAddress,
                                                Int32(outputSizes.count),
                                                outputSizes.isEmpty ? nil : outputBuf.baseAddress
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let rawHandle: OpaquePointer?
        if checkBudget {
            CompileGate.lock.lock()
            defer { CompileGate.lock.unlock() }
            if CompileBudget.isExhausted {
                throw .compileBudgetExhausted
            }
            rawHandle = compileHandle()
        } else {
            rawHandle = compileHandle()
        }

        guard let rawHandle else {
            throw Self.mapInteropCompileError()
        }
        self.handle = rawHandle
    }

    /// Convenience: single-input, single-output kernel (most common case).
    public init(
        milText: String,
        weights: [(path: String, data: Data)],
        inputBytes: Int,
        outputBytes: Int,
        checkBudget: Bool = true
    ) throws(ANEError) {
        try self.init(
            milText: milText,
            weights: weights,
            inputSizes: [inputBytes],
            outputSizes: [outputBytes],
            checkBudget: checkBudget
        )
    }

    /// Run the compiled kernel on ANE.
    /// Input data must be written to inputSurface before calling.
    /// Output data is available on outputSurface after return.
    public func eval() throws(ANEError) {
        guard ane_interop_eval(handle) else {
            throw .evaluationFailed
        }
    }

    /// Last reported on-chip execution time from `_ANEPerformanceStats` (nanoseconds).
    ///
    /// Returns `0` when perf stats are disabled (default) or unsupported on this host.
    public func lastHWExecutionTimeNS() -> UInt64 {
        ane_interop_last_hw_execution_time_ns(handle)
    }

    public func chainingProbe(
        useRealStatsSurface: Bool = false,
        skipPrepare: Bool = false,
        validateRequest: Bool = true,
        useScalarLoopbackSymbolIndices: Bool = false,
        callEnqueueSets: Bool = false,
        callBuffersReady: Bool = false,
        requestProcedureIndex: UInt32 = 0,
        requestTransactionHandle: UInt64 = 0,
        requestFWEnqueueDelay: UInt64 = 0,
        requestMemoryPoolId: UInt64 = 0,
        enqueueProcedureIndex: UInt32 = 0,
        enqueueSetIndex: UInt32 = 0,
        enqueueSignalValue: UInt64 = 0,
        enqueueSignalNotRequired: Bool = true,
        enqueueOpenLoop: Bool = false,
        readyProcedureIndex: UInt32 = 0,
        readyExecutionDelay: UInt64 = 0,
        useSharedSignalEvent: Bool = false,
        sharedSignalEventValue: UInt64 = 1,
        sharedSignalEventSymbolIndex: UInt32 = 0,
        sharedSignalEventType: Int64 = 0
    ) -> ANEChainingProbe {
        var options = ANEInteropChainingProbeOptions(
            useRealStatsSurface: useRealStatsSurface,
            skipPrepare: skipPrepare,
            validateRequest: validateRequest,
            useScalarLoopbackSymbolIndices: useScalarLoopbackSymbolIndices,
            callEnqueueSets: callEnqueueSets,
            callBuffersReady: callBuffersReady,
            requestProcedureIndex: requestProcedureIndex,
            requestTransactionHandle: requestTransactionHandle,
            requestFWEnqueueDelay: requestFWEnqueueDelay,
            requestMemoryPoolId: requestMemoryPoolId,
            enqueueProcedureIndex: enqueueProcedureIndex,
            enqueueSetIndex: enqueueSetIndex,
            enqueueSignalValue: enqueueSignalValue,
            enqueueSignalNotRequired: enqueueSignalNotRequired,
            enqueueOpenLoop: enqueueOpenLoop,
            readyProcedureIndex: readyProcedureIndex,
            readyExecutionDelay: readyExecutionDelay,
            useSharedSignalEvent: useSharedSignalEvent,
            sharedSignalEventValue: sharedSignalEventValue,
            sharedSignalEventSymbolIndex: sharedSignalEventSymbolIndex,
            sharedSignalEventType: sharedSignalEventType
        )
        var raw = ANEInteropChainingProbeResult()
        ane_interop_probe_chaining_with_options(handle, &options, &raw)
        return ANEChainingProbe(
            hasChainingRequestClass: raw.hasChainingRequestClass,
            hasPrepareSelector: raw.hasPrepareSelector,
            hasOutputSetsClass: raw.hasOutputSetsClass,
            hasOutputSetsFactory: raw.hasOutputSetsFactory,
            hasOutputSetEnqueueClass: raw.hasOutputSetEnqueueClass,
            hasInputBuffersReadyClass: raw.hasInputBuffersReadyClass,
            hasSharedSignalEventClass: raw.hasSharedSignalEventClass,
            builtOutputSet: raw.builtOutputSet,
            builtOutputSetEnqueue: raw.builtOutputSetEnqueue,
            builtInputBuffersReady: raw.builtInputBuffersReady,
            builtSharedSignalEvent: raw.builtSharedSignalEvent,
            builtRequest: raw.builtRequest,
            usedArrayLoopbackSymbolIndices: raw.usedArrayLoopbackSymbolIndices,
            usedRealStatsSurface: raw.usedRealStatsSurface,
            requestValidated: raw.requestValidated,
            requestValid: raw.requestValid,
            requestValidationFailed: raw.requestValidationFailed,
            inputBuffersReadyValidationFailed: raw.inputBuffersReadyValidationFailed,
            calledEnqueueSets: raw.calledEnqueueSets,
            enqueueSetsSucceeded: raw.enqueueSetsSucceeded,
            calledBuffersReady: raw.calledBuffersReady,
            buffersReadySucceeded: raw.buffersReadySucceeded,
            prepared: raw.prepared,
            stage: Int(raw.stage)
        )
    }

    /// Access input IOSurface (retained; safe to hold independently of kernel lifetime).
    public func inputSurface(at index: Int) throws(ANEError) -> IOSurfaceRef {
        let checkedIndex = try Self.checkedSurfaceIndex(index)
        guard let surface = ane_interop_copy_input(handle, checkedIndex) else {
            throw .inputSurfaceUnavailable(index)
        }
        return surface
    }

    /// Access output IOSurface (retained; safe to hold independently of kernel lifetime).
    public func outputSurface(at index: Int) throws(ANEError) -> IOSurfaceRef {
        let checkedIndex = try Self.checkedSurfaceIndex(index)
        guard let surface = ane_interop_copy_output(handle, checkedIndex) else {
            throw .outputSurfaceUnavailable(index)
        }
        return surface
    }

    deinit {
        ane_interop_free(handle)
    }
}
