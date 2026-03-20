import Foundation
import ANETypes
import ANEBuilder
import ANEGraphIR

/// Decode-time FFN core probe that exposes intermediate SwiGLU stages.
///
/// Input:
/// - `normalized`: `[1, dim, 1, laneSpatial]` fp16 normalized token activations
///
/// Output:
/// - One selected stage from:
///   - `gateLinear`: `W1(normalized)` `[1, hiddenDim, 1, laneSpatial]`
///   - `upLinear`: `W3(normalized)` `[1, hiddenDim, 1, laneSpatial]`
///   - `siluGate`: `SiLU(gateLinear)` `[1, hiddenDim, 1, laneSpatial]`
///   - `gated`: `SiLU(gateLinear) * upLinear` `[1, hiddenDim, 1, laneSpatial]`
///   - `down`: `W2(gated)` `[1, dim, 1, laneSpatial]`
public struct DecodeFFNStagesGenerator: MILProgramGenerator {
    public enum Stage: Sendable {
        case gateLinear
        case upLinear
        case siluGate
        case gated
        case down

        fileprivate var outputName: String {
            switch self {
            case .gateLinear: "gate_linear"
            case .upLinear: "up_linear"
            case .siluGate: "silu_gate"
            case .gated: "gated"
            case .down: "down"
            }
        }
    }

    public let dim: Int
    public let hiddenDim: Int
    public let laneSpatial: Int
    public let stage: Stage

    public init(
        dim: Int = ModelConfig.dim,
        hiddenDim: Int = ModelConfig.hidden,
        laneSpatial: Int = 32,
        stage: Stage = .down
    ) {
        precondition(dim > 0)
        precondition(hiddenDim > 0)
        precondition(laneSpatial > 0)
        self.dim = dim
        self.hiddenDim = hiddenDim
        self.laneSpatial = laneSpatial
        self.stage = stage
    }

    public var inputBytes: Int { dim * laneSpatial * 2 }
    public var inputByteSizes: [Int] { [inputBytes] }
    public var outputByteSizes: [Int] {
        [outputChannels * laneSpatial * 2]
    }

    private var outputChannels: Int {
        switch stage {
        case .gateLinear, .upLinear, .siluGate, .gated:
            hiddenDim
        case .down:
            dim
        }
    }

    public var milText: String {
        LegacyGraphSupport.emitGraph { graph in
            let normalized = try LegacyGraphSupport.input(
                &graph,
                name: "normalized",
                channels: dim,
                spatial: laneSpatial
            )
            let gateLinear = try graph.linear(
                "gate_linear",
                input: normalized,
                inDim: dim,
                outDim: hiddenDim,
                spatial: laneSpatial,
                weightPath: "@model_path/weights/w1.bin"
            )
            let upLinear = try graph.linear(
                "up_linear",
                input: normalized,
                inDim: dim,
                outDim: hiddenDim,
                spatial: laneSpatial,
                weightPath: "@model_path/weights/w3.bin"
            )
            let siluGate = try graph.silu("silu_gate", input: gateLinear)
            let gated = try graph.mul("gated", x: siluGate, y: upLinear)
            let down = try graph.linear(
                "down",
                input: gated,
                inDim: hiddenDim,
                outDim: dim,
                spatial: laneSpatial,
                weightPath: "@model_path/weights/w2.bin"
            )
            let selectedOutput: Int = switch stage {
            case .gateLinear: gateLinear
            case .upLinear: upLinear
            case .siluGate: siluGate
            case .gated: gated
            case .down: down
            }
            try LegacyGraphSupport.setOutputs(&graph, [(stage.outputName, selectedOutput)])
        }
    }
}
