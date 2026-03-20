import Foundation
import ANETypes
import ANEBuilder
import ANEGraphIR

/// Decode-time FFN kernel that assumes RMSNorm has already been applied.
///
/// Inputs:
/// - `normalized`: `[1, dim, 1, laneSpatial]` fp16 normalized token activations
/// - `residual`: `[1, dim, 1, laneSpatial]` fp16 residual stream to add back
///
/// Output:
/// - `residual + ffn(normalized)`: `[1, dim, 1, laneSpatial]` fp16
public struct DecodeFFNPostNormGenerator: MILProgramGenerator {
    public let dim: Int
    public let hiddenDim: Int
    public let laneSpatial: Int
    public let architecture: LayerWeightsArchitecture

    public init(
        dim: Int = ModelConfig.dim,
        hiddenDim: Int = ModelConfig.hidden,
        laneSpatial: Int = 32,
        architecture: LayerWeightsArchitecture = .rmsNormSwiGLU
    ) {
        precondition(dim > 0)
        precondition(hiddenDim > 0)
        precondition(laneSpatial > 0)
        self.dim = dim
        self.hiddenDim = hiddenDim
        self.laneSpatial = laneSpatial
        self.architecture = architecture
    }

    public var inputBytes: Int { dim * laneSpatial * 2 }

    public var inputByteSizes: [Int] {
        [
            dim * laneSpatial * 2,
            dim * laneSpatial * 2,
        ]
    }

    public var outputByteSizes: [Int] { [dim * laneSpatial * 2] }

    public var milText: String {
        LegacyGraphSupport.emitGraph { graph in
            let normalized = try LegacyGraphSupport.input(
                &graph,
                name: "normalized",
                channels: dim,
                spatial: laneSpatial
            )
            let residual = try LegacyGraphSupport.input(
                &graph,
                name: "residual",
                channels: dim,
                spatial: laneSpatial
            )
            let y: Int

            switch architecture {
            case .rmsNormSwiGLU:
                y = try graph.swigluFFN(
                    "ffn",
                    input: normalized,
                    inDim: dim,
                    hiddenDim: hiddenDim,
                    spatial: laneSpatial,
                    w1Path: "@model_path/weights/w1.bin",
                    w3Path: "@model_path/weights/w3.bin",
                    w2Path: "@model_path/weights/w2.bin"
                )
            case .gpt2:
                y = try graph.ffn(
                    "ffn",
                    input: normalized,
                    inDim: dim,
                    hiddenDim: hiddenDim,
                    spatial: laneSpatial,
                    w1Path: "@model_path/weights/w1.bin",
                    b1Path: "@model_path/weights/b1.bin",
                    w2Path: "@model_path/weights/w2.bin",
                    b2Path: "@model_path/weights/b2.bin",
                    activation: .gelu
                )
            }

            let out = try graph.add("out", x: residual, y: y)
            try LegacyGraphSupport.setOutputs(&graph, [("out", out)])
        }
    }
}
