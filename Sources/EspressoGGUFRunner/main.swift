import Foundation
import ANETypes
import EspressoGGUF

@main
struct GGUFRunner {
    static func main() async {
        do {
            try await run()
        } catch {
            fputs("Error: \(error)\n", stderr)
            Foundation.exit(1)
        }
    }

    private static func run() async throws {
        let args = Array(CommandLine.arguments.dropFirst())
        guard !args.isEmpty else {
            printUsage()
            Foundation.exit(1)
        }

        if args[0] == "verify-qwen" {
            let request = try parseVerifyRequest(args)
            let result = try await RunGGUF.verifyQwen(request: request)
            fputs("Verified weight dir: \(result.prepared.weightDir)\n", stderr)
            fputs("Prepare time: \(String(format: "%.0f", result.prepareMs))ms\n", stderr)
            fputs("Build time: \(String(format: "%.0f", result.buildMs))ms\n", stderr)
            fputs("Cold-start text: \(result.coldStartText)\n", stderr)
            fputs("Late-prefix token: \(result.latePrefixToken)\n", stderr)
            fputs("Hello tokens: \(result.helloTokens)\n", stderr)
            print(result.helloText)
            if request.keepWeightDir || ProcessInfo.processInfo.environment["ESPRESSO_GGUF_KEEP_WEIGHT_DIR"] == "1" {
                fputs("Keeping weight dir: \(result.prepared.weightDir)\n", stderr)
            } else {
                GGUFModelLoader.cleanup(weightDir: result.prepared.weightDir)
            }
            return
        }

        guard args.count >= 3 else {
            printUsage()
            Foundation.exit(1)
        }

        let ggufPath = args[0]
        let tokenizerDir = args[1]
        let prompt = args[2]
        let maxTokens = args.count > 3 ? Int(args[3]) ?? 32 : 32

        try await RunGGUF.main(
            ggufPath: ggufPath,
            tokenizerDir: tokenizerDir,
            prompt: prompt,
            maxTokens: maxTokens
        )
    }

    private static func parseVerifyRequest(_ args: [String]) throws -> RunGGUF.QwenVerificationRequest {
        guard args.count >= 3 else {
            throw NSError(domain: "GGUFRunner", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Usage: EspressoGGUFRunner verify-qwen <gguf-path> <tokenizer-dir> [options]"
            ])
        }

        let ggufPath = args[1]
        let tokenizerDir = args[2]
        var freshPrepare = false
        var keepWeightDir = false
        var exactFloat32SidecarMode: GGUFModelLoader.PrepareOptions.ExactFloat32SidecarMode = .full
        var selectedExactFloat32Sidecars = Set<String>()
        var helloPrompt = "Hello"
        var latePrefixPrompt = "Hello Answer, I'm sorry for the"
        var latePrefixTokens: [TokenID] = [9707, 21806, 11, 358, 2776, 14589, 369, 279]
        var latePrefixExpectedToken: TokenID = 3681
        var helloExpectedTokens: [TokenID] = [21806, 11, 358, 2776, 14589, 369, 279, 3681]
        var coldStartExpectedText = "Hello Answer"

        var index = 3
        while index < args.count {
            switch args[index] {
            case "--fresh":
                freshPrepare = true
            case "--keep-weight-dir":
                keepWeightDir = true
            case "--hello-prompt":
                index += 1
                helloPrompt = try requiredValue(args, index: index, flag: "--hello-prompt")
            case "--cold-start-text":
                index += 1
                coldStartExpectedText = try requiredValue(args, index: index, flag: "--cold-start-text")
            case "--late-prefix-prompt":
                index += 1
                latePrefixPrompt = try requiredValue(args, index: index, flag: "--late-prefix-prompt")
            case "--late-prefix-tokens":
                index += 1
                latePrefixTokens = try parseTokenCSV(
                    requiredValue(args, index: index, flag: "--late-prefix-tokens")
                )
            case "--late-prefix-expected":
                index += 1
                latePrefixExpectedToken = try parseToken(
                    requiredValue(args, index: index, flag: "--late-prefix-expected"),
                    flag: "--late-prefix-expected"
                )
            case "--hello-tokens":
                index += 1
                helloExpectedTokens = try parseTokenCSV(
                    requiredValue(args, index: index, flag: "--hello-tokens")
                )
            case "--sidecars":
                index += 1
                exactFloat32SidecarMode = try parseSidecarMode(
                    requiredValue(args, index: index, flag: "--sidecars")
                )
            case "--selected-sidecars":
                index += 1
                selectedExactFloat32Sidecars = try parseTensorNameCSV(
                    requiredValue(args, index: index, flag: "--selected-sidecars")
                )
            default:
                throw NSError(domain: "GGUFRunner", code: 2, userInfo: [
                    NSLocalizedDescriptionKey: "Unknown verify-qwen argument: \(args[index])"
                ])
            }
            index += 1
        }

        return RunGGUF.QwenVerificationRequest(
            ggufPath: ggufPath,
            tokenizerDir: tokenizerDir,
            freshPrepare: freshPrepare,
            exactFloat32SidecarMode: exactFloat32SidecarMode,
            selectedExactFloat32Sidecars: selectedExactFloat32Sidecars,
            keepWeightDir: keepWeightDir,
            helloPrompt: helloPrompt,
            latePrefixPrompt: latePrefixPrompt,
            latePrefixTokens: latePrefixTokens,
            latePrefixExpectedToken: latePrefixExpectedToken,
            helloExpectedTokens: helloExpectedTokens,
            coldStartExpectedText: coldStartExpectedText
        )
    }

    private static func requiredValue(_ args: [String], index: Int, flag: String) throws -> String {
        guard index < args.count else {
            throw NSError(domain: "GGUFRunner", code: 3, userInfo: [
                NSLocalizedDescriptionKey: "Missing value for \(flag)"
            ])
        }
        return args[index]
    }

    private static func parseSidecarMode(
        _ rawValue: String
    ) throws -> GGUFModelLoader.PrepareOptions.ExactFloat32SidecarMode {
        switch rawValue.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "automatic", "auto":
            return .automatic
        case "none", "off":
            return .none
        case "essential":
            return .essential
        case "selected":
            return .selected([])
        case "full":
            return .full
        default:
            throw NSError(domain: "GGUFRunner", code: 4, userInfo: [
                NSLocalizedDescriptionKey: "Unsupported sidecar mode '\(rawValue)'; expected automatic|none|essential|selected|full"
            ])
        }
    }

    private static func parseTensorNameCSV(_ rawValue: String) throws -> Set<String> {
        let values = rawValue
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !values.isEmpty else {
            throw NSError(domain: "GGUFRunner", code: 6, userInfo: [
                NSLocalizedDescriptionKey: "Expected at least one tensor name for --selected-sidecars"
            ])
        }
        return Set(values)
    }

    private static func parseTokenCSV(_ rawValue: String) throws -> [TokenID] {
        try rawValue
            .split(separator: ",")
            .map { try parseToken(String($0), flag: "token-list") }
    }

    private static func parseToken(_ rawValue: String, flag: String) throws -> TokenID {
        let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let value = Int(trimmed), let token = TokenID(exactly: value) else {
            throw NSError(domain: "GGUFRunner", code: 5, userInfo: [
                NSLocalizedDescriptionKey: "Invalid token value '\(rawValue)' for \(flag)"
            ])
        }
        return token
    }

    private static func printUsage() {
        fputs(
            """
            Usage:
              EspressoGGUFRunner <gguf-path> <tokenizer-dir> <prompt> [max-tokens]
              EspressoGGUFRunner verify-qwen <gguf-path> <tokenizer-dir> [--fresh] [--keep-weight-dir] [--sidecars automatic|none|essential|selected|full] [--selected-sidecars CSV] [--hello-prompt TEXT] [--cold-start-text TEXT] [--late-prefix-prompt TEXT] [--late-prefix-tokens CSV] [--late-prefix-expected TOKEN] [--hello-tokens CSV]

            """,
            stderr
        )
    }
}
