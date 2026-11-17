// Z-Image-Turbo sampler for the Core AI bundle Looking Glass exported on 2026-09-19 (pipeline
// "lookingglass.zimage.v1"). Apple's FlowTransformerPipeline can't run it: the export is our own, with its own input
// contract (spelled out in the bundle's metadata.json). This is a Swift port of the Python reference sampler
// (LookingGlass WORKSPACE/coreai-imagegen-eval/tools/zimage/zimage_coreai.py, `phase_gen`), which itself mirrors
// diffusers' ZImagePipeline:
//
//   Qwen3 chat template (enable_thinking = true) → TextEncoder → caption rows of hidden_states[-2]
//   → torch-identical noise → 9 flow-match Euler steps (shift 3.0, sigma_min 0) → VAE decode → 1024² image
//
// The transformer takes RoPE as precomputed cos/sin tables, so all the position bookkeeping lives here.
// Text-to-image only: the bundle has no VAE encoder, so it can't edit.

import CoreAI
import CoreAIDiffusionPipeline
import CoreGraphics
import Foundation
import Tokenizers

enum ZImage {
    static let captionLength = 512          // caption bucket == diffusers max_sequence_length
    static let latentSide = 128             // 1024 px / 8
    static let latentChannels = 16
    static let gridSide = latentSide / 2    // patch 2 → 64×64 image tokens
    static let imageTokens = gridSide * gridSide
    static let ropeAxes = [32, 48, 48]
    static let ropeTheta = 256.0
    static let defaultSteps = 9
    static let shift = 3.0

    /// Whether a bundle's metadata.json says it's this pipeline (checked before Apple's DiffusionBundle, which
    /// rejects the unknown "zimage" type).
    static func isZImageBundle(_ bundle: URL) -> Bool {
        guard let data = try? Data(contentsOf: bundle.appendingPathComponent("metadata.json")),
              let meta = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let diffusion = meta["diffusion"] as? [String: Any] else { return false }
        return diffusion["type"] as? String == "zimage"
    }

    struct Metadata {
        let vaeScale: Float
        let vaeShift: Float
        let defaultSteps: Int

        init(bundle: URL) throws {
            let data = try Data(contentsOf: bundle.appendingPathComponent("metadata.json"))
            guard let meta = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let d = meta["diffusion"] as? [String: Any],
                  let scale = d["vae_scaling_factor"] as? Double,
                  let shift = d["vae_shift_factor"] as? Double else {
                throw HelperError("\(bundle.lastPathComponent)/metadata.json is missing the VAE factors")
            }
            vaeScale = Float(scale)
            vaeShift = Float(shift)
            defaultSteps = d["default_steps"] as? Int ?? ZImage.defaultSteps
        }
    }

    /// Generate one 1024×1024 image. `onStep(step, total)` follows the transformer calls.
    static func generate(bundle: URL, prompt: String, seed: UInt32, steps requested: Int?,
                         onStep: (Int, Int) -> Void) async throws -> CGImage {
        let meta = try Metadata(bundle: bundle)
        let steps = requested ?? meta.defaultSteps

        // 1. Text → caption features. The text encoder is released before the transformer loads, so the two
        //    never sit in memory together.
        let tokenizer = try await AutoTokenizer.from(modelFolder: bundle.appendingPathComponent("tokenizer"))
        let (ids, captionRows) = try tokenize(prompt, tokenizer: tokenizer)
        var captionFeatures: [Float]
        let hidden: Int
        do {
            let encoder = try await CoreAIFunction(bundle.appendingPathComponent("TextEncoder.aimodel"))
            let mask = (0..<captionLength).map { $0 < captionRows ? Int32(1) : 0 }
            let embeds = try await encoder.run(["input_ids": .int32(ids), "attention_mask": .int32(mask)])
            hidden = embeds.count / captionLength
            captionFeatures = Array(embeds[0..<(captionRows * hidden)])   // valid rows only, like diffusers
        }
        captionFeatures += [Float](repeating: 0, count: (captionLength - captionRows) * hidden)

        // 2. Fixed transformer inputs.
        let capLen = captionRows + (32 - captionRows % 32) % 32           // the model pads the caption to ×32
        let padMask = (0..<captionLength).map { $0 >= captionRows }
        let valid = (0..<captionLength).map { $0 < capLen }
        let (capCos, capSin) = ropeTables(positions: (0..<captionLength).map { (Double(1 + $0), 0, 0) })
        let imagePositions = (0..<imageTokens).map { (Double(capLen + 1), Double($0 / gridSide), Double($0 % gridSide)) }
        let (imgCos, imgSin) = ropeTables(positions: imagePositions)

        // 3. Noise, identical to torch.randn(1, 16, 128, 128, generator=torch.Generator().manual_seed(seed)).
        var rng = TorchRandomSource(seed: seed)
        var latents = rng.normalArray([1, latentChannels, latentSide, latentSide])

        // 4. Denoise.
        let sigmas = schedule(steps: steps)
        let transformer = try await CoreAIFunction(bundle.appendingPathComponent("Transformer.aimodel"))
        let fixed: [String: CoreAIFunction.Input] = [
            "cap_feats": .float(captionFeatures), "cap_pad_mask": .bool(padMask), "cap_valid": .bool(valid),
            "img_cos": .float(imgCos), "img_sin": .float(imgSin), "cap_cos": .float(capCos), "cap_sin": .float(capSin),
        ]
        try await transformer.prepare(fixed)
        for i in 0..<steps {
            let sigma = sigmas[i], next = sigmas[i + 1]
            // The last step lands on sigma 0 → dt 0, so diffusers' final transformer call can't change the
            // latents. Skipping it gives the same image, ~8 s sooner.
            if next == sigma { onStep(i + 1, steps); continue }
            let t = Float((1000 - Float(sigma * 1000)) / 1000)               // the model's t: 1 − timestep/1000
            let out = try await transformer.run(["latents": .float(latents), "t": .float([t])])
            let dt = Float(next - sigma)
            for j in 0..<latents.count { latents[j] += -out[j] * dt }       // noise_pred = −model_out
            guard latents.allSatisfy({ $0.isFinite }) else {
                throw HelperError("the latents went non-finite at step \(i + 1)")
            }
            onStep(i + 1, steps)
        }

        // 5. Decode.
        let decoder = try await CoreAIFunction(bundle.appendingPathComponent("VAEDecoder.aimodel"))
        let z = latents.map { $0 / meta.vaeScale + meta.vaeShift }
        let pixels = try await decoder.run(["latent": .float(z)])
        return try DiffusionUtilities.pixelsToCGImage(pixels, height: latentSide * 8, width: latentSide * 8)
    }

    /// Qwen3 chat template with enable_thinking = true (Z-Image's choice; FLUX.2 uses false), right-padded with
    /// <|endoftext|> to 512. Returns the ids and the count of real tokens.
    static func tokenize(_ prompt: String, tokenizer: any Tokenizer) throws -> ([Int32], Int) {
        var ids = try tokenizer.applyChatTemplate(
            messages: [["role": "user", "content": prompt]], chatTemplate: nil,
            addGenerationPrompt: true, truncation: true, maxLength: captionLength, tools: nil,
            additionalContext: ["enable_thinking": true])
        if ids.count > captionLength { ids = Array(ids.prefix(captionLength)) }
        let real = ids.count
        let pad = tokenizer.convertTokenToId("<|endoftext|>") ?? 151643
        ids += [Int](repeating: pad, count: captionLength - real)
        return (ids.map(Int32.init), real)
    }

    /// RoPE cos/sin, [N, 64] row-major: per axis, outer(position, theta^(−arange(0, d, 2)/d)) in float64, then float32.
    static func ropeTables(positions: [(Double, Double, Double)]) -> ([Float], [Float]) {
        let freqs = ropeAxes.map { d in stride(from: 0, to: d, by: 2).map { 1.0 / pow(ropeTheta, Double($0) / Double(d)) } }
        let width = freqs.reduce(0) { $0 + $1.count }                       // 64
        var cos = [Float](repeating: 0, count: positions.count * width)
        var sin = cos
        for (n, p) in positions.enumerated() {
            var k = n * width
            for (axis, f) in freqs.enumerated() {
                let pos = axis == 0 ? p.0 : axis == 1 ? p.1 : p.2
                for inv in f {
                    let angle = Float(pos * inv)                            // float64 product → float32, as numpy
                    cos[k] = Foundation.cos(angle); sin[k] = Foundation.sin(angle); k += 1
                }
            }
        }
        return (cos, sin)
    }

    /// diffusers FlowMatchEulerDiscreteScheduler with shift 3.0, use_dynamic_shifting false and the pipeline's
    /// sigma_min = 0: sigmas = shift(linspace(1, 0, steps)) in float64, stored float32, then a terminal 0.
    static func schedule(steps: Int) -> [Float] {
        let raw: [Double] = steps == 1 ? [1.0] : (0..<steps).map { 1.0 - Double($0) / Double(steps - 1) }
        return raw.map { Float(shift * $0 / (1 + (shift - 1) * $0)) } + [0]
    }
}

/// One Core AI model function, with inputs built from its own descriptors (so each tensor gets the scalar type
/// the export declared: fp16, fp32, int32 or bool) and constant inputs packed once.
final class CoreAIFunction {
    enum Input {
        case float([Float])
        case int32([Int32])
        case bool([Bool])
    }

    private let model: AIModel
    private let function: InferenceFunction
    private let name: String
    private var prepared: [String: NDArray] = [:]

    init(_ url: URL) async throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw HelperError("the bundle is missing \(url.lastPathComponent)")
        }
        model = try await AIModel(contentsOf: url, options: SpecializationOptions(preferredComputeUnitKind: .gpu))
        guard let fn = try model.loadFunction(named: "main") else {
            throw HelperError("\(url.lastPathComponent) has no main function")
        }
        function = fn
        name = url.lastPathComponent
    }

    /// Pack inputs that stay the same across calls.
    func prepare(_ inputs: [String: Input]) throws {
        for (key, value) in inputs { prepared[key] = try pack(key, value) }
    }

    /// Run with `inputs` plus anything prepared; returns the first output as Float.
    func run(_ inputs: [String: Input]) async throws -> [Float] {
        var all = prepared
        for (key, value) in inputs { all[key] = try pack(key, value) }
        var outputs = try await function.run(inputs: all)
        guard let outName = function.descriptor.outputNames.first,
              let array = outputs.remove(outName)?.ndArray else {
            throw HelperError("\(name) returned no output")
        }
        return try Self.floats(array, name: name)
    }

    private func pack(_ key: String, _ value: Input) throws -> NDArray {
        guard case .ndArray(let desc) = function.descriptor.inputDescriptor(of: key) else {
            throw HelperError("\(name) has no input named \(key)")
        }
        var array = NDArray(descriptor: desc)
        let expected = desc.shape.reduce(1, *)
        func check(_ count: Int) throws {
            guard count == expected else {
                throw HelperError("\(name).\(key) expects \(expected) values (shape \(desc.shape)), got \(count)")
            }
        }
        switch (value, desc.scalarType) {
        case (.float(let v), .float16):
            try check(v.count); Self.fill(&array, Float16.self, v.count) { Float16(v[$0]) }
        case (.float(let v), .float32):
            try check(v.count); Self.fill(&array, Float.self, v.count) { v[$0] }
        case (.int32(let v), .int32):
            try check(v.count); Self.fill(&array, Int32.self, v.count) { v[$0] }
        case (.bool(let v), .bool):
            try check(v.count); Self.fill(&array, Bool.self, v.count) { v[$0] }
        default:
            throw HelperError("\(name).\(key): unexpected input type \(desc.scalarType)")
        }
        return array
    }

    /// Stride-aware fill (Core AI may pad rows for GPU alignment).
    private static func fill<T: BitwiseCopyable>(_ array: inout NDArray, _ type: T.Type, _ count: Int,
                                                 _ value: (Int) -> T) {
        array.mutableView(as: T.self).withUnsafeMutablePointer { ptr, shape, strides in
            let rank = shape.count
            var dense = true, expect = 1
            for d in stride(from: rank - 1, through: 0, by: -1) {
                if shape[d] > 1 && strides[d] != expect { dense = false }
                expect *= shape[d]
            }
            if dense { for i in 0..<count { ptr[i] = value(i) }; return }
            var index = [Int](repeating: 0, count: rank)
            for i in 0..<count {
                var offset = 0
                for d in 0..<rank { offset += index[d] * strides[d] }
                ptr[offset] = value(i)
                var d = rank - 1
                while d >= 0 { index[d] += 1; if index[d] < shape[d] { break }; index[d] = 0; d -= 1 }
            }
        }
    }

    private static func floats(_ array: NDArray, name: String) throws -> [Float] {
        switch array.scalarType {
        case .float16: return read(array, Float16.self) { Float($0) }
        case .float32: return read(array, Float.self) { $0 }
        default: throw HelperError("\(name): unexpected output type \(array.scalarType)")
        }
    }

    private static func read<T: BitwiseCopyable>(_ array: NDArray, _ type: T.Type, _ convert: (T) -> Float) -> [Float] {
        array.view(as: T.self).withUnsafePointer { ptr, shape, strides in
            let rank = shape.count
            let total = (0..<rank).reduce(1) { $0 * shape[$1] }
            var out = [Float](repeating: 0, count: total)
            var index = [Int](repeating: 0, count: rank)
            for i in 0..<total {
                var offset = 0
                for d in 0..<rank { offset += index[d] * strides[d] }
                out[i] = convert(ptr[offset])
                var d = rank - 1
                while d >= 0 { index[d] += 1; if index[d] < shape[d] { break }; index[d] = 0; d -= 1 }
            }
            return out
        }
    }
}
