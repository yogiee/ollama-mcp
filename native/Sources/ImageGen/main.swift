// ollamamcp-imagegen — OllamaMCP's on-device image generator (macOS 27+).
//
// A small command-line helper the MCP server runs once per image. It runs a FLUX.2 [klein] or Sana Sprint bundle
// through Apple's Core AI diffusion pipeline (github.com/apple/coreai-models, CoreAIDiffusionPipeline) and exits,
// so the ~7 GB it needs is returned to the system the moment the image is written: nothing stays resident between
// requests. Loading from Core AI's specialization cache is ~1 s once warm, so a process per image costs little.
//
// A separate process because the pipeline is Swift-only and the server is Python. Started from Looking Glass's
// lg-imagegen (2026-10-13), which this replaces once Looking Glass uses OllamaMCP's local_image.
// Core AI keeps one specialization cache per process name, so this helper's name must stay stable.
//
// Output is one JSON object per line on stdout, so the server can follow along:
//   {"event":"loading"}
//   {"event":"step","step":2,"total":4}
//   {"event":"done","path":"/…/image.png","seconds":23.9,"width":1024,"height":1024}
//   {"event":"error","message":"…"}            (and a non-zero exit)
//
// Two maintenance modes, each printing one JSON line:
//   --info                       the GPU working-set budget and physical memory
//   --clear-cache --model <dir>  drop Core AI's specialized copy of a bundle (~the bundle's size, and never
//                                reclaimed on its own) — run BEFORE deleting the bundle

import CoreAI
import CoreAIDiffusionPipeline
import CoreGraphics
import Foundation
import ImageIO
import Metal
import UniformTypeIdentifiers

struct Options {
    var model = ""
    var prompt = ""
    var output = ""
    var seed: UInt32 = 0
    var steps: Int?
    var inputImage: String?
    /// Edits see the input image through this token grid. `.full` keeps it 1:1 with the output; `.half`
    /// (the coreai-models README example) samples it at a quarter of the tokens, so every edit redraws
    /// the detail it can't see: softer, drifting framing, compounding over chained edits. Full measured
    /// 2026-10-13: ~50–65 s and 7.25 GB peak vs ~39 s and 6.92 GB, crisp after two chained edits.
    var referenceGrid: ReferenceGrid = .full
    var half = false
    var info = false
    var clearCache = false

    static let usage = """
        usage: ollamamcp-imagegen --model <bundle> --prompt <text> --output <file.png> [--seed N] [--steps N]
                                  [--input-image <file>] [--reference-grid full|half|quarter] [--half]
               ollamamcp-imagegen --info
               ollamamcp-imagegen --clear-cache --model <bundle>
        """

    static func parse(_ args: [String]) throws -> Options {
        var o = Options()
        var it = args.makeIterator()
        func value(_ flag: String) throws -> String {
            guard let v = it.next() else { throw HelperError("\(flag) needs a value") }
            return v
        }
        while let arg = it.next() {
            switch arg {
            case "--model": o.model = try value(arg)
            case "--prompt": o.prompt = try value(arg)
            case "--output": o.output = try value(arg)
            case "--seed":
                guard let n = UInt32(try value(arg)) else { throw HelperError("--seed must be a whole number") }
                o.seed = n
            case "--steps":
                guard let n = Int(try value(arg)), (1...50).contains(n) else { throw HelperError("--steps must be 1…50") }
                o.steps = n
            case "--input-image": o.inputImage = try value(arg)
            case "--reference-grid":
                let raw = try value(arg)
                guard let g = ReferenceGrid(rawValue: raw) else { throw HelperError("--reference-grid: full, half or quarter") }
                o.referenceGrid = g
            case "--half": o.half = true
            case "--info": o.info = true
            case "--clear-cache": o.clearCache = true
            case "-h", "--help": print(usage); exit(0)
            default: throw HelperError("unknown option \(arg)\n\(usage)")
            }
        }
        if o.info { return o }
        if o.clearCache {
            guard !o.model.isEmpty else { throw HelperError("--clear-cache needs --model") }
            return o
        }
        guard !o.model.isEmpty, !o.prompt.isEmpty, !o.output.isEmpty else { throw HelperError(usage) }
        return o
    }
}

struct HelperError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

func emit(_ object: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]),
          let line = String(data: data, encoding: .utf8) else { return }
    print(line)
    fflush(stdout)
}

func loadImage(_ path: String) throws -> CGImage {
    guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw HelperError("couldn't read the image at \(path)")
    }
    return image
}

func savePNG(_ image: CGImage, to url: URL) throws {
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
        throw HelperError("couldn't create \(url.path)")
    }
    CGImageDestinationAddImage(dest, image, nil)
    guard CGImageDestinationFinalize(dest) else { throw HelperError("couldn't write \(url.path)") }
}

func run(_ o: Options) async throws {
    let bundleURL = URL(fileURLWithPath: o.model)
    let diffusion = try DiffusionBundle(at: bundleURL).config
    guard diffusion.type == .flux2 || diffusion.type == .sanaSprint else {
        throw HelperError("\(bundleURL.lastPathComponent) isn't a FLUX.2 or Sana Sprint bundle")
    }
    let starting = try o.inputImage.map(loadImage)
    let decode: DecodeResolution = o.half ? .half : .full

    emit(["event": "loading"])
    let start = ContinuousClock.now
    let pipeline = try await FlowTransformerPipeline(from: bundleURL, mode: decode)
    if starting != nil, !pipeline.supportsImageToImage {
        throw HelperError("this bundle has no image encoder, so it can't edit images")
    }
    let config = PipelineConfiguration(
        prompt: o.prompt,
        seed: o.seed,
        stepCount: o.steps ?? diffusion.defaultSteps ?? DiffusionDefaults.Runner.steps,
        guidanceScale: diffusion.defaultGuidanceScale ?? DiffusionDefaults.Runner.guidanceScale,
        schedulerType: .discreteFlow,
        startingImage: starting,
        referenceGrid: o.referenceGrid,
        guidanceMode: .distilled,
        encoderScaleFactor: diffusion.encoderScaleFactor ?? DiffusionDefaults.Image.scaleFactor,
        decoderScaleFactor: diffusion.decoderScaleFactor ?? DiffusionDefaults.Image.scaleFactor,
        decoderShiftFactor: diffusion.decoderShiftFactor ?? DiffusionDefaults.Image.decoderShiftFactor,
        decodeResolution: decode
    )
    let result = try await pipeline.generateImages(configuration: config) { progress in
        emit(["event": "step", "step": progress.step, "total": progress.totalSteps])
        return true
    }
    guard let image = result.images.first else { throw HelperError("the pipeline returned no image") }
    let out = URL(fileURLWithPath: o.output)
    try savePNG(image, to: out)
    let seconds = (ContinuousClock.now - start).components
    emit(["event": "done", "path": out.path, "width": image.width, "height": image.height,
          "seconds": Double(seconds.seconds) + Double(seconds.attoseconds) / 1e18])
}

func info() {
    let device = MTLCreateSystemDefaultDevice()
    emit(["event": "info",
          "gpu_budget_bytes": device.map { Int($0.recommendedMaxWorkingSetSize) } ?? 0,
          "physical_memory_bytes": Int(ProcessInfo.processInfo.physicalMemory)])
}

/// Core AI keys its specialized copy by each `.aimodel` asset, so clear every one in the bundle.
func clearCache(_ model: String) throws {
    let root = URL(fileURLWithPath: model)
    let assets = (try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil))
        .filter { $0.pathExtension == "aimodel" }
    for asset in assets {
        try AIModelCache.default.deleteEntries(for: asset)
    }
    emit(["event": "cleared", "assets": assets.map(\.lastPathComponent).sorted()])
}

do {
    let options = try Options.parse(Array(CommandLine.arguments.dropFirst()))
    if options.info {
        info()
    } else if options.clearCache {
        try clearCache(options.model)
    } else {
        try await run(options)
    }
} catch {
    emit(["event": "error", "message": String(describing: error)])
    exit(1)
}
