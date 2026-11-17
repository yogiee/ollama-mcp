// ollamamcp-videogen — OllamaMCP's on-device text-to-video (macOS 27+), EXPERIMENTAL.
//
// Runs a Wan 2.1 T2V bundle (exported with apple/coreai-models' `coreai.diffusion.export
// Wan-AI/Wan2.1-T2V-1.3B-Diffusers`) through Apple's WanPipeline (CoreAIVideoDiffusionPipeline) and writes an H.264
// mp4. One process per clip, like ollamamcp-imagegen; a separate executable so the image helper stays lean and each
// keeps its own Core AI specialization cache (keyed by process name, so this name must stay stable).
//
// Output is one JSON object per line on stdout:
//   {"event":"loading"}
//   {"event":"phase","phase":"encoding|denoising|decoding|assembling"}
//   {"event":"step","step":3,"total":12}
//   {"event":"done","path":"…","seconds":95.2,"frames":33,"fps":16,"width":832,"height":480}
//   {"event":"error","message":"…"}   (and a non-zero exit)
//   --clear-cache --model <dir>  drop Core AI's specialized copy of the bundle (run BEFORE deleting it)

import AVFoundation
import CoreAI
import CoreAIVideoDiffusionPipeline
import CoreGraphics
import Foundation

struct ToolError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

func emit(_ object: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]),
          let line = String(data: data, encoding: .utf8) else { return }
    print(line)
    fflush(stdout)
}

/// Draw one frame into a pooled buffer and append it. Its own function: the buffer is noncopyable, and declaring it
/// inside the frame loop doesn't compile.
func appendFrame(_ frame: CGImage, index i: Int, fps: Int, width: Int, height: Int,
                 to receiver: AVAssetWriterInput.PixelBufferReceiver) async throws {
    guard let pool = receiver.pixelBufferPool else { throw ToolError("no pixel buffer pool") }
    guard var buffer = try? pool.makeMutablePixelBuffer() else { throw ToolError("couldn't allocate a frame buffer") }
    try buffer.accessUnsafeMutableRawPlaneBytes { planes in
        guard let plane = planes.first,
              let context = CGContext(data: plane.bytes.baseAddress, width: width, height: height,
                                      bitsPerComponent: 8, bytesPerRow: plane.properties.bytesPerRow,
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue) else {
            throw ToolError("couldn't draw frame \(i)")
        }
        context.draw(frame, in: CGRect(x: 0, y: 0, width: width, height: height))
    }
    // Awaiting the append is the back-pressure: it suspends until the encoder has room.
    try await receiver.append(CVReadOnlyPixelBuffer(buffer), with: CMTime(value: CMTimeValue(i), timescale: CMTimeScale(fps)))
}

/// H.264 mp4 from CGImages, with macOS 27's AVAssetWriter input receiver (the same approach as Apple's
/// StreamingVideoWriter in coreai-models, which isn't a public product).
func writeMP4(_ frames: [CGImage], fps: Int, to url: URL) async throws {
    guard let first = frames.first else { throw ToolError("the pipeline returned no frames") }
    let width = first.width + first.width % 2, height = first.height + first.height % 2   // H.264 needs even sizes
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try? FileManager.default.removeItem(at: url)
    let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
    let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
        AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: width, AVVideoHeightKey: height,
    ])
    guard writer.canAdd(input) else { throw ToolError("can't add a video track to \(url.lastPathComponent)") }
    let receiver = writer.inputPixelBufferReceiver(
        for: input,
        pixelBufferAttributes: CVPixelBufferCreationAttributes(
            pixelFormatType: CVPixelFormatType(rawValue: kCVPixelFormatType_32ARGB),
            size: CVImageSize(width: width, height: height)))
    try writer.start()
    writer.startSession(atSourceTime: .zero)
    for (i, frame) in frames.enumerated() {
        try await appendFrame(frame, index: i, fps: fps, width: width, height: height, to: receiver)
    }
    receiver.finish()
    await writer.finishWriting()
    if writer.status == .failed { throw ToolError("couldn't finish \(url.lastPathComponent): \(String(describing: writer.error))") }
}

/// Last phase seen by the progress handler. Safe: the pipeline calls the handler synchronously from its own loop
/// (the same pattern as Apple's videodiffusion-runner PhaseTiming).
final class PhaseBox: @unchecked Sendable { var last = "" }

let usage = """
    usage: ollamamcp-videogen --model <bundle> --prompt <text> --output <file.mp4>
                              [--quality fast|balanced|best] [--frames 17|33|49|65|81] [--steps N] [--seed N]
           ollamamcp-videogen --clear-cache --model <bundle>
    """

do {
    var it = Array(CommandLine.arguments.dropFirst()).makeIterator()
    var model = "", prompt = "", output = "", quality = QualityPreset.fast
    var frames: Int?, steps: Int?, seed: UInt32 = 0, clearCache = false
    func value(_ flag: String) throws -> String {
        guard let v = it.next() else { throw ToolError("\(flag) needs a value") }
        return v
    }
    while let arg = it.next() {
        switch arg {
        case "--model": model = try value(arg)
        case "--prompt": prompt = try value(arg)
        case "--output": output = try value(arg)
        case "--quality":
            guard let q = QualityPreset(rawValue: try value(arg)) else { throw ToolError("--quality: fast, balanced or best") }
            quality = q
        case "--frames":
            guard let n = Int(try value(arg)), [17, 33, 49, 65, 81].contains(n) else {
                throw ToolError("--frames must be 17, 33, 49, 65 or 81")
            }
            frames = n
        case "--steps":
            guard let n = Int(try value(arg)), (1...50).contains(n) else { throw ToolError("--steps must be 1…50") }
            steps = n
        case "--seed":
            guard let n = UInt32(try value(arg)) else { throw ToolError("--seed must be a whole number") }
            seed = n
        case "--clear-cache": clearCache = true
        case "-h", "--help": print(usage); exit(0)
        default: throw ToolError("unknown option \(arg)\n\(usage)")
        }
    }
    let bundle = URL(fileURLWithPath: model)
    if clearCache {
        guard !model.isEmpty else { throw ToolError("--clear-cache needs --model") }
        let assets = try FileManager.default.contentsOfDirectory(at: bundle, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "aimodel" }
        for asset in assets { try AIModelCache.default.deleteEntries(for: asset) }
        emit(["event": "cleared", "assets": assets.map(\.lastPathComponent).sorted()])
        exit(0)
    }
    guard !model.isEmpty, !prompt.isEmpty, !output.isEmpty else { throw ToolError(usage) }

    emit(["event": "loading"])
    let start = ContinuousClock.now
    let pipeline = try await WanPipeline(from: bundle, lazyModelLoading: true)
    let config = VideoConfiguration.from(
        preset: quality, prompt: prompt, seed: seed, stepCount: steps,
        guidanceScale: pipeline.defaultGuidanceScale, numFrames: frames)
    let seen = PhaseBox()
    let result = try await pipeline.generateVideo(configuration: config) { progress in
        let phase: String
        switch progress.phase {
        case .encoding: phase = "encoding"
        case .denoising: phase = "denoising"
        case .decoding: phase = "decoding"
        case .assembling: phase = "assembling"
        }
        if phase != seen.last { emit(["event": "phase", "phase": phase]); seen.last = phase }
        if progress.phase == .denoising {
            emit(["event": "step", "step": progress.step + 1, "total": progress.totalSteps])
        }
        return true
    }
    let url = URL(fileURLWithPath: output)
    try await writeMP4(result.frames, fps: result.fps, to: url)
    let c = (ContinuousClock.now - start).components
    emit(["event": "done", "path": url.path, "frames": result.frames.count, "fps": result.fps,
          "width": result.frames.first?.width ?? 0, "height": result.frames.first?.height ?? 0,
          "seconds": Double(c.seconds) + Double(c.attoseconds) / 1e18])
} catch {
    emit(["event": "error", "message": String(describing: error)])
    exit(1)
}
