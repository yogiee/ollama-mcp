// ollamamcp-imagetools — OllamaMCP's on-device image utilities (macOS 27+), built only on what macOS ships:
// nothing to download or host, no Core AI specialization cache.
//
//   --upscale <image> --output <file.png> [--sharpen S]
//       4× super-resolution with Apple's VTSuperResolutionScaler (VideoToolbox). Inputs 16…1920 px per side.
//       The scaler is faithful but a little soft, so a light luminance sharpen follows (CISharpenLuminance,
//       radius 0.75): S = 0 keeps Apple's output as is, 1.6 is the default, 3.0 the most. Ported from Looking
//       Glass's SuperResolutionService (chosen in its 2026-09-21 bake-off: best fidelity, ~0.7 s per 1024²).
//   --cutout <image> --output <file.png> [--crop]
//       Lift the foreground subject(s) onto a transparent background with Vision's foreground instance mask
//       (the same model as "Copy Subject" in Photos). --crop trims to the subject.
//   --status
//       Whether the super-resolution model is on this Mac (macOS downloads it on request; we never do).
//
// One JSON object per line on stdout, like ollamamcp-imagegen:
//   {"event":"done","path":"…","width":4096,"height":4096,"seconds":0.8}
//   {"event":"error","message":"…"}   (and a non-zero exit)

import CoreImage
import CoreVideo
import Foundation
import ImageIO
import UniformTypeIdentifiers
import Vision
import VideoToolbox

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

let sRGB = CGColorSpace(name: CGColorSpace.sRGB)!
let ciContext = CIContext(options: [.workingColorSpace: sRGB, .outputColorSpace: sRGB])

/// Read an image the right way up (applies EXIF orientation, which a plain CGImageSource read ignores).
func loadImage(_ path: String) throws -> CGImage {
    guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any] else {
        throw ToolError("couldn't read the image at \(path)")
    }
    let w = props[kCGImagePropertyPixelWidth] as? Int ?? 0, h = props[kCGImagePropertyPixelHeight] as? Int ?? 0
    let options: [CFString: Any] = [
        kCGImageSourceCreateThumbnailFromImageAlways: true,
        kCGImageSourceCreateThumbnailWithTransform: true,
        kCGImageSourceThumbnailMaxPixelSize: max(w, h, 1),
    ]
    guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else {
        throw ToolError("couldn't decode the image at \(path)")
    }
    return image
}

func savePNG(_ image: CGImage, to url: URL) throws {
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
        throw ToolError("couldn't create \(url.path)")
    }
    CGImageDestinationAddImage(dest, image, nil)
    guard CGImageDestinationFinalize(dest) else { throw ToolError("couldn't write \(url.path)") }
}

// MARK: - Upscale

enum Upscale {
    static let maxSide = 1920
    static let minSide = 16
    static let sharpenRadius = 0.75
    static let maxSharpness = 3.0
    static let defaultSharpness = 1.6

    static func configuration(width: Int, height: Int) -> VTSuperResolutionScalerConfiguration? {
        VTSuperResolutionScalerConfiguration(
            frameWidth: width, frameHeight: height, scaleFactor: 4, inputType: .image, usePrecomputedFlow: false,
            qualityPrioritization: .normal, revision: VTSuperResolutionScalerConfiguration.defaultRevision)
    }

    static func status() -> String {
        guard VTSuperResolutionScalerConfiguration.isSupported else { return "unsupported" }
        guard let config = configuration(width: 1024, height: 1024) else { return "unsupported" }
        switch config.configurationModelStatus {
        case .ready: return "ready"
        case .downloading: return "downloading"
        default: return "not_downloaded"
        }
    }

    static func run(_ src: CGImage, sharpness: Double) async throws -> CGImage {
        guard (minSide...maxSide).contains(src.width), (minSide...maxSide).contains(src.height) else {
            throw ToolError("upscale takes images \(minSide)–\(maxSide) px per side; this one is \(src.width)×\(src.height)")
        }
        switch status() {
        case "ready": break
        case "unsupported": throw ToolError("Apple's super-resolution scaler isn't available on this Mac")
        default:
            throw ToolError("Apple's super-resolution model isn't on this Mac yet (macOS downloads it on request, "
                            + "e.g. from an app that uses it)")
        }
        guard let config = configuration(width: src.width, height: src.height),
              let input = pixelBuffer(src.width, src.height, config.sourcePixelBufferAttributes),
              let output = pixelBuffer(src.width * 4, src.height * 4, config.destinationPixelBufferAttributes) else {
            throw ToolError("couldn't set up the super-resolution scaler for \(src.width)×\(src.height)")
        }
        // The scaler takes half-float RGBA; sRGB-encoded values measured better than linear light.
        ciContext.render(CIImage(cgImage: src), to: input,
                         bounds: CGRect(x: 0, y: 0, width: src.width, height: src.height), colorSpace: sRGB)
        let processor = VTFrameProcessor()   // one per image: an ended session can't start another
        try processor.startSession(configuration: config)
        defer { processor.endSession() }
        guard let from = VTFrameProcessorFrame(buffer: input, presentationTimeStamp: .zero),
              let to = VTFrameProcessorFrame(buffer: output, presentationTimeStamp: .zero),
              let params = VTSuperResolutionScalerParameters(
                sourceFrame: from, previousFrame: nil, previousOutputFrame: nil, opticalFlow: nil,
                submissionMode: .random, destinationFrame: to) else {
            throw ToolError("couldn't build the super-resolution request")
        }
        _ = try await processor.process(parameters: params)

        let extent = CGRect(x: 0, y: 0, width: src.width * 4, height: src.height * 4)
        var result = CIImage(cvPixelBuffer: output, options: [.colorSpace: sRGB])
        if sharpness > 0 {
            let sharpen = CIFilter(name: "CISharpenLuminance")!
            sharpen.setValue(result, forKey: kCIInputImageKey)
            sharpen.setValue(min(sharpness, maxSharpness), forKey: kCIInputSharpnessKey)
            sharpen.setValue(sharpenRadius, forKey: kCIInputRadiusKey)
            if let out = sharpen.outputImage { result = out }
        }
        guard let image = ciContext.createCGImage(result, from: extent, format: .RGBA8, colorSpace: sRGB) else {
            throw ToolError("couldn't render the upscaled image")
        }
        return image
    }

    static func pixelBuffer(_ width: Int, _ height: Int, _ attributes: [String: Any]) -> CVPixelBuffer? {
        let declared = attributes[kCVPixelBufferPixelFormatTypeKey as String]
        guard let format = (declared as? NSNumber) ?? (declared as? [NSNumber])?.first else { return nil }
        var attrs = attributes
        attrs[kCVPixelBufferWidthKey as String] = width
        attrs[kCVPixelBufferHeightKey as String] = height
        var buffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(nil, width, height, format.uint32Value, attrs as CFDictionary, &buffer)
        return status == kCVReturnSuccess ? buffer : nil
    }
}

// MARK: - Cutout

enum Cutout {
    /// Every foreground instance Vision finds, on transparency. Returns the image and how many subjects it kept.
    static func run(_ src: CGImage, crop: Bool) throws -> (CGImage, Int) {
        let request = VNGenerateForegroundInstanceMaskRequest()
        let handler = VNImageRequestHandler(cgImage: src)
        try handler.perform([request])
        guard let observation = request.results?.first, !observation.allInstances.isEmpty else {
            throw ToolError("no distinct subject found in the image")
        }
        let masked = try observation.generateMaskedImage(
            ofInstances: observation.allInstances, from: handler, croppedToInstancesExtent: crop)
        let ci = CIImage(cvPixelBuffer: masked)
        guard let image = ciContext.createCGImage(ci, from: ci.extent, format: .RGBA8, colorSpace: sRGB) else {
            throw ToolError("couldn't render the cutout")
        }
        return (image, observation.allInstances.count)
    }
}

// MARK: - Main

func seconds(since start: ContinuousClock.Instant) -> Double {
    let c = (ContinuousClock.now - start).components
    return Double(c.seconds) + Double(c.attoseconds) / 1e18
}

let usage = """
    usage: ollamamcp-imagetools --upscale <image> --output <file.png> [--sharpen 0…3]
           ollamamcp-imagetools --cutout <image> --output <file.png> [--crop]
           ollamamcp-imagetools --status
    """

do {
    var args = Array(CommandLine.arguments.dropFirst()).makeIterator()
    var upscale: String?, cutout: String?, output: String?, sharpen = Upscale.defaultSharpness
    var crop = false, status = false
    func value(_ flag: String) throws -> String {
        guard let v = args.next() else { throw ToolError("\(flag) needs a value") }
        return v
    }
    while let arg = args.next() {
        switch arg {
        case "--upscale": upscale = try value(arg)
        case "--cutout": cutout = try value(arg)
        case "--output": output = try value(arg)
        case "--sharpen":
            guard let s = Double(try value(arg)), (0...Upscale.maxSharpness).contains(s) else {
                throw ToolError("--sharpen must be 0…\(Upscale.maxSharpness)")
            }
            sharpen = s
        case "--crop": crop = true
        case "--status": status = true
        case "-h", "--help": print(usage); exit(0)
        default: throw ToolError("unknown option \(arg)\n\(usage)")
        }
    }
    let start = ContinuousClock.now
    if status {
        emit(["event": "status", "upscale": Upscale.status()])
    } else if let path = upscale, let output {
        let image = try await Upscale.run(try loadImage(path), sharpness: sharpen)
        try savePNG(image, to: URL(fileURLWithPath: output))
        emit(["event": "done", "path": output, "width": image.width, "height": image.height,
              "seconds": seconds(since: start)])
    } else if let path = cutout, let output {
        let (image, subjects) = try Cutout.run(try loadImage(path), crop: crop)
        try savePNG(image, to: URL(fileURLWithPath: output))
        emit(["event": "done", "path": output, "width": image.width, "height": image.height,
              "subjects": subjects, "seconds": seconds(since: start)])
    } else {
        throw ToolError(usage)
    }
} catch {
    emit(["event": "error", "message": String(describing: error)])
    exit(1)
}
