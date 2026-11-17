// swift-tools-version: 5.9
// OllamaMCP's Apple-native layer (macOS 27+). Optional: the cross-platform core never needs it.
// Build: swift build -c release --package-path native
import PackageDescription

let package = Package(
    name: "OllamaMCPNative",
    platforms: [.macOS("27.0")],
    dependencies: [
        // Apple's Core AI runtime utilities. Pinned: the FLUX.2 [klein] bundle in use was exported for this
        // revision's pipeline, and Apple changes the package without notice. Bump deliberately, then re-test.
        .package(url: "https://github.com/apple/coreai-models", revision: "be11f018217f087df49d59b60bbff9cb08230403"),
        // Already a dependency of coreai-models (resolved to the same version); named here because the Z-Image
        // sampler tokenizes prompts itself.
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.0"),
    ],
    targets: [
        // The MCP server runs this once per image (see native/image_backend.py).
        .executableTarget(
            name: "ollamamcp-imagegen",
            dependencies: [
                .product(name: "CoreAIDiffusion", package: "coreai-models"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/ImageGen"
        ),
        // EXPERIMENTAL text-to-video (local_video): Wan 2.1 through Apple's WanPipeline. Its own executable, so
        // the image helper stays lean and video keeps a separate Core AI cache.
        .executableTarget(
            name: "ollamamcp-videogen",
            dependencies: [
                .product(name: "CoreAIVideoDiffusion", package: "coreai-models"),
            ],
            path: "Sources/VideoGen"
        ),
        // Upscale and cutout (local_upscale / local_cutout): macOS's own VideoToolbox super-resolution and Vision
        // subject masks. No Core AI, so it builds fast and leaves the image generator's cache name alone.
        .executableTarget(
            name: "ollamamcp-imagetools",
            path: "Sources/ImageTools"
        ),
    ]
)
