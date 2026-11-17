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
    ],
    targets: [
        // The MCP server runs this once per image (see native/image_backend.py).
        .executableTarget(
            name: "ollamamcp-imagegen",
            dependencies: [
                .product(name: "CoreAIDiffusion", package: "coreai-models"),
            ],
            path: "Sources/ImageGen"
        ),
    ]
)
