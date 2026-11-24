# XCFramework

This fork builds the XCFramework for macOS on Apple Silicon (arm64) only. The
iOS, visionOS and tvOS slices are not built.

To build it from source, run the script from the repository root:

```bash
./build-xcframework.sh
```

The result is `build-apple/llama.xcframework`. It contains `libllama`, `ggml`
(CPU, Metal and BLAS backends) and `libmtmd`, with the public headers. It can be
used in a Swift package as a local binary target. For example:

```swift
// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MyLlamaPackage",
    platforms: [
        .macOS(.v13)
    ],
    targets: [
        .executableTarget(
            name: "MyLlamaPackage",
            dependencies: [
                "LlamaFramework"
            ]),
        .binaryTarget(
            name: "LlamaFramework",
            path: "path/to/llama.xcframework"
        )
    ]
)
```
