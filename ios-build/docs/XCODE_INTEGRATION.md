# KataGo iOS Integration Guide

This guide explains how to integrate KataGo with CoreML model conversion into a SwiftUI iOS app using **Swift/C++ interoperability** (Swift 5.9+).

## Prerequisites

- macOS 14.0+ (Sonoma)
- Xcode 15.0+
- iOS 17.0+ deployment target
- CMake 3.18+: `brew install cmake`
- Ninja: `brew install ninja`

## Architecture

This integration uses **Swift 5.9+ bidirectional C++ interoperability** - no Objective-C++ bridging required:

```
┌─────────────────────────────────────────────────────────┐
│                    SwiftUI App                          │
│                         │                               │
│                         ▼                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │         KataGoModelConverter.swift              │   │
│  │         (Pure Swift API)                        │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                               │
│            Swift/C++ Interop (Swift 5.9+)              │
│                         │                               │
│                         ▼                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │         katagocoreml-swift.h/.cpp               │   │
│  │         (C++ wrapper layer)                     │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                               │
│                         ▼                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │         libkatagocoreml.a                       │   │
│  │         (Static library)                        │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Step 1: Build iOS Dependencies

Run the master build script:

```bash
cd ios-build/scripts
./build_all_ios.sh
```

This builds:
- Abseil (~10 min)
- Protocol Buffers (~15 min)
- katagocoreml-cpp (~5 min)

Output location: `ios-build/install/ios/`

## Step 2: Create Xcode Project

### 2.1 Create New iOS App

1. File → New → Project → iOS → App
2. Product Name: `KataGoApp`
3. Interface: SwiftUI
4. Language: Swift

### 2.2 Configure Build Settings for C++ Interop

**Enable C++ Interoperability:**

1. Select your target → Build Settings
2. Search for "C++ and Objective-C Interoperability"
3. Set to: **C++ / Objective-C++**

Or add to "Other Swift Flags":
```
-cxx-interoperability-mode=default
```

**Set C++ Language Standard:**

1. Search for "C++ Language Dialect"
2. Set to: **C++17 [-std=c++17]**

### 2.3 Add Static Libraries

1. Select target → Build Phases → Link Binary With Libraries
2. Click "+" → Add Other → Add Files
3. Navigate to `ios-build/install/ios/lib/`
4. Add all `.a` files:
   - `libkatagocoreml.a`
   - `libprotobuf.a`
   - `libabsl_*.a` (multiple files)

### 2.4 Add System Frameworks

Add these frameworks in Link Binary With Libraries:
- CoreML.framework
- Metal.framework
- MetalPerformanceShaders.framework
- MetalPerformanceShadersGraph.framework
- CoreFoundation.framework
- Accelerate.framework
- libz.tbd
- libc++.tbd

### 2.5 Configure Search Paths

**Header Search Paths:**
```
$(PROJECT_DIR)/../ios-build/install/ios/include
$(PROJECT_DIR)/../ios-build/swift
```

**Library Search Paths:**
```
$(PROJECT_DIR)/../ios-build/install/ios/lib
```

**Swift Include Paths** (for module map):
```
$(PROJECT_DIR)/../ios-build/swift
```

### 2.6 Add Module Map

Add to "Other Swift Flags":
```
-Xcc -fmodule-map-file=$(PROJECT_DIR)/../ios-build/swift/module.modulemap
```

## Step 3: Add Swift/C++ Interop Files

Copy these files to your project:

```
ios-build/swift/
├── KataGoModelConverter.swift  # Pure Swift API
├── katagocoreml-swift.h        # C++ header for Swift
├── katagocoreml-swift.cpp      # C++ implementation
└── module.modulemap            # Module map for C++ import
```

Add to your Xcode project:
1. **KataGoModelConverter.swift** → Add to Swift sources
2. **katagocoreml-swift.cpp** → Add to C++ sources (Compile Sources)
3. **katagocoreml-swift.h** and **module.modulemap** → Add as headers (not compiled)

## Step 4: Swift Usage

### 4.1 Model Conversion

```swift
import SwiftUI
import CoreML

@Observable
class KataGoEngine {
    var isConverting = false
    var conversionProgress: Float = 0
    var conversionStage: String = ""
    var model: MLModel?

    private let converter = KataGoModelConverter()

    /// Get information about a model file
    func getModelInfo(from url: URL) throws -> KataGoModelInfo {
        return try converter.getModelInfo(from: url)
    }

    /// Convert a KataGo model to CoreML format
    func convertModel(from inputURL: URL, to outputURL: URL) async throws {
        isConverting = true
        defer { isConverting = false }

        let options = KataGoConversionOptions(
            boardXSize: 19,
            boardYSize: 19,
            useFP16: true,
            optimizeIdentityMask: true
        )

        try await converter.convert(
            from: inputURL,
            to: outputURL,
            options: options
        ) { [weak self] progress, stage in
            Task { @MainActor in
                self?.conversionProgress = progress
                self?.conversionStage = stage
            }
        }
    }

    /// Convert and immediately load the model
    func convertAndLoad(from url: URL) async throws {
        model = try await converter.convertAndLoad(from: url)
    }
}
```

### 4.2 SwiftUI View

```swift
import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var engine = KataGoEngine()
    @State private var showFilePicker = false
    @State private var modelInfo: KataGoModelInfo?
    @State private var errorMessage: String?

    var body: some View {
        VStack(spacing: 20) {
            Text("KataGo Model Converter")
                .font(.title)

            if let info = modelInfo {
                VStack(alignment: .leading) {
                    Text("Model: \(info.name)")
                    Text("Version: \(info.version)")
                    Text("Blocks: \(info.numBlocks)")
                    Text("Channels: \(info.trunkNumChannels)")
                }
                .font(.caption)
                .padding()
                .background(Color.gray.opacity(0.1))
                .cornerRadius(8)
            }

            if engine.isConverting {
                VStack {
                    ProgressView(value: engine.conversionProgress)
                    Text(engine.conversionStage)
                        .font(.caption)
                }
                .padding()
            }

            Button("Select Model File") {
                showFilePicker = true
            }
            .disabled(engine.isConverting)

            if let error = errorMessage {
                Text(error)
                    .foregroundColor(.red)
                    .font(.caption)
            }
        }
        .padding()
        .fileImporter(
            isPresented: $showFilePicker,
            allowedContentTypes: [.data],
            allowsMultipleSelection: false
        ) { result in
            handleFileSelection(result)
        }
    }

    private func handleFileSelection(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            guard let url = urls.first else { return }

            Task {
                do {
                    // Get model info
                    modelInfo = try engine.getModelInfo(from: url)

                    // Convert and load
                    try await engine.convertAndLoad(from: url)

                    errorMessage = nil
                } catch {
                    errorMessage = error.localizedDescription
                }
            }

        case .failure(let error):
            errorMessage = error.localizedDescription
        }
    }
}
```

## Step 5: Add KataGo MCTS (Optional)

If you need full MCTS search, add the KataGo C++ sources and Swift inference files.

### 5.1 Required C++ Source Files

Copy these directories to your Xcode project:
- `cpp/core/` (utilities)
- `cpp/game/` (board, rules)
- `cpp/search/` (MCTS)
- `cpp/neuralnet/` (neural net interface)

### 5.2 Swift Inference Files

Copy from the `coreml-backend` branch:
- `cpp/neuralnet/coremlbackend.swift`
- `cpp/neuralnet/mpsgraphlayers.swift`

These already use Swift/C++ interop and integrate seamlessly.

## Project Structure

```
KataGoApp/
├── KataGoApp.xcodeproj
├── KataGoApp/
│   ├── KataGoApp.swift              # @main App
│   ├── ContentView.swift            # Main UI
│   ├── KataGoEngine.swift           # Engine wrapper
│   ├── GoBoardView.swift            # Board UI (optional)
│   │
│   ├── Interop/                     # C++/Swift interop
│   │   ├── KataGoModelConverter.swift
│   │   ├── katagocoreml-swift.h
│   │   ├── katagocoreml-swift.cpp
│   │   └── module.modulemap
│   │
│   ├── CoreML/                      # Swift inference
│   │   ├── coremlbackend.swift
│   │   └── mpsgraphlayers.swift
│   │
│   └── Resources/
│       └── gtp_ios.cfg              # Config file
│
└── Libraries/                       # Symlink to ios-build/install/ios/
    ├── include/
    └── lib/
```

## Build Settings Summary

| Setting | Value |
|---------|-------|
| C++ Language Dialect | C++17 |
| C++ and Objective-C Interoperability | C++ / Objective-C++ |
| Other Swift Flags | `-cxx-interoperability-mode=default` |
| Header Search Paths | `$(PROJECT_DIR)/../ios-build/install/ios/include` |
| Library Search Paths | `$(PROJECT_DIR)/../ios-build/install/ios/lib` |
| Other Linker Flags | `-lc++` |

## Troubleshooting

### "No such module" error

Ensure the module.modulemap is in the Swift Include Paths and the `-Xcc -fmodule-map-file=` flag is set.

### Undefined C++ symbols

Add `-lc++` to Other Linker Flags.

### Swift can't find C++ types

Verify:
1. C++ Interoperability is enabled
2. Module map path is correct
3. Headers are in Header Search Paths

### Linker errors for protobuf/abseil

Ensure all `.a` files from `install/ios/lib/` are added to Link Binary With Libraries.

### CoreML model compilation fails on device

Check iOS deployment target matches the specification version (iOS 17+ for spec v8).

## Performance Tips

1. **Use FP16**: Reduces model size by 50%, minimal accuracy loss
2. **Enable mask optimization**: ~6.5% speedup for fixed 19x19 board
3. **Background conversion**: Always convert on background queue
4. **Cache converted models**: Store .mlpackage in app's Documents

## References

- [Swift C++ Interoperability](https://www.swift.org/documentation/cxx-interop/)
- [KataGo GitHub](https://github.com/lightvector/KataGo)
- [katagocoreml-cpp](https://github.com/ChinChangYang/katagocoreml-cpp)
- [CoreML Documentation](https://developer.apple.com/documentation/coreml)
