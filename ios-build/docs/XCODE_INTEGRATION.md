# KataGo iOS Integration Guide

This guide explains how to integrate KataGo with CoreML model conversion into a SwiftUI iOS app.

## Prerequisites

- macOS 13.0+
- Xcode 15.0+
- iOS 17.0+ deployment target
- CMake 3.18+: `brew install cmake`

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

### 2.2 Add C++ Static Libraries

1. In Project Navigator, select your project
2. Select your target → Build Phases → Link Binary With Libraries
3. Click "+" → Add Other → Add Files
4. Navigate to `ios-build/install/ios/lib/`
5. Add all `.a` files:
   - `libkatagocoreml.a`
   - `libprotobuf.a`
   - `libabsl_*.a` (multiple files)

### 2.3 Add System Frameworks

Add these frameworks in Link Binary With Libraries:
- CoreML.framework
- Metal.framework
- MetalPerformanceShaders.framework
- MetalPerformanceShadersGraph.framework
- CoreFoundation.framework
- Accelerate.framework
- libz.tbd

### 2.4 Configure Header Search Paths

1. Select target → Build Settings
2. Search for "Header Search Paths"
3. Add: `$(PROJECT_DIR)/../ios-build/install/ios/include` (recursive)

### 2.5 Configure Library Search Paths

1. Search for "Library Search Paths"
2. Add: `$(PROJECT_DIR)/../ios-build/install/ios/lib`

### 2.6 Enable C++ Interop

1. Search for "C++ and Objective-C Interoperability"
2. Set to: "C++ / Objective-C++"

Or add to Other C++ Flags: `-fcxx-modules -fmodules`

## Step 3: Create Bridging Header

### 3.1 Create Header File

Create `KataGo-Bridging-Header.h`:

```objc
#ifndef KataGo_Bridging_Header_h
#define KataGo_Bridging_Header_h

#import "KataGoBridge.h"

#endif
```

### 3.2 Configure Bridging Header

1. Build Settings → Search for "Objective-C Bridging Header"
2. Set to: `$(PROJECT_DIR)/KataGoApp/KataGo-Bridging-Header.h`

## Step 4: Create Objective-C++ Bridge

### 4.1 KataGoBridge.h

```objc
// KataGoBridge.h
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Model information returned by getModelInfo
@interface KataGoModelInfo : NSObject
@property (nonatomic, readonly) NSString *name;
@property (nonatomic, readonly) int version;
@property (nonatomic, readonly) int numInputChannels;
@property (nonatomic, readonly) int numInputGlobalChannels;
@property (nonatomic, readonly) int trunkNumChannels;
@property (nonatomic, readonly) int numBlocks;
@end

/// Options for model conversion
@interface KataGoConversionOptions : NSObject
@property (nonatomic) int boardXSize;           // Default: 19
@property (nonatomic) int boardYSize;           // Default: 19
@property (nonatomic) BOOL useFP16;             // Default: YES
@property (nonatomic) BOOL optimizeIdentityMask; // Default: YES (for exact board size)
@property (nonatomic) int specificationVersion; // Default: 8 (iOS 17+)
@end

/// Main bridge class for KataGo model conversion
@interface KataGoModelConverter : NSObject

/// Get information about a KataGo model file
+ (nullable KataGoModelInfo *)getModelInfoFromPath:(NSString *)modelPath
                                             error:(NSError **)error;

/// Convert a KataGo model (.bin.gz) to CoreML format (.mlpackage)
/// @param inputPath Path to input .bin.gz model
/// @param outputPath Path for output .mlpackage directory
/// @param options Conversion options (nil for defaults)
/// @param error Error output
/// @return YES on success, NO on failure
+ (BOOL)convertModelAtPath:(NSString *)inputPath
                    toPath:(NSString *)outputPath
                   options:(nullable KataGoConversionOptions *)options
                     error:(NSError **)error;

/// Convert model with progress callback
+ (BOOL)convertModelAtPath:(NSString *)inputPath
                    toPath:(NSString *)outputPath
                   options:(nullable KataGoConversionOptions *)options
                  progress:(void (^)(float progress, NSString *stage))progressCallback
                     error:(NSError **)error;

@end

NS_ASSUME_NONNULL_END
```

### 4.2 KataGoBridge.mm

```objc
// KataGoBridge.mm
#import "KataGoBridge.h"
#include <katagocoreml/KataGoConverter.hpp>
#include <string>

// MARK: - KataGoModelInfo

@implementation KataGoModelInfo {
    NSString *_name;
    int _version;
    int _numInputChannels;
    int _numInputGlobalChannels;
    int _trunkNumChannels;
    int _numBlocks;
}

- (instancetype)initWithName:(NSString *)name
                     version:(int)version
            numInputChannels:(int)numInputChannels
      numInputGlobalChannels:(int)numInputGlobalChannels
            trunkNumChannels:(int)trunkNumChannels
                   numBlocks:(int)numBlocks {
    self = [super init];
    if (self) {
        _name = [name copy];
        _version = version;
        _numInputChannels = numInputChannels;
        _numInputGlobalChannels = numInputGlobalChannels;
        _trunkNumChannels = trunkNumChannels;
        _numBlocks = numBlocks;
    }
    return self;
}

@end

// MARK: - KataGoConversionOptions

@implementation KataGoConversionOptions

- (instancetype)init {
    self = [super init];
    if (self) {
        _boardXSize = 19;
        _boardYSize = 19;
        _useFP16 = YES;
        _optimizeIdentityMask = YES;
        _specificationVersion = 8;
    }
    return self;
}

@end

// MARK: - KataGoModelConverter

@implementation KataGoModelConverter

+ (nullable KataGoModelInfo *)getModelInfoFromPath:(NSString *)modelPath
                                             error:(NSError **)error {
    try {
        std::string path = [modelPath UTF8String];
        auto info = katagocoreml::KataGoConverter::getModelInfo(path);

        return [[KataGoModelInfo alloc] initWithName:[NSString stringWithUTF8String:info.name.c_str()]
                                             version:info.version
                                    numInputChannels:info.numInputChannels
                              numInputGlobalChannels:info.numInputGlobalChannels
                                    trunkNumChannels:info.trunkNumChannels
                                           numBlocks:info.numBlocks];
    } catch (const std::exception& e) {
        if (error) {
            *error = [NSError errorWithDomain:@"KataGoErrorDomain"
                                         code:1
                                     userInfo:@{NSLocalizedDescriptionKey:
                                         [NSString stringWithUTF8String:e.what()]}];
        }
        return nil;
    }
}

+ (BOOL)convertModelAtPath:(NSString *)inputPath
                    toPath:(NSString *)outputPath
                   options:(nullable KataGoConversionOptions *)options
                     error:(NSError **)error {
    return [self convertModelAtPath:inputPath
                             toPath:outputPath
                            options:options
                           progress:nil
                              error:error];
}

+ (BOOL)convertModelAtPath:(NSString *)inputPath
                    toPath:(NSString *)outputPath
                   options:(nullable KataGoConversionOptions *)options
                  progress:(void (^)(float progress, NSString *stage))progressCallback
                     error:(NSError **)error {
    try {
        katagocoreml::ConversionOptions opts;

        if (options) {
            opts.board_x_size = options.boardXSize;
            opts.board_y_size = options.boardYSize;
            opts.compute_precision = options.useFP16 ? "FLOAT16" : "FLOAT32";
            opts.optimize_identity_mask = options.optimizeIdentityMask;
            opts.specification_version = options.specificationVersion;
        } else {
            // Defaults
            opts.board_x_size = 19;
            opts.board_y_size = 19;
            opts.compute_precision = "FLOAT16";
            opts.optimize_identity_mask = true;
            opts.specification_version = 8;
        }

        if (progressCallback) {
            progressCallback(0.0f, @"Loading model...");
        }

        std::string input = [inputPath UTF8String];
        std::string output = [outputPath UTF8String];

        if (progressCallback) {
            progressCallback(0.2f, @"Parsing model architecture...");
        }

        katagocoreml::KataGoConverter::convert(input, output, opts);

        if (progressCallback) {
            progressCallback(1.0f, @"Conversion complete");
        }

        return YES;

    } catch (const std::exception& e) {
        if (error) {
            *error = [NSError errorWithDomain:@"KataGoErrorDomain"
                                         code:2
                                     userInfo:@{NSLocalizedDescriptionKey:
                                         [NSString stringWithUTF8String:e.what()]}];
        }
        return NO;
    }
}

@end
```

## Step 5: Swift Usage

### 5.1 Model Conversion

```swift
import Foundation

class KataGoEngine: ObservableObject {
    @Published var isConverting = false
    @Published var conversionProgress: Float = 0
    @Published var conversionStage: String = ""

    func convertModel(from inputURL: URL, to outputURL: URL) async throws {
        await MainActor.run {
            isConverting = true
            conversionProgress = 0
        }

        let options = KataGoConversionOptions()
        options.boardXSize = 19
        options.boardYSize = 19
        options.useFP16 = true
        options.optimizeIdentityMask = true

        var conversionError: NSError?

        let success = KataGoModelConverter.convertModel(
            atPath: inputURL.path,
            toPath: outputURL.path,
            options: options,
            progress: { [weak self] progress, stage in
                DispatchQueue.main.async {
                    self?.conversionProgress = progress
                    self?.conversionStage = stage ?? ""
                }
            },
            error: &conversionError
        )

        await MainActor.run {
            isConverting = false
        }

        if !success, let error = conversionError {
            throw error
        }
    }
}
```

### 5.2 Loading and Using the Model

```swift
import CoreML

extension KataGoEngine {
    func loadConvertedModel(from mlpackageURL: URL) async throws -> MLModel {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine

        // Compile the model
        let compiledURL = try MLModel.compileModel(at: mlpackageURL)

        // Load the compiled model
        let model = try MLModel(contentsOf: compiledURL, configuration: config)

        return model
    }
}
```

## Step 6: Add KataGo C++ Sources (for MCTS)

If you also need MCTS search, add these C++ source files to your project:

### 6.1 Required Source Files

Copy these directories to your Xcode project:
- `cpp/core/` (utilities)
- `cpp/game/` (board, rules)
- `cpp/search/` (MCTS)
- `cpp/neuralnet/` (neural net interface)

### 6.2 Swift Files for Inference

Copy these Swift files:
- `cpp/neuralnet/coremlbackend.swift`
- `cpp/neuralnet/mpsgraphlayers.swift`

## Project Structure

```
KataGoApp/
├── KataGoApp.xcodeproj
├── KataGoApp/
│   ├── KataGoApp.swift          # @main App
│   ├── ContentView.swift        # Main UI
│   ├── KataGoEngine.swift       # Engine wrapper
│   ├── GoBoardView.swift        # Board UI
│   │
│   ├── Bridge/
│   │   ├── KataGo-Bridging-Header.h
│   │   ├── KataGoBridge.h
│   │   └── KataGoBridge.mm
│   │
│   ├── CoreML/                  # Swift inference (optional)
│   │   ├── coremlbackend.swift
│   │   └── mpsgraphlayers.swift
│   │
│   └── Resources/
│       └── gtp_ios.cfg          # Config file
│
└── Libraries/                   # Symlink to ios-build/install/ios/
    ├── include/
    └── lib/
```

## Troubleshooting

### Undefined symbols for C++ standard library

Add to Other Linker Flags: `-lc++`

### Protobuf symbol conflicts

Ensure all protobuf/abseil libraries are from the same build.

### "No such module" for Swift files

Ensure Swift files are added to the target's Compile Sources.

### CoreML model compilation fails

Check iOS deployment target matches spec version (iOS 17+ for spec v8).

## Performance Tips

1. **Use FP16**: Reduces model size by 50% with minimal accuracy loss
2. **Enable mask optimization**: ~6.5% speedup for fixed board sizes
3. **Hybrid execution**: Let CoreML dispatch to CPU+ANE+GPU automatically
4. **Background conversion**: Convert models on background queue

## References

- [KataGo GitHub](https://github.com/lightvector/KataGo)
- [katagocoreml-cpp](https://github.com/ChinChangYang/katagocoreml-cpp)
- [CoreML Documentation](https://developer.apple.com/documentation/coreml)
