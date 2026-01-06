# KataGo iOS Build System

This directory contains scripts and tools to build KataGo with CoreML model conversion support for iOS.

## Overview

The build system compiles the following dependencies for iOS arm64:

| Library | Purpose | Size (approx) |
|---------|---------|---------------|
| Abseil | Google C++ utilities | ~1.5 MB |
| Protocol Buffers | Serialization | ~2.5 MB |
| katagocoreml-cpp | CoreML model converter | ~500 KB |

**Total additional binary size: ~4.5 MB**

## Quick Start

```bash
# Build all dependencies
cd scripts
./build_all_ios.sh

# Build time: ~30 minutes on Apple Silicon
```

## Directory Structure

```
ios-build/
├── README.md                 # This file
├── scripts/
│   ├── build_all_ios.sh      # Master build script
│   ├── build_abseil_ios.sh   # Abseil build
│   ├── build_protobuf_ios.sh # Protobuf build
│   └── build_katagocoreml_ios.sh  # katagocoreml build
├── toolchains/
│   └── ios.toolchain.cmake   # CMake iOS toolchain
├── patches/
│   └── uuid_ios_compat.patch # iOS UUID compatibility
├── docs/
│   └── XCODE_INTEGRATION.md  # Xcode setup guide
├── deps/                     # Downloaded sources (created by scripts)
└── install/                  # Built libraries (created by scripts)
    ├── host/                 # macOS binaries (protoc)
    └── ios/                  # iOS arm64 libraries
        ├── include/          # Headers
        └── lib/              # Static libraries
```

## Requirements

- macOS 13.0 or later
- Xcode 15.0 or later
- CMake 3.18+: `brew install cmake`
- iOS 17.0+ deployment target

## Build Steps

### Option 1: Build All (Recommended)

```bash
./scripts/build_all_ios.sh
```

### Option 2: Build Individually

```bash
# 1. Build Abseil first (required by protobuf)
./scripts/build_abseil_ios.sh

# 2. Build Protobuf
./scripts/build_protobuf_ios.sh

# 3. Build katagocoreml-cpp
./scripts/build_katagocoreml_ios.sh
```

## Output

After successful build, libraries are installed to:

```
ios-build/install/ios/
├── include/
│   ├── absl/
│   ├── google/protobuf/
│   └── katagocoreml/
└── lib/
    ├── libabsl_*.a
    ├── libprotobuf.a
    └── libkatagocoreml.a
```

## Integration

See [docs/XCODE_INTEGRATION.md](docs/XCODE_INTEGRATION.md) for detailed Xcode setup instructions.

### Quick Summary

1. Add all `.a` files from `install/ios/lib/` to Xcode
2. Add `install/ios/include/` to Header Search Paths
3. Link required frameworks (CoreML, Metal, etc.)
4. Create Objective-C++ bridge (see documentation)
5. Use from Swift via the bridge

## Capabilities

Once integrated, your iOS app can:

- **Convert models on-device**: Load `.bin.gz` KataGo models and convert to CoreML `.mlpackage`
- **Run inference**: Use CoreML for neural network evaluation
- **Full MCTS search**: Combined with KataGo C++ search code

## Troubleshooting

### Build fails with "SDK not found"

Ensure Xcode is installed and command line tools are selected:
```bash
xcode-select --install
sudo xcode-select -s /Applications/Xcode.app
```

### CMake can't find iOS SDK

```bash
xcrun --sdk iphoneos --show-sdk-path
# Should output something like:
# /Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS17.0.sdk
```

### Protobuf build fails

Ensure you have a recent CMake:
```bash
brew upgrade cmake
```

## License

The build scripts are part of the KataGo project. Dependencies have their own licenses:
- Abseil: Apache 2.0
- Protocol Buffers: BSD 3-Clause
- katagocoreml-cpp: Check repository
