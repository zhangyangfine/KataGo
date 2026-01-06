#!/bin/bash
# Master build script for all KataGo iOS dependencies
# Builds: abseil, protobuf, katagocoreml-cpp

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_ROOT="${SCRIPT_DIR}/.."

echo "============================================================"
echo "KataGo iOS Dependencies Build"
echo "============================================================"
echo ""
echo "This script will build:"
echo "  1. Abseil (Google C++ common libraries)"
echo "  2. Protocol Buffers (serialization library)"
echo "  3. katagocoreml-cpp (CoreML model converter)"
echo ""
echo "Target: iOS 17.0+ arm64"
echo "Output: ${BUILD_ROOT}/install/ios/"
echo ""
echo "============================================================"

# Check for required tools
echo "Checking prerequisites..."

if ! command -v cmake &> /dev/null; then
    echo "Error: cmake not found. Install with: brew install cmake"
    exit 1
fi

if ! command -v xcrun &> /dev/null; then
    echo "Error: Xcode command line tools not found. Install with: xcode-select --install"
    exit 1
fi

# Check for iOS SDK
IOS_SDK=$(xcrun --sdk iphoneos --show-sdk-path 2>/dev/null || true)
if [ -z "$IOS_SDK" ]; then
    echo "Error: iOS SDK not found. Install Xcode from the App Store."
    exit 1
fi
echo "iOS SDK: ${IOS_SDK}"

# Check Xcode version
XCODE_VERSION=$(xcodebuild -version | head -1)
echo "Xcode: ${XCODE_VERSION}"
echo ""

# Start timer
START_TIME=$(date +%s)

# ============================================
# Step 1: Build Abseil
# ============================================
echo ""
echo "============================================================"
echo "Step 1/3: Building Abseil"
echo "============================================================"
"${SCRIPT_DIR}/build_abseil_ios.sh"

# ============================================
# Step 2: Build Protobuf
# ============================================
echo ""
echo "============================================================"
echo "Step 2/3: Building Protocol Buffers"
echo "============================================================"
"${SCRIPT_DIR}/build_protobuf_ios.sh"

# ============================================
# Step 3: Build katagocoreml-cpp
# ============================================
echo ""
echo "============================================================"
echo "Step 3/3: Building katagocoreml-cpp"
echo "============================================================"
"${SCRIPT_DIR}/build_katagocoreml_ios.sh"

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "============================================================"
echo "BUILD COMPLETE!"
echo "============================================================"
echo ""
echo "Total build time: ${MINUTES}m ${SECONDS}s"
echo ""
echo "iOS Libraries installed to: ${BUILD_ROOT}/install/ios/"
echo ""
echo "Contents:"
ls -la "${BUILD_ROOT}/install/ios/lib/" 2>/dev/null || echo "  (libraries)"
echo ""
echo "Next steps:"
echo "  1. Add libraries to your Xcode project"
echo "  2. See ios-build/docs/XCODE_INTEGRATION.md for details"
echo ""
echo "============================================================"
