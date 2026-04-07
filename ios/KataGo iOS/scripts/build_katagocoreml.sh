#!/bin/bash
# Build katagocoreml static library for the iOS/macOS Xcode project.
# This script builds katagocoreml and its dependencies (protobuf, abseil)
# for the specified platform using cmake.
#
# Usage: ./build_katagocoreml.sh [platform]
#   platform: iphoneos, iphonesimulator, macosx, xros, xrsimulator (default: macosx)
#
# Prerequisites: cmake, ninja, protobuf, abseil (install via: brew install cmake ninja protobuf abseil)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CPP_DIR="${SCRIPT_DIR}/../../cpp"
PLATFORM="${1:-macosx}"
OUTPUT_DIR="${SCRIPT_DIR}/../Libraries/${PLATFORM}"

# Skip rebuild if already built
if [ -f "${OUTPUT_DIR}/lib/libkatagocoreml.a" ]; then
    echo "katagocoreml already built for ${PLATFORM}, skipping..."
    exit 0
fi

# Check for required tools
for tool in cmake ninja protoc; do
    if ! command -v "${tool}" &> /dev/null; then
        echo "error: ${tool} is required. Install with: brew install cmake ninja protobuf abseil"
        exit 1
    fi
done

echo "Building katagocoreml for ${PLATFORM}..."

BUILD_DIR="/tmp/katagocoreml-build-${PLATFORM}"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

SDK_PATH=$(xcrun --sdk "${PLATFORM}" --show-sdk-path 2>/dev/null || echo "")

CMAKE_EXTRA_ARGS=""
case "${PLATFORM}" in
    iphoneos)
        CMAKE_EXTRA_ARGS="-DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=${SDK_PATH} -DCMAKE_OSX_ARCHITECTURES=arm64"
        ;;
    iphonesimulator)
        CMAKE_EXTRA_ARGS="-DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=${SDK_PATH} -DCMAKE_OSX_ARCHITECTURES=arm64"
        ;;
    macosx)
        CMAKE_EXTRA_ARGS="-DCMAKE_OSX_ARCHITECTURES=arm64"
        ;;
    xros)
        CMAKE_EXTRA_ARGS="-DCMAKE_SYSTEM_NAME=visionOS -DCMAKE_OSX_SYSROOT=${SDK_PATH} -DCMAKE_OSX_ARCHITECTURES=arm64"
        ;;
    xrsimulator)
        CMAKE_EXTRA_ARGS="-DCMAKE_SYSTEM_NAME=visionOS -DCMAKE_OSX_SYSROOT=${SDK_PATH} -DCMAKE_OSX_ARCHITECTURES=arm64"
        ;;
esac

cd "${BUILD_DIR}"

cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_BACKEND=METAL \
    ${CMAKE_EXTRA_ARGS} \
    "${CPP_DIR}"

# Only build the katagocoreml target
ninja katagocoreml

# Install to output directory
mkdir -p "${OUTPUT_DIR}/lib" "${OUTPUT_DIR}/include"

cp -f "${BUILD_DIR}/external/katagocoreml/libkatagocoreml.a" "${OUTPUT_DIR}/lib/"
cp -rf "${CPP_DIR}/external/katagocoreml/include/katagocoreml" "${OUTPUT_DIR}/include/"

# Copy generated proto headers
mkdir -p "${OUTPUT_DIR}/include/proto"
cp -f "${BUILD_DIR}/external/katagocoreml/proto/"*.pb.h "${OUTPUT_DIR}/include/proto/" 2>/dev/null || true

# Clean up build dir
rm -rf "${BUILD_DIR}"

echo "katagocoreml built successfully for ${PLATFORM} → ${OUTPUT_DIR}"
