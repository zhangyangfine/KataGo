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
CPP_DIR="${SCRIPT_DIR}/../../../cpp"
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
HOMEBREW_PREFIX="$(brew --prefix 2>/dev/null || echo /opt/homebrew)"

# Get protobuf version from the actual protoc binary to ensure FetchContent
# matches the host compiler. Get abseil version from the linked Homebrew cellar.
PROTOBUF_VERSION=$(protoc --version | awk '{print $2}')
ABSEIL_VERSION=$(readlink "$(brew --prefix abseil)" | xargs basename)
echo "Using protobuf ${PROTOBUF_VERSION}, abseil ${ABSEIL_VERSION}"

CMAKE_EXTRA_ARGS=""
CROSS_COMPILE_FIND_ARGS="-DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=NEVER -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=BOTH -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=BOTH -DCMAKE_PREFIX_PATH=${HOMEBREW_PREFIX}"
case "${PLATFORM}" in
    iphoneos)
        CMAKE_EXTRA_ARGS="-DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=${SDK_PATH} -DCMAKE_OSX_ARCHITECTURES=arm64 ${CROSS_COMPILE_FIND_ARGS}"
        ;;
    iphonesimulator)
        CMAKE_EXTRA_ARGS="-DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=${SDK_PATH} -DCMAKE_OSX_ARCHITECTURES=arm64 ${CROSS_COMPILE_FIND_ARGS}"
        ;;
    macosx)
        CMAKE_EXTRA_ARGS="-DCMAKE_OSX_ARCHITECTURES=arm64"
        ;;
    xros)
        CMAKE_EXTRA_ARGS="-DCMAKE_SYSTEM_NAME=visionOS -DCMAKE_OSX_SYSROOT=${SDK_PATH} -DCMAKE_OSX_ARCHITECTURES=arm64 ${CROSS_COMPILE_FIND_ARGS}"
        ;;
    xrsimulator)
        CMAKE_EXTRA_ARGS="-DCMAKE_SYSTEM_NAME=visionOS -DCMAKE_OSX_SYSROOT=${SDK_PATH} -DCMAKE_OSX_ARCHITECTURES=arm64 ${CROSS_COMPILE_FIND_ARGS}"
        ;;
esac

cd "${BUILD_DIR}"

# Build protobuf and abseil from source for all platforms, then create
# a combined static library with all dependencies bundled into one .a.
cat > "${BUILD_DIR}/CMakeLists.txt" <<WRAPPER_EOF
cmake_minimum_required(VERSION 3.18.2)
project(katagocoreml_standalone LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

# Build abseil from source (static)
set(ABSL_PROPAGATE_CXX_STD ON)
set(ABSL_ENABLE_INSTALL OFF)
set(BUILD_TESTING OFF)
FetchContent_Declare(abseil-cpp
    URL https://github.com/abseil/abseil-cpp/releases/download/${ABSEIL_VERSION}/abseil-cpp-${ABSEIL_VERSION}.tar.gz
)
FetchContent_MakeAvailable(abseil-cpp)

# Build protobuf from source (static, without tests)
set(protobuf_BUILD_TESTS OFF)
set(protobuf_BUILD_SHARED_LIBS OFF)
set(protobuf_INSTALL OFF)
set(protobuf_ABSL_PROVIDER "package")
# Use host protoc for code generation
set(protobuf_BUILD_PROTOC_BINARIES OFF)
set(Protobuf_PROTOC_EXECUTABLE "$(which protoc)")
FetchContent_Declare(protobuf
    URL https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VERSION}/protobuf-${PROTOBUF_VERSION}.tar.gz
)
FetchContent_MakeAvailable(protobuf)

add_subdirectory("\${KATAGOCOREML_SRC_DIR}" katagocoreml)
WRAPPER_EOF

cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DKATAGOCOREML_SRC_DIR="${CPP_DIR}/external/katagocoreml" \
    ${CMAKE_EXTRA_ARGS} \
    "${BUILD_DIR}"

# Build all targets so protobuf and abseil are compiled (not just katagocoreml)
ninja

# Create combined static library with all dependencies using libtool.
# libtool -static merges .a archives directly, handling member name collisions
# and avoiding ARG_MAX limits that affect ar with many files.
DEP_LIBS=$(find "${BUILD_DIR}/_deps" -name "*.a")
KATAGOCOREML_A="${BUILD_DIR}/katagocoreml/libkatagocoreml.a"

mkdir -p "${OUTPUT_DIR}/lib" "${OUTPUT_DIR}/include"
libtool -static -o "${OUTPUT_DIR}/lib/libkatagocoreml.a" "${KATAGOCOREML_A}" ${DEP_LIBS}

cp -rf "${CPP_DIR}/external/katagocoreml/include/katagocoreml" "${OUTPUT_DIR}/include/"

# Copy generated proto headers
mkdir -p "${OUTPUT_DIR}/include/proto"
cp -f "${BUILD_DIR}/katagocoreml/proto/"*.pb.h "${OUTPUT_DIR}/include/proto/" 2>/dev/null || true

# Clean up build dir
rm -rf "${BUILD_DIR}"

echo "katagocoreml built successfully for ${PLATFORM} → ${OUTPUT_DIR}"
