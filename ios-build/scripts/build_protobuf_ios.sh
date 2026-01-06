#!/bin/bash
# Build Protocol Buffers for iOS
# This script builds both host protoc and iOS arm64 static library

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_ROOT="${SCRIPT_DIR}/.."
TOOLCHAIN_FILE="${BUILD_ROOT}/toolchains/ios.toolchain.cmake"
DEPS_DIR="${BUILD_ROOT}/deps"
INSTALL_DIR="${BUILD_ROOT}/install/ios"
HOST_INSTALL_DIR="${BUILD_ROOT}/install/host"

# Protobuf version - use a version with good CMake support
PROTOBUF_VERSION="25.1"
PROTOBUF_URL="https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VERSION}/protobuf-${PROTOBUF_VERSION}.tar.gz"

# Number of parallel jobs
JOBS=$(sysctl -n hw.ncpu)

echo "=========================================="
echo "Building Protocol Buffers ${PROTOBUF_VERSION} for iOS"
echo "=========================================="

# Create directories
mkdir -p "${DEPS_DIR}"
mkdir -p "${INSTALL_DIR}"
mkdir -p "${HOST_INSTALL_DIR}"

cd "${DEPS_DIR}"

# Download protobuf if not exists
if [ ! -d "protobuf-${PROTOBUF_VERSION}" ]; then
    echo "Downloading protobuf ${PROTOBUF_VERSION}..."
    curl -L "${PROTOBUF_URL}" -o protobuf.tar.gz
    tar xzf protobuf.tar.gz
    rm protobuf.tar.gz
fi

cd "protobuf-${PROTOBUF_VERSION}"

# ============================================
# Step 1: Build host protoc (runs on macOS)
# ============================================
echo ""
echo "Step 1: Building host protoc..."
echo ""

# Verify Abseil is built for host
if [ ! -d "${HOST_INSTALL_DIR}/lib/cmake/absl" ]; then
    echo "Error: Host Abseil not found at ${HOST_INSTALL_DIR}"
    echo "Please run build_abseil_ios.sh first."
    exit 1
fi

mkdir -p build-host
cd build-host

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${HOST_INSTALL_DIR}" \
    -DCMAKE_PREFIX_PATH="${HOST_INSTALL_DIR}" \
    -Dprotobuf_BUILD_TESTS=OFF \
    -Dprotobuf_BUILD_EXAMPLES=OFF \
    -Dprotobuf_BUILD_PROTOBUF_BINARIES=ON \
    -Dprotobuf_BUILD_LIBPROTOC=OFF \
    -Dprotobuf_ABSL_PROVIDER=package \
    -Dabsl_DIR="${HOST_INSTALL_DIR}/lib/cmake/absl" \
    -Dutf8_range_DIR="${HOST_INSTALL_DIR}/lib/cmake/utf8_range" \
    -DABSL_PROPAGATE_CXX_STD=ON \
    -DCMAKE_CXX_STANDARD=17

cmake --build . --target protoc -j${JOBS}

# Copy protoc to host install
mkdir -p "${HOST_INSTALL_DIR}/bin"
cp protoc "${HOST_INSTALL_DIR}/bin/"

echo "Host protoc built at: ${HOST_INSTALL_DIR}/bin/protoc"

cd ..

# ============================================
# Step 2: Build protobuf static library for iOS
# ============================================
echo ""
echo "Step 2: Building protobuf for iOS arm64..."
echo ""

# Verify Abseil is built for iOS
if [ ! -d "${INSTALL_DIR}/lib/cmake/absl" ]; then
    echo "Error: iOS Abseil not found at ${INSTALL_DIR}"
    echo "Please run build_abseil_ios.sh first."
    exit 1
fi

mkdir -p build-ios
cd build-ios

# Configure for iOS using toolchain
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DCMAKE_PREFIX_PATH="${INSTALL_DIR}" \
    -Dprotobuf_BUILD_TESTS=OFF \
    -Dprotobuf_BUILD_EXAMPLES=OFF \
    -Dprotobuf_BUILD_PROTOBUF_BINARIES=OFF \
    -Dprotobuf_BUILD_PROTOC_BINARIES=OFF \
    -Dprotobuf_BUILD_LIBPROTOC=OFF \
    -Dprotobuf_BUILD_SHARED_LIBS=OFF \
    -Dprotobuf_ABSL_PROVIDER=package \
    -Dabsl_DIR="${INSTALL_DIR}/lib/cmake/absl" \
    -Dutf8_range_DIR="${INSTALL_DIR}/lib/cmake/utf8_range" \
    -DABSL_PROPAGATE_CXX_STD=ON \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON

cmake --build . -j${JOBS}
cmake --install .

echo ""
echo "=========================================="
echo "Protobuf iOS build complete!"
echo "  Host protoc: ${HOST_INSTALL_DIR}/bin/protoc"
echo "  iOS libraries: ${INSTALL_DIR}/lib/"
echo "  iOS headers: ${INSTALL_DIR}/include/"
echo "=========================================="
