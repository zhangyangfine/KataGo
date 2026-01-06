#!/bin/bash
# Build Abseil (Google's C++ common libraries) for iOS
# Required dependency for Protocol Buffers v22+

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_ROOT="${SCRIPT_DIR}/.."
TOOLCHAIN_FILE="${BUILD_ROOT}/toolchains/ios.toolchain.cmake"
DEPS_DIR="${BUILD_ROOT}/deps"
INSTALL_DIR="${BUILD_ROOT}/install/ios"
HOST_INSTALL_DIR="${BUILD_ROOT}/install/host"

# Abseil version - should be compatible with protobuf version
ABSEIL_VERSION="20240116.2"
ABSEIL_URL="https://github.com/abseil/abseil-cpp/releases/download/${ABSEIL_VERSION}/abseil-cpp-${ABSEIL_VERSION}.tar.gz"

# Number of parallel jobs
JOBS=$(sysctl -n hw.ncpu)

echo "=========================================="
echo "Building Abseil ${ABSEIL_VERSION} for iOS"
echo "=========================================="

# Create directories
mkdir -p "${DEPS_DIR}"
mkdir -p "${INSTALL_DIR}"
mkdir -p "${HOST_INSTALL_DIR}"

cd "${DEPS_DIR}"

# Download abseil if not exists
if [ ! -d "abseil-cpp-${ABSEIL_VERSION}" ]; then
    echo "Downloading abseil ${ABSEIL_VERSION}..."
    curl -L "${ABSEIL_URL}" -o abseil.tar.gz
    tar xzf abseil.tar.gz
    rm abseil.tar.gz
fi

cd "abseil-cpp-${ABSEIL_VERSION}"

# ============================================
# Step 1: Build Abseil for host (macOS)
# ============================================
echo ""
echo "Step 1: Building Abseil for host (macOS)..."
echo ""

mkdir -p build-host
cd build-host

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${HOST_INSTALL_DIR}" \
    -DCMAKE_CXX_STANDARD=17 \
    -DABSL_BUILD_TESTING=OFF \
    -DABSL_USE_GOOGLETEST_HEAD=OFF \
    -DABSL_PROPAGATE_CXX_STD=ON \
    -DBUILD_SHARED_LIBS=OFF

cmake --build . -j${JOBS}
cmake --install .

echo "Host Abseil installed to: ${HOST_INSTALL_DIR}"

cd ..

# ============================================
# Step 2: Build Abseil for iOS arm64
# ============================================
echo ""
echo "Step 2: Building Abseil for iOS arm64..."
echo ""

mkdir -p build-ios
cd build-ios

cmake .. \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DCMAKE_CXX_STANDARD=17 \
    -DABSL_BUILD_TESTING=OFF \
    -DABSL_USE_GOOGLETEST_HEAD=OFF \
    -DABSL_PROPAGATE_CXX_STD=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON

cmake --build . -j${JOBS}
cmake --install .

echo ""
echo "=========================================="
echo "Abseil iOS build complete!"
echo "  Host libraries: ${HOST_INSTALL_DIR}/lib/"
echo "  iOS libraries: ${INSTALL_DIR}/lib/"
echo "  iOS headers: ${INSTALL_DIR}/include/"
echo "=========================================="
