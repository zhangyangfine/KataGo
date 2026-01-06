#!/bin/bash
# Build katagocoreml-cpp for iOS
# Requires protobuf and abseil to be built first

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_ROOT="${SCRIPT_DIR}/.."
TOOLCHAIN_FILE="${BUILD_ROOT}/toolchains/ios.toolchain.cmake"
PATCHES_DIR="${BUILD_ROOT}/patches"
DEPS_DIR="${BUILD_ROOT}/deps"
INSTALL_DIR="${BUILD_ROOT}/install/ios"
HOST_INSTALL_DIR="${BUILD_ROOT}/install/host"

# katagocoreml-cpp repository
KATAGOCOREML_REPO="https://github.com/ChinChangYang/katagocoreml-cpp.git"
KATAGOCOREML_BRANCH="master"

# Number of parallel jobs
JOBS=$(sysctl -n hw.ncpu)

echo "=========================================="
echo "Building katagocoreml-cpp for iOS"
echo "=========================================="

# Verify dependencies are built
if [ ! -f "${INSTALL_DIR}/lib/libprotobuf.a" ]; then
    echo "Error: protobuf not found. Run build_abseil_ios.sh and build_protobuf_ios.sh first."
    exit 1
fi

# Create directories
mkdir -p "${DEPS_DIR}"

cd "${DEPS_DIR}"

# Clone or update katagocoreml-cpp
if [ ! -d "katagocoreml-cpp" ]; then
    echo "Cloning katagocoreml-cpp..."
    git clone "${KATAGOCOREML_REPO}" -b "${KATAGOCOREML_BRANCH}"
else
    echo "Updating katagocoreml-cpp..."
    cd katagocoreml-cpp
    git fetch origin
    git checkout "${KATAGOCOREML_BRANCH}"
    git pull origin "${KATAGOCOREML_BRANCH}"
    cd ..
fi

cd katagocoreml-cpp

# ============================================
# Apply iOS compatibility patches
# ============================================
echo ""
echo "Applying iOS compatibility patches..."
echo ""

# Create a patched version of ModelPackage.cpp for iOS UUID support
# This is done inline since the patch format may vary based on the actual source

# Check if we need to patch (look for uuid/uuid.h usage)
if grep -q "uuid/uuid.h" modelpackage/src/ModelPackage.cpp 2>/dev/null; then
    echo "Patching ModelPackage.cpp for iOS UUID compatibility..."

    # Backup original
    cp modelpackage/src/ModelPackage.cpp modelpackage/src/ModelPackage.cpp.bak

    # Create iOS-compatible version
    cat > /tmp/uuid_patch.py << 'PYTHON_PATCH'
import sys

content = open(sys.argv[1]).read()

# Replace uuid include with cross-platform version
old_include = '#include <uuid/uuid.h>'
new_include = '''#if defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_IOS || TARGET_OS_SIMULATOR
#include <CoreFoundation/CoreFoundation.h>
#define USE_CF_UUID 1
#else
#include <uuid/uuid.h>
#endif
#else
#include <uuid/uuid.h>
#endif'''

content = content.replace(old_include, new_include)

# Add helper function after includes (before first namespace or class)
uuid_helper = '''
// Cross-platform UUID generation helper
namespace {
inline std::string generate_uuid_string() {
#if defined(USE_CF_UUID)
    CFUUIDRef uuid = CFUUIDCreate(NULL);
    CFStringRef uuidStr = CFUUIDCreateString(NULL, uuid);
    char buffer[37];
    CFStringGetCString(uuidStr, buffer, sizeof(buffer), kCFStringEncodingASCII);
    CFRelease(uuidStr);
    CFRelease(uuid);
    return std::string(buffer);
#else
    uuid_t uuid;
    uuid_generate(uuid);
    char uuid_str[37];
    uuid_unparse_lower(uuid, uuid_str);
    return std::string(uuid_str);
#endif
}
} // anonymous namespace
'''

# Find a good place to insert the helper (after includes, before namespace)
import re
# Find the position after #include blocks
include_pattern = r'(#include\s*<[^>]+>\s*\n)+'
match = re.search(include_pattern, content)
if match:
    insert_pos = match.end()
    content = content[:insert_pos] + uuid_helper + content[insert_pos:]

print(content)
PYTHON_PATCH

    python3 /tmp/uuid_patch.py modelpackage/src/ModelPackage.cpp > modelpackage/src/ModelPackage.cpp.patched
    mv modelpackage/src/ModelPackage.cpp.patched modelpackage/src/ModelPackage.cpp

    echo "Patch applied successfully."
fi

# ============================================
# Build for iOS
# ============================================
echo ""
echo "Building katagocoreml-cpp for iOS arm64..."
echo ""

mkdir -p build-ios
cd build-ios

# Configure with CMake
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DCMAKE_PREFIX_PATH="${INSTALL_DIR}" \
    -DCMAKE_CXX_STANDARD=17 \
    -DKATAGOCOREML_BUILD_TESTS=OFF \
    -DKATAGOCOREML_BUILD_TOOLS=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DProtobuf_DIR="${INSTALL_DIR}/lib/cmake/protobuf" \
    -Dabsl_DIR="${INSTALL_DIR}/lib/cmake/absl" \
    -DZLIB_ROOT="${INSTALL_DIR}"

cmake --build . -j${JOBS}
cmake --install .

echo ""
echo "=========================================="
echo "katagocoreml-cpp iOS build complete!"
echo "  Library: ${INSTALL_DIR}/lib/libkatagocoreml.a"
echo "  Headers: ${INSTALL_DIR}/include/katagocoreml/"
echo "=========================================="
