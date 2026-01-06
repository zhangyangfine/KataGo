# iOS CMake Toolchain File for KataGo Dependencies
# Targets iOS 17.0+ arm64 devices

set(CMAKE_SYSTEM_NAME iOS)
set(CMAKE_SYSTEM_VERSION 17.0)
set(CMAKE_OSX_DEPLOYMENT_TARGET 17.0 CACHE STRING "Minimum iOS version")

# Target architecture - arm64 for modern iOS devices
set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "iOS architectures")

# Use the iOS SDK
set(CMAKE_OSX_SYSROOT iphoneos CACHE STRING "iOS SDK")

# Find the iOS SDK path
execute_process(
    COMMAND xcrun --sdk iphoneos --show-sdk-path
    OUTPUT_VARIABLE IOS_SDK_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Compiler settings
set(CMAKE_C_COMPILER_WORKS TRUE)
set(CMAKE_CXX_COMPILER_WORKS TRUE)

# Use clang from Xcode
execute_process(
    COMMAND xcrun --find clang
    OUTPUT_VARIABLE CMAKE_C_COMPILER
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND xcrun --find clang++
    OUTPUT_VARIABLE CMAKE_CXX_COMPILER
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Standard settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# iOS-specific flags
set(CMAKE_C_FLAGS_INIT "-fembed-bitcode-marker")
set(CMAKE_CXX_FLAGS_INIT "-fembed-bitcode-marker")

# Position independent code for static libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Skip RPATH for iOS
set(CMAKE_MACOSX_RPATH OFF)
set(CMAKE_SKIP_RPATH ON)

# Don't try to run executables during build (cross-compiling)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# Install prefix
if(NOT CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Install prefix")
endif()

# Mark as cross-compiling
set(CMAKE_CROSSCOMPILING TRUE)

# iOS specific definitions
add_definitions(-DIOS_PLATFORM=1)

message(STATUS "iOS Toolchain Configuration:")
message(STATUS "  SDK Path: ${IOS_SDK_PATH}")
message(STATUS "  Deployment Target: ${CMAKE_OSX_DEPLOYMENT_TARGET}")
message(STATUS "  Architectures: ${CMAKE_OSX_ARCHITECTURES}")
