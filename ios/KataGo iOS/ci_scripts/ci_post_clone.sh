#!/bin/sh

# Install build dependencies for katagocoreml (Metal backend model conversion)
brew install cmake ninja protobuf abseil

# Build katagocoreml for all platforms
SCRIPTS_DIR="../scripts"
"${SCRIPTS_DIR}/build_katagocoreml.sh" iphoneos
"${SCRIPTS_DIR}/build_katagocoreml.sh" iphonesimulator
"${SCRIPTS_DIR}/build_katagocoreml.sh" macosx
"${SCRIPTS_DIR}/build_katagocoreml.sh" xros
"${SCRIPTS_DIR}/build_katagocoreml.sh" xrsimulator

# Install Metal Toolchain (required for Xcode 26+; not bundled by default)
METAL_EXPORT_PATH="/tmp/metalToolchainExport"
rm -rf "$METAL_EXPORT_PATH"

xcodebuild -downloadComponent metalToolchain -exportPath "$METAL_EXPORT_PATH"

# Get current Xcode build version (e.g. "17A5241e")
XCODE_BUILD=$(xcodebuild -version | grep 'Build version' | awk '{print $3}')

# Find the downloaded bundle (e.g. MetalToolchain-17A5295f.exportedBundle)
BUNDLE_PATH=$(ls "$METAL_EXPORT_PATH"/*.exportedBundle)

# Extract the version baked into the bundle name
BUNDLE_VERSION=$(basename "$BUNDLE_PATH" .exportedBundle | sed 's/MetalToolchain-//')

# Patch ExportMetadata.plist so its version matches the running Xcode build
sed -i '' "s/${BUNDLE_VERSION}/${XCODE_BUILD}/g" "${BUNDLE_PATH}/ExportMetadata.plist"

xcodebuild -importComponent metalToolchain -importPath "$BUNDLE_PATH"

# Download built-in 18b network (Metal backend converts to CoreML on-the-fly)
DEFAULT_MODEL_GZ="default_model.bin.gz"
DEFAULT_MODEL_URL="https://github.com/ChinChangYang/KataGo/releases/download/v1.15.1-coreml2/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz"
DEFAULT_MODEL_RES="../Resources/default_model.bin.gz"

rm -f "$DEFAULT_MODEL_GZ"
curl -L -o "$DEFAULT_MODEL_GZ" "$DEFAULT_MODEL_URL"
cp -f "$DEFAULT_MODEL_GZ" "$DEFAULT_MODEL_RES"

# Download human SL model
HUMAN_MODEL_GZ="b18c384nbt-humanv0.bin.gz"
HUMAN_MODEL_URL="https://github.com/lightvector/KataGo/releases/download/v1.15.0/b18c384nbt-humanv0.bin.gz"
HUMAN_MODEL_RES="../Resources/b18c384nbt-humanv0.bin.gz"

curl -L -o "$HUMAN_MODEL_GZ" "$HUMAN_MODEL_URL"
cp -f "$HUMAN_MODEL_GZ" "$HUMAN_MODEL_RES"

# Download 9x9 opening book
BOOK_GZ="book9x9jp-20260226.kbook.gz"
BOOK_URL="https://github.com/ChinChangYang/KataGo/releases/download/v1.16.4-coreml1/book9x9jp-20260226.kbook.gz"
BOOK_RES="../Resources/book9x9jp-20260226.kbook.gz"

curl -L -o "$BOOK_GZ" "$BOOK_URL"
cp -f "$BOOK_GZ" "$BOOK_RES"
