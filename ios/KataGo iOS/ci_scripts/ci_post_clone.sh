#!/bin/sh

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

DEFAULT_MODEL_GZ="default_model.bin.gz"
DEFAULT_MODEL_URL="https://github.com/ChinChangYang/KataGo/releases/download/v1.16.4-coreml1/kata1-b28c512nbt-adam-s11165M-d5387M-null.bin.gz"
DEFAULT_MODEL_RES="../Resources/default_model.bin.gz"

HUMAN_MODEL_GZ="b18c384nbt-humanv0.bin.gz"
HUMAN_MODEL_URL="https://github.com/lightvector/KataGo/releases/download/v1.15.0/b18c384nbt-humanv0.bin.gz"
HUMAN_MODEL_RES="../Resources/b18c384nbt-humanv0.bin.gz"

FP16_MLPACKAGE_RES="../Resources/KataGoModel19x19fp16.mlpackage"
FP16_ZIP="KataGoModel19x19fp16-adam-s11165M.mlpackage.zip"
FP16_URL="https://github.com/ChinChangYang/KataGo/releases/download/v1.16.4-coreml1/KataGoModel19x19fp16-adam-s11165M.mlpackage.zip"
FP16_DIR="KataGoModel19x19fp16-adam-s11165M.mlpackage"

FP16M1_MLPACKAGE_RES="../Resources/KataGoModel19x19fp16m1.mlpackage"
FP16M1_ZIP="KataGoModel19x19fp16m1.mlpackage.zip"
FP16M1_URL="https://github.com/ChinChangYang/KataGo/releases/download/v1.16.4-coreml1/KataGoModel19x19fp16m1.mlpackage.zip"
FP16M1_DIR="KataGoModel19x19fp16m1.mlpackage"

rm -f "$DEFAULT_MODEL_GZ"
curl -L -o "$DEFAULT_MODEL_GZ" "$DEFAULT_MODEL_URL"
cp -f "$DEFAULT_MODEL_GZ" "$DEFAULT_MODEL_RES"

curl -L -o "$HUMAN_MODEL_GZ" "$HUMAN_MODEL_URL"
cp -f "$HUMAN_MODEL_GZ" "$HUMAN_MODEL_RES"

rm -rf "$FP16_MLPACKAGE_RES"
rm -f "$FP16_ZIP"
curl -L -o "$FP16_ZIP" "$FP16_URL"
unzip "$FP16_ZIP"
mv "$FP16_DIR" "$FP16_MLPACKAGE_RES"

if [ ! -f "$FP16M1_MLPACKAGE_RES" ]; then
    rm -f "$FP16M1_ZIP"
    curl -L -o "$FP16M1_ZIP" "$FP16M1_URL"
    unzip "$FP16M1_ZIP"
    mv "$FP16M1_DIR" "$FP16M1_MLPACKAGE_RES"
fi

BOOK_GZ="book9x9jp-20260226.kbook.gz"
BOOK_URL="https://github.com/ChinChangYang/KataGo/releases/download/v1.16.4-coreml1/book9x9jp-20260226.kbook.gz"
BOOK_RES="../Resources/book9x9jp-20260226.kbook.gz"

curl -L -o "$BOOK_GZ" "$BOOK_URL"
cp -f "$BOOK_GZ" "$BOOK_RES"
