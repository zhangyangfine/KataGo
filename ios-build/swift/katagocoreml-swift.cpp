// katagocoreml-swift.cpp
// Implementation of Swift-compatible wrapper for katagocoreml-cpp

#include "katagocoreml-swift.h"
#include <katagocoreml/KataGoConverter.hpp>

namespace katagocoreml {

ModelInfo KataGoConverter::getModelInfo(const std::string& modelPath) {
    // Call the actual katagocoreml-cpp library
    auto cppInfo = ::katagocoreml::KataGoConverter::getModelInfo(modelPath);

    ModelInfo info;
    info.name = cppInfo.name;
    info.version = cppInfo.version;
    info.numInputChannels = cppInfo.numInputChannels;
    info.numInputGlobalChannels = cppInfo.numInputGlobalChannels;
    info.trunkNumChannels = cppInfo.trunkNumChannels;
    info.numBlocks = cppInfo.numBlocks;

    return info;
}

void KataGoConverter::convert(
    const std::string& inputPath,
    const std::string& outputPath,
    const ConversionOptions& options
) {
    // Build the katagocoreml-cpp options structure
    ::katagocoreml::ConversionOptions cppOpts;
    cppOpts.board_x_size = options.board_x_size;
    cppOpts.board_y_size = options.board_y_size;
    cppOpts.compute_precision = options.compute_precision;
    cppOpts.optimize_identity_mask = options.optimize_identity_mask;
    cppOpts.specification_version = options.specification_version;

    // Call the actual converter
    ::katagocoreml::KataGoConverter::convert(inputPath, outputPath, cppOpts);
}

} // namespace katagocoreml
