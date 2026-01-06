// katagocoreml-swift.h
// C++ header for Swift interoperability with katagocoreml-cpp
//
// This header wraps the katagocoreml-cpp API in a Swift-friendly manner
// using Swift 5.9+ C++ interoperability.

#ifndef KATAGOCOREML_SWIFT_H
#define KATAGOCOREML_SWIFT_H

#include <string>
#include <stdexcept>

// Swift-compatible namespace for katagocoreml
namespace katagocoreml {

/// Model information structure
struct ModelInfo {
    std::string name;
    int32_t version;
    int32_t numInputChannels;
    int32_t numInputGlobalChannels;
    int32_t trunkNumChannels;
    int32_t numBlocks;

    ModelInfo() : version(0), numInputChannels(0), numInputGlobalChannels(0),
                  trunkNumChannels(0), numBlocks(0) {}
};

/// Conversion options for KataGo to CoreML conversion
struct ConversionOptions {
    int32_t board_x_size = 19;
    int32_t board_y_size = 19;
    std::string compute_precision = "FLOAT16";
    bool optimize_identity_mask = true;
    int32_t specification_version = 8;

    ConversionOptions() = default;
};

/// KataGo to CoreML model converter
///
/// This class provides static methods for converting KataGo neural network
/// models (.bin.gz format) to Apple CoreML format (.mlpackage).
class KataGoConverter {
public:
    /// Get information about a KataGo model file
    /// @param modelPath Path to the .bin.gz model file
    /// @return ModelInfo structure with model details
    /// @throws std::runtime_error if the model cannot be read
    static ModelInfo getModelInfo(const std::string& modelPath);

    /// Convert a KataGo model to CoreML format
    /// @param inputPath Path to input .bin.gz model
    /// @param outputPath Path for output .mlpackage directory
    /// @param options Conversion options
    /// @throws std::runtime_error if conversion fails
    static void convert(
        const std::string& inputPath,
        const std::string& outputPath,
        const ConversionOptions& options = ConversionOptions()
    );

private:
    KataGoConverter() = delete;
};

} // namespace katagocoreml

#endif // KATAGOCOREML_SWIFT_H
