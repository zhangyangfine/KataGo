// KataGoModelConverter.swift
// Swift wrapper for katagocoreml-cpp using C++/Swift interoperability
//
// This file provides a pure Swift interface to the katagocoreml-cpp library
// using Swift 5.9+ bidirectional C++ interoperability.

import Foundation

// MARK: - Model Information

/// Information about a KataGo model file
public struct KataGoModelInfo: Sendable {
    public let name: String
    public let version: Int32
    public let numInputChannels: Int32
    public let numInputGlobalChannels: Int32
    public let trunkNumChannels: Int32
    public let numBlocks: Int32
}

// MARK: - Conversion Options

/// Options for converting a KataGo model to CoreML format
public struct KataGoConversionOptions: Sendable {
    /// Board width (default: 19)
    public var boardXSize: Int32 = 19

    /// Board height (default: 19)
    public var boardYSize: Int32 = 19

    /// Use FP16 precision for smaller model size (default: true)
    public var useFP16: Bool = true

    /// Optimize for fixed board size - ~6.5% speedup (default: true)
    public var optimizeIdentityMask: Bool = true

    /// CoreML specification version (default: 8 for iOS 17+)
    public var specificationVersion: Int32 = 8

    public init() {}

    public init(
        boardXSize: Int32 = 19,
        boardYSize: Int32 = 19,
        useFP16: Bool = true,
        optimizeIdentityMask: Bool = true,
        specificationVersion: Int32 = 8
    ) {
        self.boardXSize = boardXSize
        self.boardYSize = boardYSize
        self.useFP16 = useFP16
        self.optimizeIdentityMask = optimizeIdentityMask
        self.specificationVersion = specificationVersion
    }
}

// MARK: - Conversion Error

/// Errors that can occur during model conversion
public enum KataGoConversionError: Error, LocalizedError {
    case fileNotFound(String)
    case invalidModel(String)
    case conversionFailed(String)
    case outputPathError(String)

    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let path):
            return "Model file not found: \(path)"
        case .invalidModel(let reason):
            return "Invalid model: \(reason)"
        case .conversionFailed(let reason):
            return "Conversion failed: \(reason)"
        case .outputPathError(let reason):
            return "Output path error: \(reason)"
        }
    }
}

// MARK: - Model Converter

/// Converts KataGo models (.bin.gz) to CoreML format (.mlpackage)
///
/// This class provides a pure Swift interface to the katagocoreml-cpp library
/// using Swift 5.9+ C++ interoperability.
///
/// Example usage:
/// ```swift
/// let converter = KataGoModelConverter()
///
/// // Get model info
/// let info = try converter.getModelInfo(from: modelURL)
/// print("Model: \(info.name), version: \(info.version)")
///
/// // Convert model
/// try await converter.convert(
///     from: modelURL,
///     to: outputURL,
///     options: KataGoConversionOptions()
/// )
/// ```
public final class KataGoModelConverter: Sendable {

    public init() {}

    /// Get information about a KataGo model file
    /// - Parameter url: URL to the .bin.gz model file
    /// - Returns: Model information including version, channels, and block count
    /// - Throws: KataGoConversionError if the model cannot be read
    public func getModelInfo(from url: URL) throws -> KataGoModelInfo {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw KataGoConversionError.fileNotFound(url.path)
        }

        // Call C++ function via Swift/C++ interop
        // The C++ types are automatically bridged
        do {
            let cppInfo = try katagocoreml.KataGoConverter.getModelInfo(std.string(url.path))

            return KataGoModelInfo(
                name: String(cppInfo.name),
                version: cppInfo.version,
                numInputChannels: cppInfo.numInputChannels,
                numInputGlobalChannels: cppInfo.numInputGlobalChannels,
                trunkNumChannels: cppInfo.trunkNumChannels,
                numBlocks: cppInfo.numBlocks
            )
        } catch {
            throw KataGoConversionError.invalidModel(String(describing: error))
        }
    }

    /// Convert a KataGo model to CoreML format
    /// - Parameters:
    ///   - inputURL: URL to the input .bin.gz model file
    ///   - outputURL: URL for the output .mlpackage directory
    ///   - options: Conversion options
    /// - Throws: KataGoConversionError if conversion fails
    public func convert(
        from inputURL: URL,
        to outputURL: URL,
        options: KataGoConversionOptions = KataGoConversionOptions()
    ) throws {
        guard FileManager.default.fileExists(atPath: inputURL.path) else {
            throw KataGoConversionError.fileNotFound(inputURL.path)
        }

        // Ensure output directory's parent exists
        let parentDir = outputURL.deletingLastPathComponent()
        if !FileManager.default.fileExists(atPath: parentDir.path) {
            try FileManager.default.createDirectory(
                at: parentDir,
                withIntermediateDirectories: true
            )
        }

        // Build C++ conversion options
        var cppOptions = katagocoreml.ConversionOptions()
        cppOptions.board_x_size = options.boardXSize
        cppOptions.board_y_size = options.boardYSize
        cppOptions.compute_precision = std.string(options.useFP16 ? "FLOAT16" : "FLOAT32")
        cppOptions.optimize_identity_mask = options.optimizeIdentityMask
        cppOptions.specification_version = options.specificationVersion

        // Call C++ converter
        do {
            try katagocoreml.KataGoConverter.convert(
                std.string(inputURL.path),
                std.string(outputURL.path),
                cppOptions
            )
        } catch {
            throw KataGoConversionError.conversionFailed(String(describing: error))
        }
    }

    /// Convert a KataGo model to CoreML format asynchronously
    /// - Parameters:
    ///   - inputURL: URL to the input .bin.gz model file
    ///   - outputURL: URL for the output .mlpackage directory
    ///   - options: Conversion options
    ///   - progress: Optional progress callback (0.0 to 1.0)
    /// - Throws: KataGoConversionError if conversion fails
    public func convert(
        from inputURL: URL,
        to outputURL: URL,
        options: KataGoConversionOptions = KataGoConversionOptions(),
        progress: (@Sendable (Float, String) -> Void)? = nil
    ) async throws {
        progress?(0.0, "Starting conversion...")

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    progress?(0.1, "Loading model...")

                    try self.convert(from: inputURL, to: outputURL, options: options)

                    progress?(1.0, "Conversion complete")
                    continuation.resume()
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}

// MARK: - Convenience Extensions

extension KataGoModelConverter {

    /// Convert model and return the compiled MLModel
    /// - Parameters:
    ///   - inputURL: URL to the input .bin.gz model file
    ///   - options: Conversion options
    /// - Returns: Compiled CoreML model ready for inference
    @available(iOS 17.0, macOS 14.0, *)
    public func convertAndLoad(
        from inputURL: URL,
        options: KataGoConversionOptions = KataGoConversionOptions()
    ) async throws -> MLModel {
        import CoreML

        // Create temporary output path
        let tempDir = FileManager.default.temporaryDirectory
        let modelName = inputURL.deletingPathExtension().deletingPathExtension().lastPathComponent
        let outputURL = tempDir.appendingPathComponent("\(modelName).mlpackage")

        // Convert
        try await convert(from: inputURL, to: outputURL, options: options)

        // Compile and load
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine

        let compiledURL = try MLModel.compileModel(at: outputURL)
        let model = try MLModel(contentsOf: compiledURL, configuration: config)

        // Cleanup temp mlpackage (compiled version is separate)
        try? FileManager.default.removeItem(at: outputURL)

        return model
    }
}
