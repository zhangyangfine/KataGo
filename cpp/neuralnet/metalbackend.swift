//
//  metalbackend.swift
//  Pure Metal 4 backend for KataGo - Maximum performance neural network inference
//
//  Requires: macOS 26 (Tahoe) / iOS 26 or later, Apple Silicon (M1/A14+)
//  Metal 4 was announced at WWDC 2025
//
//  This implementation uses pure Metal 4 compute shaders for all operations,
//  providing direct GPU control for optimal performance on Apple Silicon.
//

import Foundation
import Metal

// MARK: - Error Output

/// A class that handles output to standard error.
class StandardError: TextOutputStream {
    func write(_ string: String) {
        try? FileHandle.standardError.write(contentsOf: Data(string.utf8))
    }
}

/// A function to print error messages
func printError(_ item: Any) {
    var instance = StandardError()
    print(item, to: &instance)
}

// MARK: - Activation Kind

/// Enum representing different activation functions
@frozen
public enum ActivationKind: Int32 {
    case identity = 0
    case relu = 1
    case mish = 2
}

// MARK: - Layer Descriptors

/// A struct that represents a description of convolutional layer.
public struct SWConvLayerDesc {
    let convYSize: NSNumber
    let convXSize: NSNumber
    let inChannels: NSNumber
    let outChannels: NSNumber
    let dilationY: Int
    let dilationX: Int
    let weights: UnsafeMutablePointer<Float32>

    init(
        convYSize: NSNumber,
        convXSize: NSNumber,
        inChannels: NSNumber,
        outChannels: NSNumber,
        dilationY: Int,
        dilationX: Int,
        weights: UnsafeMutablePointer<Float32>
    ) {
        self.convYSize = convYSize
        self.convXSize = convXSize
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.dilationY = dilationY
        self.dilationX = dilationX
        self.weights = weights
    }
}

public func createSWConvLayerDesc(
    convYSize: Int32,
    convXSize: Int32,
    inChannels: Int32,
    outChannels: Int32,
    dilationY: Int32,
    dilationX: Int32,
    weights: UnsafeMutablePointer<Float32>
) -> SWConvLayerDesc {
    return SWConvLayerDesc(
        convYSize: convYSize as NSNumber,
        convXSize: convXSize as NSNumber,
        inChannels: inChannels as NSNumber,
        outChannels: outChannels as NSNumber,
        dilationY: Int(dilationY),
        dilationX: Int(dilationX),
        weights: weights)
}

/// A struct that represents a description of a batch normalization layer.
public struct SWBatchNormLayerDesc {
    let numChannels: NSNumber
    let mergedScale: UnsafeMutablePointer<Float32>
    let mergedBias: UnsafeMutablePointer<Float32>

    init(
        numChannels: NSNumber,
        mergedScale: UnsafeMutablePointer<Float32>,
        mergedBias: UnsafeMutablePointer<Float32>
    ) {
        self.numChannels = numChannels
        self.mergedScale = mergedScale
        self.mergedBias = mergedBias
    }
}

public func createSWBatchNormLayerDesc(
    numChannels: Int32,
    mergedScale: UnsafeMutablePointer<Float32>,
    mergedBias: UnsafeMutablePointer<Float32>
) -> SWBatchNormLayerDesc {
    return SWBatchNormLayerDesc(
        numChannels: numChannels as NSNumber,
        mergedScale: mergedScale,
        mergedBias: mergedBias)
}

/// A struct that represents a description of a matrix multiply layer.
public struct SWMatMulLayerDesc {
    let inChannels: NSNumber
    let outChannels: NSNumber
    let weights: UnsafeMutablePointer<Float32>

    init(
        inChannels: NSNumber,
        outChannels: NSNumber,
        weights: UnsafeMutablePointer<Float32>
    ) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.weights = weights
    }
}

public func createSWMatMulLayerDesc(
    inChannels: Int32,
    outChannels: Int32,
    weights: UnsafeMutablePointer<Float32>
) -> SWMatMulLayerDesc {
    return SWMatMulLayerDesc(
        inChannels: inChannels as NSNumber,
        outChannels: outChannels as NSNumber,
        weights: weights)
}

/// A struct that represents a description of a matrix bias layer.
public struct SWMatBiasLayerDesc {
    let numChannels: NSNumber
    let weights: UnsafeMutablePointer<Float32>

    init(
        numChannels: NSNumber,
        weights: UnsafeMutablePointer<Float32>
    ) {
        self.numChannels = numChannels
        self.weights = weights
    }
}

public func createSWMatBiasLayerDesc(
    numChannels: Int32,
    weights: UnsafeMutablePointer<Float32>
) -> SWMatBiasLayerDesc {
    return SWMatBiasLayerDesc(
        numChannels: numChannels as NSNumber,
        weights: weights)
}

// MARK: - Block Descriptors

/// Protocol for block descriptors
public protocol BlockDescriptor {}

/// Enum to identify block types for C++ interop
public enum BlockDescriptorKind: Int32 {
    case residual = 0
    case globalPooling = 1
    case nestedBottleneck = 2
}

/// Wrapper class for block descriptors - enables C++ interop
/// Swift protocols can't be exported to C++, so we use this wrapper class instead
public class BlockDescriptorWrapper {
    public let kind: BlockDescriptorKind
    public private(set) var residualBlock: SWResidualBlockDesc?
    public private(set) var globalPoolingBlock: SWGlobalPoolingResidualBlockDesc?
    public private(set) var nestedBottleneckBlock: SWNestedBottleneckResidualBlockDesc?

    public init(residualBlock: SWResidualBlockDesc) {
        self.kind = .residual
        self.residualBlock = residualBlock
    }

    public init(globalPoolingBlock: SWGlobalPoolingResidualBlockDesc) {
        self.kind = .globalPooling
        self.globalPoolingBlock = globalPoolingBlock
    }

    public init(nestedBottleneckBlock: SWNestedBottleneckResidualBlockDesc) {
        self.kind = .nestedBottleneck
        self.nestedBottleneckBlock = nestedBottleneckBlock
    }
}

/// Builder class for block descriptor arrays - enables C++ interop
public class BlockDescriptorBuilder {
    private var wrappers: [BlockDescriptorWrapper] = []

    public init() {}

    public func enqueue(_ wrapper: BlockDescriptorWrapper) {
        wrappers.append(wrapper)
    }

    public func enqueueResidualBlock(_ block: SWResidualBlockDesc) {
        wrappers.append(BlockDescriptorWrapper(residualBlock: block))
    }

    public func enqueueGlobalPoolingBlock(_ block: SWGlobalPoolingResidualBlockDesc) {
        wrappers.append(BlockDescriptorWrapper(globalPoolingBlock: block))
    }

    public func enqueueNestedBottleneckBlock(_ block: SWNestedBottleneckResidualBlockDesc) {
        wrappers.append(BlockDescriptorWrapper(nestedBottleneckBlock: block))
    }

    public func getBlockDescriptors() -> [BlockDescriptorWrapper] {
        return wrappers
    }

    public func getBlockDescriptorsAsProtocol() -> [BlockDescriptor] {
        return wrappers.compactMap { wrapper -> BlockDescriptor? in
            switch wrapper.kind {
            case .residual:
                return wrapper.residualBlock
            case .globalPooling:
                return wrapper.globalPoolingBlock
            case .nestedBottleneck:
                return wrapper.nestedBottleneckBlock
            }
        }
    }

    public var count: Int {
        return wrappers.count
    }
}

/// Factory function to create BlockDescriptorBuilder for C++ interop
public func createBlockDescriptorBuilder() -> BlockDescriptorBuilder {
    return BlockDescriptorBuilder()
}

/// A struct that describes a residual block in a neural network.
public struct SWResidualBlockDesc: BlockDescriptor {
    let preBN: SWBatchNormLayerDesc
    let preActivation: ActivationKind
    let regularConv: SWConvLayerDesc
    let midBN: SWBatchNormLayerDesc
    let midActivation: ActivationKind
    let finalConv: SWConvLayerDesc

    init(
        preBN: SWBatchNormLayerDesc,
        preActivation: ActivationKind,
        regularConv: SWConvLayerDesc,
        midBN: SWBatchNormLayerDesc,
        midActivation: ActivationKind,
        finalConv: SWConvLayerDesc
    ) {
        self.preBN = preBN
        self.preActivation = preActivation
        self.regularConv = regularConv
        self.midBN = midBN
        self.midActivation = midActivation
        self.finalConv = finalConv
    }
}

public func createSWResidualBlockDesc(
    preBN: SWBatchNormLayerDesc,
    preActivation: ActivationKind,
    regularConv: SWConvLayerDesc,
    midBN: SWBatchNormLayerDesc,
    midActivation: ActivationKind,
    finalConv: SWConvLayerDesc
) -> SWResidualBlockDesc {
    return SWResidualBlockDesc(
        preBN: preBN,
        preActivation: preActivation,
        regularConv: regularConv,
        midBN: midBN,
        midActivation: midActivation,
        finalConv: finalConv)
}

/// A struct that describes a global pooling residual block.
public struct SWGlobalPoolingResidualBlockDesc: BlockDescriptor {
    let preBN: SWBatchNormLayerDesc
    let preActivation: ActivationKind
    let regularConv: SWConvLayerDesc
    let gpoolConv: SWConvLayerDesc
    let gpoolBN: SWBatchNormLayerDesc
    let gpoolActivation: ActivationKind
    let gpoolToBiasMul: SWMatMulLayerDesc
    let midBN: SWBatchNormLayerDesc
    let midActivation: ActivationKind
    let finalConv: SWConvLayerDesc

    init(
        preBN: SWBatchNormLayerDesc,
        preActivation: ActivationKind,
        regularConv: SWConvLayerDesc,
        gpoolConv: SWConvLayerDesc,
        gpoolBN: SWBatchNormLayerDesc,
        gpoolActivation: ActivationKind,
        gpoolToBiasMul: SWMatMulLayerDesc,
        midBN: SWBatchNormLayerDesc,
        midActivation: ActivationKind,
        finalConv: SWConvLayerDesc
    ) {
        self.preBN = preBN
        self.preActivation = preActivation
        self.regularConv = regularConv
        self.gpoolConv = gpoolConv
        self.gpoolBN = gpoolBN
        self.gpoolActivation = gpoolActivation
        self.gpoolToBiasMul = gpoolToBiasMul
        self.midBN = midBN
        self.midActivation = midActivation
        self.finalConv = finalConv
    }
}

public func createSWGlobalPoolingResidualBlockDesc(
    preBN: SWBatchNormLayerDesc,
    preActivation: ActivationKind,
    regularConv: SWConvLayerDesc,
    gpoolConv: SWConvLayerDesc,
    gpoolBN: SWBatchNormLayerDesc,
    gpoolActivation: ActivationKind,
    gpoolToBiasMul: SWMatMulLayerDesc,
    midBN: SWBatchNormLayerDesc,
    midActivation: ActivationKind,
    finalConv: SWConvLayerDesc
) -> SWGlobalPoolingResidualBlockDesc {
    return SWGlobalPoolingResidualBlockDesc(
        preBN: preBN,
        preActivation: preActivation,
        regularConv: regularConv,
        gpoolConv: gpoolConv,
        gpoolBN: gpoolBN,
        gpoolActivation: gpoolActivation,
        gpoolToBiasMul: gpoolToBiasMul,
        midBN: midBN,
        midActivation: midActivation,
        finalConv: finalConv)
}

/// A struct that describes a nested bottleneck residual block.
public struct SWNestedBottleneckResidualBlockDesc: BlockDescriptor {
    let preBN: SWBatchNormLayerDesc
    let preActivation: ActivationKind
    let preConv: SWConvLayerDesc
    let blockDescriptors: [BlockDescriptorWrapper]
    let postBN: SWBatchNormLayerDesc
    let postActivation: ActivationKind
    let postConv: SWConvLayerDesc

    init(
        preBN: SWBatchNormLayerDesc,
        preActivation: ActivationKind,
        preConv: SWConvLayerDesc,
        blockDescriptors: [BlockDescriptorWrapper],
        postBN: SWBatchNormLayerDesc,
        postActivation: ActivationKind,
        postConv: SWConvLayerDesc
    ) {
        self.preBN = preBN
        self.preActivation = preActivation
        self.preConv = preConv
        self.blockDescriptors = blockDescriptors
        self.postBN = postBN
        self.postActivation = postActivation
        self.postConv = postConv
    }
}

public func createSWNestedBottleneckResidualBlockDesc(
    preBN: SWBatchNormLayerDesc,
    preActivation: ActivationKind,
    preConv: SWConvLayerDesc,
    blockDescriptors: [BlockDescriptorWrapper],
    postBN: SWBatchNormLayerDesc,
    postActivation: ActivationKind,
    postConv: SWConvLayerDesc
) -> SWNestedBottleneckResidualBlockDesc {
    return SWNestedBottleneckResidualBlockDesc(
        preBN: preBN,
        preActivation: preActivation,
        preConv: preConv,
        blockDescriptors: blockDescriptors,
        postBN: postBN,
        postActivation: postActivation,
        postConv: postConv)
}

// MARK: - SGF Metadata Encoder

/// A struct that describes an SGF metadata encoder.
public struct SWSGFMetadataEncoderDesc {
    let version: Int
    let numInputMetaChannels: Int
    let mul1: SWMatMulLayerDesc
    let bias1: SWMatBiasLayerDesc
    let act1: ActivationKind
    let mul2: SWMatMulLayerDesc
    let bias2: SWMatBiasLayerDesc
    let act2: ActivationKind
    let mul3: SWMatMulLayerDesc

    init(
        version: Int,
        numInputMetaChannels: Int,
        mul1: SWMatMulLayerDesc,
        bias1: SWMatBiasLayerDesc,
        act1: ActivationKind,
        mul2: SWMatMulLayerDesc,
        bias2: SWMatBiasLayerDesc,
        act2: ActivationKind,
        mul3: SWMatMulLayerDesc
    ) {
        self.version = version
        self.numInputMetaChannels = numInputMetaChannels
        self.mul1 = mul1
        self.bias1 = bias1
        self.act1 = act1
        self.mul2 = mul2
        self.bias2 = bias2
        self.act2 = act2
        self.mul3 = mul3
    }
}

public func createSWSGFMetadataEncoderDesc(
    version: Int32,
    numInputMetaChannels: Int32,
    mul1: SWMatMulLayerDesc,
    bias1: SWMatBiasLayerDesc,
    act1: ActivationKind,
    mul2: SWMatMulLayerDesc,
    bias2: SWMatBiasLayerDesc,
    act2: ActivationKind,
    mul3: SWMatMulLayerDesc
) -> SWSGFMetadataEncoderDesc {
    return SWSGFMetadataEncoderDesc(
        version: Int(version),
        numInputMetaChannels: Int(numInputMetaChannels),
        mul1: mul1,
        bias1: bias1,
        act1: act1,
        mul2: mul2,
        bias2: bias2,
        act2: act2,
        mul3: mul3)
}

/// Wrap a non-optional SWSGFMetadataEncoderDesc into an optional for C++ interop
public func wrapOptionalSGFMetadataEncoder(_ encoder: SWSGFMetadataEncoderDesc) -> SWSGFMetadataEncoderDesc? {
    return encoder
}

// MARK: - Trunk Descriptor

/// A struct that describes the trunk of a neural network.
public struct SWTrunkDesc {
    let version: Int
    let trunkNumChannels: NSNumber
    let midNumChannels: NSNumber
    let regularNumChannels: NSNumber
    let gpoolNumChannels: NSNumber
    let initialConv: SWConvLayerDesc
    let initialMatMul: SWMatMulLayerDesc
    let blockDescriptors: [BlockDescriptorWrapper]
    let trunkTipBN: SWBatchNormLayerDesc
    let trunkTipActivation: ActivationKind
    let sgfMetadataEncoder: SWSGFMetadataEncoderDesc?

    init(
        version: Int,
        trunkNumChannels: NSNumber,
        midNumChannels: NSNumber,
        regularNumChannels: NSNumber,
        gpoolNumChannels: NSNumber,
        initialConv: SWConvLayerDesc,
        initialMatMul: SWMatMulLayerDesc,
        blockDescriptors: [BlockDescriptorWrapper],
        trunkTipBN: SWBatchNormLayerDesc,
        trunkTipActivation: ActivationKind,
        sgfMetadataEncoder: SWSGFMetadataEncoderDesc?
    ) {
        self.version = version
        self.trunkNumChannels = trunkNumChannels
        self.midNumChannels = midNumChannels
        self.regularNumChannels = regularNumChannels
        self.gpoolNumChannels = gpoolNumChannels
        self.initialConv = initialConv
        self.initialMatMul = initialMatMul
        self.blockDescriptors = blockDescriptors
        self.trunkTipBN = trunkTipBN
        self.trunkTipActivation = trunkTipActivation
        self.sgfMetadataEncoder = sgfMetadataEncoder
    }
}

public func createSWTrunkDesc(
    version: Int32,
    trunkNumChannels: Int32,
    midNumChannels: Int32,
    regularNumChannels: Int32,
    gpoolNumChannels: Int32,
    initialConv: SWConvLayerDesc,
    initialMatMul: SWMatMulLayerDesc,
    blockDescriptors: [BlockDescriptorWrapper],
    trunkTipBN: SWBatchNormLayerDesc,
    trunkTipActivation: ActivationKind,
    sgfMetadataEncoder: SWSGFMetadataEncoderDesc?
) -> SWTrunkDesc {
    return SWTrunkDesc(
        version: Int(version),
        trunkNumChannels: trunkNumChannels as NSNumber,
        midNumChannels: midNumChannels as NSNumber,
        regularNumChannels: regularNumChannels as NSNumber,
        gpoolNumChannels: gpoolNumChannels as NSNumber,
        initialConv: initialConv,
        initialMatMul: initialMatMul,
        blockDescriptors: blockDescriptors,
        trunkTipBN: trunkTipBN,
        trunkTipActivation: trunkTipActivation,
        sgfMetadataEncoder: sgfMetadataEncoder)
}

// MARK: - Policy Head Descriptor

/// A struct that describes the policy head of a neural network.
public struct SWPolicyHeadDesc {
    let version: Int
    let p1Conv: SWConvLayerDesc
    let g1Conv: SWConvLayerDesc
    let g1BN: SWBatchNormLayerDesc
    let g1Activation: ActivationKind
    let gpoolToBiasMul: SWMatMulLayerDesc
    let p1BN: SWBatchNormLayerDesc
    let p1Activation: ActivationKind
    let p2Conv: SWConvLayerDesc
    let gpoolToPassMul: SWMatMulLayerDesc
    let gpoolToPassBias: SWMatBiasLayerDesc?
    let passActivation: ActivationKind?
    let gpoolToPassMul2: SWMatMulLayerDesc?

    init(
        version: Int,
        p1Conv: SWConvLayerDesc,
        g1Conv: SWConvLayerDesc,
        g1BN: SWBatchNormLayerDesc,
        g1Activation: ActivationKind,
        gpoolToBiasMul: SWMatMulLayerDesc,
        p1BN: SWBatchNormLayerDesc,
        p1Activation: ActivationKind,
        p2Conv: SWConvLayerDesc,
        gpoolToPassMul: SWMatMulLayerDesc,
        gpoolToPassBias: SWMatBiasLayerDesc?,
        passActivation: ActivationKind?,
        gpoolToPassMul2: SWMatMulLayerDesc?
    ) {
        self.version = version
        self.p1Conv = p1Conv
        self.g1Conv = g1Conv
        self.g1BN = g1BN
        self.g1Activation = g1Activation
        self.gpoolToBiasMul = gpoolToBiasMul
        self.p1BN = p1BN
        self.p1Activation = p1Activation
        self.p2Conv = p2Conv
        self.gpoolToPassMul = gpoolToPassMul
        self.gpoolToPassBias = gpoolToPassBias
        self.passActivation = passActivation
        self.gpoolToPassMul2 = gpoolToPassMul2
    }
}

public func createSWPolicyHeadDesc(
    version: Int32,
    p1Conv: SWConvLayerDesc,
    g1Conv: SWConvLayerDesc,
    g1BN: SWBatchNormLayerDesc,
    g1Activation: ActivationKind,
    gpoolToBiasMul: SWMatMulLayerDesc,
    p1BN: SWBatchNormLayerDesc,
    p1Activation: ActivationKind,
    p2Conv: SWConvLayerDesc,
    gpoolToPassMul: SWMatMulLayerDesc,
    gpoolToPassBias: SWMatBiasLayerDesc,
    passActivation: ActivationKind,
    gpoolToPassMul2: SWMatMulLayerDesc
) -> SWPolicyHeadDesc {
    if version >= 15 {
        return SWPolicyHeadDesc(
            version: Int(version),
            p1Conv: p1Conv,
            g1Conv: g1Conv,
            g1BN: g1BN,
            g1Activation: g1Activation,
            gpoolToBiasMul: gpoolToBiasMul,
            p1BN: p1BN,
            p1Activation: p1Activation,
            p2Conv: p2Conv,
            gpoolToPassMul: gpoolToPassMul,
            gpoolToPassBias: gpoolToPassBias,
            passActivation: passActivation,
            gpoolToPassMul2: gpoolToPassMul2)
    } else {
        return SWPolicyHeadDesc(
            version: Int(version),
            p1Conv: p1Conv,
            g1Conv: g1Conv,
            g1BN: g1BN,
            g1Activation: g1Activation,
            gpoolToBiasMul: gpoolToBiasMul,
            p1BN: p1BN,
            p1Activation: p1Activation,
            p2Conv: p2Conv,
            gpoolToPassMul: gpoolToPassMul,
            gpoolToPassBias: nil,
            passActivation: nil,
            gpoolToPassMul2: nil)
    }
}

// MARK: - Value Head Descriptor

/// A struct that describes the value head of a neural network.
public struct SWValueHeadDesc {
    let version: Int
    let v1Conv: SWConvLayerDesc
    let v1BN: SWBatchNormLayerDesc
    let v1Activation: ActivationKind
    let v2Mul: SWMatMulLayerDesc
    let v2Bias: SWMatBiasLayerDesc
    let v2Activation: ActivationKind
    let v3Mul: SWMatMulLayerDesc
    let v3Bias: SWMatBiasLayerDesc
    let sv3Mul: SWMatMulLayerDesc
    let sv3Bias: SWMatBiasLayerDesc
    let vOwnershipConv: SWConvLayerDesc

    init(
        version: Int,
        v1Conv: SWConvLayerDesc,
        v1BN: SWBatchNormLayerDesc,
        v1Activation: ActivationKind,
        v2Mul: SWMatMulLayerDesc,
        v2Bias: SWMatBiasLayerDesc,
        v2Activation: ActivationKind,
        v3Mul: SWMatMulLayerDesc,
        v3Bias: SWMatBiasLayerDesc,
        sv3Mul: SWMatMulLayerDesc,
        sv3Bias: SWMatBiasLayerDesc,
        vOwnershipConv: SWConvLayerDesc
    ) {
        self.version = version
        self.v1Conv = v1Conv
        self.v1BN = v1BN
        self.v1Activation = v1Activation
        self.v2Mul = v2Mul
        self.v2Bias = v2Bias
        self.v2Activation = v2Activation
        self.v3Mul = v3Mul
        self.v3Bias = v3Bias
        self.sv3Mul = sv3Mul
        self.sv3Bias = sv3Bias
        self.vOwnershipConv = vOwnershipConv
    }
}

public func createSWValueHeadDesc(
    version: Int32,
    v1Conv: SWConvLayerDesc,
    v1BN: SWBatchNormLayerDesc,
    v1Activation: ActivationKind,
    v2Mul: SWMatMulLayerDesc,
    v2Bias: SWMatBiasLayerDesc,
    v2Activation: ActivationKind,
    v3Mul: SWMatMulLayerDesc,
    v3Bias: SWMatBiasLayerDesc,
    sv3Mul: SWMatMulLayerDesc,
    sv3Bias: SWMatBiasLayerDesc,
    vOwnershipConv: SWConvLayerDesc
) -> SWValueHeadDesc {
    return SWValueHeadDesc(
        version: Int(version),
        v1Conv: v1Conv,
        v1BN: v1BN,
        v1Activation: v1Activation,
        v2Mul: v2Mul,
        v2Bias: v2Bias,
        v2Activation: v2Activation,
        v3Mul: v3Mul,
        v3Bias: v3Bias,
        sv3Mul: sv3Mul,
        sv3Bias: sv3Bias,
        vOwnershipConv: vOwnershipConv)
}

// MARK: - Model Descriptor

/// A struct that describes a neural network model for playing the game of Go.
public struct SWModelDesc {
    let version: Int
    let name: String
    let numInputChannels: NSNumber
    let numInputGlobalChannels: NSNumber
    let numInputMetaChannels: NSNumber
    let numValueChannels: NSNumber
    let numScoreValueChannels: NSNumber
    let numOwnershipChannels: NSNumber
    let trunk: SWTrunkDesc
    let policyHead: SWPolicyHeadDesc
    let valueHead: SWValueHeadDesc

    init(
        version: Int,
        name: String,
        numInputChannels: NSNumber,
        numInputGlobalChannels: NSNumber,
        numInputMetaChannels: NSNumber,
        numValueChannels: NSNumber,
        numScoreValueChannels: NSNumber,
        numOwnershipChannels: NSNumber,
        trunk: SWTrunkDesc,
        policyHead: SWPolicyHeadDesc,
        valueHead: SWValueHeadDesc
    ) {
        self.version = version
        self.name = name
        self.numInputChannels = numInputChannels
        self.numInputGlobalChannels = numInputGlobalChannels
        self.numInputMetaChannels = numInputMetaChannels
        self.numValueChannels = numValueChannels
        self.numScoreValueChannels = numScoreValueChannels
        self.numOwnershipChannels = numOwnershipChannels
        self.trunk = trunk
        self.policyHead = policyHead
        self.valueHead = valueHead
    }
}

public func createSWModelDesc(
    version: Int32,
    name: String,
    numInputChannels: Int32,
    numInputGlobalChannels: Int32,
    numInputMetaChannels: Int32,
    numValueChannels: Int32,
    numScoreValueChannels: Int32,
    numOwnershipChannels: Int32,
    trunk: SWTrunkDesc,
    policyHead: SWPolicyHeadDesc,
    valueHead: SWValueHeadDesc
) -> SWModelDesc {
    return SWModelDesc(
        version: Int(version),
        name: name,
        numInputChannels: numInputChannels as NSNumber,
        numInputGlobalChannels: numInputGlobalChannels as NSNumber,
        numInputMetaChannels: numInputMetaChannels as NSNumber,
        numValueChannels: numValueChannels as NSNumber,
        numScoreValueChannels: numScoreValueChannels as NSNumber,
        numOwnershipChannels: numOwnershipChannels as NSNumber,
        trunk: trunk,
        policyHead: policyHead,
        valueHead: valueHead)
}

// MARK: - Compute Context

/// A class that represents context of GPU devices.
public class MetalComputeContext {
    public let nnXLen: Int32
    public let nnYLen: Int32

    init(nnXLen: Int32, nnYLen: Int32) {
        self.nnXLen = nnXLen
        self.nnYLen = nnYLen
    }
}

public func createMetalComputeContext(
    nnXLen: Int32,
    nnYLen: Int32
) -> MetalComputeContext {
    return MetalComputeContext(
        nnXLen: nnXLen,
        nnYLen: nnYLen)
}

// MARK: - Metal Compute Handle

/// A class that represents a handle to the GPU device using pure Metal 4 compute shaders.
/// This implementation provides maximum performance by using direct Metal compute pipelines
/// instead of MPSGraph for all neural network operations.
public class MetalComputeHandle {
    let model: PureMetalModel

    init(model: PureMetalModel) {
        self.model = model
    }

    public func apply(
        input inputPointer: UnsafeMutablePointer<Float32>,
        inputGlobal inputGlobalPointer: UnsafeMutablePointer<Float32>,
        inputMeta inputMetaPointer: UnsafeMutablePointer<Float32>,
        policy: UnsafeMutablePointer<Float32>,
        policyPass: UnsafeMutablePointer<Float32>,
        value: UnsafeMutablePointer<Float32>,
        scoreValue: UnsafeMutablePointer<Float32>,
        ownership: UnsafeMutablePointer<Float32>,
        batchSize: Int
    ) {
        autoreleasepool {
            model.apply(
                input: inputPointer,
                inputGlobal: inputGlobalPointer,
                inputMeta: inputMetaPointer,
                policy: policy,
                policyPass: policyPass,
                value: value,
                scoreValue: scoreValue,
                ownership: ownership,
                batchSize: batchSize)
        }
    }
}

/// Creates a Metal compute handle using pure Metal 4 compute shaders for maximum performance.
public func maybeCreateMetalComputeHandle(
    condition: Bool,
    serverThreadIdx: Int = 0,
    descriptor: SWModelDesc,
    context: MetalComputeContext
) -> MetalComputeHandle? {
    guard condition else { return nil }

    guard let device = MTLCreateSystemDefaultDevice() else {
        printError("Failed to create Metal device")
        return nil
    }

    do {
        let pureModel = try PureMetalModel(
            device: device,
            descriptor: descriptor,
            nnXLen: Int(context.nnXLen),
            nnYLen: Int(context.nnYLen))

        let handle = MetalComputeHandle(model: pureModel)

        printError(
            "Pure Metal 4 backend \(serverThreadIdx): \(device.name), Model version \(descriptor.version) \(descriptor.name), \(context.nnXLen)x\(context.nnYLen)"
        )

        return handle
    } catch {
        printError("Failed to create Pure Metal backend: \(error)")
        return nil
    }
}

/// Prints available Metal devices.
public func printMetalDevices() {
    if let device = MTLCreateSystemDefaultDevice() {
        printError("Found Metal Device: \(device.name)")
    } else {
        printError("No Metal devices found")
    }
}

// MARK: - Enable Option

/// An enum to represent enabled/disabled/auto option of a feature.
public enum SWEnable {
    case False
    case True
    case Auto
}

// MARK: - Test Functions (for unit testing)

public func testConvLayer(
    descriptor: SWConvLayerDesc,
    nnXLen: Int32,
    nnYLen: Int32,
    batchSize: Int32,
    input: UnsafeMutablePointer<Float32>,
    output: UnsafeMutablePointer<Float32>
) {
    guard let device = MTLCreateSystemDefaultDevice() else {
        printError("testConvLayer: No Metal device found")
        return
    }

    do {
        let pipelineManager = try MetalPipelineManager(device: device)
        let dispatcher = MetalComputeDispatcher(pipelineManager: pipelineManager)

        let inChannels = descriptor.inChannels.intValue
        let outChannels = descriptor.outChannels.intValue
        let height = Int(nnYLen)
        let width = Int(nnXLen)
        let batch = Int(batchSize)

        let inputSize = batch * inChannels * height * width * 4
        let outputSize = batch * outChannels * height * width * 4
        let weightSize = outChannels * inChannels * descriptor.convYSize.intValue * descriptor.convXSize.intValue * 4

        guard let inputBuffer = device.makeBuffer(bytes: input, length: inputSize, options: .storageModeShared),
              let weightBuffer = device.makeBuffer(bytes: descriptor.weights, length: weightSize, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: outputSize, options: .storageModeShared) else {
            printError("testConvLayer: Failed to create buffers")
            return
        }

        // Add buffers to residency set and commit BEFORE creating command buffer (required for Metal 4)
        pipelineManager.addToResidencySet(inputBuffer)
        pipelineManager.addToResidencySet(weightBuffer)
        pipelineManager.addToResidencySet(outputBuffer)
        pipelineManager.addToResidencySet(pipelineManager.constantsBuffer)
        pipelineManager.commitResidency()

        guard let commandBuffer = pipelineManager.makeCommandBuffer() else {
            printError("testConvLayer: Failed to create command buffer")
            return
        }

        // Reset constants offset for this command buffer
        dispatcher.resetConstantsOffset()

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            printError("testConvLayer: Failed to create encoder")
            return
        }

        dispatcher.dispatchConv2D(
            encoder: encoder,
            input: inputBuffer,
            weights: weightBuffer,
            output: outputBuffer,
            batchSize: batch,
            inChannels: inChannels,
            outChannels: outChannels,
            height: height,
            width: width,
            kernelH: descriptor.convYSize.intValue,
            kernelW: descriptor.convXSize.intValue,
            dilationH: descriptor.dilationY,
            dilationW: descriptor.dilationX)

        encoder.endEncoding()
        pipelineManager.submit(commandBuffer)
        pipelineManager.waitForCompletion()

        memcpy(output, outputBuffer.contents(), outputSize)
    } catch {
        printError("Test conv layer failed: \(error)")
    }
}

public func testBatchNormLayer(
    descriptor: SWBatchNormLayerDesc,
    nnXLen: Int32,
    nnYLen: Int32,
    batchSize: Int32,
    input: UnsafeMutablePointer<Float32>,
    mask: UnsafeMutablePointer<Float32>,
    output: UnsafeMutablePointer<Float32>
) {
    guard let device = MTLCreateSystemDefaultDevice() else { return }

    do {
        let pipelineManager = try MetalPipelineManager(device: device)
        let dispatcher = MetalComputeDispatcher(pipelineManager: pipelineManager)

        let channels = descriptor.numChannels.intValue
        let height = Int(nnYLen)
        let width = Int(nnXLen)
        let batch = Int(batchSize)

        let inputSize = batch * channels * height * width * 4
        let maskSize = batch * height * width * 4
        let scaleSize = channels * 4

        guard let inputBuffer = device.makeBuffer(bytes: input, length: inputSize, options: .storageModeShared),
              let maskBuffer = device.makeBuffer(bytes: mask, length: maskSize, options: .storageModeShared),
              let scaleBuffer = device.makeBuffer(bytes: descriptor.mergedScale, length: scaleSize, options: .storageModeShared),
              let biasBuffer = device.makeBuffer(bytes: descriptor.mergedBias, length: scaleSize, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: inputSize, options: .storageModeShared) else {
            return
        }

        // Add buffers to residency set and commit BEFORE creating command buffer (required for Metal 4)
        pipelineManager.addToResidencySet(inputBuffer)
        pipelineManager.addToResidencySet(maskBuffer)
        pipelineManager.addToResidencySet(scaleBuffer)
        pipelineManager.addToResidencySet(biasBuffer)
        pipelineManager.addToResidencySet(outputBuffer)
        pipelineManager.addToResidencySet(pipelineManager.constantsBuffer)
        pipelineManager.commitResidency()

        guard let commandBuffer = pipelineManager.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }

        // Reset constants offset for this command buffer
        dispatcher.resetConstantsOffset()

        dispatcher.dispatchBatchNorm(
            encoder: encoder,
            input: inputBuffer,
            scale: scaleBuffer,
            bias: biasBuffer,
            mask: maskBuffer,
            output: outputBuffer,
            batchSize: batch,
            channels: channels,
            height: height,
            width: width,
            activation: .identity)

        encoder.endEncoding()
        pipelineManager.submit(commandBuffer)
        pipelineManager.waitForCompletion()

        memcpy(output, outputBuffer.contents(), inputSize)
    } catch {
        printError("Test batch norm layer failed: \(error)")
    }
}

public func testResidualBlock(
    descriptor: SWResidualBlockDesc,
    batchSize: Int32,
    nnXLen: Int32,
    nnYLen: Int32,
    input: UnsafeMutablePointer<Float32>,
    mask: UnsafeMutablePointer<Float32>,
    output: UnsafeMutablePointer<Float32>
) {
    guard let device = MTLCreateSystemDefaultDevice() else { return }

    do {
        let pipelineManager = try MetalPipelineManager(device: device)
        let dispatcher = MetalComputeDispatcher(pipelineManager: pipelineManager)

        let batch = Int(batchSize)
        let height = Int(nnYLen)
        let width = Int(nnXLen)

        // Extract channel counts from descriptors
        let trunkChannels = descriptor.preBN.numChannels.intValue
        let midChannels = descriptor.regularConv.outChannels.intValue

        // Calculate buffer sizes
        let trunkSize = batch * trunkChannels * height * width * 4
        let midSize = batch * midChannels * height * width * 4
        let maskSize = batch * height * width * 4
        let preBNScaleSize = trunkChannels * 4
        let midBNScaleSize = midChannels * 4
        let regularConvWeightSize = midChannels * trunkChannels * descriptor.regularConv.convYSize.intValue * descriptor.regularConv.convXSize.intValue * 4
        let finalConvWeightSize = trunkChannels * midChannels * descriptor.finalConv.convYSize.intValue * descriptor.finalConv.convXSize.intValue * 4

        // Create buffers
        guard let inputBuffer = device.makeBuffer(bytes: input, length: trunkSize, options: .storageModeShared),
              let maskBuffer = device.makeBuffer(bytes: mask, length: maskSize, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: trunkSize, options: .storageModeShared),
              let bnOutBuffer = device.makeBuffer(length: max(trunkSize, midSize), options: .storageModeShared),
              let convOutBuffer = device.makeBuffer(length: max(trunkSize, midSize), options: .storageModeShared),
              let preBNScaleBuffer = device.makeBuffer(bytes: descriptor.preBN.mergedScale, length: preBNScaleSize, options: .storageModeShared),
              let preBNBiasBuffer = device.makeBuffer(bytes: descriptor.preBN.mergedBias, length: preBNScaleSize, options: .storageModeShared),
              let regularConvWeightBuffer = device.makeBuffer(bytes: descriptor.regularConv.weights, length: regularConvWeightSize, options: .storageModeShared),
              let midBNScaleBuffer = device.makeBuffer(bytes: descriptor.midBN.mergedScale, length: midBNScaleSize, options: .storageModeShared),
              let midBNBiasBuffer = device.makeBuffer(bytes: descriptor.midBN.mergedBias, length: midBNScaleSize, options: .storageModeShared),
              let finalConvWeightBuffer = device.makeBuffer(bytes: descriptor.finalConv.weights, length: finalConvWeightSize, options: .storageModeShared) else {
            printError("testResidualBlock: Failed to create buffers")
            return
        }

        // Add buffers to residency set
        pipelineManager.addToResidencySet(inputBuffer)
        pipelineManager.addToResidencySet(maskBuffer)
        pipelineManager.addToResidencySet(outputBuffer)
        pipelineManager.addToResidencySet(bnOutBuffer)
        pipelineManager.addToResidencySet(convOutBuffer)
        pipelineManager.addToResidencySet(preBNScaleBuffer)
        pipelineManager.addToResidencySet(preBNBiasBuffer)
        pipelineManager.addToResidencySet(regularConvWeightBuffer)
        pipelineManager.addToResidencySet(midBNScaleBuffer)
        pipelineManager.addToResidencySet(midBNBiasBuffer)
        pipelineManager.addToResidencySet(finalConvWeightBuffer)
        pipelineManager.addToResidencySet(pipelineManager.constantsBuffer)
        pipelineManager.commitResidency()

        guard let commandBuffer = pipelineManager.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            printError("testResidualBlock: Failed to create command buffer or encoder")
            return
        }

        dispatcher.resetConstantsOffset()

        // Pre BN + activation: input -> bnOut
        dispatcher.dispatchBatchNorm(
            encoder: encoder,
            input: inputBuffer,
            scale: preBNScaleBuffer,
            bias: preBNBiasBuffer,
            mask: maskBuffer,
            output: bnOutBuffer,
            batchSize: batch,
            channels: trunkChannels,
            height: height,
            width: width,
            activation: descriptor.preActivation)

        // Barrier to ensure preBN completes before regularConv reads
        dispatcher.insertBarrier(encoder: encoder)

        // Regular conv: bnOut -> convOut
        dispatcher.dispatchConv2D(
            encoder: encoder,
            input: bnOutBuffer,
            weights: regularConvWeightBuffer,
            output: convOutBuffer,
            batchSize: batch,
            inChannels: trunkChannels,
            outChannels: midChannels,
            height: height,
            width: width,
            kernelH: descriptor.regularConv.convYSize.intValue,
            kernelW: descriptor.regularConv.convXSize.intValue,
            dilationH: descriptor.regularConv.dilationY,
            dilationW: descriptor.regularConv.dilationX)

        // Barrier to ensure regularConv completes before midBN reads
        dispatcher.insertBarrier(encoder: encoder)

        // Mid BN + activation: convOut -> bnOut
        dispatcher.dispatchBatchNorm(
            encoder: encoder,
            input: convOutBuffer,
            scale: midBNScaleBuffer,
            bias: midBNBiasBuffer,
            mask: maskBuffer,
            output: bnOutBuffer,
            batchSize: batch,
            channels: midChannels,
            height: height,
            width: width,
            activation: descriptor.midActivation)

        // Barrier to ensure midBN completes before finalConv reads
        dispatcher.insertBarrier(encoder: encoder)

        // Final conv: bnOut -> convOut
        dispatcher.dispatchConv2D(
            encoder: encoder,
            input: bnOutBuffer,
            weights: finalConvWeightBuffer,
            output: convOutBuffer,
            batchSize: batch,
            inChannels: midChannels,
            outChannels: trunkChannels,
            height: height,
            width: width,
            kernelH: descriptor.finalConv.convYSize.intValue,
            kernelW: descriptor.finalConv.convXSize.intValue,
            dilationH: descriptor.finalConv.dilationY,
            dilationW: descriptor.finalConv.dilationX)

        // Barrier to ensure finalConv completes before residual add reads
        dispatcher.insertBarrier(encoder: encoder)

        // Residual add: input + convOut -> output
        dispatcher.dispatchElementwiseAdd(
            encoder: encoder,
            a: inputBuffer,
            b: convOutBuffer,
            output: outputBuffer,
            size: batch * trunkChannels * height * width)

        encoder.endEncoding()
        pipelineManager.submit(commandBuffer)
        pipelineManager.waitForCompletion()

        memcpy(output, outputBuffer.contents(), trunkSize)
    } catch {
        printError("testResidualBlock failed: \(error)")
    }
}

public func testGlobalPoolingResidualBlock(
    descriptor: SWGlobalPoolingResidualBlockDesc,
    batchSize: Int32,
    nnXLen: Int32,
    nnYLen: Int32,
    input: UnsafeMutablePointer<Float32>,
    mask: UnsafeMutablePointer<Float32>,
    output: UnsafeMutablePointer<Float32>
) {
    guard let device = MTLCreateSystemDefaultDevice() else { return }

    do {
        let pipelineManager = try MetalPipelineManager(device: device)
        let dispatcher = MetalComputeDispatcher(pipelineManager: pipelineManager)

        let batch = Int(batchSize)
        let height = Int(nnYLen)
        let width = Int(nnXLen)

        // Extract channel counts from descriptors
        let trunkChannels = descriptor.preBN.numChannels.intValue
        let regularConvOutChannels = descriptor.regularConv.outChannels.intValue
        let gpoolConvOutChannels = descriptor.gpoolConv.outChannels.intValue
        let gpoolToBiasInChannels = descriptor.gpoolToBiasMul.inChannels.intValue
        let gpoolToBiasOutChannels = descriptor.gpoolToBiasMul.outChannels.intValue
        let midChannels = descriptor.midBN.numChannels.intValue

        // Calculate buffer sizes
        let trunkSize = batch * trunkChannels * height * width * 4
        let regularConvOutSize = batch * regularConvOutChannels * height * width * 4
        let gpoolConvOutSize = batch * gpoolConvOutChannels * height * width * 4
        let gpoolOutSize = batch * gpoolToBiasInChannels * 4  // global pooling output (3 values per channel)
        let matmulOutSize = batch * gpoolToBiasOutChannels * 4
        let midSize = batch * midChannels * height * width * 4
        let maskSize = batch * height * width * 4
        let maskSumSize = batch * 4

        // Weight buffer sizes
        let preBNScaleSize = trunkChannels * 4
        let regularConvWeightSize = regularConvOutChannels * trunkChannels * descriptor.regularConv.convYSize.intValue * descriptor.regularConv.convXSize.intValue * 4
        let gpoolConvWeightSize = gpoolConvOutChannels * trunkChannels * descriptor.gpoolConv.convYSize.intValue * descriptor.gpoolConv.convXSize.intValue * 4
        let gpoolBNScaleSize = gpoolConvOutChannels * 4
        let gpoolToBiasMulWeightSize = gpoolToBiasInChannels * gpoolToBiasOutChannels * 4
        let midBNScaleSize = midChannels * 4
        let finalConvWeightSize = trunkChannels * midChannels * descriptor.finalConv.convYSize.intValue * descriptor.finalConv.convXSize.intValue * 4

        // Create buffers
        guard let inputBuffer = device.makeBuffer(bytes: input, length: trunkSize, options: .storageModeShared),
              let maskBuffer = device.makeBuffer(bytes: mask, length: maskSize, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: trunkSize, options: .storageModeShared),
              let bnOutBuffer = device.makeBuffer(length: max(trunkSize, regularConvOutSize, gpoolConvOutSize), options: .storageModeShared),
              let convOutBuffer = device.makeBuffer(length: max(trunkSize, regularConvOutSize, midSize), options: .storageModeShared),
              let gpoolConvOutBuffer = device.makeBuffer(length: gpoolConvOutSize, options: .storageModeShared),
              let gpoolOutBuffer = device.makeBuffer(length: gpoolOutSize, options: .storageModeShared),
              let matmulOutBuffer = device.makeBuffer(length: matmulOutSize, options: .storageModeShared),
              let maskSumBuffer = device.makeBuffer(length: maskSumSize, options: .storageModeShared),
              let maskSumSqrtS14M01Buffer = device.makeBuffer(length: maskSumSize, options: .storageModeShared),
              let preBNScaleBuffer = device.makeBuffer(bytes: descriptor.preBN.mergedScale, length: preBNScaleSize, options: .storageModeShared),
              let preBNBiasBuffer = device.makeBuffer(bytes: descriptor.preBN.mergedBias, length: preBNScaleSize, options: .storageModeShared),
              let regularConvWeightBuffer = device.makeBuffer(bytes: descriptor.regularConv.weights, length: regularConvWeightSize, options: .storageModeShared),
              let gpoolConvWeightBuffer = device.makeBuffer(bytes: descriptor.gpoolConv.weights, length: gpoolConvWeightSize, options: .storageModeShared),
              let gpoolBNScaleBuffer = device.makeBuffer(bytes: descriptor.gpoolBN.mergedScale, length: gpoolBNScaleSize, options: .storageModeShared),
              let gpoolBNBiasBuffer = device.makeBuffer(bytes: descriptor.gpoolBN.mergedBias, length: gpoolBNScaleSize, options: .storageModeShared),
              let gpoolToBiasMulWeightBuffer = device.makeBuffer(bytes: descriptor.gpoolToBiasMul.weights, length: gpoolToBiasMulWeightSize, options: .storageModeShared),
              let midBNScaleBuffer = device.makeBuffer(bytes: descriptor.midBN.mergedScale, length: midBNScaleSize, options: .storageModeShared),
              let midBNBiasBuffer = device.makeBuffer(bytes: descriptor.midBN.mergedBias, length: midBNScaleSize, options: .storageModeShared),
              let finalConvWeightBuffer = device.makeBuffer(bytes: descriptor.finalConv.weights, length: finalConvWeightSize, options: .storageModeShared) else {
            printError("testGlobalPoolingResidualBlock: Failed to create buffers")
            return
        }

        // Add buffers to residency set
        pipelineManager.addToResidencySet(inputBuffer)
        pipelineManager.addToResidencySet(maskBuffer)
        pipelineManager.addToResidencySet(outputBuffer)
        pipelineManager.addToResidencySet(bnOutBuffer)
        pipelineManager.addToResidencySet(convOutBuffer)
        pipelineManager.addToResidencySet(gpoolConvOutBuffer)
        pipelineManager.addToResidencySet(gpoolOutBuffer)
        pipelineManager.addToResidencySet(matmulOutBuffer)
        pipelineManager.addToResidencySet(maskSumBuffer)
        pipelineManager.addToResidencySet(maskSumSqrtS14M01Buffer)
        pipelineManager.addToResidencySet(preBNScaleBuffer)
        pipelineManager.addToResidencySet(preBNBiasBuffer)
        pipelineManager.addToResidencySet(regularConvWeightBuffer)
        pipelineManager.addToResidencySet(gpoolConvWeightBuffer)
        pipelineManager.addToResidencySet(gpoolBNScaleBuffer)
        pipelineManager.addToResidencySet(gpoolBNBiasBuffer)
        pipelineManager.addToResidencySet(gpoolToBiasMulWeightBuffer)
        pipelineManager.addToResidencySet(midBNScaleBuffer)
        pipelineManager.addToResidencySet(midBNBiasBuffer)
        pipelineManager.addToResidencySet(finalConvWeightBuffer)
        pipelineManager.addToResidencySet(pipelineManager.constantsBuffer)
        pipelineManager.commitResidency()

        guard let commandBuffer = pipelineManager.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            printError("testGlobalPoolingResidualBlock: Failed to create command buffer or encoder")
            return
        }

        dispatcher.resetConstantsOffset()

        // Compute mask sum and transformed mask sum
        dispatcher.dispatchMaskSum(encoder: encoder, mask: maskBuffer, maskSum: maskSumBuffer, batchSize: batch, height: height, width: width)

        // Barrier: maskSum must complete before maskSumSqrtS14M01
        dispatcher.insertBarrier(encoder: encoder)

        dispatcher.dispatchMaskSumSqrtS14M01(encoder: encoder, maskSum: maskSumBuffer, output: maskSumSqrtS14M01Buffer, batchSize: batch)

        // Pre BN + activation: input -> bnOut
        dispatcher.dispatchBatchNorm(
            encoder: encoder,
            input: inputBuffer,
            scale: preBNScaleBuffer,
            bias: preBNBiasBuffer,
            mask: maskBuffer,
            output: bnOutBuffer,
            batchSize: batch,
            channels: trunkChannels,
            height: height,
            width: width,
            activation: descriptor.preActivation)

        // Barrier: preBN must complete before regularConv and gpoolConv
        dispatcher.insertBarrier(encoder: encoder)

        // Regular conv: bnOut -> convOut
        dispatcher.dispatchConv2D(
            encoder: encoder,
            input: bnOutBuffer,
            weights: regularConvWeightBuffer,
            output: convOutBuffer,
            batchSize: batch,
            inChannels: trunkChannels,
            outChannels: regularConvOutChannels,
            height: height,
            width: width,
            kernelH: descriptor.regularConv.convYSize.intValue,
            kernelW: descriptor.regularConv.convXSize.intValue,
            dilationH: descriptor.regularConv.dilationY,
            dilationW: descriptor.regularConv.dilationX)

        // Global pooling branch
        // Gpool conv: bnOut -> gpoolConvOut
        dispatcher.dispatchConv2D(
            encoder: encoder,
            input: bnOutBuffer,
            weights: gpoolConvWeightBuffer,
            output: gpoolConvOutBuffer,
            batchSize: batch,
            inChannels: trunkChannels,
            outChannels: gpoolConvOutChannels,
            height: height,
            width: width,
            kernelH: descriptor.gpoolConv.convYSize.intValue,
            kernelW: descriptor.gpoolConv.convXSize.intValue,
            dilationH: descriptor.gpoolConv.dilationY,
            dilationW: descriptor.gpoolConv.dilationX)

        // Barrier: gpoolConv must complete before gpoolBN
        dispatcher.insertBarrier(encoder: encoder)

        // Gpool BN + activation: gpoolConvOut -> gpoolConvOut (in-place)
        dispatcher.dispatchBatchNorm(
            encoder: encoder,
            input: gpoolConvOutBuffer,
            scale: gpoolBNScaleBuffer,
            bias: gpoolBNBiasBuffer,
            mask: maskBuffer,
            output: gpoolConvOutBuffer,
            batchSize: batch,
            channels: gpoolConvOutChannels,
            height: height,
            width: width,
            activation: descriptor.gpoolActivation)

        // Barrier: gpoolBN and maskSumSqrtS14M01 must complete before globalPooling
        dispatcher.insertBarrier(encoder: encoder)

        // Global pooling: gpoolConvOut -> gpoolOut
        dispatcher.dispatchGlobalPooling(
            encoder: encoder,
            input: gpoolConvOutBuffer,
            mask: maskBuffer,
            maskSum: maskSumBuffer,
            maskSumSqrtS14M01: maskSumSqrtS14M01Buffer,
            output: gpoolOutBuffer,
            batchSize: batch,
            channels: gpoolConvOutChannels,
            height: height,
            width: width)

        // Barrier: globalPooling must complete before matmul
        dispatcher.insertBarrier(encoder: encoder)

        // Gpool to bias matmul: gpoolOut -> matmulOut
        dispatcher.dispatchMatMul(
            encoder: encoder,
            A: gpoolOutBuffer,
            B: gpoolToBiasMulWeightBuffer,
            C: matmulOutBuffer,
            M: batch,
            K: gpoolToBiasInChannels,
            N: gpoolToBiasOutChannels)

        // Barrier: matmul and regularConv must complete before addNCBias
        dispatcher.insertBarrier(encoder: encoder)

        // Add bias to regular conv output: convOut + matmulOut -> convOut
        dispatcher.dispatchAddNCBias(
            encoder: encoder,
            input: convOutBuffer,
            bias: matmulOutBuffer,
            output: convOutBuffer,
            batchSize: batch,
            channels: gpoolToBiasOutChannels,
            height: height,
            width: width)

        // Barrier: addNCBias must complete before midBN
        dispatcher.insertBarrier(encoder: encoder)

        // Mid BN + activation: convOut -> bnOut
        dispatcher.dispatchBatchNorm(
            encoder: encoder,
            input: convOutBuffer,
            scale: midBNScaleBuffer,
            bias: midBNBiasBuffer,
            mask: maskBuffer,
            output: bnOutBuffer,
            batchSize: batch,
            channels: midChannels,
            height: height,
            width: width,
            activation: descriptor.midActivation)

        // Barrier: midBN must complete before finalConv
        dispatcher.insertBarrier(encoder: encoder)

        // Final conv: bnOut -> convOut
        dispatcher.dispatchConv2D(
            encoder: encoder,
            input: bnOutBuffer,
            weights: finalConvWeightBuffer,
            output: convOutBuffer,
            batchSize: batch,
            inChannels: midChannels,
            outChannels: trunkChannels,
            height: height,
            width: width,
            kernelH: descriptor.finalConv.convYSize.intValue,
            kernelW: descriptor.finalConv.convXSize.intValue,
            dilationH: descriptor.finalConv.dilationY,
            dilationW: descriptor.finalConv.dilationX)

        // Barrier: finalConv must complete before residualAdd
        dispatcher.insertBarrier(encoder: encoder)

        // Residual add: input + convOut -> output
        dispatcher.dispatchElementwiseAdd(
            encoder: encoder,
            a: inputBuffer,
            b: convOutBuffer,
            output: outputBuffer,
            size: batch * trunkChannels * height * width)

        encoder.endEncoding()
        pipelineManager.submit(commandBuffer)
        pipelineManager.waitForCompletion()

        memcpy(output, outputBuffer.contents(), trunkSize)
    } catch {
        printError("testGlobalPoolingResidualBlock failed: \(error)")
    }
}
