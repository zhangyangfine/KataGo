//
//  metalbackend_model.swift
//  Pure Metal 4 Model implementation for KataGo
//

import Foundation
import Metal

// MARK: - Weight Buffers

/// Holds all weight buffers for a convolution layer
struct ConvWeightBuffers {
    let weights: MTLBuffer
    let inChannels: Int
    let outChannels: Int
    let kernelH: Int
    let kernelW: Int
    let dilationH: Int
    let dilationW: Int
}

/// Holds all weight buffers for a batch norm layer
struct BatchNormWeightBuffers {
    let scale: MTLBuffer
    let bias: MTLBuffer
    let channels: Int
}

/// Holds all weight buffers for a matmul layer
struct MatMulWeightBuffers {
    let weights: MTLBuffer
    let inChannels: Int
    let outChannels: Int
}

/// Holds all weight buffers for a bias layer
struct BiasWeightBuffers {
    let bias: MTLBuffer
    let channels: Int
}

// MARK: - Pure Metal Model

/// Pure Metal implementation of the KataGo neural network model
class PureMetalModel {
    let device: MTLDevice
    let pipelineManager: MetalPipelineManager
    let dispatcher: MetalComputeDispatcher
    let bufferManager: MetalBufferManager

    let nnXLen: Int
    let nnYLen: Int
    let version: Int
    let maxBatchSize: Int

    // Model descriptor
    let descriptor: SWModelDesc

    // Weight buffers
    var convWeights: [String: ConvWeightBuffers] = [:]
    var bnWeights: [String: BatchNormWeightBuffers] = [:]
    var matmulWeights: [String: MatMulWeightBuffers] = [:]
    var biasWeights: [String: BiasWeightBuffers] = [:]

    // Intermediate buffers (allocated per batch)
    var intermediateBuffers: [String: MTLBuffer] = [:]

    init(device: MTLDevice, descriptor: SWModelDesc, nnXLen: Int, nnYLen: Int, maxBatchSize: Int = 16) throws {
        self.device = device
        self.pipelineManager = try MetalPipelineManager(device: device)
        self.dispatcher = MetalComputeDispatcher(pipelineManager: pipelineManager)
        self.bufferManager = MetalBufferManager(device: device)
        self.nnXLen = nnXLen
        self.nnYLen = nnYLen
        self.version = descriptor.version
        self.maxBatchSize = maxBatchSize
        self.descriptor = descriptor

        // Load weights from descriptor
        try loadWeights()

        // Allocate intermediate buffers
        allocateIntermediateBuffers()
    }

    private func loadWeights() throws {
        let trunk = descriptor.trunk

        // Initial conv
        convWeights["trunk.initialConv"] = try loadConvWeights(trunk.initialConv, name: "trunk.initialConv")

        // Initial matmul
        matmulWeights["trunk.initialMatMul"] = try loadMatMulWeights(trunk.initialMatMul, name: "trunk.initialMatMul")

        // SGF metadata encoder (if present)
        if let sgf = trunk.sgfMetadataEncoder, sgf.numInputMetaChannels > 0 {
            matmulWeights["trunk.sgf.mul1"] = try loadMatMulWeights(sgf.mul1, name: "trunk.sgf.mul1")
            biasWeights["trunk.sgf.bias1"] = try loadBiasWeights(sgf.bias1, name: "trunk.sgf.bias1")
            matmulWeights["trunk.sgf.mul2"] = try loadMatMulWeights(sgf.mul2, name: "trunk.sgf.mul2")
            biasWeights["trunk.sgf.bias2"] = try loadBiasWeights(sgf.bias2, name: "trunk.sgf.bias2")
            matmulWeights["trunk.sgf.mul3"] = try loadMatMulWeights(sgf.mul3, name: "trunk.sgf.mul3")
        }

        // Residual blocks
        for (i, block) in trunk.blockDescriptors.enumerated() {
            try loadBlockWeights(block, index: i)
        }

        // Trunk tip
        bnWeights["trunk.tipBN"] = try loadBNWeights(trunk.trunkTipBN, name: "trunk.tipBN")

        // Policy head
        let policy = descriptor.policyHead
        convWeights["policy.p1Conv"] = try loadConvWeights(policy.p1Conv, name: "policy.p1Conv")
        convWeights["policy.g1Conv"] = try loadConvWeights(policy.g1Conv, name: "policy.g1Conv")
        bnWeights["policy.g1BN"] = try loadBNWeights(policy.g1BN, name: "policy.g1BN")
        matmulWeights["policy.gpoolToBiasMul"] = try loadMatMulWeights(policy.gpoolToBiasMul, name: "policy.gpoolToBiasMul")
        bnWeights["policy.p1BN"] = try loadBNWeights(policy.p1BN, name: "policy.p1BN")
        convWeights["policy.p2Conv"] = try loadConvWeights(policy.p2Conv, name: "policy.p2Conv")
        matmulWeights["policy.gpoolToPassMul"] = try loadMatMulWeights(policy.gpoolToPassMul, name: "policy.gpoolToPassMul")

        if let gpoolToPassBias = policy.gpoolToPassBias {
            biasWeights["policy.gpoolToPassBias"] = try loadBiasWeights(gpoolToPassBias, name: "policy.gpoolToPassBias")
        }
        if let gpoolToPassMul2 = policy.gpoolToPassMul2 {
            matmulWeights["policy.gpoolToPassMul2"] = try loadMatMulWeights(gpoolToPassMul2, name: "policy.gpoolToPassMul2")
        }

        // Value head
        let value = descriptor.valueHead
        convWeights["value.v1Conv"] = try loadConvWeights(value.v1Conv, name: "value.v1Conv")
        bnWeights["value.v1BN"] = try loadBNWeights(value.v1BN, name: "value.v1BN")
        matmulWeights["value.v2Mul"] = try loadMatMulWeights(value.v2Mul, name: "value.v2Mul")
        biasWeights["value.v2Bias"] = try loadBiasWeights(value.v2Bias, name: "value.v2Bias")
        matmulWeights["value.v3Mul"] = try loadMatMulWeights(value.v3Mul, name: "value.v3Mul")
        biasWeights["value.v3Bias"] = try loadBiasWeights(value.v3Bias, name: "value.v3Bias")
        matmulWeights["value.sv3Mul"] = try loadMatMulWeights(value.sv3Mul, name: "value.sv3Mul")
        biasWeights["value.sv3Bias"] = try loadBiasWeights(value.sv3Bias, name: "value.sv3Bias")
        convWeights["value.vOwnershipConv"] = try loadConvWeights(value.vOwnershipConv, name: "value.vOwnershipConv")
    }

    private func loadBlockWeights(_ block: BlockDescriptor, index: Int) throws {
        let prefix = "trunk.block\(index)"

        if let resBlock = block as? SWResidualBlockDesc {
            bnWeights["\(prefix).preBN"] = try loadBNWeights(resBlock.preBN, name: "\(prefix).preBN")
            convWeights["\(prefix).regularConv"] = try loadConvWeights(resBlock.regularConv, name: "\(prefix).regularConv")
            bnWeights["\(prefix).midBN"] = try loadBNWeights(resBlock.midBN, name: "\(prefix).midBN")
            convWeights["\(prefix).finalConv"] = try loadConvWeights(resBlock.finalConv, name: "\(prefix).finalConv")
        } else if let gpoolBlock = block as? SWGlobalPoolingResidualBlockDesc {
            bnWeights["\(prefix).preBN"] = try loadBNWeights(gpoolBlock.preBN, name: "\(prefix).preBN")
            convWeights["\(prefix).regularConv"] = try loadConvWeights(gpoolBlock.regularConv, name: "\(prefix).regularConv")
            convWeights["\(prefix).gpoolConv"] = try loadConvWeights(gpoolBlock.gpoolConv, name: "\(prefix).gpoolConv")
            bnWeights["\(prefix).gpoolBN"] = try loadBNWeights(gpoolBlock.gpoolBN, name: "\(prefix).gpoolBN")
            matmulWeights["\(prefix).gpoolToBiasMul"] = try loadMatMulWeights(gpoolBlock.gpoolToBiasMul, name: "\(prefix).gpoolToBiasMul")
            bnWeights["\(prefix).midBN"] = try loadBNWeights(gpoolBlock.midBN, name: "\(prefix).midBN")
            convWeights["\(prefix).finalConv"] = try loadConvWeights(gpoolBlock.finalConv, name: "\(prefix).finalConv")
        } else if let nestedBlock = block as? SWNestedBottleneckResidualBlockDesc {
            bnWeights["\(prefix).preBN"] = try loadBNWeights(nestedBlock.preBN, name: "\(prefix).preBN")
            convWeights["\(prefix).preConv"] = try loadConvWeights(nestedBlock.preConv, name: "\(prefix).preConv")
            for (j, innerBlock) in nestedBlock.blockDescriptors.enumerated() {
                try loadBlockWeights(innerBlock, index: index * 100 + j)
            }
            bnWeights["\(prefix).postBN"] = try loadBNWeights(nestedBlock.postBN, name: "\(prefix).postBN")
            convWeights["\(prefix).postConv"] = try loadConvWeights(nestedBlock.postConv, name: "\(prefix).postConv")
        }
    }

    private func loadConvWeights(_ desc: SWConvLayerDesc, name: String) throws -> ConvWeightBuffers {
        let size = desc.outChannels.intValue * desc.inChannels.intValue *
                   desc.convYSize.intValue * desc.convXSize.intValue * 4
        guard let buffer = device.makeBuffer(bytes: desc.weights, length: size, options: .storageModeShared) else {
            throw MetalError.bufferCreationFailed
        }
        return ConvWeightBuffers(
            weights: buffer,
            inChannels: desc.inChannels.intValue,
            outChannels: desc.outChannels.intValue,
            kernelH: desc.convYSize.intValue,
            kernelW: desc.convXSize.intValue,
            dilationH: desc.dilationY,
            dilationW: desc.dilationX
        )
    }

    private func loadBNWeights(_ desc: SWBatchNormLayerDesc, name: String) throws -> BatchNormWeightBuffers {
        let size = desc.numChannels.intValue * 4
        guard let scaleBuffer = device.makeBuffer(bytes: desc.mergedScale, length: size, options: .storageModeShared),
              let biasBuffer = device.makeBuffer(bytes: desc.mergedBias, length: size, options: .storageModeShared) else {
            throw MetalError.bufferCreationFailed
        }
        return BatchNormWeightBuffers(
            scale: scaleBuffer,
            bias: biasBuffer,
            channels: desc.numChannels.intValue
        )
    }

    private func loadMatMulWeights(_ desc: SWMatMulLayerDesc, name: String) throws -> MatMulWeightBuffers {
        let size = desc.inChannels.intValue * desc.outChannels.intValue * 4
        guard let buffer = device.makeBuffer(bytes: desc.weights, length: size, options: .storageModeShared) else {
            throw MetalError.bufferCreationFailed
        }
        return MatMulWeightBuffers(
            weights: buffer,
            inChannels: desc.inChannels.intValue,
            outChannels: desc.outChannels.intValue
        )
    }

    private func loadBiasWeights(_ desc: SWMatBiasLayerDesc, name: String) throws -> BiasWeightBuffers {
        let size = desc.numChannels.intValue * 4
        guard let buffer = device.makeBuffer(bytes: desc.weights, length: size, options: .storageModeShared) else {
            throw MetalError.bufferCreationFailed
        }
        return BiasWeightBuffers(
            bias: buffer,
            channels: desc.numChannels.intValue
        )
    }

    private func allocateIntermediateBuffers() {
        let hw = nnXLen * nnYLen
        let trunkChannels = descriptor.trunk.trunkNumChannels.intValue

        // Calculate required buffer sizes
        let spatialSize = maxBatchSize * trunkChannels * hw * 4
        let globalSize = maxBatchSize * trunkChannels * 3 * 4  // For global pooling concat

        // Allocate reusable intermediate buffers
        intermediateBuffers["trunk0"] = device.makeBuffer(length: spatialSize, options: .storageModeShared)
        intermediateBuffers["trunk1"] = device.makeBuffer(length: spatialSize, options: .storageModeShared)
        intermediateBuffers["conv_out"] = device.makeBuffer(length: spatialSize, options: .storageModeShared)
        intermediateBuffers["bn_out"] = device.makeBuffer(length: spatialSize, options: .storageModeShared)
        intermediateBuffers["gpool_out"] = device.makeBuffer(length: globalSize, options: .storageModeShared)
        intermediateBuffers["matmul_out"] = device.makeBuffer(length: maxBatchSize * trunkChannels * 4, options: .storageModeShared)

        // Mask-related buffers
        intermediateBuffers["mask"] = device.makeBuffer(length: maxBatchSize * hw * 4, options: .storageModeShared)
        intermediateBuffers["maskSum"] = device.makeBuffer(length: maxBatchSize * 4, options: .storageModeShared)
        intermediateBuffers["maskSumSqrtS14M01"] = device.makeBuffer(length: maxBatchSize * 4, options: .storageModeShared)
        intermediateBuffers["maskSumSqrtS14M01SquareS01"] = device.makeBuffer(length: maxBatchSize * 4, options: .storageModeShared)

        // Policy head buffers
        let policyChannels = descriptor.policyHead.p2Conv.outChannels.intValue
        intermediateBuffers["policy_p1"] = device.makeBuffer(length: maxBatchSize * descriptor.policyHead.p1Conv.outChannels.intValue * hw * 4, options: .storageModeShared)
        intermediateBuffers["policy_g1"] = device.makeBuffer(length: maxBatchSize * descriptor.policyHead.g1Conv.outChannels.intValue * hw * 4, options: .storageModeShared)
        intermediateBuffers["policy_gpool"] = device.makeBuffer(length: maxBatchSize * descriptor.policyHead.gpoolToBiasMul.inChannels.intValue * 4, options: .storageModeShared)

        // Value head buffers
        let valueChannels = descriptor.valueHead.v1Conv.outChannels.intValue
        intermediateBuffers["value_v1"] = device.makeBuffer(length: maxBatchSize * valueChannels * hw * 4, options: .storageModeShared)
        intermediateBuffers["value_gpool"] = device.makeBuffer(length: maxBatchSize * descriptor.valueHead.v2Mul.inChannels.intValue * 4, options: .storageModeShared)
        intermediateBuffers["value_v2"] = device.makeBuffer(length: maxBatchSize * descriptor.valueHead.v2Mul.outChannels.intValue * 4, options: .storageModeShared)
    }

    /// Apply the model to input data
    func apply(
        input: UnsafeMutablePointer<Float32>,
        inputGlobal: UnsafeMutablePointer<Float32>,
        inputMeta: UnsafeMutablePointer<Float32>,
        policy: UnsafeMutablePointer<Float32>,
        policyPass: UnsafeMutablePointer<Float32>,
        value: UnsafeMutablePointer<Float32>,
        scoreValue: UnsafeMutablePointer<Float32>,
        ownership: UnsafeMutablePointer<Float32>,
        batchSize: Int
    ) {
        guard let commandBuffer = pipelineManager.commandQueue.makeCommandBuffer() else {
            return
        }

        let hw = nnXLen * nnYLen
        let inputChannels = descriptor.numInputChannels.intValue
        let globalChannels = descriptor.numInputGlobalChannels.intValue
        let metaChannels = descriptor.numInputMetaChannels.intValue

        // Create input buffers
        let inputSize = batchSize * inputChannels * hw * 4
        let globalSize = batchSize * globalChannels * 4
        let metaSize = batchSize * metaChannels * 4

        guard let inputBuffer = device.makeBuffer(bytes: input, length: inputSize, options: .storageModeShared),
              let globalBuffer = device.makeBuffer(bytes: inputGlobal, length: globalSize, options: .storageModeShared),
              let metaBuffer = device.makeBuffer(bytes: inputMeta, length: metaSize, options: .storageModeShared) else {
            return
        }

        // Extract mask from first channel of input
        let maskBuffer = intermediateBuffers["mask"]!
        extractMask(input: inputBuffer, mask: maskBuffer, batchSize: batchSize, channels: inputChannels, hw: hw)

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }

        // Compute mask statistics
        computeMaskStatistics(encoder: encoder, batchSize: batchSize)

        // Run trunk
        let trunkOutput = runTrunk(encoder: encoder, input: inputBuffer, global: globalBuffer, meta: metaBuffer, batchSize: batchSize)

        // Run policy head
        runPolicyHead(encoder: encoder, trunkOutput: trunkOutput, policy: policy, policyPass: policyPass, batchSize: batchSize)

        // Run value head
        runValueHead(encoder: encoder, trunkOutput: trunkOutput, value: value, scoreValue: scoreValue, ownership: ownership, batchSize: batchSize)

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    private func extractMask(input: MTLBuffer, mask: MTLBuffer, batchSize: Int, channels: Int, hw: Int) {
        // Copy first channel from input to mask (stride copy)
        let inputPtr = input.contents().assumingMemoryBound(to: Float32.self)
        let maskPtr = mask.contents().assumingMemoryBound(to: Float32.self)

        for b in 0..<batchSize {
            for i in 0..<hw {
                maskPtr[b * hw + i] = inputPtr[b * channels * hw + i]
            }
        }
    }

    private func computeMaskStatistics(encoder: MTLComputeCommandEncoder, batchSize: Int) {
        let mask = intermediateBuffers["mask"]!
        let maskSum = intermediateBuffers["maskSum"]!
        let maskSumSqrtS14M01 = intermediateBuffers["maskSumSqrtS14M01"]!
        let maskSumSqrtS14M01SquareS01 = intermediateBuffers["maskSumSqrtS14M01SquareS01"]!

        dispatcher.dispatchMaskSum(encoder: encoder, mask: mask, maskSum: maskSum, batchSize: batchSize, height: nnYLen, width: nnXLen)
        dispatcher.dispatchMaskSumSqrtS14M01(encoder: encoder, maskSum: maskSum, output: maskSumSqrtS14M01, batchSize: batchSize)
        dispatcher.dispatchMaskSumSqrtS14M01SquareS01(encoder: encoder, maskSumSqrtS14M01: maskSumSqrtS14M01, output: maskSumSqrtS14M01SquareS01, batchSize: batchSize)
    }

    private func runTrunk(encoder: MTLComputeCommandEncoder, input: MTLBuffer, global: MTLBuffer, meta: MTLBuffer, batchSize: Int) -> MTLBuffer {
        let trunk = descriptor.trunk
        let trunkChannels = trunk.trunkNumChannels.intValue
        let mask = intermediateBuffers["mask"]!

        // Initial conv
        let initialConvW = convWeights["trunk.initialConv"]!
        var currentBuffer = intermediateBuffers["trunk0"]!

        dispatcher.dispatchConv2D(
            encoder: encoder,
            input: input,
            weights: initialConvW.weights,
            output: currentBuffer,
            batchSize: batchSize,
            inChannels: initialConvW.inChannels,
            outChannels: initialConvW.outChannels,
            height: nnYLen,
            width: nnXLen,
            kernelH: initialConvW.kernelH,
            kernelW: initialConvW.kernelW
        )

        // Initial matmul + add bias
        let initialMatMulW = matmulWeights["trunk.initialMatMul"]!
        let matmulOut = intermediateBuffers["matmul_out"]!

        dispatcher.dispatchMatMul(
            encoder: encoder,
            A: global,
            B: initialMatMulW.weights,
            C: matmulOut,
            M: batchSize,
            K: initialMatMulW.inChannels,
            N: initialMatMulW.outChannels
        )

        dispatcher.dispatchAddNCBias(
            encoder: encoder,
            input: currentBuffer,
            bias: matmulOut,
            output: currentBuffer,
            batchSize: batchSize,
            channels: trunkChannels,
            height: nnYLen,
            width: nnXLen
        )

        // Process residual blocks
        var nextBuffer = intermediateBuffers["trunk1"]!

        for (i, block) in trunk.blockDescriptors.enumerated() {
            if let resBlock = block as? SWResidualBlockDesc {
                runResidualBlock(encoder: encoder, block: resBlock, index: i, input: currentBuffer, output: nextBuffer, batchSize: batchSize)
            } else if let gpoolBlock = block as? SWGlobalPoolingResidualBlockDesc {
                runGlobalPoolingResidualBlock(encoder: encoder, block: gpoolBlock, index: i, input: currentBuffer, output: nextBuffer, batchSize: batchSize)
            }
            swap(&currentBuffer, &nextBuffer)
        }

        // Trunk tip BN + activation
        let tipBN = bnWeights["trunk.tipBN"]!
        dispatcher.dispatchBatchNorm(
            encoder: encoder,
            input: currentBuffer,
            scale: tipBN.scale,
            bias: tipBN.bias,
            mask: mask,
            output: currentBuffer,
            batchSize: batchSize,
            channels: trunkChannels,
            height: nnYLen,
            width: nnXLen,
            activation: trunk.trunkTipActivation
        )

        return currentBuffer
    }

    private func runResidualBlock(encoder: MTLComputeCommandEncoder, block: SWResidualBlockDesc, index: Int, input: MTLBuffer, output: MTLBuffer, batchSize: Int) {
        let prefix = "trunk.block\(index)"
        let mask = intermediateBuffers["mask"]!
        let convOut = intermediateBuffers["conv_out"]!
        let bnOut = intermediateBuffers["bn_out"]!
        let channels = block.preBN.numChannels.intValue

        // Pre BN + activation
        let preBN = bnWeights["\(prefix).preBN"]!
        dispatcher.dispatchBatchNorm(encoder: encoder, input: input, scale: preBN.scale, bias: preBN.bias, mask: mask, output: bnOut, batchSize: batchSize, channels: channels, height: nnYLen, width: nnXLen, activation: block.preActivation)

        // Regular conv
        let regularConv = convWeights["\(prefix).regularConv"]!
        dispatcher.dispatchConv2D(encoder: encoder, input: bnOut, weights: regularConv.weights, output: convOut, batchSize: batchSize, inChannels: regularConv.inChannels, outChannels: regularConv.outChannels, height: nnYLen, width: nnXLen, kernelH: regularConv.kernelH, kernelW: regularConv.kernelW)

        // Mid BN + activation
        let midBN = bnWeights["\(prefix).midBN"]!
        dispatcher.dispatchBatchNorm(encoder: encoder, input: convOut, scale: midBN.scale, bias: midBN.bias, mask: mask, output: bnOut, batchSize: batchSize, channels: midBN.channels, height: nnYLen, width: nnXLen, activation: block.midActivation)

        // Final conv
        let finalConv = convWeights["\(prefix).finalConv"]!
        dispatcher.dispatchConv2D(encoder: encoder, input: bnOut, weights: finalConv.weights, output: convOut, batchSize: batchSize, inChannels: finalConv.inChannels, outChannels: finalConv.outChannels, height: nnYLen, width: nnXLen, kernelH: finalConv.kernelH, kernelW: finalConv.kernelW)

        // Residual add
        dispatcher.dispatchElementwiseAdd(encoder: encoder, a: input, b: convOut, output: output, size: batchSize * channels * nnYLen * nnXLen)
    }

    private func runGlobalPoolingResidualBlock(encoder: MTLComputeCommandEncoder, block: SWGlobalPoolingResidualBlockDesc, index: Int, input: MTLBuffer, output: MTLBuffer, batchSize: Int) {
        let prefix = "trunk.block\(index)"
        let mask = intermediateBuffers["mask"]!
        let maskSum = intermediateBuffers["maskSum"]!
        let maskSumSqrtS14M01 = intermediateBuffers["maskSumSqrtS14M01"]!
        let convOut = intermediateBuffers["conv_out"]!
        let bnOut = intermediateBuffers["bn_out"]!
        let gpoolOut = intermediateBuffers["gpool_out"]!
        let matmulOut = intermediateBuffers["matmul_out"]!
        let channels = block.preBN.numChannels.intValue

        // Pre BN + activation
        let preBN = bnWeights["\(prefix).preBN"]!
        dispatcher.dispatchBatchNorm(encoder: encoder, input: input, scale: preBN.scale, bias: preBN.bias, mask: mask, output: bnOut, batchSize: batchSize, channels: channels, height: nnYLen, width: nnXLen, activation: block.preActivation)

        // Regular conv
        let regularConv = convWeights["\(prefix).regularConv"]!
        dispatcher.dispatchConv2D(encoder: encoder, input: bnOut, weights: regularConv.weights, output: convOut, batchSize: batchSize, inChannels: regularConv.inChannels, outChannels: regularConv.outChannels, height: nnYLen, width: nnXLen, kernelH: regularConv.kernelH, kernelW: regularConv.kernelW)

        // Global pooling branch
        let gpoolConv = convWeights["\(prefix).gpoolConv"]!
        let gpoolBN = bnWeights["\(prefix).gpoolBN"]!
        let gpoolToBiasMul = matmulWeights["\(prefix).gpoolToBiasMul"]!

        // Gpool conv -> BN -> activation -> global pooling
        dispatcher.dispatchConv2D(encoder: encoder, input: bnOut, weights: gpoolConv.weights, output: intermediateBuffers["policy_g1"]!, batchSize: batchSize, inChannels: gpoolConv.inChannels, outChannels: gpoolConv.outChannels, height: nnYLen, width: nnXLen, kernelH: gpoolConv.kernelH, kernelW: gpoolConv.kernelW)

        dispatcher.dispatchBatchNorm(encoder: encoder, input: intermediateBuffers["policy_g1"]!, scale: gpoolBN.scale, bias: gpoolBN.bias, mask: mask, output: intermediateBuffers["policy_g1"]!, batchSize: batchSize, channels: gpoolBN.channels, height: nnYLen, width: nnXLen, activation: block.gpoolActivation)

        dispatcher.dispatchGlobalPooling(encoder: encoder, input: intermediateBuffers["policy_g1"]!, mask: mask, maskSum: maskSum, maskSumSqrtS14M01: maskSumSqrtS14M01, output: gpoolOut, batchSize: batchSize, channels: gpoolConv.outChannels, height: nnYLen, width: nnXLen)

        // Gpool to bias matmul
        dispatcher.dispatchMatMul(encoder: encoder, A: gpoolOut, B: gpoolToBiasMul.weights, C: matmulOut, M: batchSize, K: gpoolToBiasMul.inChannels, N: gpoolToBiasMul.outChannels)

        // Add bias to regular conv output
        dispatcher.dispatchAddNCBias(encoder: encoder, input: convOut, bias: matmulOut, output: convOut, batchSize: batchSize, channels: gpoolToBiasMul.outChannels, height: nnYLen, width: nnXLen)

        // Mid BN + activation
        let midBN = bnWeights["\(prefix).midBN"]!
        dispatcher.dispatchBatchNorm(encoder: encoder, input: convOut, scale: midBN.scale, bias: midBN.bias, mask: mask, output: bnOut, batchSize: batchSize, channels: midBN.channels, height: nnYLen, width: nnXLen, activation: block.midActivation)

        // Final conv
        let finalConv = convWeights["\(prefix).finalConv"]!
        dispatcher.dispatchConv2D(encoder: encoder, input: bnOut, weights: finalConv.weights, output: convOut, batchSize: batchSize, inChannels: finalConv.inChannels, outChannels: finalConv.outChannels, height: nnYLen, width: nnXLen, kernelH: finalConv.kernelH, kernelW: finalConv.kernelW)

        // Residual add
        dispatcher.dispatchElementwiseAdd(encoder: encoder, a: input, b: convOut, output: output, size: batchSize * channels * nnYLen * nnXLen)
    }

    private func runPolicyHead(encoder: MTLComputeCommandEncoder, trunkOutput: MTLBuffer, policy: UnsafeMutablePointer<Float32>, policyPass: UnsafeMutablePointer<Float32>, batchSize: Int) {
        let policyDesc = descriptor.policyHead
        let mask = intermediateBuffers["mask"]!
        let maskSum = intermediateBuffers["maskSum"]!
        let maskSumSqrtS14M01 = intermediateBuffers["maskSumSqrtS14M01"]!
        let hw = nnXLen * nnYLen

        // P1 conv
        let p1ConvW = convWeights["policy.p1Conv"]!
        let p1Out = intermediateBuffers["policy_p1"]!
        dispatcher.dispatchConv2D(encoder: encoder, input: trunkOutput, weights: p1ConvW.weights, output: p1Out, batchSize: batchSize, inChannels: p1ConvW.inChannels, outChannels: p1ConvW.outChannels, height: nnYLen, width: nnXLen, kernelH: p1ConvW.kernelH, kernelW: p1ConvW.kernelW)

        // G1 conv -> BN -> activation -> global pooling
        let g1ConvW = convWeights["policy.g1Conv"]!
        let g1BN = bnWeights["policy.g1BN"]!
        let g1Out = intermediateBuffers["policy_g1"]!
        let gpoolOut = intermediateBuffers["gpool_out"]!

        dispatcher.dispatchConv2D(encoder: encoder, input: trunkOutput, weights: g1ConvW.weights, output: g1Out, batchSize: batchSize, inChannels: g1ConvW.inChannels, outChannels: g1ConvW.outChannels, height: nnYLen, width: nnXLen, kernelH: g1ConvW.kernelH, kernelW: g1ConvW.kernelW)

        dispatcher.dispatchBatchNorm(encoder: encoder, input: g1Out, scale: g1BN.scale, bias: g1BN.bias, mask: mask, output: g1Out, batchSize: batchSize, channels: g1BN.channels, height: nnYLen, width: nnXLen, activation: policyDesc.g1Activation)

        dispatcher.dispatchGlobalPooling(encoder: encoder, input: g1Out, mask: mask, maskSum: maskSum, maskSumSqrtS14M01: maskSumSqrtS14M01, output: gpoolOut, batchSize: batchSize, channels: g1ConvW.outChannels, height: nnYLen, width: nnXLen)

        // Gpool to bias
        let gpoolToBiasMul = matmulWeights["policy.gpoolToBiasMul"]!
        let matmulOut = intermediateBuffers["matmul_out"]!
        dispatcher.dispatchMatMul(encoder: encoder, A: gpoolOut, B: gpoolToBiasMul.weights, C: matmulOut, M: batchSize, K: gpoolToBiasMul.inChannels, N: gpoolToBiasMul.outChannels)

        // Add bias to P1
        dispatcher.dispatchAddNCBias(encoder: encoder, input: p1Out, bias: matmulOut, output: p1Out, batchSize: batchSize, channels: gpoolToBiasMul.outChannels, height: nnYLen, width: nnXLen)

        // P1 BN + activation
        let p1BN = bnWeights["policy.p1BN"]!
        dispatcher.dispatchBatchNorm(encoder: encoder, input: p1Out, scale: p1BN.scale, bias: p1BN.bias, mask: mask, output: p1Out, batchSize: batchSize, channels: p1BN.channels, height: nnYLen, width: nnXLen, activation: policyDesc.p1Activation)

        // P2 conv (final policy)
        let p2ConvW = convWeights["policy.p2Conv"]!
        let policyChannels = p2ConvW.outChannels
        let policySize = batchSize * policyChannels * hw * 4
        guard let policyBuffer = device.makeBuffer(length: policySize, options: .storageModeShared) else { return }

        dispatcher.dispatchConv2D(encoder: encoder, input: p1Out, weights: p2ConvW.weights, output: policyBuffer, batchSize: batchSize, inChannels: p2ConvW.inChannels, outChannels: policyChannels, height: nnYLen, width: nnXLen, kernelH: p2ConvW.kernelH, kernelW: p2ConvW.kernelW)

        // Pass logit
        let gpoolToPassMul = matmulWeights["policy.gpoolToPassMul"]!
        let passSize = batchSize * gpoolToPassMul.outChannels * 4
        guard let passBuffer = device.makeBuffer(length: passSize, options: .storageModeShared) else { return }

        dispatcher.dispatchMatMul(encoder: encoder, A: gpoolOut, B: gpoolToPassMul.weights, C: passBuffer, M: batchSize, K: gpoolToPassMul.inChannels, N: gpoolToPassMul.outChannels)

        // For version >= 15, apply bias + activation + another matmul
        if policyDesc.version >= 15, let gpoolToPassBias = biasWeights["policy.gpoolToPassBias"],
           let passActivation = policyDesc.passActivation,
           let gpoolToPassMul2 = matmulWeights["policy.gpoolToPassMul2"] {
            dispatcher.dispatchMatBiasAdd(encoder: encoder, input: passBuffer, bias: gpoolToPassBias.bias, output: passBuffer, batchSize: batchSize, channels: gpoolToPassBias.channels)
            dispatcher.dispatchActivation(encoder: encoder, input: passBuffer, output: passBuffer, size: batchSize * gpoolToPassBias.channels, activation: passActivation)
            let pass2Size = batchSize * gpoolToPassMul2.outChannels * 4
            guard let pass2Buffer = device.makeBuffer(length: pass2Size, options: .storageModeShared) else { return }
            dispatcher.dispatchMatMul(encoder: encoder, A: passBuffer, B: gpoolToPassMul2.weights, C: pass2Buffer, M: batchSize, K: gpoolToPassMul2.inChannels, N: gpoolToPassMul2.outChannels)
            // Copy to output
            encoder.endEncoding()
            memcpy(policyPass, pass2Buffer.contents(), pass2Size)
        } else {
            encoder.endEncoding()
            memcpy(policyPass, passBuffer.contents(), passSize)
        }

        memcpy(policy, policyBuffer.contents(), policySize)
    }

    private func runValueHead(encoder: MTLComputeCommandEncoder, trunkOutput: MTLBuffer, value: UnsafeMutablePointer<Float32>, scoreValue: UnsafeMutablePointer<Float32>, ownership: UnsafeMutablePointer<Float32>, batchSize: Int) {
        let valueDesc = descriptor.valueHead
        let mask = intermediateBuffers["mask"]!
        let maskSum = intermediateBuffers["maskSum"]!
        let maskSumSqrtS14M01 = intermediateBuffers["maskSumSqrtS14M01"]!
        let maskSumSqrtS14M01SquareS01 = intermediateBuffers["maskSumSqrtS14M01SquareS01"]!
        let hw = nnXLen * nnYLen

        // V1 conv -> BN -> activation
        let v1ConvW = convWeights["value.v1Conv"]!
        let v1BN = bnWeights["value.v1BN"]!
        let v1Out = intermediateBuffers["value_v1"]!

        dispatcher.dispatchConv2D(encoder: encoder, input: trunkOutput, weights: v1ConvW.weights, output: v1Out, batchSize: batchSize, inChannels: v1ConvW.inChannels, outChannels: v1ConvW.outChannels, height: nnYLen, width: nnXLen, kernelH: v1ConvW.kernelH, kernelW: v1ConvW.kernelW)

        dispatcher.dispatchBatchNorm(encoder: encoder, input: v1Out, scale: v1BN.scale, bias: v1BN.bias, mask: mask, output: v1Out, batchSize: batchSize, channels: v1BN.channels, height: nnYLen, width: nnXLen, activation: valueDesc.v1Activation)

        // Global pooling for value
        let gpoolOut = intermediateBuffers["value_gpool"]!
        dispatcher.dispatchGlobalPoolingValue(encoder: encoder, input: v1Out, maskSum: maskSum, maskSumSqrtS14M01: maskSumSqrtS14M01, maskSumSqrtS14M01SquareS01: maskSumSqrtS14M01SquareS01, output: gpoolOut, batchSize: batchSize, channels: v1ConvW.outChannels, height: nnYLen, width: nnXLen)

        // V2 matmul -> bias -> activation
        let v2Mul = matmulWeights["value.v2Mul"]!
        let v2Bias = biasWeights["value.v2Bias"]!
        let v2Out = intermediateBuffers["value_v2"]!

        dispatcher.dispatchMatMul(encoder: encoder, A: gpoolOut, B: v2Mul.weights, C: v2Out, M: batchSize, K: v2Mul.inChannels, N: v2Mul.outChannels)
        dispatcher.dispatchMatBiasAdd(encoder: encoder, input: v2Out, bias: v2Bias.bias, output: v2Out, batchSize: batchSize, channels: v2Bias.channels)
        dispatcher.dispatchActivation(encoder: encoder, input: v2Out, output: v2Out, size: batchSize * v2Bias.channels, activation: valueDesc.v2Activation)

        // V3 matmul -> bias (value output)
        let v3Mul = matmulWeights["value.v3Mul"]!
        let v3Bias = biasWeights["value.v3Bias"]!
        let valueSize = batchSize * v3Mul.outChannels * 4
        guard let valueBuffer = device.makeBuffer(length: valueSize, options: .storageModeShared) else { return }

        dispatcher.dispatchMatMul(encoder: encoder, A: v2Out, B: v3Mul.weights, C: valueBuffer, M: batchSize, K: v3Mul.inChannels, N: v3Mul.outChannels)
        dispatcher.dispatchMatBiasAdd(encoder: encoder, input: valueBuffer, bias: v3Bias.bias, output: valueBuffer, batchSize: batchSize, channels: v3Bias.channels)

        // SV3 matmul -> bias (score value output)
        let sv3Mul = matmulWeights["value.sv3Mul"]!
        let sv3Bias = biasWeights["value.sv3Bias"]!
        let scoreValueSize = batchSize * sv3Mul.outChannels * 4
        guard let scoreValueBuffer = device.makeBuffer(length: scoreValueSize, options: .storageModeShared) else { return }

        dispatcher.dispatchMatMul(encoder: encoder, A: v2Out, B: sv3Mul.weights, C: scoreValueBuffer, M: batchSize, K: sv3Mul.inChannels, N: sv3Mul.outChannels)
        dispatcher.dispatchMatBiasAdd(encoder: encoder, input: scoreValueBuffer, bias: sv3Bias.bias, output: scoreValueBuffer, batchSize: batchSize, channels: sv3Bias.channels)

        // Ownership conv
        let vOwnershipConvW = convWeights["value.vOwnershipConv"]!
        let ownershipSize = batchSize * vOwnershipConvW.outChannels * hw * 4
        guard let ownershipBuffer = device.makeBuffer(length: ownershipSize, options: .storageModeShared) else { return }

        dispatcher.dispatchConv2D(encoder: encoder, input: v1Out, weights: vOwnershipConvW.weights, output: ownershipBuffer, batchSize: batchSize, inChannels: vOwnershipConvW.inChannels, outChannels: vOwnershipConvW.outChannels, height: nnYLen, width: nnXLen, kernelH: vOwnershipConvW.kernelH, kernelW: vOwnershipConvW.kernelW)

        // Copy outputs
        memcpy(value, valueBuffer.contents(), valueSize)
        memcpy(scoreValue, scoreValueBuffer.contents(), scoreValueSize)
        memcpy(ownership, ownershipBuffer.contents(), ownershipSize)
    }
}
