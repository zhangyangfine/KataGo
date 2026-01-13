//
//  metalbackend_pure.swift
//  Pure Metal 4 backend for KataGo - replaces MPSGraph with compute shaders
//

import Foundation
import Metal

// MARK: - Pipeline Manager

/// Manages Metal compute pipeline states for all kernels
class MetalPipelineManager {
    let device: MTLDevice
    let library: MTLLibrary
    let commandQueue: MTLCommandQueue

    // Compute pipelines for each kernel
    var conv2dPipeline: MTLComputePipelineState!
    var conv2d1x1Pipeline: MTLComputePipelineState!
    var conv2d3x3TiledPipeline: MTLComputePipelineState!
    var batchnormMaskPipeline: MTLComputePipelineState!
    var reluPipeline: MTLComputePipelineState!
    var mishPipeline: MTLComputePipelineState!
    var batchnormReluPipeline: MTLComputePipelineState!
    var batchnormMishPipeline: MTLComputePipelineState!
    var matmulPipeline: MTLComputePipelineState!
    var matmulTiledPipeline: MTLComputePipelineState!
    var elementwiseAddPipeline: MTLComputePipelineState!
    var elementwiseMulPipeline: MTLComputePipelineState!
    var addNCBiasPipeline: MTLComputePipelineState!
    var reductionSumHWPipeline: MTLComputePipelineState!
    var reductionMaxHWMaskedPipeline: MTLComputePipelineState!
    var maskSumPipeline: MTLComputePipelineState!
    var maskSumSqrtS14M01Pipeline: MTLComputePipelineState!
    var maskSumSqrtS14M01SquareS01Pipeline: MTLComputePipelineState!
    var globalPoolingPipeline: MTLComputePipelineState!
    var globalPoolingValuePipeline: MTLComputePipelineState!
    var matBiasAddPipeline: MTLComputePipelineState!
    var residualAddPipeline: MTLComputePipelineState!
    var copyBufferPipeline: MTLComputePipelineState!

    init(device: MTLDevice) throws {
        self.device = device
        self.commandQueue = device.makeCommandQueue()!

        // Load the Metal library from the compiled .metallib or source
        guard let libraryPath = Bundle.main.path(forResource: "metalbackend", ofType: "metallib"),
              let lib = try? device.makeLibrary(filepath: libraryPath) else {
            // Try to compile from source
            let sourcePath = Bundle.main.path(forResource: "metalbackend", ofType: "metal")
            if let path = sourcePath, let source = try? String(contentsOfFile: path) {
                self.library = try device.makeLibrary(source: source, options: nil)
            } else {
                // Use default library
                self.library = device.makeDefaultLibrary()!
            }
            try createPipelines()
            return
        }
        self.library = lib
        try createPipelines()
    }

    private func createPipelines() throws {
        conv2dPipeline = try createPipeline("conv2d_nchw")
        conv2d1x1Pipeline = try createPipeline("conv2d_1x1_nchw")
        conv2d3x3TiledPipeline = try createPipeline("conv2d_3x3_nchw_tiled")
        batchnormMaskPipeline = try createPipeline("batchnorm_mask")
        reluPipeline = try createPipeline("relu_activation")
        mishPipeline = try createPipeline("mish_activation")
        batchnormReluPipeline = try createPipeline("batchnorm_relu")
        batchnormMishPipeline = try createPipeline("batchnorm_mish")
        matmulPipeline = try createPipeline("matmul")
        matmulTiledPipeline = try createPipeline("matmul_tiled")
        elementwiseAddPipeline = try createPipeline("elementwise_add")
        elementwiseMulPipeline = try createPipeline("elementwise_mul")
        addNCBiasPipeline = try createPipeline("add_nc_bias")
        reductionSumHWPipeline = try createPipeline("reduction_sum_hw")
        reductionMaxHWMaskedPipeline = try createPipeline("reduction_max_hw_masked")
        maskSumPipeline = try createPipeline("mask_sum")
        maskSumSqrtS14M01Pipeline = try createPipeline("mask_sum_sqrt_s14_m01")
        maskSumSqrtS14M01SquareS01Pipeline = try createPipeline("mask_sum_sqrt_s14_m01_square_s01")
        globalPoolingPipeline = try createPipeline("global_pooling")
        globalPoolingValuePipeline = try createPipeline("global_pooling_value")
        matBiasAddPipeline = try createPipeline("mat_bias_add")
        residualAddPipeline = try createPipeline("residual_add")
        copyBufferPipeline = try createPipeline("copy_buffer")
    }

    private func createPipeline(_ name: String) throws -> MTLComputePipelineState {
        guard let function = library.makeFunction(name: name) else {
            throw MetalError.functionNotFound(name)
        }
        return try device.makeComputePipelineState(function: function)
    }
}

enum MetalError: Error {
    case functionNotFound(String)
    case bufferCreationFailed
    case commandBufferCreationFailed
}

// MARK: - Buffer Manager

/// Manages GPU buffers for weights and intermediate tensors
class MetalBufferManager {
    let device: MTLDevice
    var buffers: [String: MTLBuffer] = [:]

    init(device: MTLDevice) {
        self.device = device
    }

    func createBuffer(name: String, size: Int) -> MTLBuffer? {
        guard let buffer = device.makeBuffer(length: size, options: .storageModeShared) else {
            return nil
        }
        buffers[name] = buffer
        return buffer
    }

    func createBuffer(name: String, data: UnsafeRawPointer, size: Int) -> MTLBuffer? {
        guard let buffer = device.makeBuffer(bytes: data, length: size, options: .storageModeShared) else {
            return nil
        }
        buffers[name] = buffer
        return buffer
    }

    func getBuffer(_ name: String) -> MTLBuffer? {
        return buffers[name]
    }

    func releaseBuffer(_ name: String) {
        buffers.removeValue(forKey: name)
    }

    func releaseAll() {
        buffers.removeAll()
    }
}

// MARK: - Compute Dispatcher

/// Dispatches compute kernels with optimal thread configuration
class MetalComputeDispatcher {
    let pipelineManager: MetalPipelineManager

    init(pipelineManager: MetalPipelineManager) {
        self.pipelineManager = pipelineManager
    }

    func dispatchConv2D(
        encoder: MTLComputeCommandEncoder,
        input: MTLBuffer,
        weights: MTLBuffer,
        output: MTLBuffer,
        batchSize: Int,
        inChannels: Int,
        outChannels: Int,
        height: Int,
        width: Int,
        kernelH: Int,
        kernelW: Int,
        dilationH: Int = 1,
        dilationW: Int = 1
    ) {
        let pipeline: MTLComputePipelineState
        if kernelH == 1 && kernelW == 1 {
            pipeline = pipelineManager.conv2d1x1Pipeline
        } else if kernelH == 3 && kernelW == 3 {
            pipeline = pipelineManager.conv2d3x3TiledPipeline
        } else {
            pipeline = pipelineManager.conv2dPipeline
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(weights, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)

        var bs = Int32(batchSize)
        var ic = Int32(inChannels)
        var oc = Int32(outChannels)
        var h = Int32(height)
        var w = Int32(width)
        var kh = Int32(kernelH)
        var kw = Int32(kernelW)
        var dh = Int32(dilationH)
        var dw = Int32(dilationW)

        encoder.setBytes(&bs, length: 4, index: 3)
        encoder.setBytes(&ic, length: 4, index: 4)
        encoder.setBytes(&oc, length: 4, index: 5)
        encoder.setBytes(&h, length: 4, index: 6)
        encoder.setBytes(&w, length: 4, index: 7)

        if kernelH != 1 || kernelW != 1 {
            encoder.setBytes(&kh, length: 4, index: 8)
            encoder.setBytes(&kw, length: 4, index: 9)
            encoder.setBytes(&dh, length: 4, index: 10)
            encoder.setBytes(&dw, length: 4, index: 11)
        }

        let threadsPerGrid = MTLSize(width: width, height: height, depth: batchSize * outChannels)
        let threadsPerGroup = MTLSize(width: min(16, width), height: min(16, height), depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)
    }

    func dispatchBatchNorm(
        encoder: MTLComputeCommandEncoder,
        input: MTLBuffer,
        scale: MTLBuffer,
        bias: MTLBuffer,
        mask: MTLBuffer,
        output: MTLBuffer,
        batchSize: Int,
        channels: Int,
        height: Int,
        width: Int,
        activation: ActivationKind = .identity
    ) {
        let pipeline: MTLComputePipelineState
        switch activation {
        case .relu:
            pipeline = pipelineManager.batchnormReluPipeline
        case .mish:
            pipeline = pipelineManager.batchnormMishPipeline
        default:
            pipeline = pipelineManager.batchnormMaskPipeline
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(scale, offset: 0, index: 1)
        encoder.setBuffer(bias, offset: 0, index: 2)
        encoder.setBuffer(mask, offset: 0, index: 3)
        encoder.setBuffer(output, offset: 0, index: 4)

        var bs = Int32(batchSize)
        var c = Int32(channels)
        var h = Int32(height)
        var w = Int32(width)

        encoder.setBytes(&bs, length: 4, index: 5)
        encoder.setBytes(&c, length: 4, index: 6)
        encoder.setBytes(&h, length: 4, index: 7)
        encoder.setBytes(&w, length: 4, index: 8)

        let threadsPerGrid = MTLSize(width: width, height: height, depth: batchSize * channels)
        let threadsPerGroup = MTLSize(width: min(16, width), height: min(16, height), depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)
    }

    func dispatchActivation(
        encoder: MTLComputeCommandEncoder,
        input: MTLBuffer,
        output: MTLBuffer,
        size: Int,
        activation: ActivationKind
    ) {
        let pipeline: MTLComputePipelineState
        switch activation {
        case .relu:
            pipeline = pipelineManager.reluPipeline
        case .mish:
            pipeline = pipelineManager.mishPipeline
        default:
            // Identity - just copy
            pipeline = pipelineManager.copyBufferPipeline
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)

        var s = Int32(size)
        encoder.setBytes(&s, length: 4, index: 2)

        let threadsPerGrid = MTLSize(width: size, height: 1, depth: 1)
        let threadsPerGroup = MTLSize(width: min(256, size), height: 1, depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)
    }

    func dispatchMatMul(
        encoder: MTLComputeCommandEncoder,
        A: MTLBuffer,
        B: MTLBuffer,
        C: MTLBuffer,
        M: Int,
        K: Int,
        N: Int
    ) {
        encoder.setComputePipelineState(pipelineManager.matmulPipeline)
        encoder.setBuffer(A, offset: 0, index: 0)
        encoder.setBuffer(B, offset: 0, index: 1)
        encoder.setBuffer(C, offset: 0, index: 2)

        var m = Int32(M)
        var k = Int32(K)
        var n = Int32(N)

        encoder.setBytes(&m, length: 4, index: 3)
        encoder.setBytes(&k, length: 4, index: 4)
        encoder.setBytes(&n, length: 4, index: 5)

        let threadsPerGrid = MTLSize(width: N, height: M, depth: 1)
        let threadsPerGroup = MTLSize(width: min(16, N), height: min(16, M), depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)
    }

    func dispatchAddNCBias(
        encoder: MTLComputeCommandEncoder,
        input: MTLBuffer,
        bias: MTLBuffer,
        output: MTLBuffer,
        batchSize: Int,
        channels: Int,
        height: Int,
        width: Int
    ) {
        encoder.setComputePipelineState(pipelineManager.addNCBiasPipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(bias, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)

        var bs = Int32(batchSize)
        var c = Int32(channels)
        var h = Int32(height)
        var w = Int32(width)

        encoder.setBytes(&bs, length: 4, index: 3)
        encoder.setBytes(&c, length: 4, index: 4)
        encoder.setBytes(&h, length: 4, index: 5)
        encoder.setBytes(&w, length: 4, index: 6)

        let threadsPerGrid = MTLSize(width: width, height: height, depth: batchSize * channels)
        let threadsPerGroup = MTLSize(width: min(16, width), height: min(16, height), depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)
    }

    func dispatchElementwiseAdd(
        encoder: MTLComputeCommandEncoder,
        a: MTLBuffer,
        b: MTLBuffer,
        output: MTLBuffer,
        size: Int
    ) {
        encoder.setComputePipelineState(pipelineManager.elementwiseAddPipeline)
        encoder.setBuffer(a, offset: 0, index: 0)
        encoder.setBuffer(b, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)

        var s = Int32(size)
        encoder.setBytes(&s, length: 4, index: 3)

        let threadsPerGrid = MTLSize(width: size, height: 1, depth: 1)
        let threadsPerGroup = MTLSize(width: min(256, size), height: 1, depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)
    }

    func dispatchMatBiasAdd(
        encoder: MTLComputeCommandEncoder,
        input: MTLBuffer,
        bias: MTLBuffer,
        output: MTLBuffer,
        batchSize: Int,
        channels: Int
    ) {
        encoder.setComputePipelineState(pipelineManager.matBiasAddPipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(bias, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)

        var bs = Int32(batchSize)
        var c = Int32(channels)

        encoder.setBytes(&bs, length: 4, index: 3)
        encoder.setBytes(&c, length: 4, index: 4)

        let threadsPerGrid = MTLSize(width: channels, height: batchSize, depth: 1)
        let threadsPerGroup = MTLSize(width: min(256, channels), height: 1, depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)
    }

    func dispatchGlobalPooling(
        encoder: MTLComputeCommandEncoder,
        input: MTLBuffer,
        mask: MTLBuffer,
        maskSum: MTLBuffer,
        maskSumSqrtS14M01: MTLBuffer,
        output: MTLBuffer,
        batchSize: Int,
        channels: Int,
        height: Int,
        width: Int
    ) {
        encoder.setComputePipelineState(pipelineManager.globalPoolingPipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(mask, offset: 0, index: 1)
        encoder.setBuffer(maskSum, offset: 0, index: 2)
        encoder.setBuffer(maskSumSqrtS14M01, offset: 0, index: 3)
        encoder.setBuffer(output, offset: 0, index: 4)

        var bs = Int32(batchSize)
        var c = Int32(channels)
        var h = Int32(height)
        var w = Int32(width)

        encoder.setBytes(&bs, length: 4, index: 5)
        encoder.setBytes(&c, length: 4, index: 6)
        encoder.setBytes(&h, length: 4, index: 7)
        encoder.setBytes(&w, length: 4, index: 8)

        let threadgroupSize = 256
        encoder.setThreadgroupMemoryLength(threadgroupSize * 4, index: 0)  // sharedSum
        encoder.setThreadgroupMemoryLength(threadgroupSize * 4, index: 1)  // sharedMax

        let threadsPerGrid = MTLSize(width: threadgroupSize, height: batchSize * channels, depth: 1)
        let threadsPerGroup = MTLSize(width: threadgroupSize, height: 1, depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)
    }

    func dispatchGlobalPoolingValue(
        encoder: MTLComputeCommandEncoder,
        input: MTLBuffer,
        maskSum: MTLBuffer,
        maskSumSqrtS14M01: MTLBuffer,
        maskSumSqrtS14M01SquareS01: MTLBuffer,
        output: MTLBuffer,
        batchSize: Int,
        channels: Int,
        height: Int,
        width: Int
    ) {
        encoder.setComputePipelineState(pipelineManager.globalPoolingValuePipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(maskSum, offset: 0, index: 1)
        encoder.setBuffer(maskSumSqrtS14M01, offset: 0, index: 2)
        encoder.setBuffer(maskSumSqrtS14M01SquareS01, offset: 0, index: 3)
        encoder.setBuffer(output, offset: 0, index: 4)

        var bs = Int32(batchSize)
        var c = Int32(channels)
        var h = Int32(height)
        var w = Int32(width)

        encoder.setBytes(&bs, length: 4, index: 5)
        encoder.setBytes(&c, length: 4, index: 6)
        encoder.setBytes(&h, length: 4, index: 7)
        encoder.setBytes(&w, length: 4, index: 8)

        let threadgroupSize = 256
        encoder.setThreadgroupMemoryLength(threadgroupSize * 4, index: 0)

        let threadsPerGrid = MTLSize(width: threadgroupSize, height: batchSize * channels, depth: 1)
        let threadsPerGroup = MTLSize(width: threadgroupSize, height: 1, depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)
    }

    func dispatchMaskSum(
        encoder: MTLComputeCommandEncoder,
        mask: MTLBuffer,
        maskSum: MTLBuffer,
        batchSize: Int,
        height: Int,
        width: Int
    ) {
        encoder.setComputePipelineState(pipelineManager.maskSumPipeline)
        encoder.setBuffer(mask, offset: 0, index: 0)
        encoder.setBuffer(maskSum, offset: 0, index: 1)

        var bs = Int32(batchSize)
        var h = Int32(height)
        var w = Int32(width)

        encoder.setBytes(&bs, length: 4, index: 2)
        encoder.setBytes(&h, length: 4, index: 3)
        encoder.setBytes(&w, length: 4, index: 4)

        let threadgroupSize = 256
        encoder.setThreadgroupMemoryLength(threadgroupSize * 4, index: 0)

        let threadsPerGrid = MTLSize(width: batchSize, height: 1, depth: 1)
        let threadsPerGroup = MTLSize(width: 1, height: 1, depth: 1)
        encoder.dispatchThreadgroups(MTLSize(width: batchSize, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
    }

    func dispatchMaskSumSqrtS14M01(
        encoder: MTLComputeCommandEncoder,
        maskSum: MTLBuffer,
        output: MTLBuffer,
        batchSize: Int
    ) {
        encoder.setComputePipelineState(pipelineManager.maskSumSqrtS14M01Pipeline)
        encoder.setBuffer(maskSum, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)

        var bs = Int32(batchSize)
        encoder.setBytes(&bs, length: 4, index: 2)

        let threadsPerGrid = MTLSize(width: batchSize, height: 1, depth: 1)
        let threadsPerGroup = MTLSize(width: min(256, batchSize), height: 1, depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)
    }

    func dispatchMaskSumSqrtS14M01SquareS01(
        encoder: MTLComputeCommandEncoder,
        maskSumSqrtS14M01: MTLBuffer,
        output: MTLBuffer,
        batchSize: Int
    ) {
        encoder.setComputePipelineState(pipelineManager.maskSumSqrtS14M01SquareS01Pipeline)
        encoder.setBuffer(maskSumSqrtS14M01, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)

        var bs = Int32(batchSize)
        encoder.setBytes(&bs, length: 4, index: 2)

        let threadsPerGrid = MTLSize(width: batchSize, height: 1, depth: 1)
        let threadsPerGroup = MTLSize(width: min(256, batchSize), height: 1, depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)
    }
}
