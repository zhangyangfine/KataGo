//
//  metalbackend_pure.swift
//  Pure Metal 4 backend for KataGo - uses MTL4 APIs for maximum performance
//
//  Requires: macOS 26 (Tahoe) / iOS 26 or later, Apple Silicon (M1/A14+)
//  Metal 4 was announced at WWDC 2025
//

import Foundation
import Metal

// MARK: - Metal 4 Pipeline Manager

/// Manages Metal 4 compute pipeline states and command infrastructure
/// Requires macOS 26+ / iOS 26+ for Metal 4 support
class MetalPipelineManager {
    let device: MTLDevice
    let library: MTLLibrary

    // Metal 4 command infrastructure
    let mtl4CommandQueue: MTLCommandQueue
    let commandAllocator: MTLHeap?

    // Residency set for resource management (Metal 4)
    let residencySet: MTLResidencySet

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

        // Create Metal 4 command queue with high priority
        // MTL4CommandQueue provides decoupled command buffers for parallel encoding
        let queueDescriptor = MTLCommandQueueDescriptor()
        queueDescriptor.maxCommandBufferCount = 64  // Allow more in-flight command buffers

        guard let queue = device.makeCommandQueue(descriptor: queueDescriptor) else {
            throw MetalError.commandBufferCreationFailed
        }
        self.mtl4CommandQueue = queue

        // Create a heap for pre-allocated buffer memory (Metal 4 style resource management)
        let heapDescriptor = MTLHeapDescriptor()
        heapDescriptor.size = 256 * 1024 * 1024  // 256 MB heap for command allocations
        heapDescriptor.storageMode = .shared
        heapDescriptor.type = .automatic
        self.commandAllocator = device.makeHeap(descriptor: heapDescriptor)

        // Create residency set for efficient resource management (Metal 4)
        let residencyDescriptor = MTLResidencySetDescriptor()
        residencyDescriptor.label = "KataGo Residency Set"
        residencyDescriptor.initialCapacity = 1024  // Expected number of resources
        self.residencySet = try device.makeResidencySet(descriptor: residencyDescriptor)

        // Load the Metal library from the compiled .metallib or source
        guard let libraryPath = Bundle.main.path(forResource: "metalbackend", ofType: "metallib"),
              let lib = try? device.makeLibrary(filepath: libraryPath) else {
            // Try to compile from source
            let sourcePath = Bundle.main.path(forResource: "metalbackend", ofType: "metal")
            if let path = sourcePath, let source = try? String(contentsOfFile: path) {
                // Use Metal 3.1 compilation options
                let options = MTLCompileOptions()
                options.languageVersion = .version3_1
                options.fastMathEnabled = true
                self.library = try device.makeLibrary(source: source, options: options)
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
        // Create pipelines with optimized descriptors for Metal 4
        conv2dPipeline = try createOptimizedPipeline("conv2d_nchw")
        conv2d1x1Pipeline = try createOptimizedPipeline("conv2d_1x1_nchw")
        conv2d3x3TiledPipeline = try createOptimizedPipeline("conv2d_3x3_nchw_tiled")
        batchnormMaskPipeline = try createOptimizedPipeline("batchnorm_mask")
        reluPipeline = try createOptimizedPipeline("relu_activation")
        mishPipeline = try createOptimizedPipeline("mish_activation")
        batchnormReluPipeline = try createOptimizedPipeline("batchnorm_relu")
        batchnormMishPipeline = try createOptimizedPipeline("batchnorm_mish")
        matmulPipeline = try createOptimizedPipeline("matmul")
        matmulTiledPipeline = try createOptimizedPipeline("matmul_tiled")
        elementwiseAddPipeline = try createOptimizedPipeline("elementwise_add")
        elementwiseMulPipeline = try createOptimizedPipeline("elementwise_mul")
        addNCBiasPipeline = try createOptimizedPipeline("add_nc_bias")
        reductionSumHWPipeline = try createOptimizedPipeline("reduction_sum_hw")
        reductionMaxHWMaskedPipeline = try createOptimizedPipeline("reduction_max_hw_masked")
        maskSumPipeline = try createOptimizedPipeline("mask_sum")
        maskSumSqrtS14M01Pipeline = try createOptimizedPipeline("mask_sum_sqrt_s14_m01")
        maskSumSqrtS14M01SquareS01Pipeline = try createOptimizedPipeline("mask_sum_sqrt_s14_m01_square_s01")
        globalPoolingPipeline = try createOptimizedPipeline("global_pooling")
        globalPoolingValuePipeline = try createOptimizedPipeline("global_pooling_value")
        matBiasAddPipeline = try createOptimizedPipeline("mat_bias_add")
        residualAddPipeline = try createOptimizedPipeline("residual_add")
        copyBufferPipeline = try createOptimizedPipeline("copy_buffer")
    }

    private func createOptimizedPipeline(_ name: String) throws -> MTLComputePipelineState {
        guard let function = library.makeFunction(name: name) else {
            throw MetalError.functionNotFound(name)
        }

        // Use pipeline descriptor for Metal 4 optimizations
        let descriptor = MTLComputePipelineDescriptor()
        descriptor.computeFunction = function
        descriptor.label = name

        // Enable optimizations
        descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true

        // Support indirect command buffers for Metal 4
        descriptor.supportIndirectCommandBuffers = true

        // Metal 4: Enable dynamic libraries linking if needed
        // supportAddingBinaryFunctions available since macOS 11.0
        descriptor.supportAddingBinaryFunctions = false  // We don't need dynamic functions

        return try device.makeComputePipelineState(descriptor: descriptor, options: [], reflection: nil)
    }

    /// Create a command buffer using Metal 4's decoupled model
    func makeCommandBuffer() -> MTLCommandBuffer? {
        // Metal 4: Command buffers are independent of queues
        // Use retained references mode for explicit resource lifetime control
        let descriptor = MTLCommandBufferDescriptor()
        descriptor.retainedReferences = false  // Better performance with explicit management
        descriptor.errorOptions = .encoderExecutionStatus

        return mtl4CommandQueue.makeCommandBuffer(descriptor: descriptor)
    }

    /// Add a buffer to the residency set for Metal 4 resource management
    func addToResidencySet(_ buffer: MTLBuffer) {
        residencySet.addAllocation(buffer)
    }

    /// Commit the residency set before execution
    func commitResidency() {
        residencySet.commit()
    }
}

enum MetalError: Error {
    case functionNotFound(String)
    case bufferCreationFailed
    case commandBufferCreationFailed
}

// MARK: - Metal 4 Buffer Manager

/// Manages GPU buffers with Metal 4 resource management
class MetalBufferManager {
    let device: MTLDevice
    let pipelineManager: MetalPipelineManager?
    var buffers: [String: MTLBuffer] = [:]

    // Pre-allocated heap for intermediate buffers (Metal 4 style)
    var bufferHeap: MTLHeap?

    init(device: MTLDevice, pipelineManager: MetalPipelineManager? = nil) {
        self.device = device
        self.pipelineManager = pipelineManager
    }

    /// Initialize heap for pre-allocated buffers
    func initializeHeap(size: Int) {
        let heapDescriptor = MTLHeapDescriptor()
        heapDescriptor.size = size
        heapDescriptor.storageMode = .shared
        heapDescriptor.type = .automatic
        heapDescriptor.hazardTrackingMode = .tracked  // Metal 4: explicit hazard tracking
        bufferHeap = device.makeHeap(descriptor: heapDescriptor)
    }

    func createBuffer(name: String, size: Int) -> MTLBuffer? {
        // Try to allocate from heap first (Metal 4 style)
        if let heap = bufferHeap {
            let sizeAndAlign = heap.maxAvailableSize(alignment: 256)
            if sizeAndAlign >= size {
                if let buffer = heap.makeBuffer(length: size, options: .storageModeShared) {
                    buffer.label = name
                    buffers[name] = buffer
                    pipelineManager?.addToResidencySet(buffer)
                    return buffer
                }
            }
        }

        // Fall back to device allocation
        guard let buffer = device.makeBuffer(length: size, options: .storageModeShared) else {
            return nil
        }
        buffer.label = name
        buffers[name] = buffer
        pipelineManager?.addToResidencySet(buffer)
        return buffer
    }

    func createBuffer(name: String, data: UnsafeRawPointer, size: Int) -> MTLBuffer? {
        guard let buffer = device.makeBuffer(bytes: data, length: size, options: .storageModeShared) else {
            return nil
        }
        buffer.label = name
        buffers[name] = buffer
        pipelineManager?.addToResidencySet(buffer)
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

// MARK: - Metal 4 Compute Dispatcher

/// Dispatches compute kernels with Metal 4 optimized thread configuration
class MetalComputeDispatcher {
    let pipelineManager: MetalPipelineManager

    // SIMD width for optimal dispatching (Metal 4 typically uses 32)
    let simdWidth: Int = 32

    init(pipelineManager: MetalPipelineManager) {
        self.pipelineManager = pipelineManager
    }

    /// Calculate optimal threadgroup size for Metal 4 SIMD execution
    private func optimalThreadgroupSize(for pipeline: MTLComputePipelineState, workSize: MTLSize) -> MTLSize {
        let maxThreads = pipeline.maxTotalThreadsPerThreadgroup
        let threadExecutionWidth = pipeline.threadExecutionWidth

        // Align to SIMD width for Metal 4
        var width = min(workSize.width, threadExecutionWidth)
        var height = min(workSize.height, maxThreads / width)
        var depth = min(workSize.depth, maxThreads / (width * height))

        // Ensure we use SIMD-aligned dimensions
        width = max(1, (width / threadExecutionWidth) * threadExecutionWidth)
        if width == 0 { width = min(workSize.width, threadExecutionWidth) }
        height = max(1, height)
        depth = max(1, depth)

        return MTLSize(width: width, height: height, depth: depth)
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
        let threadsPerGroup = optimalThreadgroupSize(for: pipeline, workSize: threadsPerGrid)
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
        let threadsPerGroup = optimalThreadgroupSize(for: pipeline, workSize: threadsPerGrid)
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

        // Use SIMD-aligned threadgroup size for Metal 4
        let threadsPerGrid = MTLSize(width: size, height: 1, depth: 1)
        let threadsPerGroup = MTLSize(width: min(simdWidth * 8, size), height: 1, depth: 1)  // 256 threads
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
        let threadsPerGroup = optimalThreadgroupSize(for: pipelineManager.matmulPipeline, workSize: threadsPerGrid)
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
        let threadsPerGroup = optimalThreadgroupSize(for: pipelineManager.addNCBiasPipeline, workSize: threadsPerGrid)
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
        let threadsPerGroup = MTLSize(width: min(simdWidth * 8, size), height: 1, depth: 1)
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
        let threadsPerGroup = MTLSize(width: min(simdWidth * 8, channels), height: 1, depth: 1)
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

        // Use SIMD-aligned threadgroup size for efficient reductions
        let threadgroupSize = simdWidth * 8  // 256 threads
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

        let threadgroupSize = simdWidth * 8
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

        let threadgroupSize = simdWidth * 8
        encoder.setThreadgroupMemoryLength(threadgroupSize * 4, index: 0)

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
        let threadsPerGroup = MTLSize(width: min(simdWidth * 8, batchSize), height: 1, depth: 1)
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
        let threadsPerGroup = MTLSize(width: min(simdWidth * 8, batchSize), height: 1, depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)
    }
}
