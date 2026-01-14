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
/// Uses MTL4* types for true Metal 4 implementation
class MetalPipelineManager {
    let device: MTLDevice
    let library: MTLLibrary

    // Metal 4 command infrastructure
    let mtl4CommandQueue: MTL4CommandQueue
    let commandAllocator: MTL4CommandAllocator

    // Metal 4 compiler for pipeline creation
    let mtl4Compiler: MTL4Compiler

    // Residency set for resource management
    let residencySet: MTLResidencySet

    // Pool of Metal 4 argument tables for resource binding (one per dispatch in a command buffer)
    var argumentTables: [MTL4ArgumentTable] = []
    var currentTableIndex: Int = 0
    let maxDispatchesPerCommandBuffer = 256  // Should be enough for a full forward pass

    // Constants buffer for shader parameters (replaces setBytes)
    let constantsBuffer: MTLBuffer
    let constantsBufferSize = 64 * 1024  // 64KB to handle deep networks with many dispatch calls

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
    var extractMaskPipeline: MTLComputePipelineState!

    // Shared event for CPU-GPU synchronization
    let sharedEvent: MTLSharedEvent
    var eventValue: UInt64 = 0

    init(device: MTLDevice) throws {
        self.device = device

        // Create Metal 4 command queue
        guard let queue = device.makeMTL4CommandQueue() else {
            throw MetalError.commandBufferCreationFailed
        }
        self.mtl4CommandQueue = queue

        // Create Metal 4 command allocator for explicit memory management
        let allocatorDescriptor = MTL4CommandAllocatorDescriptor()
        self.commandAllocator = try device.makeCommandAllocator(descriptor: allocatorDescriptor)

        // Create Metal 4 compiler for pipeline creation
        let compilerDescriptor = MTL4CompilerDescriptor()
        self.mtl4Compiler = try device.makeCompiler(descriptor: compilerDescriptor)

        // Create shared event for synchronization
        guard let event = device.makeSharedEvent() else {
            throw MetalError.commandBufferCreationFailed
        }
        self.sharedEvent = event

        // Create residency set for efficient resource management
        let residencyDescriptor = MTLResidencySetDescriptor()
        residencyDescriptor.label = "KataGo Residency Set"
        residencyDescriptor.initialCapacity = 1024
        self.residencySet = try device.makeResidencySet(descriptor: residencyDescriptor)

        // Create pool of Metal 4 argument tables for resource binding (one per dispatch)
        let argTableDescriptor = MTL4ArgumentTableDescriptor()
        argTableDescriptor.maxBufferBindCount = 32  // Up to 32 buffer bindings
        for _ in 0..<maxDispatchesPerCommandBuffer {
            let table = try device.makeArgumentTable(descriptor: argTableDescriptor)
            self.argumentTables.append(table)
        }

        // Create constants buffer for shader parameters (replaces setBytes)
        guard let constBuffer = device.makeBuffer(length: constantsBufferSize, options: .storageModeShared) else {
            throw MetalError.bufferCreationFailed
        }
        constBuffer.label = "Constants Buffer"
        self.constantsBuffer = constBuffer

        // Load the Metal library
        guard let libraryPath = Bundle.main.path(forResource: "metalbackend", ofType: "metallib"),
              let lib = try? device.makeLibrary(filepath: libraryPath) else {
            // Try to compile from source
            let sourcePath = Bundle.main.path(forResource: "metalbackend", ofType: "metal")
            if let path = sourcePath, let source = try? String(contentsOfFile: path) {
                let options = MTLCompileOptions()
                options.languageVersion = .version4_0  // MSL 4.0 for Metal 4
                options.fastMathEnabled = true
                self.library = try device.makeLibrary(source: source, options: options)
            } else {
                guard let defaultLibrary = device.makeDefaultLibrary() else {
                    throw MetalError.functionNotFound("default library")
                }
                self.library = defaultLibrary
            }
            try createPipelines()
            return
        }
        self.library = lib
        try createPipelines()
    }

    private func createPipelines() throws {
        // Create pipelines using Metal 4 compiler
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
        extractMaskPipeline = try createOptimizedPipeline("extract_mask")
    }

    private func createOptimizedPipeline(_ name: String) throws -> MTLComputePipelineState {
        // Metal 4: Use function descriptors instead of MTLFunction directly
        let functionDescriptor = MTL4LibraryFunctionDescriptor()
        functionDescriptor.name = name
        functionDescriptor.library = library

        // Use MTL4Compiler for pipeline creation with MTL4 descriptor
        let descriptor = MTL4ComputePipelineDescriptor()
        descriptor.computeFunctionDescriptor = functionDescriptor
        descriptor.label = name

        return try mtl4Compiler.makeComputePipelineState(descriptor: descriptor)
    }

    /// Create a Metal 4 command buffer from device (not queue)
    func makeCommandBuffer() -> MTL4CommandBuffer? {
        // Metal 4: Command buffers are created from device, then linked to allocator
        guard let commandBuffer = device.makeCommandBuffer() else {
            return nil
        }
        commandBuffer.beginCommandBuffer(allocator: commandAllocator)
        return commandBuffer
    }

    /// Add a buffer to the residency set
    func addToResidencySet(_ buffer: MTLBuffer) {
        residencySet.addAllocation(buffer)
    }

    /// Commit the residency set and add to command queue
    func commitResidency() {
        residencySet.commit()
        // Metal 4: Must add residency set to command queue for resources to be available
        mtl4CommandQueue.addResidencySet(residencySet)
    }

    /// Get the next argument table from the pool and advance the index
    func getNextArgumentTable() -> MTL4ArgumentTable {
        let table = argumentTables[currentTableIndex]
        currentTableIndex = (currentTableIndex + 1) % maxDispatchesPerCommandBuffer
        return table
    }

    /// Reset the argument table pool index at the start of each command buffer
    func resetArgumentTableIndex() {
        currentTableIndex = 0
    }

    /// Submit command buffer to Metal 4 queue and signal event
    func submit(_ commandBuffer: MTL4CommandBuffer) {
        eventValue += 1
        commandBuffer.endCommandBuffer()  // Metal 4: Must end command buffer before commit
        mtl4CommandQueue.commit([commandBuffer])  // Metal 4: commit takes array of command buffers
        mtl4CommandQueue.signalEvent(sharedEvent, value: eventValue)
    }

    /// Wait for all submitted commands to complete
    func waitForCompletion() {
        _ = sharedEvent.wait(untilSignaledValue: eventValue, timeoutMS: 10000)
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

/// Dispatches compute kernels using MTL4ComputeCommandEncoder with argument tables
/// MTL4ComputeCommandEncoder is a unified encoder that handles compute, blits, and acceleration structures
class MetalComputeDispatcher {
    let pipelineManager: MetalPipelineManager

    // SIMD width for optimal dispatching (Metal 4 typically uses 32)
    let simdWidth: Int = 32

    // Current offset into constants buffer for parameter storage
    var constantsOffset: Int = 0
    var debugMode: Bool = false

    init(pipelineManager: MetalPipelineManager) {
        self.pipelineManager = pipelineManager
    }

    /// Reset constants buffer offset at start of each command buffer
    func resetConstantsOffset() {
        constantsOffset = 0
    }

    /// Write Int32 constant to buffer and return its GPU address
    func writeConstant(_ value: Int32) -> UInt64 {
        // Bounds check - should never overflow with 64KB buffer
        guard constantsOffset + 4 <= pipelineManager.constantsBufferSize else {
            fatalError("Constants buffer overflow at offset \(constantsOffset)")
        }
        let ptr = pipelineManager.constantsBuffer.contents().advanced(by: constantsOffset)
        ptr.storeBytes(of: value, as: Int32.self)
        let address = pipelineManager.constantsBuffer.gpuAddress + UInt64(constantsOffset)
        constantsOffset += 4
        return address
    }

    /// Configure argument table with buffer bindings
    /// Uses a pool of argument tables to ensure each dispatch has a unique table state
    private func bindResources(encoder: MTL4ComputeCommandEncoder, buffers: [(MTLBuffer, Int)], constants: [(Int32, Int)], debugLabel: String = "") {
        // Get a fresh argument table from the pool for this dispatch
        let argTable = pipelineManager.getNextArgumentTable()
        let showDebug = debugMode && !debugLabel.isEmpty

        if showDebug {
            print("DEBUG bindResources: \(debugLabel)")
        }

        // Bind data buffers using GPU addresses
        for (buffer, index) in buffers {
            argTable.setAddress(buffer.gpuAddress, index: index)
            if showDebug {
                print("  Buffer[\(index)] = 0x\(String(format: "%llx", buffer.gpuAddress))")
            }
        }

        // Bind constants - write to constants buffer and set GPU address
        for (value, index) in constants {
            guard constantsOffset + 4 <= pipelineManager.constantsBufferSize else {
                fatalError("Constants buffer overflow at offset \(constantsOffset)")
            }
            let ptr = pipelineManager.constantsBuffer.contents().advanced(by: constantsOffset)
            ptr.storeBytes(of: value, as: Int32.self)
            let address = pipelineManager.constantsBuffer.gpuAddress + UInt64(constantsOffset)
            argTable.setAddress(address, index: index)
            constantsOffset += 4

            if showDebug {
                print("  Constant[\(index)] = \(value)")
            }
        }

        // Set the argument table for this dispatch
        encoder.setArgumentTable(argTable)
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
        encoder: MTL4ComputeCommandEncoder,
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

        // Configure argument table with buffers and constants
        let buffers: [(MTLBuffer, Int)] = [
            (input, 0),
            (weights, 1),
            (output, 2)
        ]

        var constants: [(Int32, Int)] = [
            (Int32(batchSize), 3),
            (Int32(inChannels), 4),
            (Int32(outChannels), 5),
            (Int32(height), 6),
            (Int32(width), 7)
        ]

        // Add kernel-specific parameters
        if kernelH == 1 && kernelW == 1 {
            // 1x1 conv: no additional params
        } else if kernelH == 3 && kernelW == 3 {
            // 3x3 tiled conv: dilation at indices 8-9
            constants.append((Int32(dilationH), 8))
            constants.append((Int32(dilationW), 9))
        } else {
            // General conv: kernel size at 8-9, dilation at 10-11
            constants.append((Int32(kernelH), 8))
            constants.append((Int32(kernelW), 9))
            constants.append((Int32(dilationH), 10))
            constants.append((Int32(dilationW), 11))
        }

        bindResources(encoder: encoder, buffers: buffers, constants: constants, debugLabel: "Conv2D \(kernelH)x\(kernelW)")

        let threadsPerGrid = MTLSize(width: width, height: height, depth: batchSize * outChannels)
        let threadsPerGroup = optimalThreadgroupSize(for: pipeline, workSize: threadsPerGrid)
        encoder.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)

        // Insert barrier to ensure this dispatch completes before dependent dispatches read output
        insertBarrier(encoder: encoder)
    }

    func dispatchBatchNorm(
        encoder: MTL4ComputeCommandEncoder,
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

        bindResources(
            encoder: encoder,
            buffers: [
                (input, 0),
                (scale, 1),
                (bias, 2),
                (mask, 3),
                (output, 4)
            ],
            constants: [
                (Int32(batchSize), 5),
                (Int32(channels), 6),
                (Int32(height), 7),
                (Int32(width), 8)
            ],
            debugLabel: "BatchNorm"
        )

        let threadsPerGrid = MTLSize(width: width, height: height, depth: batchSize * channels)
        let threadsPerGroup = optimalThreadgroupSize(for: pipeline, workSize: threadsPerGrid)
        encoder.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)

        // Insert barrier to ensure this dispatch completes before dependent dispatches read output
        insertBarrier(encoder: encoder)
    }

    /// Insert a barrier to ensure previous dispatches complete before subsequent ones read their output
    func insertBarrier(encoder: MTL4ComputeCommandEncoder) {
        // Metal 4 barrier: wait for dispatch stage to complete before allowing dispatch stage to proceed
        // Use .device visibility to ensure buffer writes are visible to subsequent reads
        encoder.barrier(afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch, visibilityOptions: .device)
    }

    func dispatchActivation(
        encoder: MTL4ComputeCommandEncoder,
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

        bindResources(
            encoder: encoder,
            buffers: [(input, 0), (output, 1)],
            constants: [(Int32(size), 2)]
        )

        let threadsPerGrid = MTLSize(width: size, height: 1, depth: 1)
        let threadsPerGroup = MTLSize(width: min(simdWidth * 8, size), height: 1, depth: 1)
        encoder.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)
        insertBarrier(encoder: encoder)
    }

    func dispatchMatMul(
        encoder: MTL4ComputeCommandEncoder,
        A: MTLBuffer,
        B: MTLBuffer,
        C: MTLBuffer,
        M: Int,
        K: Int,
        N: Int
    ) {
        encoder.setComputePipelineState(pipelineManager.matmulPipeline)

        bindResources(
            encoder: encoder,
            buffers: [(A, 0), (B, 1), (C, 2)],
            constants: [(Int32(M), 3), (Int32(K), 4), (Int32(N), 5)]
        )

        let threadsPerGrid = MTLSize(width: N, height: M, depth: 1)
        let threadsPerGroup = optimalThreadgroupSize(for: pipelineManager.matmulPipeline, workSize: threadsPerGrid)
        encoder.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)
        insertBarrier(encoder: encoder)
    }

    func dispatchAddNCBias(
        encoder: MTL4ComputeCommandEncoder,
        input: MTLBuffer,
        bias: MTLBuffer,
        output: MTLBuffer,
        batchSize: Int,
        channels: Int,
        height: Int,
        width: Int
    ) {
        encoder.setComputePipelineState(pipelineManager.addNCBiasPipeline)

        bindResources(
            encoder: encoder,
            buffers: [(input, 0), (bias, 1), (output, 2)],
            constants: [
                (Int32(batchSize), 3),
                (Int32(channels), 4),
                (Int32(height), 5),
                (Int32(width), 6)
            ]
        )

        let threadsPerGrid = MTLSize(width: width, height: height, depth: batchSize * channels)
        let threadsPerGroup = optimalThreadgroupSize(for: pipelineManager.addNCBiasPipeline, workSize: threadsPerGrid)
        encoder.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)
        insertBarrier(encoder: encoder)
    }

    func dispatchElementwiseAdd(
        encoder: MTL4ComputeCommandEncoder,
        a: MTLBuffer,
        b: MTLBuffer,
        output: MTLBuffer,
        size: Int
    ) {
        encoder.setComputePipelineState(pipelineManager.elementwiseAddPipeline)

        bindResources(
            encoder: encoder,
            buffers: [(a, 0), (b, 1), (output, 2)],
            constants: [(Int32(size), 3)]
        )

        let threadsPerGrid = MTLSize(width: size, height: 1, depth: 1)
        let threadsPerGroup = MTLSize(width: min(simdWidth * 8, size), height: 1, depth: 1)
        encoder.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)
        insertBarrier(encoder: encoder)
    }

    func dispatchMatBiasAdd(
        encoder: MTL4ComputeCommandEncoder,
        input: MTLBuffer,
        bias: MTLBuffer,
        output: MTLBuffer,
        batchSize: Int,
        channels: Int
    ) {
        encoder.setComputePipelineState(pipelineManager.matBiasAddPipeline)

        bindResources(
            encoder: encoder,
            buffers: [(input, 0), (bias, 1), (output, 2)],
            constants: [(Int32(batchSize), 3), (Int32(channels), 4)]
        )

        let threadsPerGrid = MTLSize(width: channels, height: batchSize, depth: 1)
        let threadsPerGroup = MTLSize(width: min(simdWidth * 8, channels), height: 1, depth: 1)
        encoder.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)
        insertBarrier(encoder: encoder)
    }

    func dispatchGlobalPooling(
        encoder: MTL4ComputeCommandEncoder,
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

        bindResources(
            encoder: encoder,
            buffers: [
                (input, 0),
                (mask, 1),
                (maskSum, 2),
                (maskSumSqrtS14M01, 3),
                (output, 4)
            ],
            constants: [
                (Int32(batchSize), 5),
                (Int32(channels), 6),
                (Int32(height), 7),
                (Int32(width), 8)
            ]
        )

        // Use SIMD-aligned threadgroup size for efficient reductions
        let threadgroupSize = simdWidth * 8  // 256 threads
        encoder.setThreadgroupMemoryLength(threadgroupSize * 4, index: 0)  // sharedSum
        encoder.setThreadgroupMemoryLength(threadgroupSize * 4, index: 1)  // sharedMax

        let threadsPerGrid = MTLSize(width: threadgroupSize, height: batchSize * channels, depth: 1)
        let threadsPerGroup = MTLSize(width: threadgroupSize, height: 1, depth: 1)
        encoder.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)
        insertBarrier(encoder: encoder)
    }

    func dispatchGlobalPoolingValue(
        encoder: MTL4ComputeCommandEncoder,
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

        bindResources(
            encoder: encoder,
            buffers: [
                (input, 0),
                (maskSum, 1),
                (maskSumSqrtS14M01, 2),
                (maskSumSqrtS14M01SquareS01, 3),
                (output, 4)
            ],
            constants: [
                (Int32(batchSize), 5),
                (Int32(channels), 6),
                (Int32(height), 7),
                (Int32(width), 8)
            ]
        )

        let threadgroupSize = simdWidth * 8
        encoder.setThreadgroupMemoryLength(threadgroupSize * 4, index: 0)

        let threadsPerGrid = MTLSize(width: threadgroupSize, height: batchSize * channels, depth: 1)
        let threadsPerGroup = MTLSize(width: threadgroupSize, height: 1, depth: 1)
        encoder.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)
        insertBarrier(encoder: encoder)
    }

    func dispatchMaskSum(
        encoder: MTL4ComputeCommandEncoder,
        mask: MTLBuffer,
        maskSum: MTLBuffer,
        batchSize: Int,
        height: Int,
        width: Int
    ) {
        encoder.setComputePipelineState(pipelineManager.maskSumPipeline)

        // Use simple per-thread dispatch (no threadgroup memory needed)
        bindResources(
            encoder: encoder,
            buffers: [(mask, 0), (maskSum, 1)],
            constants: [
                (Int32(batchSize), 2),
                (Int32(height), 3),
                (Int32(width), 4)
            ]
        )

        // Simple dispatch - one thread per batch element
        let threadsPerGrid = MTLSize(width: batchSize, height: 1, depth: 1)
        let threadsPerGroup = MTLSize(width: min(simdWidth, batchSize), height: 1, depth: 1)
        encoder.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)
        insertBarrier(encoder: encoder)
    }

    func dispatchMaskSumSqrtS14M01(
        encoder: MTL4ComputeCommandEncoder,
        maskSum: MTLBuffer,
        output: MTLBuffer,
        batchSize: Int
    ) {
        encoder.setComputePipelineState(pipelineManager.maskSumSqrtS14M01Pipeline)

        bindResources(
            encoder: encoder,
            buffers: [(maskSum, 0), (output, 1)],
            constants: [(Int32(batchSize), 2)]
        )

        let threadsPerGrid = MTLSize(width: batchSize, height: 1, depth: 1)
        let threadsPerGroup = MTLSize(width: min(simdWidth * 8, batchSize), height: 1, depth: 1)
        encoder.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)
        insertBarrier(encoder: encoder)
    }

    func dispatchMaskSumSqrtS14M01SquareS01(
        encoder: MTL4ComputeCommandEncoder,
        maskSumSqrtS14M01: MTLBuffer,
        output: MTLBuffer,
        batchSize: Int
    ) {
        encoder.setComputePipelineState(pipelineManager.maskSumSqrtS14M01SquareS01Pipeline)

        bindResources(
            encoder: encoder,
            buffers: [(maskSumSqrtS14M01, 0), (output, 1)],
            constants: [(Int32(batchSize), 2)]
        )

        let threadsPerGrid = MTLSize(width: batchSize, height: 1, depth: 1)
        let threadsPerGroup = MTLSize(width: min(simdWidth * 8, batchSize), height: 1, depth: 1)
        encoder.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)
        insertBarrier(encoder: encoder)
    }

    func dispatchExtractMask(
        encoder: MTL4ComputeCommandEncoder,
        input: MTLBuffer,
        mask: MTLBuffer,
        batchSize: Int,
        channels: Int,
        hw: Int
    ) {
        encoder.setComputePipelineState(pipelineManager.extractMaskPipeline)

        bindResources(
            encoder: encoder,
            buffers: [(input, 0), (mask, 1)],
            constants: [
                (Int32(batchSize), 2),
                (Int32(channels), 3),
                (Int32(hw), 4)
            ]
        )

        let threadsPerGrid = MTLSize(width: hw, height: batchSize, depth: 1)
        let threadsPerGroup = optimalThreadgroupSize(for: pipelineManager.extractMaskPipeline, workSize: threadsPerGrid)
        encoder.dispatchThreads(threadsPerGrid: threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)
        insertBarrier(encoder: encoder)
    }
}
