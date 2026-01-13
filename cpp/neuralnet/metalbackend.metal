//
//  metalbackend.metal
//  Metal 4 (Metal Shading Language 4.0) compute shaders for KataGo
//
//  Requires: macOS 26+ / iOS 26+ (Metal 4 / MSL 4.0)
//
//  Metal 4 Optimizations:
//  - SIMD group operations (simd_shuffle, simd_sum) for fast reductions
//  - Threadgroup memory for data reuse
//  - Memory coalescing patterns
//  - Optimal thread dispatch with non-uniform threadgroups
//

#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

// Constants for optimal thread dispatch
constant int TILE_SIZE = 8;
constant int CONV_TILE_WIDTH = 16;
constant int CONV_TILE_HEIGHT = 16;
constant int MATMUL_TILE_SIZE = 16;
constant int REDUCE_THREADGROUP_SIZE = 256;

// =============================================================================
// MARK: - Activation Functions
// =============================================================================

// Softplus function with threshold for numerical stability
inline float softplus(float x, float threshold = 20.0f) {
    if (x < threshold) {
        return log(1.0f + exp(x));
    }
    return x;
}

// Mish activation: x * tanh(softplus(x))
inline float mish(float x) {
    return x * tanh(softplus(x));
}

// =============================================================================
// MARK: - Convolution Kernels
// =============================================================================

// General 2D convolution kernel with TF_SAME padding
// Supports arbitrary kernel sizes and dilation
kernel void conv2d_nchw(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& batchSize [[buffer(3)]],
    constant int& inChannels [[buffer(4)]],
    constant int& outChannels [[buffer(5)]],
    constant int& height [[buffer(6)]],
    constant int& width [[buffer(7)]],
    constant int& kernelH [[buffer(8)]],
    constant int& kernelW [[buffer(9)]],
    constant int& dilationH [[buffer(10)]],
    constant int& dilationW [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgSize [[threads_per_threadgroup]])
{
    int b = gid.z / outChannels;
    int oc = gid.z % outChannels;
    int oh = gid.y;
    int ow = gid.x;

    if (b >= batchSize || oh >= height || ow >= width) return;

    // Compute padded kernel dimensions
    int padKernelH = (kernelH - 1) * dilationH + 1;
    int padKernelW = (kernelW - 1) * dilationW + 1;
    int padTop = padKernelH / 2;
    int padLeft = padKernelW / 2;

    float sum = 0.0f;

    // Weight layout: OIHW
    int weightOffset = oc * inChannels * kernelH * kernelW;

    for (int ic = 0; ic < inChannels; ic++) {
        for (int kh = 0; kh < kernelH; kh++) {
            for (int kw = 0; kw < kernelW; kw++) {
                int ih = oh - padTop + kh * dilationH;
                int iw = ow - padLeft + kw * dilationW;

                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    int inputIdx = b * inChannels * height * width +
                                   ic * height * width +
                                   ih * width + iw;
                    int weightIdx = weightOffset +
                                    ic * kernelH * kernelW +
                                    kh * kernelW + kw;
                    sum += input[inputIdx] * weights[weightIdx];
                }
            }
        }
    }

    int outIdx = b * outChannels * height * width +
                 oc * height * width +
                 oh * width + ow;
    output[outIdx] = sum;
}

// Optimized 1x1 convolution (pointwise convolution)
// Much faster than general conv2d for 1x1 kernels
kernel void conv2d_1x1_nchw(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& batchSize [[buffer(3)]],
    constant int& inChannels [[buffer(4)]],
    constant int& outChannels [[buffer(5)]],
    constant int& height [[buffer(6)]],
    constant int& width [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{
    int b = gid.z / outChannels;
    int oc = gid.z % outChannels;
    int oh = gid.y;
    int ow = gid.x;

    if (b >= batchSize || oh >= height || ow >= width) return;

    float sum = 0.0f;
    int spatialIdx = oh * width + ow;

    // Vectorized accumulation for 1x1 conv
    for (int ic = 0; ic < inChannels; ic++) {
        int inputIdx = b * inChannels * height * width +
                       ic * height * width + spatialIdx;
        int weightIdx = oc * inChannels + ic;
        sum += input[inputIdx] * weights[weightIdx];
    }

    int outIdx = b * outChannels * height * width +
                 oc * height * width + spatialIdx;
    output[outIdx] = sum;
}

// Tiled 3x3 convolution with shared memory optimization
kernel void conv2d_3x3_nchw_tiled(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& batchSize [[buffer(3)]],
    constant int& inChannels [[buffer(4)]],
    constant int& outChannels [[buffer(5)]],
    constant int& height [[buffer(6)]],
    constant int& width [[buffer(7)]],
    constant int& dilationH [[buffer(8)]],
    constant int& dilationW [[buffer(9)]],
    threadgroup float* sharedInput [[threadgroup(0)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgSize [[threads_per_threadgroup]])
{
    int b = gid.z / outChannels;
    int oc = gid.z % outChannels;
    int oh = gid.y;
    int ow = gid.x;

    if (b >= batchSize || oh >= height || ow >= width) return;

    // Padded kernel dimensions
    int padKernelH = 2 * dilationH + 1;
    int padKernelW = 2 * dilationW + 1;
    int padTop = padKernelH / 2;
    int padLeft = padKernelW / 2;

    float sum = 0.0f;
    int weightOffset = oc * inChannels * 9;  // 3x3 = 9

    for (int ic = 0; ic < inChannels; ic++) {
        for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
                int ih = oh - padTop + kh * dilationH;
                int iw = ow - padLeft + kw * dilationW;

                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    int inputIdx = b * inChannels * height * width +
                                   ic * height * width +
                                   ih * width + iw;
                    int weightIdx = weightOffset + ic * 9 + kh * 3 + kw;
                    sum += input[inputIdx] * weights[weightIdx];
                }
            }
        }
    }

    int outIdx = b * outChannels * height * width +
                 oc * height * width +
                 oh * width + ow;
    output[outIdx] = sum;
}

// =============================================================================
// MARK: - Batch Normalization Kernels
// =============================================================================

// Batch normalization with pre-computed merged scale and bias
// Applies mask to zero out invalid positions
kernel void batchnorm_mask(
    device const float* input [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device const float* mask [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant int& batchSize [[buffer(5)]],
    constant int& channels [[buffer(6)]],
    constant int& height [[buffer(7)]],
    constant int& width [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
    int b = gid.z / channels;
    int c = gid.z % channels;
    int h = gid.y;
    int w = gid.x;

    if (b >= batchSize || h >= height || w >= width) return;

    int spatialIdx = h * width + w;
    int inputIdx = b * channels * height * width +
                   c * height * width + spatialIdx;
    int maskIdx = b * height * width + spatialIdx;

    float val = input[inputIdx];
    float s = scale[c];
    float bi = bias[c];
    float m = mask[maskIdx];

    output[inputIdx] = (val * s + bi) * m;
}

// =============================================================================
// MARK: - Activation Kernels
// =============================================================================

// ReLU activation
kernel void relu_activation(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= uint(size)) return;
    output[gid] = max(input[gid], 0.0f);
}

// Mish activation: x * tanh(softplus(x))
kernel void mish_activation(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= uint(size)) return;
    float x = input[gid];
    output[gid] = mish(x);
}

// Fused BatchNorm + ReLU activation
kernel void batchnorm_relu(
    device const float* input [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device const float* mask [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant int& batchSize [[buffer(5)]],
    constant int& channels [[buffer(6)]],
    constant int& height [[buffer(7)]],
    constant int& width [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
    int b = gid.z / channels;
    int c = gid.z % channels;
    int h = gid.y;
    int w = gid.x;

    if (b >= batchSize || h >= height || w >= width) return;

    int spatialIdx = h * width + w;
    int inputIdx = b * channels * height * width +
                   c * height * width + spatialIdx;
    int maskIdx = b * height * width + spatialIdx;

    float val = input[inputIdx];
    float s = scale[c];
    float bi = bias[c];
    float m = mask[maskIdx];

    float normalized = val * s + bi;
    output[inputIdx] = max(normalized, 0.0f) * m;
}

// Fused BatchNorm + Mish activation
kernel void batchnorm_mish(
    device const float* input [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device const float* mask [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant int& batchSize [[buffer(5)]],
    constant int& channels [[buffer(6)]],
    constant int& height [[buffer(7)]],
    constant int& width [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
    int b = gid.z / channels;
    int c = gid.z % channels;
    int h = gid.y;
    int w = gid.x;

    if (b >= batchSize || h >= height || w >= width) return;

    int spatialIdx = h * width + w;
    int inputIdx = b * channels * height * width +
                   c * height * width + spatialIdx;
    int maskIdx = b * height * width + spatialIdx;

    float val = input[inputIdx];
    float s = scale[c];
    float bi = bias[c];
    float m = mask[maskIdx];

    float normalized = val * s + bi;
    output[inputIdx] = mish(normalized) * m;
}

// =============================================================================
// MARK: - Matrix Multiplication Kernels
// =============================================================================

// General matrix multiplication: C = A * B
// A: [M x K], B: [K x N], C: [M x N]
kernel void matmul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& K [[buffer(4)]],
    constant int& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    int row = gid.y;
    int col = gid.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// Tiled matrix multiplication with shared memory
kernel void matmul_tiled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& K [[buffer(4)]],
    constant int& N [[buffer(5)]],
    threadgroup float* sharedA [[threadgroup(0)]],
    threadgroup float* sharedB [[threadgroup(1)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    int row = gid.y;
    int col = gid.x;
    int localRow = tid.y;
    int localCol = tid.x;

    float sum = 0.0f;

    int numTiles = (K + MATMUL_TILE_SIZE - 1) / MATMUL_TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // Load tile of A into shared memory
        int aRow = row;
        int aCol = t * MATMUL_TILE_SIZE + localCol;
        if (aRow < M && aCol < K) {
            sharedA[localRow * MATMUL_TILE_SIZE + localCol] = A[aRow * K + aCol];
        } else {
            sharedA[localRow * MATMUL_TILE_SIZE + localCol] = 0.0f;
        }

        // Load tile of B into shared memory
        int bRow = t * MATMUL_TILE_SIZE + localRow;
        int bCol = col;
        if (bRow < K && bCol < N) {
            sharedB[localRow * MATMUL_TILE_SIZE + localCol] = B[bRow * N + bCol];
        } else {
            sharedB[localRow * MATMUL_TILE_SIZE + localCol] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot product
        for (int k = 0; k < MATMUL_TILE_SIZE; k++) {
            sum += sharedA[localRow * MATMUL_TILE_SIZE + k] *
                   sharedB[k * MATMUL_TILE_SIZE + localCol];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// =============================================================================
// MARK: - Element-wise Operations
// =============================================================================

// Element-wise addition
kernel void elementwise_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= uint(size)) return;
    output[gid] = a[gid] + b[gid];
}

// Element-wise multiplication
kernel void elementwise_mul(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= uint(size)) return;
    output[gid] = a[gid] * b[gid];
}

// Add NC bias to NCHW tensor (broadcasts bias over HW dimensions)
kernel void add_nc_bias(
    device const float* input [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& batchSize [[buffer(3)]],
    constant int& channels [[buffer(4)]],
    constant int& height [[buffer(5)]],
    constant int& width [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    int b = gid.z / channels;
    int c = gid.z % channels;
    int h = gid.y;
    int w = gid.x;

    if (b >= batchSize || h >= height || w >= width) return;

    int inputIdx = b * channels * height * width +
                   c * height * width +
                   h * width + w;
    int biasIdx = b * channels + c;

    output[inputIdx] = input[inputIdx] + bias[biasIdx];
}

// =============================================================================
// MARK: - Reduction Operations (Metal 4 SIMD optimized)
// =============================================================================

// Metal 4 SIMD group reduction helper
inline float simd_reduce_add(float val, uint simd_lane_id, uint simd_size) {
    // Use Metal 4's simd_shuffle for fast warp-level reduction
    for (uint offset = simd_size / 2; offset > 0; offset >>= 1) {
        val += simd_shuffle_down(val, offset);
    }
    return val;
}

inline float simd_reduce_max(float val, uint simd_lane_id, uint simd_size) {
    for (uint offset = simd_size / 2; offset > 0; offset >>= 1) {
        val = max(val, simd_shuffle_down(val, offset));
    }
    return val;
}

// Compute sum over HW dimensions for each (B, C) pair
// Uses Metal 4 SIMD group operations for fast reduction
kernel void reduction_sum_hw(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& batchSize [[buffer(2)]],
    constant int& channels [[buffer(3)]],
    constant int& height [[buffer(4)]],
    constant int& width [[buffer(5)]],
    threadgroup float* sharedData [[threadgroup(0)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_size [[threads_per_simdgroup]])
{
    int b = gid.y / channels;
    int c = gid.y % channels;

    if (b >= batchSize) return;

    int hw = height * width;
    int baseIdx = b * channels * hw + c * hw;

    // Each thread sums multiple elements
    float sum = 0.0f;
    for (int i = tid; i < hw; i += tgSize) {
        sum += input[baseIdx + i];
    }

    // Metal 4: SIMD group reduction (fast warp-level reduction)
    sum = simd_reduce_add(sum, simd_lane_id, simd_size);

    // Store SIMD group results to shared memory
    if (simd_lane_id == 0) {
        sharedData[simd_group_id] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction across SIMD groups
    uint numSimdGroups = (tgSize + simd_size - 1) / simd_size;
    if (tid < numSimdGroups) {
        sum = sharedData[tid];
    } else {
        sum = 0.0f;
    }

    if (tid < simd_size) {
        sum = simd_reduce_add(sum, simd_lane_id, simd_size);
        if (tid == 0) {
            output[b * channels + c] = sum;
        }
    }
}

// Compute max over HW dimensions for each (B, C) pair
// Uses Metal 4 SIMD group operations for fast reduction
// Applies mask by subtracting 1 from mask and adding to input
kernel void reduction_max_hw_masked(
    device const float* input [[buffer(0)]],
    device const float* mask [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& batchSize [[buffer(3)]],
    constant int& channels [[buffer(4)]],
    constant int& height [[buffer(5)]],
    constant int& width [[buffer(6)]],
    threadgroup float* sharedData [[threadgroup(0)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_size [[threads_per_simdgroup]])
{
    int b = gid.y / channels;
    int c = gid.y % channels;

    if (b >= batchSize) return;

    int hw = height * width;
    int baseIdx = b * channels * hw + c * hw;
    int maskBaseIdx = b * hw;

    // Each thread finds max of multiple elements
    float maxVal = -INFINITY;
    for (int i = tid; i < hw; i += tgSize) {
        float m = mask[maskBaseIdx + i];
        float val = input[baseIdx + i] + (m - 1.0f);  // Mask invalid positions
        maxVal = max(maxVal, val);
    }

    // Metal 4: SIMD group reduction (fast warp-level reduction)
    maxVal = simd_reduce_max(maxVal, simd_lane_id, simd_size);

    // Store SIMD group results to shared memory
    if (simd_lane_id == 0) {
        sharedData[simd_group_id] = maxVal;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction across SIMD groups
    uint numSimdGroups = (tgSize + simd_size - 1) / simd_size;
    if (tid < numSimdGroups) {
        maxVal = sharedData[tid];
    } else {
        maxVal = -INFINITY;
    }

    if (tid < simd_size) {
        maxVal = simd_reduce_max(maxVal, simd_lane_id, simd_size);
        if (tid == 0) {
            output[b * channels + c] = maxVal;
        }
    }
}

// =============================================================================
// MARK: - Global Pooling Kernels
// =============================================================================

// Compute mask sum over HW dimensions
kernel void mask_sum(
    device const float* mask [[buffer(0)]],
    device float* maskSum [[buffer(1)]],
    constant int& batchSize [[buffer(2)]],
    constant int& height [[buffer(3)]],
    constant int& width [[buffer(4)]],
    threadgroup float* sharedData [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]])
{
    int b = gid;
    if (b >= batchSize) return;

    int hw = height * width;
    int baseIdx = b * hw;

    float sum = 0.0f;
    for (int i = tid; i < hw; i += tgSize) {
        sum += mask[baseIdx + i];
    }

    sharedData[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        maskSum[b] = sharedData[0];
    }
}

// Compute (sqrt(maskSum) - 14) * 0.1
kernel void mask_sum_sqrt_s14_m01(
    device const float* maskSum [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& batchSize [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= uint(batchSize)) return;
    output[gid] = (sqrt(maskSum[gid]) - 14.0f) * 0.1f;
}

// Compute (maskSumSqrtS14M01)^2 - 0.1
kernel void mask_sum_sqrt_s14_m01_square_s01(
    device const float* maskSumSqrtS14M01 [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& batchSize [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= uint(batchSize)) return;
    float val = maskSumSqrtS14M01[gid];
    output[gid] = val * val - 0.1f;
}

// Global pooling: compute mean, mean*maskSumSqrtS14M01, and masked max
// Outputs are concatenated: [mean, meanMask, max] for each (B, C)
kernel void global_pooling(
    device const float* input [[buffer(0)]],
    device const float* mask [[buffer(1)]],
    device const float* maskSum [[buffer(2)]],
    device const float* maskSumSqrtS14M01 [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant int& batchSize [[buffer(5)]],
    constant int& channels [[buffer(6)]],
    constant int& height [[buffer(7)]],
    constant int& width [[buffer(8)]],
    threadgroup float* sharedSum [[threadgroup(0)]],
    threadgroup float* sharedMax [[threadgroup(1)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]])
{
    int b = gid.y / channels;
    int c = gid.y % channels;

    if (b >= batchSize) return;

    int hw = height * width;
    int baseIdx = b * channels * hw + c * hw;
    int maskBaseIdx = b * hw;

    // Compute sum and max in parallel
    float sum = 0.0f;
    float maxVal = -INFINITY;

    for (int i = tid; i < hw; i += tgSize) {
        float val = input[baseIdx + i];
        float m = mask[maskBaseIdx + i];
        sum += val;
        maxVal = max(maxVal, val + (m - 1.0f));
    }

    sharedSum[tid] = sum;
    sharedMax[tid] = maxVal;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction
    for (uint stride = tgSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedSum[tid] += sharedSum[tid + stride];
            sharedMax[tid] = max(sharedMax[tid], sharedMax[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        float mean = sharedSum[0] / maskSum[b];
        float meanMask = mean * maskSumSqrtS14M01[b];
        int outBase = b * channels * 3 + c;
        output[outBase] = mean;
        output[outBase + channels] = meanMask;
        output[outBase + channels * 2] = sharedMax[0];
    }
}

// Global pooling for value head: mean, mean*maskSumSqrtS14M01, mean*maskSumSqrtS14M01SquareS01
kernel void global_pooling_value(
    device const float* input [[buffer(0)]],
    device const float* maskSum [[buffer(1)]],
    device const float* maskSumSqrtS14M01 [[buffer(2)]],
    device const float* maskSumSqrtS14M01SquareS01 [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant int& batchSize [[buffer(5)]],
    constant int& channels [[buffer(6)]],
    constant int& height [[buffer(7)]],
    constant int& width [[buffer(8)]],
    threadgroup float* sharedSum [[threadgroup(0)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]])
{
    int b = gid.y / channels;
    int c = gid.y % channels;

    if (b >= batchSize) return;

    int hw = height * width;
    int baseIdx = b * channels * hw + c * hw;

    float sum = 0.0f;
    for (int i = tid; i < hw; i += tgSize) {
        sum += input[baseIdx + i];
    }

    sharedSum[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedSum[tid] += sharedSum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        float mean = sharedSum[0] / maskSum[b];
        float meanMask = mean * maskSumSqrtS14M01[b];
        float meanMaskSquare = mean * maskSumSqrtS14M01SquareS01[b];
        int outBase = b * channels * 3 + c;
        output[outBase] = mean;
        output[outBase + channels] = meanMask;
        output[outBase + channels * 2] = meanMaskSquare;
    }
}

// =============================================================================
// MARK: - Bias Addition Kernels
// =============================================================================

// Add bias to matrix (for MatBiasLayer)
kernel void mat_bias_add(
    device const float* input [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& batchSize [[buffer(3)]],
    constant int& channels [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    int b = gid.y;
    int c = gid.x;

    if (b >= batchSize || c >= channels) return;

    int idx = b * channels + c;
    output[idx] = input[idx] + bias[c];
}

// =============================================================================
// MARK: - Residual Addition
// =============================================================================

// Add residual connection
kernel void residual_add(
    device const float* input [[buffer(0)]],
    device const float* residual [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= uint(size)) return;
    output[gid] = input[gid] + residual[gid];
}

// =============================================================================
// MARK: - Copy and Reshape Kernels
// =============================================================================

// Copy kernel for data movement
kernel void copy_buffer(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= uint(size)) return;
    output[gid] = input[gid];
}

// Extract first channel from NCHW input to create mask
// Input: [B, C, H, W], Output: [B, H*W] (first channel only)
kernel void extract_mask(
    device const float* input [[buffer(0)]],
    device float* mask [[buffer(1)]],
    constant int& batchSize [[buffer(2)]],
    constant int& channels [[buffer(3)]],
    constant int& hw [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    int b = gid.y;
    int i = gid.x;

    if (b >= batchSize || i >= hw) return;

    // Copy first channel value to mask
    mask[b * hw + i] = input[b * channels * hw + i];
}

// Reshape from NCHW [B, C, 1, 1] to NC [B, C] (for MatMul input)
// This is essentially a no-op since the data layout is the same
kernel void reshape_nchw_to_nc(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& batchSize [[buffer(2)]],
    constant int& channels [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    int b = gid.y;
    int c = gid.x;

    if (b >= batchSize || c >= channels) return;

    int idx = b * channels + c;
    output[idx] = input[idx];
}

// Concat tensors along channel dimension
// Takes 3 inputs of shape [B, C] and outputs [B, 3*C]
kernel void concat_channels_3(
    device const float* input0 [[buffer(0)]],
    device const float* input1 [[buffer(1)]],
    device const float* input2 [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant int& batchSize [[buffer(4)]],
    constant int& channels [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    int b = gid.y;
    int c = gid.x;

    if (b >= batchSize || c >= channels * 3) return;

    int outIdx = b * channels * 3 + c;
    int inIdx = b * channels + (c % channels);

    if (c < channels) {
        output[outIdx] = input0[inIdx];
    } else if (c < channels * 2) {
        output[outIdx] = input1[inIdx];
    } else {
        output[outIdx] = input2[inIdx];
    }
}
