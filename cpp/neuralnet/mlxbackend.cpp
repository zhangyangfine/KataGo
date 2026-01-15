#ifdef USE_MLX_BACKEND

/**
 * MLX backend for KataGo.
 * Uses Apple's MLX framework for neural network inference on Apple Silicon.
 * Only supports float32 computation with NHWC memory layout.
 */

#include "../neuralnet/nninterface.h"
#include "../neuralnet/desc.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/activations.h"
#include "../core/global.h"
#include "../core/test.h"

#include <mlx/mlx.h>
#include <iostream>
#include <cstring>
#include <memory>
#include <mutex>
#include <map>
#include <tuple>

namespace mx = mlx::core;
using namespace std;


// LoadedModel / ModelDesc ---------------------------------------------------------------------------------------------

struct LoadedModel {
  ModelDesc modelDesc;

  LoadedModel(const string& fileName, const string& expectedSha256) {
    ModelDesc::loadFromFileMaybeGZipped(fileName, modelDesc, expectedSha256);
  }

  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

LoadedModel* NeuralNet::loadModelFile(const string& file, const string& expectedSha256) {
  LoadedModel* loadedModel = new LoadedModel(file, expectedSha256);
  return loadedModel;
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel;
}

const ModelDesc& NeuralNet::getModelDesc(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc;
}

// Helpers --------------------------------------------------------------------------------------------------------------

// Convert convolution weights from OIHW to OHWI (MLX conv2d weight format)
static mx::array convertConvWeightsOIHWtoOHWI(const vector<float>& weights,
                                               int outChannels, int inChannels,
                                               int kH, int kW) {
  // Original: [outC, inC, kH, kW] - stored in column-major order
  // Target: [outC, kH, kW, inC]
  vector<float> converted(weights.size());
  for (int oc = 0; oc < outChannels; oc++) {
    for (int ic = 0; ic < inChannels; ic++) {
      for (int h = 0; h < kH; h++) {
        for (int w = 0; w < kW; w++) {
          int srcIdx = ((oc * inChannels + ic) * kH + h) * kW + w;
          int dstIdx = ((oc * kH + h) * kW + w) * inChannels + ic;
          converted[dstIdx] = weights[srcIdx];
        }
      }
    }
  }
  mx::Shape shape = {outChannels, kH, kW, inChannels};
  return mx::array(converted.data(), shape, mx::float32);
}

// Mish activation: x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
static mx::array applyMish(const mx::array& x) {
  // softplus(x) = log(1 + exp(x)) = log(exp(0) + exp(x)) = logaddexp(0, x)
  // logaddexp handles numerical stability internally
  mx::array softplus = mx::logaddexp(mx::array(0.0f), x);
  return x * mx::tanh(softplus);
}

// Apply activation function
static mx::array applyActivation(const mx::array& x, int activationType) {
  switch(activationType) {
    case ACTIVATION_RELU:
      return mx::maximum(x, mx::array(0.0f));
    case ACTIVATION_MISH:
      return applyMish(x);
    case ACTIVATION_IDENTITY:
    default:
      return x;
  }
}

// Fused matmul + bias: result = input @ weights + bias
// Uses addmm for better performance (single kernel instead of matmul + add)
static mx::array matmulBias(const mx::array& input, const mx::array& weights, const mx::array& bias) {
  // addmm(c, a, b, alpha, beta) = alpha * (a @ b) + beta * c
  return mx::addmm(bias, input, weights, 1.0f, 1.0f);
}

// Layers --------------------------------------------------------------------------------------------------------------

struct ConvLayer {
  const string name;
  const int convYSize;
  const int convXSize;
  const int inChannels;
  const int outChannels;
  const int dilationY;
  const int dilationX;
  mx::array weights; // OHWI format

  ConvLayer() = delete;
  ConvLayer(const ConvLayer&) = delete;
  ConvLayer& operator=(const ConvLayer&) = delete;

  ConvLayer(const ConvLayerDesc& desc)
    : name(desc.name),
      convYSize(desc.convYSize),
      convXSize(desc.convXSize),
      inChannels(desc.inChannels),
      outChannels(desc.outChannels),
      dilationY(desc.dilationY),
      dilationX(desc.dilationX),
      weights(convertConvWeightsOIHWtoOHWI(desc.weights, outChannels, inChannels, convYSize, convXSize))
  {}

  mx::array apply(const mx::array& input) const {
    // MLX conv2d: input NHWC, weights OHWI
    // Compute padding to maintain spatial dimensions (same padding)
    int padY = (convYSize - 1) * dilationY / 2;
    int padX = (convXSize - 1) * dilationX / 2;

    return mx::conv2d(
      input,
      weights,
      /*stride=*/std::make_pair(1, 1),
      /*padding=*/std::make_pair(padY, padX),
      /*dilation=*/std::make_pair(dilationY, dilationX),
      /*groups=*/1
    );
  }
};

struct BatchNormLayer {
  const string name;
  const int numChannels;
  const int activation;
  mx::array mergedScale; // Shape: [C]
  mx::array mergedBias;  // Shape: [C]

  BatchNormLayer() = delete;
  BatchNormLayer(const BatchNormLayer&) = delete;
  BatchNormLayer& operator=(const BatchNormLayer&) = delete;

  static mx::array createArray1D(const std::vector<float>& data, int size) {
    mx::Shape shape = {size};
    return mx::array(data.data(), shape, mx::float32);
  }

  static std::vector<float> getMergedScale(const BatchNormLayerDesc& desc) {
    // If mergedScale is already computed, use it
    if(!desc.mergedScale.empty()) {
      return desc.mergedScale;
    }
    // Otherwise compute from mean/variance/scale/bias (for tests)
    std::vector<float> mergedScale(desc.numChannels);
    for(int c = 0; c < desc.numChannels; c++) {
      mergedScale[c] = desc.scale[c] / sqrt(desc.variance[c] + desc.epsilon);
    }
    return mergedScale;
  }

  static std::vector<float> getMergedBias(const BatchNormLayerDesc& desc) {
    // If mergedBias is already computed, use it
    if(!desc.mergedBias.empty()) {
      return desc.mergedBias;
    }
    // Otherwise compute from mean/variance/scale/bias (for tests)
    std::vector<float> mergedBias(desc.numChannels);
    for(int c = 0; c < desc.numChannels; c++) {
      float ms = desc.scale[c] / sqrt(desc.variance[c] + desc.epsilon);
      mergedBias[c] = desc.bias[c] - ms * desc.mean[c];
    }
    return mergedBias;
  }

  BatchNormLayer(const BatchNormLayerDesc& desc, int activationType)
    : name(desc.name),
      numChannels(desc.numChannels),
      activation(activationType),
      mergedScale(createArray1D(getMergedScale(desc), desc.numChannels)),
      mergedBias(createArray1D(getMergedBias(desc), desc.numChannels))
  {}

  mx::array apply(const mx::array& input, const mx::array& mask, bool useMask) const {
    // input: NHWC [N, H, W, C]
    // mask: NHW1 [N, H, W, 1]
    // BN: output = input * scale + bias
    mx::array normalized = input * mergedScale + mergedBias;
    mx::array activated = applyActivation(normalized, activation);
    // Apply mask (zero out padded regions) - skip when useMask=false
    if(useMask)
      return activated * mask;
    return activated;
  }
};

struct MatMulLayer {
  const string name;
  const int inChannels;
  const int outChannels;
  mx::array weights; // [inC, outC]

  MatMulLayer() = delete;
  MatMulLayer(const MatMulLayer&) = delete;
  MatMulLayer& operator=(const MatMulLayer&) = delete;

  static mx::array createWeights(const MatMulLayerDesc& desc) {
    if(desc.inChannels > 0 && desc.outChannels > 0) {
      // Original weights: [inC, outC] (column-major)
      mx::Shape shape = {desc.inChannels, desc.outChannels};
      return mx::array(desc.weights.data(), shape, mx::float32);
    }
    std::vector<float> dummy = {0.0f};
    mx::Shape shape = {1};
    return mx::array(dummy.data(), shape, mx::float32);
  }

  MatMulLayer(const MatMulLayerDesc& desc)
    : name(desc.name),
      inChannels(desc.inChannels),
      outChannels(desc.outChannels),
      weights(createWeights(desc))
  {}

  mx::array apply(const mx::array& input) const {
    // input: [N, inC]
    // output: [N, outC]
    return mx::matmul(input, weights);
  }
};

struct MatBiasLayer {
  const string name;
  const int numChannels;
  mx::array bias;

  MatBiasLayer() = delete;
  MatBiasLayer(const MatBiasLayer&) = delete;
  MatBiasLayer& operator=(const MatBiasLayer&) = delete;

  static mx::array createBias(const MatBiasLayerDesc& desc) {
    mx::Shape shape = {desc.numChannels};
    return mx::array(desc.weights.data(), shape, mx::float32);
  }

  MatBiasLayer(const MatBiasLayerDesc& desc)
    : name(desc.name),
      numChannels(desc.numChannels),
      bias(createBias(desc))
  {}

  mx::array apply(const mx::array& input) const {
    return input + bias;
  }
};

// Global pooling: computes [mean, mean * (sqrt(maskSum) - 14) * 0.1, max] concatenated along channel axis
static mx::array applyGlobalPooling(const mx::array& input, const mx::array& mask, const mx::array& maskSum, bool useMask) {
  // input: NHWC [N, H, W, C]
  // mask: NHW1 [N, H, W, 1]
  // maskSum: N111 [N, 1, 1, 1]

  // Compute sum over spatial dims
  std::vector<int> spatialAxes = {1, 2};
  mx::array spatialSum = mx::sum(input, spatialAxes, /*keepdims=*/true); // [N, 1, 1, C]

  // Mean = sum / maskSum
  mx::array mean = spatialSum / maskSum; // [N, 1, 1, C]

  // sqrt(maskSum) - 14) * 0.1
  mx::array sqrtMaskSum = mx::sqrt(maskSum);
  mx::array scaleFactor = (sqrtMaskSum - mx::array(14.0f)) * mx::array(0.1f);
  mx::array meanScaled = mean * scaleFactor;

  // Max - skip mask adjustment when useMask=false (all positions valid)
  mx::array maxVal = useMask
    ? mx::max(input - (mx::array(1.0f) - mask) * mx::array(1e9f), spatialAxes, /*keepdims=*/true)
    : mx::max(input, spatialAxes, /*keepdims=*/true);

  // Concatenate along channel axis (axis 3 for NHWC)
  std::vector<mx::array> concatInputs = {mean, meanScaled, maxVal};
  return mx::concatenate(concatInputs, /*axis=*/3);
}

// Value head pooling: computes [mean, mean * (sqrt(maskSum) - 14) * 0.1, mean * ((sqrt-14)^2 * 0.01 - 0.1)]
static mx::array applyValueHeadPooling(const mx::array& input, const mx::array& maskSum) {
  // input: NHWC [N, H, W, C]
  // maskSum: N111 [N, 1, 1, 1]

  std::vector<int> spatialAxes = {1, 2};
  mx::array spatialSum = mx::sum(input, spatialAxes, /*keepdims=*/true);
  mx::array mean = spatialSum / maskSum;

  mx::array sqrtMaskSum = mx::sqrt(maskSum);
  mx::array diff = sqrtMaskSum - mx::array(14.0f);
  mx::array meanScaled1 = mean * diff * mx::array(0.1f);
  mx::array meanScaled2 = mean * (diff * diff * mx::array(0.01f) - mx::array(0.1f));

  std::vector<mx::array> concatInputs = {mean, meanScaled1, meanScaled2};
  return mx::concatenate(concatInputs, /*axis=*/3);
}

// Residual Block
struct ResidualBlock {
  const string name;
  const BatchNormLayer preBN;
  const ConvLayer regularConv;
  const BatchNormLayer midBN;
  const ConvLayer finalConv;

  ResidualBlock() = delete;
  ResidualBlock(const ResidualBlock&) = delete;
  ResidualBlock& operator=(const ResidualBlock&) = delete;

  ResidualBlock(const ResidualBlockDesc& desc)
    : name(desc.name),
      preBN(desc.preBN, desc.preActivation.activation),
      regularConv(desc.regularConv),
      midBN(desc.midBN, desc.midActivation.activation),
      finalConv(desc.finalConv)
  {}

  mx::array apply(const mx::array& input, const mx::array& mask, bool useMask) const {
    mx::array out = preBN.apply(input, mask, useMask);
    out = regularConv.apply(out);
    out = midBN.apply(out, mask, useMask);
    out = finalConv.apply(out);
    return input + out;
  }
};

// Global Pooling Residual Block
struct GlobalPoolingResidualBlock {
  const string name;
  const BatchNormLayer preBN;
  const ConvLayer regularConv;
  const ConvLayer gpoolConv;
  const BatchNormLayer gpoolBN;
  const MatMulLayer gpoolToBiasMul;
  const BatchNormLayer midBN;
  const ConvLayer finalConv;

  GlobalPoolingResidualBlock() = delete;
  GlobalPoolingResidualBlock(const GlobalPoolingResidualBlock&) = delete;
  GlobalPoolingResidualBlock& operator=(const GlobalPoolingResidualBlock&) = delete;

  GlobalPoolingResidualBlock(const GlobalPoolingResidualBlockDesc& desc)
    : name(desc.name),
      preBN(desc.preBN, desc.preActivation.activation),
      regularConv(desc.regularConv),
      gpoolConv(desc.gpoolConv),
      gpoolBN(desc.gpoolBN, desc.gpoolActivation.activation),
      gpoolToBiasMul(desc.gpoolToBiasMul),
      midBN(desc.midBN, desc.midActivation.activation),
      finalConv(desc.finalConv)
  {}

  mx::array apply(const mx::array& input, const mx::array& mask, const mx::array& maskSum, bool useMask) const {
    mx::array preOut = preBN.apply(input, mask, useMask);

    // Regular path
    mx::array regularOut = regularConv.apply(preOut);

    // Global pooling path
    mx::array gpoolOut = gpoolConv.apply(preOut);
    gpoolOut = gpoolBN.apply(gpoolOut, mask, useMask);
    mx::array pooled = applyGlobalPooling(gpoolOut, mask, maskSum, useMask);

    // Squeeze spatial dims for matmul: [N, 1, 1, C*3] -> [N, C*3]
    std::vector<int> squeezeAxes = {1, 2};
    mx::array pooledFlat = mx::squeeze(pooled, squeezeAxes);
    mx::array bias = gpoolToBiasMul.apply(pooledFlat);

    // Add bias to regular path (broadcast): [N, outC] -> [N, 1, 1, outC]
    mx::Shape biasShape = {static_cast<int>(bias.shape()[0]), 1, 1, static_cast<int>(bias.shape()[1])};
    bias = mx::reshape(bias, biasShape);
    mx::array combined = regularOut + bias;

    combined = midBN.apply(combined, mask, useMask);
    mx::array finalOut = finalConv.apply(combined);

    return input + finalOut;
  }
};

// Nested Bottleneck Residual Block (simplified - forward declaration for recursive types)
struct NestedBottleneckResidualBlock;

// Block variant type for trunk
struct BlockVariant {
  enum Type { REGULAR, GLOBAL_POOLING, NESTED_BOTTLENECK };
  Type type;
  unique_ptr<ResidualBlock> regular;
  unique_ptr<GlobalPoolingResidualBlock> globalPooling;
  unique_ptr<NestedBottleneckResidualBlock> nestedBottleneck;

  BlockVariant(const ResidualBlockDesc& desc)
    : type(REGULAR), regular(make_unique<ResidualBlock>(desc)) {}

  BlockVariant(const GlobalPoolingResidualBlockDesc& desc)
    : type(GLOBAL_POOLING), globalPooling(make_unique<GlobalPoolingResidualBlock>(desc)) {}

  // Forward declaration - defined after NestedBottleneckResidualBlock
  BlockVariant(const NestedBottleneckResidualBlockDesc& desc);

  mx::array apply(const mx::array& input, const mx::array& mask, const mx::array& maskSum, bool useMask) const;
};

struct NestedBottleneckResidualBlock {
  const string name;
  const BatchNormLayer preBN;
  const ConvLayer preConv;
  vector<BlockVariant> blocks;
  const BatchNormLayer postBN;
  const ConvLayer postConv;

  NestedBottleneckResidualBlock() = delete;
  NestedBottleneckResidualBlock(const NestedBottleneckResidualBlock&) = delete;
  NestedBottleneckResidualBlock& operator=(const NestedBottleneckResidualBlock&) = delete;

  NestedBottleneckResidualBlock(const NestedBottleneckResidualBlockDesc& desc)
    : name(desc.name),
      preBN(desc.preBN, desc.preActivation.activation),
      preConv(desc.preConv),
      postBN(desc.postBN, desc.postActivation.activation),
      postConv(desc.postConv)
  {
    for(size_t i = 0; i < desc.blocks.size(); i++) {
      int blockKind = desc.blocks[i].first;
      if(blockKind == ORDINARY_BLOCK_KIND) {
        blocks.emplace_back(*static_cast<ResidualBlockDesc*>(desc.blocks[i].second.get()));
      }
      else if(blockKind == GLOBAL_POOLING_BLOCK_KIND) {
        blocks.emplace_back(*static_cast<GlobalPoolingResidualBlockDesc*>(desc.blocks[i].second.get()));
      }
    }
  }

  mx::array apply(const mx::array& input, const mx::array& mask, const mx::array& maskSum, bool useMask) const {
    mx::array out = preBN.apply(input, mask, useMask);
    out = preConv.apply(out);

    for(const auto& block : blocks) {
      out = block.apply(out, mask, maskSum, useMask);
    }

    out = postBN.apply(out, mask, useMask);
    out = postConv.apply(out);

    return input + out;
  }
};

// Define BlockVariant constructor for NestedBottleneckResidualBlock now that it's complete
BlockVariant::BlockVariant(const NestedBottleneckResidualBlockDesc& desc)
  : type(NESTED_BOTTLENECK), nestedBottleneck(make_unique<NestedBottleneckResidualBlock>(desc)) {}

mx::array BlockVariant::apply(const mx::array& input, const mx::array& mask, const mx::array& maskSum, bool useMask) const {
  switch(type) {
    case REGULAR:
      return regular->apply(input, mask, useMask);
    case GLOBAL_POOLING:
      return globalPooling->apply(input, mask, maskSum, useMask);
    case NESTED_BOTTLENECK:
      return nestedBottleneck->apply(input, mask, maskSum, useMask);
    default:
      return input;
  }
}

// SGF Metadata Encoder
struct SGFMetadataEncoder {
  const int metaEncoderVersion;
  const int numInputMetaChannels;
  const MatMulLayer mul1;
  const MatBiasLayer bias1;
  const int act1;
  const MatMulLayer mul2;
  const MatBiasLayer bias2;
  const int act2;
  const MatMulLayer mul3;

  SGFMetadataEncoder() = delete;
  SGFMetadataEncoder(const SGFMetadataEncoder&) = delete;
  SGFMetadataEncoder& operator=(const SGFMetadataEncoder&) = delete;

  SGFMetadataEncoder(const SGFMetadataEncoderDesc& desc)
    : metaEncoderVersion(desc.metaEncoderVersion),
      numInputMetaChannels(desc.numInputMetaChannels),
      mul1(desc.mul1),
      bias1(desc.bias1),
      act1(desc.act1.activation),
      mul2(desc.mul2),
      bias2(desc.bias2),
      act2(desc.act2.activation),
      mul3(desc.mul3)
  {}

  mx::array apply(const mx::array& metaInput) const {
    // Fuse matmul + bias with addmm for better performance
    mx::array out = matmulBias(metaInput, mul1.weights, bias1.bias);
    out = applyActivation(out, act1);
    out = matmulBias(out, mul2.weights, bias2.bias);
    out = applyActivation(out, act2);
    out = mul3.apply(out);  // Last layer has no bias
    return out;
  }
};

// Trunk
struct Trunk {
  const string name;
  const int trunkNumChannels;
  const ConvLayer initialConv;
  const MatMulLayer initialMatMul;
  unique_ptr<SGFMetadataEncoder> sgfMetadataEncoder;
  vector<BlockVariant> blocks;
  const BatchNormLayer trunkTipBN;

  Trunk() = delete;
  Trunk(const Trunk&) = delete;
  Trunk& operator=(const Trunk&) = delete;

  Trunk(const TrunkDesc& desc)
    : name(desc.name),
      trunkNumChannels(desc.trunkNumChannels),
      initialConv(desc.initialConv),
      initialMatMul(desc.initialMatMul),
      trunkTipBN(desc.trunkTipBN, desc.trunkTipActivation.activation)
  {
    if(desc.sgfMetadataEncoder.metaEncoderVersion > 0 && desc.sgfMetadataEncoder.numInputMetaChannels > 0) {
      sgfMetadataEncoder = make_unique<SGFMetadataEncoder>(desc.sgfMetadataEncoder);
    }

    for(size_t i = 0; i < desc.blocks.size(); i++) {
      int blockKind = desc.blocks[i].first;
      if(blockKind == ORDINARY_BLOCK_KIND) {
        blocks.emplace_back(*static_cast<ResidualBlockDesc*>(desc.blocks[i].second.get()));
      }
      else if(blockKind == GLOBAL_POOLING_BLOCK_KIND) {
        blocks.emplace_back(*static_cast<GlobalPoolingResidualBlockDesc*>(desc.blocks[i].second.get()));
      }
      else if(blockKind == NESTED_BOTTLENECK_BLOCK_KIND) {
        blocks.emplace_back(*static_cast<NestedBottleneckResidualBlockDesc*>(desc.blocks[i].second.get()));
      }
    }
  }

  mx::array apply(
    const mx::array& input,
    const mx::array& inputGlobal,
    const mx::array* inputMeta,
    const mx::array& mask,
    const mx::array& maskSum,
    bool useMask
  ) const {
    // Initial conv
    mx::array trunk = initialConv.apply(input);

    // Add global input bias
    mx::array globalBias = initialMatMul.apply(inputGlobal);
    // Reshape from [N, C] to [N, 1, 1, C] for broadcasting
    mx::Shape globalBiasShape = {static_cast<int>(globalBias.shape()[0]), 1, 1, static_cast<int>(globalBias.shape()[1])};
    globalBias = mx::reshape(globalBias, globalBiasShape);
    trunk = trunk + globalBias;

    // Add SGF metadata if present
    if(sgfMetadataEncoder && inputMeta != nullptr) {
      mx::array metaBias = sgfMetadataEncoder->apply(*inputMeta);
      mx::Shape metaBiasShape = {static_cast<int>(metaBias.shape()[0]), 1, 1, static_cast<int>(metaBias.shape()[1])};
      metaBias = mx::reshape(metaBias, metaBiasShape);
      trunk = trunk + metaBias;
    }

    // Apply mask - skip when useMask=false (all positions valid)
    if(useMask)
      trunk = trunk * mask;

    // Apply residual blocks
    for(const auto& block : blocks) {
      trunk = block.apply(trunk, mask, maskSum, useMask);
    }

    // Final BN + activation
    trunk = trunkTipBN.apply(trunk, mask, useMask);

    return trunk;
  }
};

// Policy Head
struct PolicyHead {
  const string name;
  const int modelVersion;
  const ConvLayer p1Conv;
  const ConvLayer g1Conv;
  const BatchNormLayer g1BN;
  const MatMulLayer gpoolToBiasMul;
  const BatchNormLayer p1BN;
  const ConvLayer p2Conv;
  const MatMulLayer gpoolToPassMul;

  PolicyHead() = delete;
  PolicyHead(const PolicyHead&) = delete;
  PolicyHead& operator=(const PolicyHead&) = delete;

  PolicyHead(const PolicyHeadDesc& desc)
    : name(desc.name),
      modelVersion(desc.modelVersion),
      p1Conv(desc.p1Conv),
      g1Conv(desc.g1Conv),
      g1BN(desc.g1BN, desc.g1Activation.activation),
      gpoolToBiasMul(desc.gpoolToBiasMul),
      p1BN(desc.p1BN, desc.p1Activation.activation),
      p2Conv(desc.p2Conv),
      gpoolToPassMul(desc.gpoolToPassMul)
  {}

  std::pair<mx::array, mx::array> apply(
    const mx::array& trunk,
    const mx::array& mask,
    const mx::array& maskSum,
    bool useMask
  ) const {
    // Policy conv
    mx::array p1Out = p1Conv.apply(trunk);

    // Global pooling path
    mx::array g1Out = g1Conv.apply(trunk);
    g1Out = g1BN.apply(g1Out, mask, useMask);
    mx::array pooled = applyGlobalPooling(g1Out, mask, maskSum, useMask);
    std::vector<int> squeezeAxes = {1, 2};
    mx::array pooledFlat = mx::squeeze(pooled, squeezeAxes);

    // Add bias from global pooling
    mx::array bias = gpoolToBiasMul.apply(pooledFlat);
    mx::Shape biasShape = {static_cast<int>(bias.shape()[0]), 1, 1, static_cast<int>(bias.shape()[1])};
    bias = mx::reshape(bias, biasShape);
    p1Out = p1Out + bias;

    p1Out = p1BN.apply(p1Out, mask, useMask);

    // Final policy conv
    mx::array policy = p2Conv.apply(p1Out);

    // Pass policy
    mx::array policyPass = gpoolToPassMul.apply(pooledFlat);

    return {policyPass, policy};
  }
};

// Value Head
struct ValueHead {
  const string name;
  const int modelVersion;
  const ConvLayer v1Conv;
  const BatchNormLayer v1BN;
  const MatMulLayer v2Mul;
  const MatBiasLayer v2Bias;
  const int v2Activation;
  const MatMulLayer v3Mul;
  const MatBiasLayer v3Bias;
  const MatMulLayer sv3Mul;
  const MatBiasLayer sv3Bias;
  const ConvLayer vOwnershipConv;

  ValueHead() = delete;
  ValueHead(const ValueHead&) = delete;
  ValueHead& operator=(const ValueHead&) = delete;

  ValueHead(const ValueHeadDesc& desc)
    : name(desc.name),
      modelVersion(desc.modelVersion),
      v1Conv(desc.v1Conv),
      v1BN(desc.v1BN, desc.v1Activation.activation),
      v2Mul(desc.v2Mul),
      v2Bias(desc.v2Bias),
      v2Activation(desc.v2Activation.activation),
      v3Mul(desc.v3Mul),
      v3Bias(desc.v3Bias),
      sv3Mul(desc.sv3Mul),
      sv3Bias(desc.sv3Bias),
      vOwnershipConv(desc.vOwnershipConv)
  {}

  std::tuple<mx::array, mx::array, mx::array> apply(
    const mx::array& trunk,
    const mx::array& mask,
    const mx::array& maskSum,
    bool useMask
  ) const {
    mx::array v1Out = v1Conv.apply(trunk);
    v1Out = v1BN.apply(v1Out, mask, useMask);

    // Value head pooling (only uses maskSum, not mask)
    mx::array v1Mean = applyValueHeadPooling(v1Out, maskSum);
    std::vector<int> squeezeAxes = {1, 2};
    mx::array v1MeanFlat = mx::squeeze(v1Mean, squeezeAxes);

    // Fuse matmul + bias with addmm for better performance
    mx::array v2Out = matmulBias(v1MeanFlat, v2Mul.weights, v2Bias.bias);
    v2Out = applyActivation(v2Out, v2Activation);

    mx::array value = matmulBias(v2Out, v3Mul.weights, v3Bias.bias);
    mx::array scoreValue = matmulBias(v2Out, sv3Mul.weights, sv3Bias.bias);

    mx::array ownership = vOwnershipConv.apply(v1Out);

    return {value, scoreValue, ownership};
  }
};

// Model
struct Model {
  const string name;
  const int modelVersion;
  const int numInputChannels;
  const int numInputGlobalChannels;
  const int numInputMetaChannels;
  const int numPolicyChannels;
  const int numValueChannels;
  const int numScoreValueChannels;
  const int numOwnershipChannels;

  const Trunk trunk;
  const PolicyHead policyHead;
  const ValueHead valueHead;

  Model() = delete;
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  Model(const ModelDesc& desc)
    : name(desc.name),
      modelVersion(desc.modelVersion),
      numInputChannels(desc.numInputChannels),
      numInputGlobalChannels(desc.numInputGlobalChannels),
      numInputMetaChannels(desc.numInputMetaChannels),
      numPolicyChannels(desc.numPolicyChannels),
      numValueChannels(desc.numValueChannels),
      numScoreValueChannels(desc.numScoreValueChannels),
      numOwnershipChannels(desc.numOwnershipChannels),
      trunk(desc.trunk),
      policyHead(desc.policyHead),
      valueHead(desc.valueHead)
  {}

  void apply(
    const float* inputSpatial,
    const float* inputGlobal,
    const float* inputMeta,
    int batchSize,
    int nnXLen,
    int nnYLen,
    bool requireExactNNLen,
    float* policyOut,
    float* policyPassOut,
    float* valueOut,
    float* scoreValueOut,
    float* ownershipOut
  ) const {
    // When requireExactNNLen=true, all boards are exactly nnXLen x nnYLen,
    // so all mask values are 1 and we can skip mask operations
    const bool useMask = !requireExactNNLen;

    // Create input tensors - NHWC format
    mx::Shape inputShape = {batchSize, nnYLen, nnXLen, numInputChannels};
    mx::array input = mx::array(inputSpatial, inputShape, mx::float32);
    mx::Shape globalShape = {batchSize, numInputGlobalChannels};
    mx::array inputGlobalArr = mx::array(inputGlobal, globalShape, mx::float32);

    // Extract mask from first channel of input
    mx::Shape sliceStart = {0, 0, 0, 0};
    mx::Shape sliceEnd = {batchSize, nnYLen, nnXLen, 1};
    mx::array mask = mx::slice(input, sliceStart, sliceEnd);

    // Compute mask sum - needed for pooling normalization even when useMask=false
    // Pre-compute fixed maskSum = nnXLen * nnYLen when all mask values are 1
    std::vector<int> sumAxes = {1, 2};
    mx::array maskSum = requireExactNNLen
      ? mx::full({batchSize, 1, 1, 1}, static_cast<float>(nnXLen * nnYLen))
      : mx::sum(mask, sumAxes, /*keepdims=*/true);

    // Optional metadata input
    unique_ptr<mx::array> inputMetaArr;
    if(numInputMetaChannels > 0 && inputMeta != nullptr) {
      mx::Shape metaShape = {batchSize, numInputMetaChannels};
      inputMetaArr = make_unique<mx::array>(mx::array(inputMeta, metaShape, mx::float32));
    }

    // Apply trunk
    mx::array trunkOut = trunk.apply(input, inputGlobalArr, inputMetaArr.get(), mask, maskSum, useMask);

    // Apply policy head
    auto [policyPass, policy] = policyHead.apply(trunkOut, mask, maskSum, useMask);

    // Apply value head
    auto [value, scoreValue, ownership] = valueHead.apply(trunkOut, mask, maskSum, useMask);

    // Force evaluation of all outputs
    std::vector<mx::array> outputs = {policy, policyPass, value, scoreValue, ownership};
    mx::eval(outputs);

    // Copy results to output buffers
    memcpy(policyOut, policy.data<float>(), batchSize * numPolicyChannels * nnXLen * nnYLen * sizeof(float));
    memcpy(policyPassOut, policyPass.data<float>(), batchSize * numPolicyChannels * sizeof(float));
    memcpy(valueOut, value.data<float>(), batchSize * numValueChannels * sizeof(float));
    memcpy(scoreValueOut, scoreValue.data<float>(), batchSize * numScoreValueChannels * sizeof(float));
    memcpy(ownershipOut, ownership.data<float>(), batchSize * numOwnershipChannels * nnXLen * nnYLen * sizeof(float));
  }
};

// ComputeContext and ComputeHandle ------------------------------------------------------------------------------------

struct ComputeContext {
  const int nnXLen;
  const int nnYLen;

  std::mutex cachedModelsMutex;
  std::map<std::string, std::shared_ptr<const Model>> cachedModels;
  std::map<std::string, int> cachedModelsRefCount;

  ComputeContext() = delete;
  ComputeContext(const ComputeContext&) = delete;
  ComputeContext& operator=(const ComputeContext&) = delete;

  ComputeContext(int nnX, int nnY)
    : nnXLen(nnX),
      nnYLen(nnY),
      cachedModelsMutex(),
      cachedModels(),
      cachedModelsRefCount()
  {}

  ~ComputeContext() {
    assert(cachedModels.size() == 0);
  }
};

struct ComputeHandle {
  ComputeContext* context;
  bool inputsUseNHWC;
  bool requireExactNNLen;
  const std::string modelCacheKey;
  std::shared_ptr<const Model> model;
  const int modelVersion;

  ComputeHandle() = delete;
  ComputeHandle(const ComputeHandle&) = delete;
  ComputeHandle& operator=(const ComputeHandle&) = delete;

  ComputeHandle(ComputeContext* ctx, const LoadedModel& loadedModel, bool iNHWC, bool requireExactNNLen_)
    : context(ctx),
      inputsUseNHWC(iNHWC),
      requireExactNNLen(requireExactNNLen_),
      modelCacheKey(loadedModel.modelDesc.name + "-" + loadedModel.modelDesc.sha256),
      model(nullptr),
      modelVersion(loadedModel.modelDesc.modelVersion)
  {
    {
      std::lock_guard<std::mutex> lock(context->cachedModelsMutex);
      if(context->cachedModels.find(modelCacheKey) == context->cachedModels.end()) {
        context->cachedModels[modelCacheKey] = std::make_shared<const Model>(loadedModel.modelDesc);
      }
      model = context->cachedModels[modelCacheKey];
      context->cachedModelsRefCount[modelCacheKey] += 1;
    }
  }

  ~ComputeHandle() {
    std::lock_guard<std::mutex> lock(context->cachedModelsMutex);
    context->cachedModelsRefCount[modelCacheKey] -= 1;
    assert(context->cachedModelsRefCount[modelCacheKey] >= 0);
    if(context->cachedModelsRefCount[modelCacheKey] == 0) {
      context->cachedModelsRefCount.erase(modelCacheKey);
      context->cachedModels.erase(modelCacheKey);
    }
  }
};

// InputBuffers --------------------------------------------------------------------------------------------------------

struct InputBuffers {
  int maxBatchSize;

  size_t singleInputElts;
  size_t singleInputGlobalElts;
  size_t singleInputMetaElts;

  size_t singlePolicyPassResultElts;
  size_t singlePolicyResultElts;
  size_t singleValueResultElts;
  size_t singleScoreValueResultElts;
  size_t singleOwnershipResultElts;

  std::vector<float> spatialInput;
  std::vector<float> globalInput;
  std::vector<float> metaInput;
  std::vector<float> policyResults;
  std::vector<float> policyPassResults;
  std::vector<float> valueResults;
  std::vector<float> scoreValueResults;
  std::vector<float> ownershipResults;

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
    const ModelDesc& m = loadedModel->modelDesc;

    maxBatchSize = maxBatchSz;
    singleInputElts = m.numInputChannels * nnXLen * nnYLen;
    singleInputGlobalElts = m.numInputGlobalChannels;
    singleInputMetaElts = m.numInputMetaChannels;

    singlePolicyPassResultElts = (size_t)(m.numPolicyChannels);
    singlePolicyResultElts = (size_t)(m.numPolicyChannels * nnXLen * nnYLen);
    singleValueResultElts = (size_t)m.numValueChannels;
    singleScoreValueResultElts = (size_t)m.numScoreValueChannels;
    singleOwnershipResultElts = (size_t)m.numOwnershipChannels * nnXLen * nnYLen;

    assert(NNModelVersion::getNumSpatialFeatures(m.modelVersion) == m.numInputChannels);
    assert(NNModelVersion::getNumGlobalFeatures(m.modelVersion) == m.numInputGlobalChannels);
    if(m.numInputMetaChannels > 0) {
      assert(SGFMetadata::METADATA_INPUT_NUM_CHANNELS == m.numInputMetaChannels);
    }

    spatialInput.resize(m.numInputChannels * nnXLen * nnYLen * maxBatchSize);
    globalInput.resize(m.numInputGlobalChannels * maxBatchSize);
    if(m.numInputMetaChannels > 0)
      metaInput.resize(m.numInputMetaChannels * maxBatchSize);
    else
      metaInput.resize(1);

    policyResults.resize(singlePolicyResultElts * maxBatchSize);
    policyPassResults.resize(singlePolicyPassResultElts * maxBatchSize);
    valueResults.resize(singleValueResultElts * maxBatchSize);
    scoreValueResults.resize(singleScoreValueResultElts * maxBatchSize);
    ownershipResults.resize(singleOwnershipResultElts * maxBatchSize);
  }

  ~InputBuffers() {}

  InputBuffers() = delete;
  InputBuffers(const InputBuffers&) = delete;
  InputBuffers& operator=(const InputBuffers&) = delete;
};

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  return new InputBuffers(loadedModel, maxBatchSize, nnXLen, nnYLen);
}

void NeuralNet::freeInputBuffers(InputBuffers* inputBuffers) {
  delete inputBuffers;
}

// NeuralNet Interface -------------------------------------------------------------------------------------------------

void NeuralNet::globalInitialize() {
  // MLX initializes automatically
}

void NeuralNet::globalCleanup() {
  // MLX cleans up automatically
}

ComputeContext* NeuralNet::createComputeContext(
  const std::vector<int>& gpuIdxs,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  const string& openCLTunerFile,
  const string& homeDataDirOverride,
  bool openCLReTunePerBoardSize,
  enabled_t useFP16Mode,
  enabled_t useNHWCMode,
  const LoadedModel* loadedModel
) {
  (void)gpuIdxs;
  (void)logger;
  (void)openCLTunerFile;
  (void)homeDataDirOverride;
  (void)openCLReTunePerBoardSize;
  (void)loadedModel;

  bool useFP16 = useFP16Mode == enabled_t::True ? true : false;
  bool useNHWC = useNHWCMode == enabled_t::False ? false : true;

  if(useFP16)
    throw StringError("MLX backend: useFP16 = true not yet supported");
  if(!useNHWC)
    throw StringError("MLX backend: useNHWC = false not supported");

  ComputeContext* context = new ComputeContext(nnXLen, nnYLen);
  return context;
}

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}

ComputeHandle* NeuralNet::createComputeHandle(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  Logger* logger,
  int maxBatchSize,
  bool requireExactNNLen,
  bool inputsUseNHWC,
  int gpuIdxForThisThread,
  int serverThreadIdx
) {
  if(logger != NULL) {
    logger->write("MLX backend thread " + Global::intToString(serverThreadIdx) + ": Model version " + Global::intToString(loadedModel->modelDesc.modelVersion));
    logger->write("MLX backend thread " + Global::intToString(serverThreadIdx) + ": Model name: " + loadedModel->modelDesc.name);
  }

  (void)maxBatchSize;
  (void)gpuIdxForThisThread;

  if(!inputsUseNHWC)
    throw StringError("MLX backend: inputsUseNHWC = false unsupported");

  return new ComputeHandle(context, *loadedModel, inputsUseNHWC, requireExactNNLen);
}

void NeuralNet::freeComputeHandle(ComputeHandle* gpuHandle) {
  delete gpuHandle;
}

bool NeuralNet::isUsingFP16(const ComputeHandle* handle) {
  (void)handle;
  return false;
}

void NeuralNet::getOutput(
  ComputeHandle* computeHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs
) {
  assert(numBatchEltsFilled <= inputBuffers->maxBatchSize);
  assert(numBatchEltsFilled > 0);
  const int batchSize = numBatchEltsFilled;
  const int nnXLen = computeHandle->context->nnXLen;
  const int nnYLen = computeHandle->context->nnYLen;
  const int modelVersion = computeHandle->modelVersion;

  const int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
  const int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
  const int numMetaFeatures = inputBuffers->singleInputMetaElts;
  assert(numSpatialFeatures == computeHandle->model->numInputChannels);
  assert(numSpatialFeatures * nnXLen * nnYLen == inputBuffers->singleInputElts);
  assert(numGlobalFeatures == inputBuffers->singleInputGlobalElts);
  const int numPolicyChannels = computeHandle->model->numPolicyChannels;

  // Copy input data to buffers
  for(int nIdx = 0; nIdx < batchSize; nIdx++) {
    float* rowSpatialInput = inputBuffers->spatialInput.data() + (inputBuffers->singleInputElts * nIdx);
    float* rowGlobalInput = inputBuffers->globalInput.data() + (inputBuffers->singleInputGlobalElts * nIdx);
    float* rowMetaInput = inputBuffers->metaInput.data() + (inputBuffers->singleInputMetaElts * nIdx);

    const float* rowGlobal = inputBufs[nIdx]->rowGlobalBuf.data();
    const float* rowSpatial = inputBufs[nIdx]->rowSpatialBuf.data();
    const float* rowMeta = inputBufs[nIdx]->rowMetaBuf.data();
    const bool hasRowMeta = inputBufs[nIdx]->hasRowMeta;

    std::copy(rowGlobal, rowGlobal + numGlobalFeatures, rowGlobalInput);

    if(numMetaFeatures > 0) {
      testAssert(rowMeta != NULL);
      testAssert(hasRowMeta);
      std::copy(rowMeta, rowMeta + numMetaFeatures, rowMetaInput);
    }
    else {
      testAssert(!hasRowMeta);
    }

    SymmetryHelpers::copyInputsWithSymmetry(rowSpatial, rowSpatialInput, 1, nnYLen, nnXLen, numSpatialFeatures, computeHandle->inputsUseNHWC, inputBufs[nIdx]->symmetry);
  }

  // Run model
  computeHandle->model->apply(
    inputBuffers->spatialInput.data(),
    inputBuffers->globalInput.data(),
    (numMetaFeatures > 0 ? inputBuffers->metaInput.data() : nullptr),
    batchSize,
    nnXLen,
    nnYLen,
    computeHandle->requireExactNNLen,
    inputBuffers->policyResults.data(),
    inputBuffers->policyPassResults.data(),
    inputBuffers->valueResults.data(),
    inputBuffers->scoreValueResults.data(),
    inputBuffers->ownershipResults.data()
  );

  assert(inputBuffers->singlePolicyPassResultElts == numPolicyChannels);
  assert(inputBuffers->singlePolicyResultElts == numPolicyChannels * nnXLen * nnYLen);
  assert(outputs.size() == batchSize);

  float policyProbsTmp[NNPos::MAX_NN_POLICY_SIZE];

  float* policyData = inputBuffers->policyResults.data();
  float* policyPassData = inputBuffers->policyPassResults.data();
  float* valueData = inputBuffers->valueResults.data();
  float* scoreValueData = inputBuffers->scoreValueResults.data();
  float* ownershipData = inputBuffers->ownershipResults.data();

  for(int row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];
    assert(output->nnXLen == nnXLen);
    assert(output->nnYLen == nnYLen);
    float policyOptimism = (float)inputBufs[row]->policyOptimism;

    const float* policyPassSrcBuf = policyPassData + row * numPolicyChannels;
    const float* policySrcBuf = policyData + row * numPolicyChannels * nnXLen * nnYLen;
    float* policyProbs = output->policyProbs;

    // Handle policy optimism (version >= 12)
    if(numPolicyChannels == 2 || (numPolicyChannels == 4 && modelVersion >= 16)) {
      // MLX output is NHWC
      for(int i = 0; i < nnXLen * nnYLen; i++) {
        float p = policySrcBuf[i * numPolicyChannels];
        float pOpt = policySrcBuf[i * numPolicyChannels + 1];
        policyProbsTmp[i] = p + (pOpt - p) * policyOptimism;
      }
      SymmetryHelpers::copyOutputsWithSymmetry(policyProbsTmp, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
      policyProbs[nnXLen * nnYLen] = policyPassSrcBuf[0] + (policyPassSrcBuf[1] - policyPassSrcBuf[0]) * policyOptimism;
    }
    else {
      assert(numPolicyChannels == 1);
      SymmetryHelpers::copyOutputsWithSymmetry(policySrcBuf, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
      policyProbs[inputBuffers->singlePolicyResultElts] = policyPassSrcBuf[0];
    }

    int numValueChannels = computeHandle->model->numValueChannels;
    assert(numValueChannels == 3);
    output->whiteWinProb = valueData[row * numValueChannels];
    output->whiteLossProb = valueData[row * numValueChannels + 1];
    output->whiteNoResultProb = valueData[row * numValueChannels + 2];

    if(output->whiteOwnerMap != NULL) {
      const float* ownershipSrcBuf = ownershipData + row * nnXLen * nnYLen;
      assert(computeHandle->model->numOwnershipChannels == 1);
      SymmetryHelpers::copyOutputsWithSymmetry(ownershipSrcBuf, output->whiteOwnerMap, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    }

    if(modelVersion >= 9) {
      int numScoreValueChannels = computeHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 6);
      output->whiteScoreMean = scoreValueData[row * numScoreValueChannels];
      output->whiteScoreMeanSq = scoreValueData[row * numScoreValueChannels + 1];
      output->whiteLead = scoreValueData[row * numScoreValueChannels + 2];
      output->varTimeLeft = scoreValueData[row * numScoreValueChannels + 3];
      output->shorttermWinlossError = scoreValueData[row * numScoreValueChannels + 4];
      output->shorttermScoreError = scoreValueData[row * numScoreValueChannels + 5];
    }
    else if(modelVersion >= 8) {
      int numScoreValueChannels = computeHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 4);
      output->whiteScoreMean = scoreValueData[row * numScoreValueChannels];
      output->whiteScoreMeanSq = scoreValueData[row * numScoreValueChannels + 1];
      output->whiteLead = scoreValueData[row * numScoreValueChannels + 2];
      output->varTimeLeft = scoreValueData[row * numScoreValueChannels + 3];
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    }
    else if(modelVersion >= 4) {
      int numScoreValueChannels = computeHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 2);
      output->whiteScoreMean = scoreValueData[row * numScoreValueChannels];
      output->whiteScoreMeanSq = scoreValueData[row * numScoreValueChannels + 1];
      output->whiteLead = output->whiteScoreMean;
      output->varTimeLeft = 0;
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    }
    else if(modelVersion >= 3) {
      int numScoreValueChannels = computeHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 1);
      output->whiteScoreMean = scoreValueData[row * numScoreValueChannels];
      output->whiteScoreMeanSq = output->whiteScoreMean * output->whiteScoreMean;
      output->whiteLead = output->whiteScoreMean;
      output->varTimeLeft = 0;
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    }
    else {
      ASSERT_UNREACHABLE;
    }
  }
}

void NeuralNet::printDevices() {
  cout << "MLX Backend (Apple Silicon)" << endl;
  cout << "Default device: " << mx::default_device() << endl;
}

// FOR TESTING ---------------------------------------------------------------------------------------------------------

bool NeuralNet::testEvaluateConv(
  const ConvLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  vector<float>& outputBuffer
) {
  (void)useFP16;
  if(!useNHWC) {
    return false; // MLX only supports NHWC
  }

  size_t numOutputFloats = (size_t)batchSize * nnXLen * nnYLen * desc->outChannels;
  outputBuffer.resize(numOutputFloats);

  ConvLayer layer(*desc);
  mx::Shape inputShape = {batchSize, nnYLen, nnXLen, desc->inChannels};
  mx::array input = mx::array(inputBuffer.data(), inputShape, mx::float32);
  mx::array output = layer.apply(input);
  mx::eval(output);

  memcpy(outputBuffer.data(), output.data<float>(), numOutputFloats * sizeof(float));
  return true;
}

bool NeuralNet::testEvaluateBatchNorm(
  const BatchNormLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer
) {
  (void)useFP16;
  if(!useNHWC) {
    return false;
  }

  size_t numOutputFloats = (size_t)batchSize * nnXLen * nnYLen * desc->numChannels;
  outputBuffer.resize(numOutputFloats);

  BatchNormLayer layer(*desc, ACTIVATION_IDENTITY);
  mx::Shape inputShape = {batchSize, nnYLen, nnXLen, desc->numChannels};
  mx::Shape maskShape = {batchSize, nnYLen, nnXLen, 1};
  mx::array input = mx::array(inputBuffer.data(), inputShape, mx::float32);
  mx::array mask = mx::array(maskBuffer.data(), maskShape, mx::float32);
  mx::array output = layer.apply(input, mask, /*useMask=*/true);
  mx::eval(output);

  memcpy(outputBuffer.data(), output.data<float>(), numOutputFloats * sizeof(float));
  return true;
}

bool NeuralNet::testEvaluateResidualBlock(
  const ResidualBlockDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer
) {
  (void)useFP16;
  if(!useNHWC) {
    return false;
  }

  size_t numOutputFloats = (size_t)batchSize * nnXLen * nnYLen * desc->preBN.numChannels;
  outputBuffer.resize(numOutputFloats);

  ResidualBlock block(*desc);
  mx::Shape inputShape = {batchSize, nnYLen, nnXLen, desc->preBN.numChannels};
  mx::Shape maskShape = {batchSize, nnYLen, nnXLen, 1};
  mx::array input = mx::array(inputBuffer.data(), inputShape, mx::float32);
  mx::array mask = mx::array(maskBuffer.data(), maskShape, mx::float32);
  mx::array output = block.apply(input, mask, /*useMask=*/true);
  mx::eval(output);

  memcpy(outputBuffer.data(), output.data<float>(), numOutputFloats * sizeof(float));
  return true;
}

bool NeuralNet::testEvaluateGlobalPoolingResidualBlock(
  const GlobalPoolingResidualBlockDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer
) {
  (void)useFP16;
  if(!useNHWC) {
    return false;
  }

  size_t numOutputFloats = (size_t)batchSize * nnXLen * nnYLen * desc->preBN.numChannels;
  outputBuffer.resize(numOutputFloats);

  GlobalPoolingResidualBlock block(*desc);
  mx::Shape inputShape = {batchSize, nnYLen, nnXLen, desc->preBN.numChannels};
  mx::Shape maskShape = {batchSize, nnYLen, nnXLen, 1};
  mx::array input = mx::array(inputBuffer.data(), inputShape, mx::float32);
  mx::array mask = mx::array(maskBuffer.data(), maskShape, mx::float32);
  std::vector<int> sumAxes = {1, 2};
  mx::array maskSum = mx::sum(mask, sumAxes, /*keepdims=*/true);
  mx::array output = block.apply(input, mask, maskSum, /*useMask=*/true);
  mx::eval(output);

  memcpy(outputBuffer.data(), output.data<float>(), numOutputFloats * sizeof(float));
  return true;
}

#endif // USE_MLX_BACKEND
