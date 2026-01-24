"""
Quantization Aware Training (QAT) for FastVIT models.

Simulates INT8 group-wise quantization during training (matching MLX
nn.quantize(bits=8, group_size=32)) so weights learn to be robust to
quantization noise. Uses Straight-Through Estimator (STE) for gradient flow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


class FakeQuantize(nn.Module):
    """
    Fake quantization module that simulates symmetric per-group INT8 quantization.

    During forward, weights are quantized and dequantized (round-trip) to inject
    quantization noise. Gradients flow through via STE (straight-through estimator).

    Args:
        bits: Number of quantization bits (default: 8)
        group_size: Group size for per-group quantization (default: 32)
    """

    def __init__(self, bits: int = 8, group_size: int = 32):
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        self.qmax = 2 ** (bits - 1) - 1  # 127 for 8-bit
        self.enabled = False

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return weight

        shape = weight.shape
        # Flatten to 2D: [out_features, in_features]
        w = weight.view(shape[0], -1)
        in_features = w.shape[1]

        # Pad if not divisible by group_size
        remainder = in_features % self.group_size
        if remainder != 0:
            pad_size = self.group_size - remainder
            w = F.pad(w, (0, pad_size))
            in_features = w.shape[1]

        # Reshape to groups: [out_features, num_groups, group_size]
        num_groups = in_features // self.group_size
        w_grouped = w.view(shape[0], num_groups, self.group_size)

        # Compute per-group scale: max(|group|) / qmax
        scale = w_grouped.abs().amax(dim=-1, keepdim=True) / self.qmax
        # Avoid division by zero
        scale = scale.clamp(min=1e-10)

        # Quantize: round(w / scale), clamp to [-qmax, qmax]
        w_q = (w_grouped / scale).round().clamp(-self.qmax, self.qmax)

        # Dequantize: w_q * scale
        w_dq = w_q * scale

        # Flatten back and remove padding
        w_dq = w_dq.view(shape[0], -1)
        if remainder != 0:
            w_dq = w_dq[:, :weight.view(shape[0], -1).shape[1]]

        w_dq = w_dq.view(shape)

        # STE: forward uses quantized, backward passes gradient through
        return weight + (w_dq - weight).detach()


class QuantizedLinear(nn.Module):
    """
    Wrapper around nn.Linear that applies fake quantization to weights.

    Args:
        linear: The nn.Linear module to wrap
        bits: Number of quantization bits
        group_size: Group size for per-group quantization
    """

    def __init__(self, linear: nn.Linear, bits: int = 8, group_size: int = 32):
        super().__init__()
        self.linear = linear
        self.fake_quantize = FakeQuantize(bits=bits, group_size=group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_weight = self.fake_quantize(self.linear.weight)
        return F.linear(x, q_weight, self.linear.bias)


class QuantizedConv2d(nn.Module):
    """
    Wrapper around 1x1 nn.Conv2d that applies fake quantization to weights.

    Weight shape [out_c, in_c, 1, 1] is squeezed to [out_c, in_c] for
    quantization, then reshaped back.

    Args:
        conv: The nn.Conv2d module to wrap (must be kernel_size=1)
        bits: Number of quantization bits
        group_size: Group size for per-group quantization
    """

    def __init__(self, conv: nn.Conv2d, bits: int = 8, group_size: int = 32):
        super().__init__()
        assert conv.kernel_size == (1, 1), "QuantizedConv2d only supports 1x1 convolutions"
        self.conv = conv
        self.fake_quantize = FakeQuantize(bits=bits, group_size=group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_weight = self.fake_quantize(self.conv.weight)
        return F.conv2d(
            x, q_weight, self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )


def apply_qat_to_model(model: nn.Module, bits: int = 8, group_size: int = 32) -> int:
    """
    Walk model modules and wrap target layers with fake quantization.

    Targets:
    - SelfAttention.qkv (nn.Linear)
    - SelfAttention.proj (nn.Linear)
    - FeedForward.fc1 (nn.Conv2d 1x1)
    - FeedForward.fc2 (nn.Conv2d 1x1)

    Args:
        model: The model to apply QAT to
        bits: Number of quantization bits
        group_size: Group size for per-group quantization

    Returns:
        Number of layers wrapped
    """
    from katago.train.fastvit import SelfAttention, FeedForward

    count = 0
    for name, module in model.named_modules():
        if isinstance(module, SelfAttention):
            module.qkv = QuantizedLinear(module.qkv, bits=bits, group_size=group_size)
            module.proj = QuantizedLinear(module.proj, bits=bits, group_size=group_size)
            count += 2
        elif isinstance(module, FeedForward):
            module.fc1 = QuantizedConv2d(module.fc1, bits=bits, group_size=group_size)
            module.fc2 = QuantizedConv2d(module.fc2, bits=bits, group_size=group_size)
            count += 2

    logging.info(f"QAT: Wrapped {count} layers (bits={bits}, group_size={group_size})")
    return count


def set_qat_enabled(model: nn.Module, enabled: bool) -> None:
    """
    Toggle all FakeQuantize modules on or off.

    Args:
        model: The model containing FakeQuantize modules
        enabled: Whether to enable fake quantization
    """
    for module in model.modules():
        if isinstance(module, FakeQuantize):
            module.enabled = enabled


def count_qat_layers(model: nn.Module) -> dict:
    """
    Count QAT-wrapped layers for debug logging.

    Returns:
        Dict with counts of QuantizedLinear, QuantizedConv2d, and enabled status
    """
    n_linear = 0
    n_conv = 0
    enabled = None
    for module in model.modules():
        if isinstance(module, QuantizedLinear):
            n_linear += 1
        elif isinstance(module, QuantizedConv2d):
            n_conv += 1
        elif isinstance(module, FakeQuantize):
            if enabled is None:
                enabled = module.enabled
    return {
        "quantized_linear": n_linear,
        "quantized_conv2d": n_conv,
        "total": n_linear + n_conv,
        "enabled": enabled,
    }
