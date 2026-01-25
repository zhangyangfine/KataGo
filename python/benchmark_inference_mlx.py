"""
MLX inference benchmark for FastVIT models (FP32 and INT8 quantized).

Reimplements the FastVIT architecture in MLX to benchmark inference on Apple Silicon.
MLX uses NHWC format: all inputs/intermediates are (B, H, W, C).
FeedForward's 1x1 convolutions are implemented as nn.Linear so they get quantized.
"""

import argparse
import time
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn


# Model configs: (layers, mixers, embed_dims, mlp_ratios, pos_emb_stages)
MODEL_CONFIGS = {
    "ft6c96": {
        "type": "fastvit",
        "layers": [2, 2, 2],
        "mixers": ["mixer", "mixer", "attention"],
        "embed_dims": [96, 96, 96],
        "mlp_ratios": [3.0, 3.0, 3.0],
        "pos_emb_stages": [False, False, True],
    },
    "ft6c96a": {
        "type": "fastvit",
        "layers": [2, 2, 2],
        "mixers": ["attention", "attention", "attention"],
        "embed_dims": [96, 96, 96],
        "mlp_ratios": [3.0, 3.0, 3.0],
        "pos_emb_stages": [True, True, True],
    },
    "ft8c96a": {
        "type": "fastvit",
        "layers": [2, 2, 2, 2],
        "mixers": ["attention", "attention", "attention", "attention"],
        "embed_dims": [96, 96, 96, 96],
        "mlp_ratios": [4.0, 4.0, 4.0, 4.0],
        "pos_emb_stages": [True, True, True, True],
    },
    "b6c96": {
        "type": "resnet",
        "trunk_channels": 96,
        "mid_channels": 96,
        "gpool_channels": 32,
        "blocks": ["regular", "regular", "gpool", "regular", "gpool", "regular"],
    },
}


class PositionalEncoding(nn.Module):
    """Depthwise 3x3 conv that adds positional info. Not quantizable."""

    def __init__(self, embed_dim: int):
        super().__init__()
        # Grouped depthwise conv: groups=embed_dim
        self.pe = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=True
        )

    def __call__(self, x):
        # x: (B, H, W, C)
        return x + self.pe(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention. qkv and proj are Linear (quantizable)."""

    def __init__(self, embed_dim: int, head_dim: int = 32):
        super().__init__()
        assert embed_dim % head_dim == 0
        self.head_dim = head_dim
        self.num_heads = embed_dim // head_dim
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def __call__(self, x):
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        N = H * W
        x = x.reshape(B, N, C)  # (B, N, C)

        qkv = self.qkv(x)  # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        # Split q, k, v along dim 2, each is (B, N, heads, head_dim)
        q = qkv[:, :, 0, :, :]
        k = qkv[:, :, 1, :, :]
        v = qkv[:, :, 2, :, :]
        # Transpose to (B, heads, N, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        attn = (q * self.scale) @ k.transpose(0, 1, 3, 2)  # (B, heads, N, N)
        attn = mx.softmax(attn, axis=-1)
        out = attn @ v  # (B, heads, N, head_dim)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, C)  # (B, N, C)
        out = self.proj(out)
        return out.reshape(B, H, W, C)


class FeedForward(nn.Module):
    """
    FFN with depthwise 7x7 conv + two pointwise projections.
    fc1/fc2 are Linear (quantizable). conv is depthwise Conv2d (not quantizable).
    """

    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        # Depthwise 7x7 conv
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=7, padding=3,
            groups=in_channels, bias=False
        )
        # Pointwise projections as Linear (quantizable)
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, in_channels)

    def __call__(self, x):
        # x: (B, H, W, C)
        x = self.conv(x)
        x = self.fc1(x)
        x = nn.gelu(x)
        x = self.fc2(x)
        return x


class Mixer(nn.Module):
    """Depthwise 3x3 conv mixer (simplified reparameterized form). Not quantizable."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=3, padding=1,
            groups=embed_dim, bias=True
        )

    def __call__(self, x):
        return self.conv(x)


class MixerBlock(nn.Module):
    """Mixer block: depthwise conv mixer + FFN with residual."""

    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mixer = Mixer(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.feed_forward = FeedForward(embed_dim, hidden_dim)

    def __call__(self, x):
        x = self.mixer(x)
        x = x + self.feed_forward(x)
        return x


class AttentionBlock(nn.Module):
    """Attention block: LayerNorm + self-attention + FFN with residuals."""

    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.feed_forward = FeedForward(embed_dim, hidden_dim)

    def __call__(self, x):
        # x: (B, H, W, C)
        x = x + self.attention(self.norm(x))
        x = x + self.feed_forward(x)
        return x


class MLXKataGoModel(nn.Module):
    """
    FastVIT model for KataGo inference benchmarking in MLX.

    Input processing:
      - conv_spatial: Conv2d(22 -> embed_dim, 3x3)
      - linear_global: Linear(19 -> embed_dim), broadcast-added to spatial

    Trunk: stages of MixerBlock/AttentionBlock with optional PositionalEncoding.
    Head: Conv2d(embed_dim -> 1, 1x1) for minimal timing overhead.
    """

    def __init__(self, config: dict):
        super().__init__()
        layers = config["layers"]
        mixers = config["mixers"]
        embed_dims = config["embed_dims"]
        mlp_ratios = config["mlp_ratios"]
        pos_emb_stages = config["pos_emb_stages"]

        embed_dim = embed_dims[0]

        # Input processing
        self.conv_spatial = nn.Conv2d(22, embed_dim, kernel_size=3, padding=1)
        self.linear_global = nn.Linear(19, embed_dim)

        # Build stages
        self.stages = []
        for stage_idx in range(len(layers)):
            stage_modules = []
            if pos_emb_stages[stage_idx]:
                stage_modules.append(PositionalEncoding(embed_dims[stage_idx]))
            for _ in range(layers[stage_idx]):
                if mixers[stage_idx] == "mixer":
                    stage_modules.append(MixerBlock(embed_dims[stage_idx], mlp_ratios[stage_idx]))
                else:
                    stage_modules.append(AttentionBlock(embed_dims[stage_idx], mlp_ratios[stage_idx]))
            self.stages.append(stage_modules)

        # Minimal head for timing
        self.head = nn.Conv2d(embed_dim, 1, kernel_size=1)

    def __call__(self, spatial, global_feat):
        # spatial: (B, H, W, 22), global_feat: (B, 19)
        x = self.conv_spatial(spatial)  # (B, H, W, C)
        g = self.linear_global(global_feat)  # (B, C)
        x = x + g.reshape(g.shape[0], 1, 1, g.shape[1])  # broadcast add

        for stage in self.stages:
            for layer in stage:
                x = layer(x)

        x = self.head(x)  # (B, H, W, 1)
        return x


# ============================================================================
# ResNet Architecture (b6c96)
# ============================================================================


class KataGPool(nn.Module):
    """
    KataGo global pooling: mean, scaled mean (by 1/10), and max pooled features.
    Returns concatenation of [mean, scaled_mean, max] along channel dimension.
    """

    def __call__(self, x, mask=None):
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        if mask is not None:
            # mask: (B, H, W, 1)
            x_masked = x * mask
            num_valid = mx.sum(mask, axis=(1, 2), keepdims=True)  # (B, 1, 1, 1)
            num_valid = mx.maximum(num_valid, 1.0)  # avoid division by zero
            mean_pool = mx.sum(x_masked, axis=(1, 2), keepdims=False) / num_valid.squeeze((2, 3))  # (B, C)
        else:
            mean_pool = mx.mean(x, axis=(1, 2))  # (B, C)
            num_valid = H * W

        # Scaled mean by 1/10 (normalized by sqrt(board_area) in original KataGo)
        scale = 0.1
        scaled_mean = mean_pool * scale

        # Max pooling
        max_pool = mx.max(x, axis=(1, 2))  # (B, C)

        # Concatenate: (B, 3*C)
        return mx.concatenate([mean_pool, scaled_mean, max_pool], axis=-1)


class KataConvAndGPool(nn.Module):
    """
    Convolution with global pooling branch (for gpool residual blocks).
    Regular path: conv_r (3x3 conv)
    GPool path: conv_g (3x3 conv) -> gpool -> linear_g -> add to regular output
    """

    def __init__(self, c_in: int, c_out: int, c_gpool: int = 32):
        super().__init__()
        # Regular path
        self.conv_r = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False)
        # Global pooling path
        self.conv_g = nn.Conv2d(c_in, c_gpool, kernel_size=3, padding=1, bias=False)
        self.gpool = KataGPool()
        self.linear_g = nn.Linear(3 * c_gpool, c_out, bias=False)  # Quantizable

    def __call__(self, x, mask=None):
        # Regular path
        out_r = self.conv_r(x)  # (B, H, W, c_out)

        # Global pooling path
        g = self.conv_g(x)  # (B, H, W, c_gpool)
        g = nn.relu(g)
        g = self.gpool(g, mask)  # (B, 3*c_gpool)
        g = self.linear_g(g)  # (B, c_out)
        g = g.reshape(g.shape[0], 1, 1, g.shape[1])  # (B, 1, 1, c_out)

        return out_r + g  # broadcast add


class ResBlock(nn.Module):
    """Basic residual block with two 3x3 convolutions."""

    def __init__(self, c_main: int, c_mid: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c_main, c_mid, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(c_mid, c_main, kernel_size=3, padding=1, bias=False)

    def __call__(self, x, mask=None):
        out = nn.relu(x)
        out = self.conv1(out)
        out = nn.relu(out)
        out = self.conv2(out)
        return x + out  # residual


class ResBlockGPool(nn.Module):
    """Residual block with global pooling in first conv."""

    def __init__(self, c_main: int, c_mid: int, c_gpool: int = 32):
        super().__init__()
        self.conv_and_gpool = KataConvAndGPool(c_main, c_mid, c_gpool)
        self.conv2 = nn.Conv2d(c_mid, c_main, kernel_size=3, padding=1, bias=False)

    def __call__(self, x, mask=None):
        out = nn.relu(x)
        out = self.conv_and_gpool(out, mask)
        out = nn.relu(out)
        out = self.conv2(out)
        return x + out  # residual


class MLXKataGoResNet(nn.Module):
    """
    KataGo ResNet model for MLX inference benchmarking.

    Input processing:
      - conv_spatial: Conv2d(22 -> trunk_channels, 3x3)
      - linear_global: Linear(19 -> trunk_channels), broadcast-added to spatial

    Trunk: sequence of ResBlock and ResBlockGPool based on config.
    Head: Conv2d(trunk_channels -> 1, 1x1) for minimal timing overhead.
    """

    def __init__(self, config: dict):
        super().__init__()
        trunk_channels = config["trunk_channels"]
        mid_channels = config["mid_channels"]
        gpool_channels = config["gpool_channels"]
        blocks = config["blocks"]

        # Input processing
        self.conv_spatial = nn.Conv2d(22, trunk_channels, kernel_size=3, padding=1)
        self.linear_global = nn.Linear(19, trunk_channels)

        # Build trunk blocks
        self.blocks = []
        for block_type in blocks:
            if block_type == "regular":
                self.blocks.append(ResBlock(trunk_channels, mid_channels))
            elif block_type == "gpool":
                self.blocks.append(ResBlockGPool(trunk_channels, mid_channels, gpool_channels))
            else:
                raise ValueError(f"Unknown block type: {block_type}")

        # Minimal head for timing
        self.head = nn.Conv2d(trunk_channels, 1, kernel_size=1)

    def __call__(self, spatial, global_feat, mask=None):
        # spatial: (B, H, W, 22), global_feat: (B, 19)
        x = self.conv_spatial(spatial)  # (B, H, W, C)
        g = self.linear_global(global_feat)  # (B, C)
        x = x + g.reshape(g.shape[0], 1, 1, g.shape[1])  # broadcast add

        for block in self.blocks:
            x = block(x, mask)

        x = self.head(x)  # (B, H, W, 1)
        return x


def create_model(config: dict):
    """Factory function to create the appropriate model type."""
    model_type = config.get("type", "fastvit")
    if model_type == "fastvit":
        return MLXKataGoModel(config)
    elif model_type == "resnet":
        return MLXKataGoResNet(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model):
    """Count total parameters in the model."""
    params = model.parameters()
    total = 0

    def count_leaf(p):
        if isinstance(p, mx.array):
            return p.size
        return 0

    def recurse(obj):
        nonlocal total
        if isinstance(obj, mx.array):
            total += obj.size
        elif isinstance(obj, dict):
            for v in obj.values():
                recurse(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                recurse(v)

    recurse(params)
    return total


def benchmark_model(model, spatial, global_feat, num_warmup=20, num_runs=100):
    """Benchmark model inference with warmup and timed runs."""
    # Warmup
    for _ in range(num_warmup):
        out = model(spatial, global_feat)
        mx.eval(out)

    # Timed runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        out = model(spatial, global_feat)
        mx.eval(out)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    return times


def main():
    parser = argparse.ArgumentParser(description="MLX FastVIT Inference Benchmark")
    parser.add_argument("--pos-len", type=int, default=9, help="Board size")
    parser.add_argument("--models", nargs="+", default=["ft6c96", "ft6c96a", "ft8c96a"],
                        help="Models to benchmark")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of timed runs")
    parser.add_argument("--num-warmup", type=int, default=20, help="Number of warmup runs")
    args = parser.parse_args()

    H = W = args.pos_len
    # Input tensors (NHWC for MLX)
    spatial = mx.random.normal((1, H, W, 22))
    global_feat = mx.random.normal((1, 19))

    print(f"Board size: {H}x{W}, Warmup: {args.num_warmup}, Runs: {args.num_runs}")
    print(f"{'='*80}")

    results = {}
    for model_name in args.models:
        if model_name not in MODEL_CONFIGS:
            print(f"Unknown model: {model_name}, skipping")
            continue

        config = MODEL_CONFIGS[model_name]
        print(f"\n{'─'*80}")
        print(f"Model: {model_name}")

        # FP32 benchmark
        model_fp32 = create_model(config)
        mx.eval(model_fp32.parameters())
        params_fp32 = count_parameters(model_fp32)
        model_type = config.get("type", "fastvit")
        print(f"  Type: {model_type}, FP32 parameters: {params_fp32:,}")

        times_fp32 = benchmark_model(model_fp32, spatial, global_feat,
                                     args.num_warmup, args.num_runs)
        mean_fp32 = sum(times_fp32) / len(times_fp32)
        median_fp32 = sorted(times_fp32)[len(times_fp32) // 2]
        print(f"  FP32 - Mean: {mean_fp32:.3f} ms, Median: {median_fp32:.3f} ms")

        # INT8 quantized benchmark
        model_int8 = create_model(config)
        mx.eval(model_int8.parameters())
        # Quantize Linear layers (group_size=32 for embed_dim=96;
        # skip linear_global: input dim 19 not group-divisible)
        if model_type == "fastvit":
            for stage in model_int8.stages:
                for layer in stage:
                    nn.quantize(layer, bits=8, group_size=32)
        elif model_type == "resnet":
            # Quantize Linear layers in gpool blocks
            for block in model_int8.blocks:
                if hasattr(block, 'conv_and_gpool'):
                    nn.quantize(block.conv_and_gpool.linear_g, bits=8, group_size=32)
        mx.eval(model_int8.parameters())
        params_int8 = count_parameters(model_int8)
        print(f"  INT8 parameters: {params_int8:,} (quantized Linear layers)")

        times_int8 = benchmark_model(model_int8, spatial, global_feat,
                                     args.num_warmup, args.num_runs)
        mean_int8 = sum(times_int8) / len(times_int8)
        median_int8 = sorted(times_int8)[len(times_int8) // 2]
        print(f"  INT8 - Mean: {mean_int8:.3f} ms, Median: {median_int8:.3f} ms")

        speedup = median_fp32 / median_int8 if median_int8 > 0 else 0
        print(f"  Speedup (median): {speedup:.2f}x")

        results[model_name] = {
            "params_fp32": params_fp32,
            "fp32_mean": mean_fp32,
            "fp32_median": median_fp32,
            "int8_mean": mean_int8,
            "int8_median": median_int8,
            "speedup": speedup,
        }

    # Summary table
    print(f"\n{'='*80}")
    print("Summary: MLX Inference Benchmark (9x9, Apple Silicon, batch=1)")
    print(f"{'─'*80}")
    print(f"{'Model':<10} {'FP32 Mean':>10} {'FP32 Med':>10} {'INT8 Mean':>10} {'INT8 Med':>10} {'Speedup':>8}")
    print(f"{'─'*80}")
    for name, r in results.items():
        print(f"{name:<10} {r['fp32_mean']:>8.3f}ms {r['fp32_median']:>8.3f}ms "
              f"{r['int8_mean']:>8.3f}ms {r['int8_median']:>8.3f}ms {r['speedup']:>7.2f}x")
    print(f"{'─'*80}")


if __name__ == "__main__":
    main()
