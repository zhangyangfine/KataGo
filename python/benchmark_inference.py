#!/usr/bin/env python3
"""Benchmark inference latency for FastVIT model variants."""

import argparse
import time
import numpy as np
import torch

from katago.train import modelconfigs
from katago.train.model_pytorch import Model


def benchmark_model(model, pos_len, device, num_warmup=20, num_runs=100):
    """Benchmark forward pass latency for a model."""
    model.eval()
    batch_size = 1
    num_bin_features = 22
    num_global_features = 19
    # Create random input matching KataGo input format
    # Channel 0 is the mask (1 for valid positions)
    bin_input = torch.randn(batch_size, num_bin_features, pos_len, pos_len, device=device)
    bin_input[:, 0, :, :] = 1.0  # mask channel
    global_input = torch.randn(batch_size, num_global_features, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            model(bin_input, global_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()

            start = time.perf_counter()
            model(bin_input, global_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

    return np.array(times)


def main():
    parser = argparse.ArgumentParser(description="Benchmark FastVIT inference")
    parser.add_argument("--pos-len", type=int, default=9, help="Board size")
    parser.add_argument("--models", nargs="+", default=["ft6c96", "ft6c96a", "ft8c96a"],
                        help="Model names to benchmark")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of timed runs")
    parser.add_argument("--num-warmup", type=int, default=20, help="Number of warmup runs")
    args = parser.parse_args()

    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Board size: {args.pos_len}x{args.pos_len}")
    print(f"Runs: {args.num_runs} (warmup: {args.num_warmup})")
    print()

    results = {}
    for model_name in args.models:
        config = modelconfigs.config_of_name[model_name]
        model = Model(config, args.pos_len)
        model.initialize()
        model.to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        times = benchmark_model(model, args.pos_len, device,
                                num_warmup=args.num_warmup, num_runs=args.num_runs)

        results[model_name] = {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "median_ms": np.median(times),
        }

        print(f"{model_name}:")
        print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")
        print(f"  Latency: {np.mean(times):.2f} +/- {np.std(times):.2f} ms "
              f"(min={np.min(times):.2f}, median={np.median(times):.2f})")
        print()

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Summary table
    print("=" * 70)
    print(f"{'Model':<12} {'Params':>10} {'Mean (ms)':>10} {'Std (ms)':>10} {'Min (ms)':>10} {'Median':>10}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<12} {r['total_params']:>10,} {r['mean_ms']:>10.2f} {r['std_ms']:>10.2f} "
              f"{r['min_ms']:>10.2f} {r['median_ms']:>10.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
