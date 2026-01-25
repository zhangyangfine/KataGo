#!/usr/bin/env python3
"""
Plot Comparison of Multiple Distillation Training Runs

Creates a 4-panel plot comparing loss curves across training variants:
1. Soft Policy Loss - all runs
2. Soft Value Loss - all runs
3. Learning Rate (cosine decay) - all runs
4. MLX Inference Time (FP32 vs INT8 bar chart)

Usage:
    python plot_distill_comparison.py --save output.png
    python plot_distill_comparison.py  # interactive display
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# Define runs to compare
RUNS = [
    {
        "dir": "/Users/chinchangyang/Code/KataGo-Trainings/distill-ft6c96-9x9/variant-f",
        "label": "F: ResNet b6c96",
        "color": "#2ECC71",
    },
    {
        "dir": "/Users/chinchangyang/Code/KataGo-Trainings/distill-ft6c96-9x9/variant-x",
        "label": "X: ExtremeFFN ft6c96x",
        "color": "#E74C3C",
    },
    {
        "dir": "/Users/chinchangyang/Code/KataGo-Trainings/distill-ft6c96-9x9/variant-w",
        "label": "W: WideChannels ft6c384 (QAT)",
        "color": "#8E44AD",
    },
    {
        "dir": "/Users/chinchangyang/Code/KataGo-Trainings/distill-ft6c96-9x9/variant-g",
        "label": "G: ResNet b10c128",
        "color": "#3498DB",
    },
    {
        "dir": "/Users/chinchangyang/Code/KataGo-Trainings/distill-ft6c96-9x9/variant-y",
        "label": "Y: Softer ft6c384 (T=2, WD=0.02)",
        "color": "#F39C12",
    },
    {
        "dir": "/Users/chinchangyang/Code/KataGo-Trainings/distill-ft6c96-9x9/variant-z",
        "label": "Z: Aggressive ft6c384 (LR=2x, QAT@5)",
        "color": "#1ABC9C",
    },
]

# MLX 8-bit quantized inference benchmark data (9x9, batch=1, median latency in ms)
MLX_BENCHMARK = {
    "b6c96":   {"fp32": 1.329, "int8": 1.143},
    "b10c128": {"fp32": 2.001, "int8": 2.072},
    "ft6c96x": {"fp32": 2.165, "int8": 1.958},
    "ft6c384": {"fp32": 1.995, "int8": 1.477},
}


def load_metrics(metrics_file):
    """Load metrics from JSON-lines file."""
    metrics = defaultdict(list)

    if not os.path.exists(metrics_file):
        return metrics

    with open(metrics_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        metrics[key].append(value)
            except json.JSONDecodeError:
                continue

    return metrics


def smooth_data(data, window=20):
    """Apply exponential moving average smoothing."""
    if len(data) < 2:
        return data
    smoothed = []
    alpha = 2.0 / (window + 1)
    ema = data[0]
    for val in data:
        ema = alpha * val + (1 - alpha) * ema
        smoothed.append(ema)
    return smoothed


def create_comparison_plot(runs, save_path=None, smooth_window=20):
    """Create multi-panel comparison plot for distillation training runs."""

    # Load all metrics
    run_data = []
    for run in runs:
        metrics_file = os.path.join(run["dir"], "metrics_train.json")
        metrics = load_metrics(metrics_file)
        if metrics and 'global_step_samples' in metrics:
            run_data.append({
                "metrics": metrics,
                "label": run["label"],
                "color": run["color"],
                "samples": np.array(metrics['global_step_samples']),
            })
        else:
            print(f"Warning: No metrics found in {metrics_file}")

    if not run_data:
        print("No valid metrics found in any run directory")
        return False

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('9x9 Distillation Training Comparison (FastVIT variants)', fontsize=14, fontweight='bold')

    linestyles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1))]

    # ==========================
    # Panel 1: Soft Policy Loss
    # ==========================
    ax1 = axes[0, 0]
    for i, rd in enumerate(run_data):
        if 'soft_policy_loss' in rd['metrics']:
            data = np.array(rd['metrics']['soft_policy_loss'])
            ax1.plot(rd['samples'], smooth_data(data, smooth_window),
                     color=rd['color'], linewidth=2, linestyle=linestyles[i % len(linestyles)],
                     label=rd['label'])
            ax1.plot(rd['samples'], data, alpha=0.15, color=rd['color'], linewidth=0.5)

    ax1.set_xlabel('Samples')
    ax1.set_ylabel('Soft Policy Loss')
    ax1.set_title('Soft Policy Loss')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ==========================
    # Panel 2: Soft Value Loss
    # ==========================
    ax2 = axes[0, 1]
    for i, rd in enumerate(run_data):
        if 'soft_value_loss' in rd['metrics']:
            data = np.array(rd['metrics']['soft_value_loss'])
            ax2.plot(rd['samples'], smooth_data(data, smooth_window),
                     color=rd['color'], linewidth=2, linestyle=linestyles[i % len(linestyles)],
                     label=rd['label'])
            ax2.plot(rd['samples'], data, alpha=0.15, color=rd['color'], linewidth=0.5)

    ax2.set_xlabel('Samples')
    ax2.set_ylabel('Soft Value Loss')
    ax2.set_title('Soft Value Loss')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ==========================
    # Panel 3: Learning Rate (Cosine Decay)
    # ==========================
    ax3 = axes[1, 0]
    for i, rd in enumerate(run_data):
        if 'lr' in rd['metrics']:
            data = np.array(rd['metrics']['lr'])
            ax3.plot(rd['samples'], data, color=rd['color'],
                     linewidth=2, linestyle=linestyles[i % len(linestyles)],
                     label=rd['label'])

    ax3.set_xlabel('Samples')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate (Cosine Decay)')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    # ==========================
    # Panel 4: MLX Inference Time (Bar Chart)
    # ==========================
    ax4 = axes[1, 1]
    models = list(MLX_BENCHMARK.keys())
    x = np.arange(len(models))
    width = 0.35

    fp32_vals = [MLX_BENCHMARK[m]['fp32'] for m in models]
    int8_vals = [MLX_BENCHMARK[m]['int8'] for m in models]

    ax4.bar(x - width/2, fp32_vals, width, label='FP32', color='#3498DB')
    ax4.bar(x + width/2, int8_vals, width, label='INT8', color='#2ECC71')

    ax4.set_xlabel('Model')
    ax4.set_ylabel('Latency (ms)')
    ax4.set_title('MLX Inference Time (9x9, batch=1, median)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # Add speedup annotations
    for i, (f, i8) in enumerate(zip(fp32_vals, int8_vals)):
        speedup = f / i8
        ax4.annotate(f'{speedup:.2f}x', xy=(x[i] + width/2, i8),
                     xytext=(0, 3), textcoords='offset points',
                     ha='center', fontsize=7)

    # Add final metrics summary
    summary_parts = []
    for rd in run_data:
        if 'soft_policy_loss' in rd['metrics']:
            final_soft = rd['metrics']['soft_policy_loss'][-1]
            min_soft = min(rd['metrics']['soft_policy_loss'])
            summary_parts.append(f"{rd['label']}: final={final_soft:.3f}, min={min_soft:.3f}")

    if summary_parts:
        fig.text(0.02, 0.02, "Soft Policy Loss  |  " + "  |  ".join(summary_parts),
                 fontsize=8, family='monospace')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")

    return True


def main():
    parser = argparse.ArgumentParser(description='Plot distillation training comparison')
    parser.add_argument('--save', type=str,
                        default="/Users/chinchangyang/Code/KataGo-Trainings/distill-ft6c96-9x9/comparison_curves.png",
                        help='Save plot to file')
    parser.add_argument('--smooth', type=int, default=20,
                        help='Smoothing window size (default: 20)')
    args = parser.parse_args()

    create_comparison_plot(RUNS, save_path=args.save, smooth_window=args.smooth)


if __name__ == "__main__":
    main()
