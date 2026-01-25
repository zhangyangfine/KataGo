#!/usr/bin/env python3
"""
Plot Comparison of Multiple Distillation Training Runs

Creates a 4-panel plot comparing loss curves across training variants:
1. Policy Loss (combined) - all runs
2. Value Loss (combined) - all runs
3. Soft Policy Loss - all runs (key distillation quality metric)
4. Total Loss - all runs

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


# Define runs to compare (top 3 variants)
RUNS = [
    {
        "dir": "/Users/chinchangyang/Code/KataGo-Trainings/distill-ft6c96-9x9/variant-a",
        "label": r"A: Pure ($\alpha$=1.0, T=1)",
        "color": "#1ABC9C",
    },
    {
        "dir": "/Users/chinchangyang/Code/KataGo-Trainings/distill-ft6c96-9x9/variant-b",
        "label": r"B: Sharp ($\alpha$=0.9, T=1)",
        "color": "#9467BD",
    },
    {
        "dir": "/Users/chinchangyang/Code/KataGo-Trainings/distill-ft6c96-9x9/variant-c",
        "label": r"C: Sharp+Sched ($\alpha$=0.9, T=1, BS=64)",
        "color": "#E74C3C",
    },
    {
        "dir": "/Users/chinchangyang/Code/KataGo-Trainings/distill-ft6c96-9x9/variant-d",
        "label": "D: AllAttn ft6c96a",
        "color": "#3498DB",
    },
    {
        "dir": "/Users/chinchangyang/Code/KataGo-Trainings/distill-ft6c96-9x9/variant-e",
        "label": "E: DeepAttn ft8c96a",
        "color": "#F39C12",
    },
    {
        "dir": "/Users/chinchangyang/Code/KataGo-Trainings/distill-ft6c96-9x9/variant-f",
        "label": "F: ResNet b6c96",
        "color": "#2ECC71",
    },
]


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
    # Panel 1: Policy Loss (combined)
    # ==========================
    ax1 = axes[0, 0]
    for i, rd in enumerate(run_data):
        if 'policy_loss' in rd['metrics']:
            data = np.array(rd['metrics']['policy_loss'])
            ax1.plot(rd['samples'], smooth_data(data, smooth_window),
                     color=rd['color'], linewidth=2, linestyle=linestyles[i % len(linestyles)],
                     label=rd['label'])
            ax1.plot(rd['samples'], data, alpha=0.15, color=rd['color'], linewidth=0.5)

    ax1.set_xlabel('Samples')
    ax1.set_ylabel('Policy Loss')
    ax1.set_title('Policy Loss (Combined)')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ==========================
    # Panel 2: Value Loss (combined)
    # ==========================
    ax2 = axes[0, 1]
    for i, rd in enumerate(run_data):
        if 'value_loss' in rd['metrics']:
            data = np.array(rd['metrics']['value_loss'])
            ax2.plot(rd['samples'], smooth_data(data, smooth_window),
                     color=rd['color'], linewidth=2, linestyle=linestyles[i % len(linestyles)],
                     label=rd['label'])
            ax2.plot(rd['samples'], data, alpha=0.15, color=rd['color'], linewidth=0.5)

    ax2.set_xlabel('Samples')
    ax2.set_ylabel('Value Loss')
    ax2.set_title('Value Loss (Combined)')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ==========================
    # Panel 3: Soft Policy Loss (key metric)
    # ==========================
    ax3 = axes[1, 0]
    for i, rd in enumerate(run_data):
        if 'soft_policy_loss' in rd['metrics']:
            data = np.array(rd['metrics']['soft_policy_loss'])
            ax3.plot(rd['samples'], smooth_data(data, smooth_window),
                     color=rd['color'], linewidth=2, linestyle=linestyles[i % len(linestyles)],
                     label=rd['label'])
            ax3.plot(rd['samples'], data, alpha=0.15, color=rd['color'], linewidth=0.5)

    ax3.set_xlabel('Samples')
    ax3.set_ylabel('Soft Policy Loss')
    ax3.set_title('Soft Policy Loss (Distillation Quality)')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ==========================
    # Panel 4: Total Loss
    # ==========================
    ax4 = axes[1, 1]
    for i, rd in enumerate(run_data):
        if 'total_loss' in rd['metrics']:
            data = np.array(rd['metrics']['total_loss'])
            ax4.plot(rd['samples'], smooth_data(data, smooth_window),
                     color=rd['color'], linewidth=2, linestyle=linestyles[i % len(linestyles)],
                     label=rd['label'])
            ax4.plot(rd['samples'], data, alpha=0.15, color=rd['color'], linewidth=0.5)

    ax4.set_xlabel('Samples')
    ax4.set_ylabel('Total Loss')
    ax4.set_title('Total Loss')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)

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
    plt.subplots_adjust(bottom=0.06)

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
