#!/usr/bin/env python3
"""
Plot Training Metrics for Knowledge Distillation

This script creates a multi-panel plot showing:
1. Policy Loss (hard + soft distillation)
2. Value Loss (hard + soft distillation)
3. Total Loss
4. Learning Rate

Features:
- Real-time updates (configurable refresh interval)
- Log-scale x-axis (samples)
- Linear y-axis for losses
- Publication-quality styling

Usage:
    python plot_distill_loss.py /path/to/traindir
    python plot_distill_loss.py /path/to/traindir --refresh 5 --save plot.png
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive display
import matplotlib.pyplot as plt
import numpy as np


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


def smooth_data(data, window=50):
    """Apply exponential moving average smoothing."""
    if len(data) < window:
        return data
    smoothed = []
    alpha = 2.0 / (window + 1)
    ema = data[0]
    for val in data:
        ema = alpha * val + (1 - alpha) * ema
        smoothed.append(ema)
    return smoothed


def create_distillation_plot(traindir, save_path=None, smooth_window=50):
    """Create multi-panel plot for distillation training metrics."""

    metrics_file = os.path.join(traindir, "metrics_train.json")
    metrics = load_metrics(metrics_file)

    if not metrics or 'global_step_samples' not in metrics:
        print(f"No valid metrics found in {metrics_file}")
        return False

    samples = np.array(metrics['global_step_samples'])

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Knowledge Distillation Training Progress', fontsize=14, fontweight='bold')

    # Color scheme
    colors = {
        'total': '#2E86AB',      # Blue
        'hard': '#A23B72',       # Magenta
        'soft': '#F18F01',       # Orange
        'lr': '#C73E1D',         # Red
        'smooth': '#1B4332',     # Dark green
    }

    # ==========================
    # Panel 1: Policy Loss
    # ==========================
    ax1 = axes[0, 0]
    if 'policy_loss' in metrics:
        policy_loss = np.array(metrics['policy_loss'])
        ax1.plot(samples, policy_loss, alpha=0.3, color=colors['total'], linewidth=0.5)
        ax1.plot(samples, smooth_data(policy_loss, smooth_window), color=colors['total'],
                 linewidth=2, label='Combined Policy')

    if 'hard_policy_loss' in metrics:
        hard_policy = np.array(metrics['hard_policy_loss'])
        ax1.plot(samples, smooth_data(hard_policy, smooth_window), color=colors['hard'],
                 linewidth=1.5, linestyle='--', label='Hard Policy')

    if 'soft_policy_loss' in metrics:
        soft_policy = np.array(metrics['soft_policy_loss'])
        ax1.plot(samples, smooth_data(soft_policy, smooth_window), color=colors['soft'],
                 linewidth=1.5, linestyle=':', label='Soft Policy (Distill)')

    ax1.set_xlabel('Samples')
    ax1.set_ylabel('Policy Loss')
    ax1.set_title('Policy Loss')
    ax1.set_xscale('log')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=max(1, samples[0]))

    # ==========================
    # Panel 2: Value Loss
    # ==========================
    ax2 = axes[0, 1]
    if 'value_loss' in metrics:
        value_loss = np.array(metrics['value_loss'])
        ax2.plot(samples, value_loss, alpha=0.3, color=colors['total'], linewidth=0.5)
        ax2.plot(samples, smooth_data(value_loss, smooth_window), color=colors['total'],
                 linewidth=2, label='Combined Value')

    if 'hard_value_loss' in metrics:
        hard_value = np.array(metrics['hard_value_loss'])
        ax2.plot(samples, smooth_data(hard_value, smooth_window), color=colors['hard'],
                 linewidth=1.5, linestyle='--', label='Hard Value')

    if 'soft_value_loss' in metrics:
        soft_value = np.array(metrics['soft_value_loss'])
        ax2.plot(samples, smooth_data(soft_value, smooth_window), color=colors['soft'],
                 linewidth=1.5, linestyle=':', label='Soft Value (Distill)')

    ax2.set_xlabel('Samples')
    ax2.set_ylabel('Value Loss')
    ax2.set_title('Value Loss')
    ax2.set_xscale('log')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=max(1, samples[0]))

    # ==========================
    # Panel 3: Total Loss
    # ==========================
    ax3 = axes[1, 0]
    if 'total_loss' in metrics:
        total_loss = np.array(metrics['total_loss'])
        ax3.plot(samples, total_loss, alpha=0.3, color=colors['total'], linewidth=0.5)
        ax3.plot(samples, smooth_data(total_loss, smooth_window), color=colors['total'],
                 linewidth=2, label='Total Loss')

    if 'ownership_loss' in metrics:
        ownership_loss = np.array(metrics['ownership_loss'])
        ax3.plot(samples, smooth_data(ownership_loss, smooth_window), color=colors['soft'],
                 linewidth=1.5, linestyle=':', label='Ownership Loss')

    ax3.set_xlabel('Samples')
    ax3.set_ylabel('Loss')
    ax3.set_title('Total Loss')
    ax3.set_xscale('log')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(left=max(1, samples[0]))

    # ==========================
    # Panel 4: Learning Rate
    # ==========================
    ax4 = axes[1, 1]
    if 'lr' in metrics:
        lr = np.array(metrics['lr'])
        ax4.plot(samples, lr, color=colors['lr'], linewidth=2, label='Learning Rate')
        ax4.set_ylabel('Learning Rate')
    else:
        ax4.text(0.5, 0.5, 'No LR data available', ha='center', va='center',
                 transform=ax4.transAxes, fontsize=12, color='gray')

    ax4.set_xlabel('Samples')
    ax4.set_title('Learning Rate Schedule')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(left=max(1, samples[0]))

    # Add summary statistics
    if 'total_loss' in metrics:
        last_loss = metrics['total_loss'][-1]
        min_loss = min(metrics['total_loss'])
        epoch = metrics.get('epoch', [0])[-1]
        fig.text(0.02, 0.02,
                 f'Epoch: {epoch} | Last Loss: {last_loss:.4f} | Min Loss: {min_loss:.4f} | '
                 f'Samples: {samples[-1]:,.0f}',
                 fontsize=10, family='monospace')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    return True


def main():
    parser = argparse.ArgumentParser(description='Plot distillation training metrics')
    parser.add_argument('traindir', help='Training directory containing metrics_train.json')
    parser.add_argument('--refresh', type=int, default=0,
                        help='Refresh interval in seconds (0 for single plot)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save plot to file instead of displaying')
    parser.add_argument('--smooth', type=int, default=50,
                        help='Smoothing window size (default: 50)')
    args = parser.parse_args()

    if not os.path.isdir(args.traindir):
        print(f"Error: {args.traindir} is not a directory")
        sys.exit(1)

    metrics_file = os.path.join(args.traindir, "metrics_train.json")
    if not os.path.exists(metrics_file):
        print(f"Error: {metrics_file} not found")
        sys.exit(1)

    if args.save:
        # Single plot to file
        create_distillation_plot(args.traindir, save_path=args.save, smooth_window=args.smooth)
    elif args.refresh > 0:
        # Interactive mode with refresh
        plt.ion()
        print(f"Plotting metrics from {args.traindir}")
        print(f"Refreshing every {args.refresh} seconds. Press Ctrl+C to stop.")

        try:
            while True:
                plt.clf()
                if not create_distillation_plot(args.traindir, smooth_window=args.smooth):
                    print("Waiting for metrics...")
                plt.pause(args.refresh)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        # Single interactive plot
        create_distillation_plot(args.traindir, smooth_window=args.smooth)
        plt.show()


if __name__ == "__main__":
    main()
