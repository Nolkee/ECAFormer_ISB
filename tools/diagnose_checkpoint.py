#!/usr/bin/env python3
"""
Checkpoint Diagnostic Tool for r43a/r38c Instability Analysis
==============================================================

Extracts channel statistics, AdaLN parameters, and gradient norms from saved
checkpoints to diagnose training instability at 3500-6000 iterations.

Usage:
    # Single experiment analysis
    python tools/diagnose_checkpoint.py \
        --exp_dir experiments/ISB_ecaformer_r43a_identity_scale \
        --iters 1000 3000 5000 7000 10000 \
        --output diagnosis_r43a.png

    # Compare r42a (stable) vs r43a (unstable)
    python tools/diagnose_checkpoint.py \
        --exp_dirs experiments/ISB_ecaformer_r42a_per_ch_res experiments/ISB_ecaformer_r43a_identity_scale \
        --labels "r42a(stable)" "r43a(unstable)" \
        --iters 1000 3000 5000 7000 10000 \
        --output r42a_vs_r43a_adaln.png
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def load_checkpoint(ckpt_path):
    """Load checkpoint with error handling."""
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        return ckpt
    except Exception as e:
        print(f"Error loading {ckpt_path}: {e}")
        return None


def extract_adaln_stats(state_dict):
    """Extract AdaLN projection layer statistics."""
    adaln_params = {}
    for name, param in state_dict.items():
        # Match adaln_proj, time_mlp, or similar AdaLN-related layers
        if any(keyword in name.lower() for keyword in ['adaln', 'time_mlp', 'proj']):
            if 'weight' in name or 'bias' in name:
                adaln_params[name] = {
                    'mean': param.mean().item(),
                    'std': param.std().item(),
                    'norm': param.norm().item(),
                    'shape': list(param.shape)
                }
    return adaln_params


def extract_channel_scales(state_dict):
    """Extract all channel-aware scale parameters."""
    scales = {}
    for name in ['identity_scale', 'channel_scale', 'residual_scale']:
        if name in state_dict:
            scales[name] = state_dict[name].squeeze().cpu().numpy()
        # Also check in nested module structure
        for key in state_dict.keys():
            if name in key and 'scale' in key:
                scales[name] = state_dict[key].squeeze().cpu().numpy()
                break
    return scales


def analyze_checkpoint_sequence(ckpt_dir, iters, label="experiment"):
    """Analyze checkpoint evolution across iterations."""
    ckpt_dir = Path(ckpt_dir)

    results = {
        'label': label,
        'iters': [],
        'adaln_norms': [],
        'adaln_means': [],
        'channel_scales': defaultdict(list),
    }

    for iter_num in iters:
        # Try both net_g and net_g_ema naming conventions
        ckpt_paths = [
            ckpt_dir / 'models' / f'net_g_{iter_num}.pth',
            ckpt_dir / 'models' / f'net_g_ema_{iter_num}.pth',
            ckpt_dir / f'{iter_num}_G.pth',
        ]

        ckpt = None
        for ckpt_path in ckpt_paths:
            if ckpt_path.exists():
                ckpt = load_checkpoint(ckpt_path)
                break

        if ckpt is None:
            print(f"Warning: No checkpoint found for iter {iter_num} in {ckpt_dir}")
            continue

        # Extract state dict (handle different checkpoint formats)
        if 'params_ema' in ckpt:
            state = ckpt['params_ema']
        elif 'params' in ckpt:
            state = ckpt['params']
        elif 'state_dict' in ckpt:
            state = ckpt['state_dict']
        else:
            state = ckpt

        results['iters'].append(iter_num)

        # AdaLN evolution
        adaln_stats = extract_adaln_stats(state)
        if adaln_stats:
            avg_norm = np.mean([v['norm'] for v in adaln_stats.values()])
            avg_mean = np.mean([abs(v['mean']) for v in adaln_stats.values()])
            results['adaln_norms'].append(avg_norm)
            results['adaln_means'].append(avg_mean)
        else:
            results['adaln_norms'].append(0.0)
            results['adaln_means'].append(0.0)

        # Channel scales
        scales = extract_channel_scales(state)
        for name, val in scales.items():
            if len(val.shape) == 0:  # Scalar
                val = np.array([val, val, val])
            results['channel_scales'][name].append(val)

    return results


def plot_diagnosis(results_list, output='diagnosis.png', highlight_window=(3500, 6000)):
    """Visualize diagnostic results for one or more experiments."""
    n_exp = len(results_list)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    colors = ['blue', 'red', 'green', 'orange', 'purple']

    # Plot 1: AdaLN norm evolution
    ax = axes[0, 0]
    for i, results in enumerate(results_list):
        if results['adaln_norms']:
            ax.plot(results['iters'], results['adaln_norms'],
                   f'{colors[i % len(colors)]}o-', label=results['label'], linewidth=2, markersize=6)
    ax.axvspan(highlight_window[0], highlight_window[1], alpha=0.2, color='yellow', label='Instability Window')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Avg AdaLN Norm', fontsize=12)
    ax.set_title('AdaLN Parameter Magnitude Evolution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Channel scales evolution
    ax = axes[0, 1]
    for results in results_list:
        for scale_name, scale_history in results['channel_scales'].items():
            if len(scale_history) == 0:
                continue
            scale_arr = np.array(scale_history)  # [num_iters, 3]
            if scale_arr.ndim == 1:
                scale_arr = scale_arr.reshape(-1, 1)
            for ch, ch_name, color in zip([0, 1, 2], ['R', 'G', 'B'], ['r', 'g', 'b']):
                if ch < scale_arr.shape[1]:
                    label = f"{results['label']}_{scale_name}[{ch_name}]"
                    ax.plot(results['iters'], scale_arr[:, ch],
                           f'{color}o-', label=label, alpha=0.7, linewidth=1.5, markersize=4)
    ax.axvspan(highlight_window[0], highlight_window[1], alpha=0.2, color='yellow')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Scale Value', fontsize=12)
    ax.set_title('Channel Scale Evolution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    # Plot 3: AdaLN mean evolution
    ax = axes[1, 0]
    for i, results in enumerate(results_list):
        if results['adaln_means']:
            ax.plot(results['iters'], results['adaln_means'],
                   f'{colors[i % len(colors)]}s-', label=results['label'], linewidth=2, markersize=6)
    ax.axvspan(highlight_window[0], highlight_window[1], alpha=0.2, color='yellow')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Avg |AdaLN Mean|', fontsize=12)
    ax.set_title('AdaLN Parameter Mean (Absolute)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Summary text
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = "Diagnostic Summary\n" + "="*40 + "\n\n"

    for results in results_list:
        summary_text += f"{results['label']}:\n"
        if results['adaln_norms']:
            norm_growth = results['adaln_norms'][-1] / (results['adaln_norms'][0] + 1e-8)
            summary_text += f"  AdaLN norm growth: {norm_growth:.2f}x\n"

        for scale_name, scale_history in results['channel_scales'].items():
            if len(scale_history) > 0:
                scale_arr = np.array(scale_history)
                if scale_arr.ndim == 2 and scale_arr.shape[1] >= 3:
                    init_val = scale_arr[0]
                    final_val = scale_arr[-1]
                    summary_text += f"  {scale_name}: {init_val} → {final_val}\n"
        summary_text += "\n"

    summary_text += f"\nHighlight window: {highlight_window[0]}-{highlight_window[1]} iter\n"
    summary_text += "Expected behavior if hypothesis correct:\n"
    summary_text += "  - r42a: Stable AdaLN norm growth\n"
    summary_text += "  - r43a: AdaLN norm spike at window\n"

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"✓ Saved diagnosis plot to {output}")


def main():
    parser = argparse.ArgumentParser(description='Diagnose checkpoint evolution to identify training instability')
    parser.add_argument('--exp_dir', type=str, help='Single experiment directory')
    parser.add_argument('--exp_dirs', type=str, nargs='+', help='Multiple experiment directories for comparison')
    parser.add_argument('--labels', type=str, nargs='+', help='Labels for experiments (same order as exp_dirs)')
    parser.add_argument('--iters', type=int, nargs='+', default=[1000, 3000, 5000, 7000, 10000],
                       help='Iterations to analyze')
    parser.add_argument('--output', type=str, default='diagnosis.png', help='Output plot path')
    parser.add_argument('--highlight', type=int, nargs=2, default=[3500, 6000],
                       help='Highlight window (start end)')

    args = parser.parse_args()

    # Determine experiment directories
    if args.exp_dirs:
        exp_dirs = args.exp_dirs
        labels = args.labels if args.labels else [f"exp_{i}" for i in range(len(exp_dirs))]
    elif args.exp_dir:
        exp_dirs = [args.exp_dir]
        labels = [Path(args.exp_dir).name]
    else:
        parser.error("Must provide either --exp_dir or --exp_dirs")

    # Analyze each experiment
    results_list = []
    for exp_dir, label in zip(exp_dirs, labels):
        print(f"Analyzing {exp_dir} ({label})...")
        results = analyze_checkpoint_sequence(exp_dir, args.iters, label=label)
        results_list.append(results)

    # Plot
    plot_diagnosis(results_list, output=args.output, highlight_window=tuple(args.highlight))


if __name__ == '__main__':
    main()
