#!/usr/bin/env python
"""Export validation curves (iter, psnr, ssim, lpips) from training artifacts.

Sources (both produced by this repo's training loop):
  - root train_<name>.log files: validation rows are embedded as CSV lines
    `iter,psnr,ssim,lpips`
  - experiments/<name>/metric.csv: same rows plus header lines

Usage:
  python tools/export_curves.py train_ISB_ecaformer_r52b_late_ema_anchor_9k.log \
      --out paper_pack/figures_raw/r52b_curve.csv
  python tools/export_curves.py experiments/ECAFormer_baseline_lolv1_fair_paper/metric.csv \
      --out paper_pack/figures_raw/base250k_curve.csv
  # optional quick plot (requires matplotlib):
  python tools/export_curves.py <src> --out <csv> --plot <png> --mark-best
"""
import argparse
import csv
import re
from os import path as osp
import os

ROW_RE = re.compile(r'^(\d+),([-\d.eE]+),([-\d.eE]+),([-\d.eE]+)\s*$')


def parse_rows(src_path):
    rows = []
    with open(src_path, 'r', errors='replace') as f:
        for line in f:
            m = ROW_RE.match(line.strip())
            if m:
                rows.append({
                    'iter': int(m.group(1)),
                    'psnr': float(m.group(2)),
                    'ssim': float(m.group(3)),
                    'lpips': float(m.group(4)),
                })
    # de-duplicate on iter (resumes can repeat), keep the last occurrence
    dedup = {}
    for r in rows:
        dedup[r['iter']] = r
    return [dedup[k] for k in sorted(dedup)]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('src', help='train_*.log or metric.csv')
    p.add_argument('--out', required=True, help='tidy CSV output path')
    p.add_argument('--plot', default=None, help='optional PNG path (matplotlib)')
    p.add_argument('--mark-best', action='store_true',
                   help='annotate best-PSNR iter on the plot')
    args = p.parse_args()

    rows = parse_rows(args.src)
    if not rows:
        raise SystemExit(f'no validation rows found in {args.src}')

    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['iter', 'psnr', 'ssim', 'lpips'])
        w.writeheader()
        w.writerows(rows)

    best = max(rows, key=lambda r: r['psnr'])
    print(f'{len(rows)} rows -> {args.out}; '
          f"best_psnr={best['psnr']:.3f} @ iter {best['iter']} "
          f"(ssim {best['ssim']:.4f}, lpips {best['lpips']:.4f})")

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
        its = [r['iter'] for r in rows]
        for ax, key in zip(axes, ('psnr', 'ssim', 'lpips')):
            ax.plot(its, [r[key] for r in rows], lw=1.5)
            ax.set_ylabel(key.upper())
            ax.grid(alpha=0.3)
            if args.mark_best:
                ax.axvline(best['iter'], color='tab:red', ls='--', lw=1)
        axes[0].set_title(osp.basename(args.src))
        axes[-1].set_xlabel('iteration')
        fig.tight_layout()
        fig.savefig(args.plot, dpi=200)
        print(f'plot -> {args.plot}')


if __name__ == '__main__':
    main()
