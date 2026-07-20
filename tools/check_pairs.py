#!/usr/bin/env python
"""Sanity-check a paired low-light dataset: are the LQ images actually dark?

Prints per-pair mean brightness and LQ-vs-GT PSNR for the first N pairs plus
dataset-level stats. A healthy low-light test split has LQ mean ~0.03-0.15,
GT mean ~0.3-0.5, and LQ-vs-GT PSNR well under 15 dB. LQ mean above ~0.3 or
pair PSNR above ~20 dB means the "low" folder is not low-light data.

Usage:
  python tools/check_pairs.py data/LOLv2Real/Test/Low data/LOLv2Real/Test/Normal
"""
import argparse
import glob
import os
from os import path as osp

import cv2
import numpy as np


def load(p):
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f'cannot read {p}')
    return img.astype(np.float64) / 255.0


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return float('inf')
    return 10.0 * np.log10(1.0 / mse)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('lq_dir')
    p.add_argument('gt_dir')
    p.add_argument('-n', type=int, default=5, help='pairs to print individually')
    args = p.parse_args()

    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.JPG', '*.PNG')
    lq_files = sorted(f for e in exts for f in glob.glob(osp.join(args.lq_dir, e)))
    gt_files = sorted(f for e in exts for f in glob.glob(osp.join(args.gt_dir, e)))
    print(f'LQ: {len(lq_files)} files in {args.lq_dir}')
    print(f'GT: {len(gt_files)} files in {args.gt_dir}')
    if not lq_files or not gt_files:
        raise SystemExit('empty folder')
    if len(lq_files) != len(gt_files):
        print('WARNING: file-count mismatch')

    lq_means, gt_means, pair_psnrs = [], [], []
    for i, (lf, gf) in enumerate(zip(lq_files, gt_files)):
        lq, gt = load(lf), load(gf)
        same_size = lq.shape == gt.shape
        lm, gm = float(lq.mean()), float(gt.mean())
        pp = psnr(lq, gt) if same_size else float('nan')
        lq_means.append(lm)
        gt_means.append(gm)
        if same_size:
            pair_psnrs.append(pp)
        if i < args.n:
            print(f'  [{i}] {osp.basename(lf)} vs {osp.basename(gf)}: '
                  f'lq_mean={lm:.4f} gt_mean={gm:.4f} pair_psnr={pp:.2f} '
                  f'{"" if same_size else "SIZE-MISMATCH"}')

    lq_mu = float(np.mean(lq_means))
    gt_mu = float(np.mean(gt_means))
    pp_mu = float(np.mean(pair_psnrs)) if pair_psnrs else float('nan')
    print(f'\nDATASET: lq_mean={lq_mu:.4f}  gt_mean={gt_mu:.4f}  '
          f'mean_pair_psnr={pp_mu:.2f} dB  (n={len(lq_files)})')
    verdict = 'OK: LQ looks like genuine low-light data'
    if lq_mu > 0.3 or (not np.isnan(pp_mu) and pp_mu > 20):
        verdict = 'SUSPECT: LQ folder does NOT look like low-light data'
    elif lq_mu > 0.2 or (not np.isnan(pp_mu) and pp_mu > 15):
        verdict = 'BORDERLINE: check samples visually'
    print('VERDICT:', verdict)


if __name__ == '__main__':
    main()
