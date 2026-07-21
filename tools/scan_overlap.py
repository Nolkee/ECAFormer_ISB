#!/usr/bin/env python
"""Scan for near-duplicate scene overlap between two image folders.

Downsamples every image to a small grayscale thumbnail and, for each image in
dir_a, finds its nearest neighbour in dir_b under (1) raw L1 distance and
(2) z-normalized L1 (exposure-invariant, catches same scene at different
exposure). Reports per-image nearest matches and the overlap count under a
threshold.

Usage (does LOLv2-Real test leak LOLv1 train scenes?):
  python tools/scan_overlap.py data/LOLv2Real/Test/Normal data/LOLv1/Train/target
  python tools/scan_overlap.py data/LOLv2Real/Test/Low    data/LOLv1/Train/input --norm
"""
import argparse
import glob
from os import path as osp

import cv2
import numpy as np

THUMB = 32


def thumbs(folder):
    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.JPG', '*.PNG')
    files = sorted(f for e in exts for f in glob.glob(osp.join(folder, e)))
    arrs = []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        t = cv2.resize(img, (THUMB, THUMB), interpolation=cv2.INTER_AREA)
        arrs.append(t.astype(np.float64) / 255.0)
    return files, np.stack(arrs)


def znorm(x):
    mu = x.mean(axis=(1, 2), keepdims=True)
    sd = x.std(axis=(1, 2), keepdims=True) + 1e-8
    return (x - mu) / sd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dir_a', help='probe set (e.g. LOLv2Real test)')
    p.add_argument('dir_b', help='reference set (e.g. LOLv1 train)')
    p.add_argument('--thresh-raw', type=float, default=0.02,
                   help='raw L1/px duplicate threshold (default 0.02)')
    p.add_argument('--thresh-norm', type=float, default=0.25,
                   help='z-normalized L1/px same-scene threshold (default 0.25)')
    p.add_argument('--show', type=int, default=10, help='closest pairs to print')
    p.add_argument('--dump-matches', default=None, metavar='JSON',
                   help='write matched pairs (under thresholds) to a JSON file')
    args = p.parse_args()

    files_a, A = thumbs(args.dir_a)
    files_b, B = thumbs(args.dir_b)
    print(f'A: {len(files_a)} images  B: {len(files_b)} images')

    # raw L1 per pixel
    d_raw = np.abs(A[:, None] - B[None]).mean(axis=(2, 3))
    # exposure-invariant
    An, Bn = znorm(A), znorm(B)
    d_nrm = np.abs(An[:, None] - Bn[None]).mean(axis=(2, 3))

    nn_raw = d_raw.min(axis=1)
    nn_nrm = d_nrm.min(axis=1)
    idx_nrm = d_nrm.argmin(axis=1)

    order = np.argsort(nn_nrm)
    print(f'\nClosest {args.show} matches (z-normalized L1, exposure-invariant):')
    for i in order[:args.show]:
        j = idx_nrm[i]
        print(f'  {osp.basename(files_a[i])} ~ {osp.basename(files_b[j])}: '
              f'norm_l1={nn_nrm[i]:.4f} raw_l1={nn_raw[i]:.4f}')

    dup_raw = int((nn_raw < args.thresh_raw).sum())
    dup_nrm = int((nn_nrm < args.thresh_norm).sum())
    n = len(files_a)

    if args.dump_matches:
        import json
        import os
        matches = []
        for i in range(n):
            if nn_nrm[i] < args.thresh_norm or nn_raw[i] < args.thresh_raw:
                j = int(idx_nrm[i])
                matches.append({
                    'a': osp.basename(files_a[i]),
                    'b_nearest': osp.basename(files_b[j]),
                    'raw_l1': round(float(nn_raw[i]), 5),
                    'norm_l1': round(float(nn_nrm[i]), 5),
                    'near_exact': bool(nn_raw[i] < args.thresh_raw),
                })
        payload = {
            'dir_a': args.dir_a, 'dir_b': args.dir_b,
            'n_a': n, 'n_b': len(files_b),
            'thresh_raw': args.thresh_raw, 'thresh_norm': args.thresh_norm,
            'n_near_exact': dup_raw, 'n_same_scene': dup_nrm,
            'matches': matches,
        }
        os.makedirs(osp.dirname(osp.abspath(args.dump_matches)), exist_ok=True)
        with open(args.dump_matches, 'w') as f:
            json.dump(payload, f, indent=1)
        print(f'match list -> {args.dump_matches} ({len(matches)} entries)')

    print(f'\nOVERLAP: {dup_raw}/{n} near-exact (raw_l1 < {args.thresh_raw}); '
          f'{dup_nrm}/{n} same-scene (norm_l1 < {args.thresh_norm})')
    if dup_nrm > 0.2 * n:
        print('VERDICT: HEAVY overlap — dir_a is NOT a clean held-out set w.r.t. dir_b')
    elif dup_nrm > 0:
        print('VERDICT: partial overlap — filter the matched scenes before cross-set claims')
    else:
        print('VERDICT: no overlap detected at these thresholds')


if __name__ == '__main__':
    main()
