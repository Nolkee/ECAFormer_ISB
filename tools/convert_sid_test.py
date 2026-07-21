#!/usr/bin/env python
"""Convert the Retinexformer-processed SID test split (npy) into the repo's
paired-folder layout so tools/eval_lol.py can consume it unchanged.

Input  (from sid_processed.zip, test scenes = folders starting with '1'):
  <src>/short_sid2/1xxxx/<scene>_<frame>_<exp>.npy   (BGR uint8, low-light)
  <src>/long_sid2/1xxxx/<scene>_00_10s.npy           (BGR uint8, GT, 1/scene)

Output:
  <dst>/Low/<scene>_<frame>_<exp>.png
  <dst>/Normal/<scene>_<frame>_<exp>.png   (scene GT duplicated per short)

Usage:
  python tools/convert_sid_test.py data/SID_raw data/SID_test
"""
import argparse
import glob
import os
from os import path as osp

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('src', help='dir containing short_sid2/ and long_sid2/')
    p.add_argument('dst', help='output dir (Low/ and Normal/ created inside)')
    args = p.parse_args()

    low_dir = osp.join(args.dst, 'Low')
    gt_dir = osp.join(args.dst, 'Normal')
    os.makedirs(low_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    scenes = sorted(
        d for d in glob.glob(osp.join(args.src, 'short_sid2', '*'))
        if osp.isdir(d) and osp.basename(d).startswith('1'))
    n_pairs = 0
    for scene_dir in scenes:
        scene = osp.basename(scene_dir)
        longs = sorted(glob.glob(osp.join(args.src, 'long_sid2', scene, '*.npy')))
        if not longs:
            print(f'WARNING: no GT for scene {scene}, skipped')
            continue
        gt = np.load(longs[0])  # BGR uint8
        for short_path in sorted(glob.glob(osp.join(scene_dir, '*.npy'))):
            name = osp.splitext(osp.basename(short_path))[0]
            lq = np.load(short_path)
            cv2.imwrite(osp.join(low_dir, f'{name}.png'), lq)
            cv2.imwrite(osp.join(gt_dir, f'{name}.png'), gt)
            n_pairs += 1
    print(f'{len(scenes)} test scenes -> {n_pairs} pairs in {args.dst}')


if __name__ == '__main__':
    main()
