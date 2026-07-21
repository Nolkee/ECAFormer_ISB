#!/usr/bin/env python
"""Count model parameters, either exactly from a checkpoint state_dict or by
instantiating network_g from a training yml.

Usage:
  python tools/count_params.py --ckpt experiments/<run>/best_psnr_*.pth [--key params]
  python tools/count_params.py --opt Options/<config>.yml
"""
import argparse
import sys
from os import path as osp

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def from_ckpt(path, key):
    import torch
    d = torch.load(path, map_location='cpu')
    if key not in d:
        key = 'params' if 'params' in d else list(d.keys())[0]
    sd = d[key]
    n = sum(v.numel() for v in sd.values())
    return n, key


def from_opt(path):
    from basicsr.utils.options import parse
    from basicsr.models.archs import define_network
    from copy import deepcopy
    opt = parse(path, is_train=False)
    net = define_network(deepcopy(opt['network_g']))
    n = sum(p.numel() for p in net.parameters())
    return n, opt['network_g']['type']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default=None)
    p.add_argument('--key', default='params')
    p.add_argument('--opt', default=None)
    args = p.parse_args()
    if args.ckpt:
        n, key = from_ckpt(args.ckpt, args.key)
        print(f'PARAMS {n} ({n/1e6:.3f}M) source=ckpt:{osp.basename(args.ckpt)} key={key}')
    if args.opt:
        n, t = from_opt(args.opt)
        print(f'PARAMS {n} ({n/1e6:.3f}M) source=opt:{osp.basename(args.opt)} type={t}')
    if not args.ckpt and not args.opt:
        p.error('need --ckpt or --opt')


if __name__ == '__main__':
    main()
