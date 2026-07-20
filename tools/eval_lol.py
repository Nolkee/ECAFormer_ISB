#!/usr/bin/env python
"""Unified LOL evaluation entry (paper protocol: NO GT-mean).

Evaluates any checkpoint through the SAME validation code path used during
training (window-size padding, tensor-domain metrics, LPIPS-alex, crop_border
0), so numbers are directly comparable with training-time val logs.

Works for both ImageISBModel (ECAFormerISB) and ImageCleanModel
(ECAFormerBaseline) configs. In test mode no EMA net is built, so the chosen
checkpoint key is loaded straight into net_g and scored exactly like
training-time validation scored net_g_ema.

Examples:
  # Reproduction check (tool calibration against the logged val curve):
  python tools/eval_lol.py \
      --opt Options/ISB_ecaformer_r52b_late_ema_anchor_9k.yml \
      --ckpt experiments/ISB_ecaformer_r52b_late_ema_anchor_9k/models/net_g_latest.pth \
      --param-key params_ema --tag r52b_latest_ema_lolv1 \
      --out paper_pack/metrics/eval.csv

  # Best-checkpoint reproduction:
  python tools/eval_lol.py \
      --opt Options/ISB_ecaformer_r52b_late_ema_anchor_9k.yml \
      --ckpt experiments/ISB_ecaformer_r52b_late_ema_anchor_9k/best_psnr_22.69_11500.pth \
      --tag r52b_best_lolv1 --out paper_pack/metrics/eval.csv

  # Zero-shot cross-dataset probe (LOLv1-trained -> LOLv2-Real test):
  python tools/eval_lol.py \
      --opt Options/ISB_ecaformer_r52b_late_ema_anchor_9k.yml \
      --ckpt experiments/ISB_ecaformer_r52b_late_ema_anchor_9k/best_psnr_22.69_11500.pth \
      --dataroot-lq data/LOLv2Real/Test/Low --dataroot-gt data/LOLv2Real/Test/Normal \
      --dataset-name LOLv2Real_zeroshot --tag r52b_best_zeroshot_lolv2 \
      --out paper_pack/metrics/eval.csv

  # NFE sweep without retraining (ISB only):
  python tools/eval_lol.py --opt ... --ckpt ... --inference-steps 4 --tag r52b_nfe4 ...
"""
import argparse
import csv
import json
import statistics
import sys
import time
from os import path as osp
import os

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from basicsr.utils.options import parse  # noqa: E402
from basicsr.models import create_model  # noqa: E402
from basicsr.data import create_dataset, create_dataloader  # noqa: E402


def apply_overrides(opt, overrides):
    """Apply dotted-path overrides like network_g.use_eca=false onto opt."""
    import yaml
    for item in overrides or []:
        if '=' not in item:
            raise ValueError(f'--override expects key=value, got: {item}')
        key_path, raw_val = item.split('=', 1)
        value = yaml.safe_load(raw_val)
        node = opt
        keys = key_path.split('.')
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = value


def build_opt(args):
    opt = parse(args.opt, is_train=False)
    opt['dist'] = False
    opt['rank'] = 0
    opt['world_size'] = 1
    opt['path']['pretrain_network_g'] = args.ckpt
    opt['path']['strict_load_g'] = True
    # load_network falls back to 'params' when the requested key is absent,
    # so 'params_ema' is a safe default for both old (single-key) and new
    # (dual-key) best checkpoints.
    opt['path']['param_key'] = args.param_key
    if args.inference_steps > 0:
        opt['val']['inference_steps'] = args.inference_steps
    apply_overrides(opt, args.override)
    return opt


def build_val_loader(opt, args):
    val_opt = opt['datasets']['val']
    if args.dataroot_lq:
        val_opt['dataroot_lq'] = args.dataroot_lq
    if args.dataroot_gt:
        val_opt['dataroot_gt'] = args.dataroot_gt
    if args.dataset_name:
        val_opt['name'] = args.dataset_name
    val_opt['phase'] = 'val'
    val_opt.setdefault('scale', opt.get('scale', 1))
    dataset = create_dataset(val_opt)
    loader = create_dataloader(
        dataset, val_opt, num_gpu=opt.get('num_gpu', 1), dist=False,
        sampler=None, seed=None)
    return dataset, loader


def summarize(model, dataset, args, opt, wall_s):
    results = {k: float(v) for k, v in model.metric_results.items()}
    dists = getattr(model, 'metric_distributions', None) or {}
    stds = {}
    for name, values in dists.items():
        stds[name] = statistics.pstdev(values) if len(values) > 1 else 0.0

    bare = model.get_bare_model(model.net_g)
    nfe = getattr(bare, 'nfe', None)
    effective_nfe = args.inference_steps if args.inference_steps > 0 else nfe

    row = {
        'tag': args.tag or osp.basename(args.ckpt),
        'config': osp.basename(args.opt),
        'ckpt': args.ckpt,
        'param_key': args.param_key,
        'dataroot_lq': opt['datasets']['val']['dataroot_lq'],
        'n_images': len(dataset),
        'nfe': effective_nfe,
        'wall_s': round(wall_s, 1),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    for name in ('psnr', 'ssim', 'lpips'):
        if name in results:
            row[name] = round(results[name], 6)
            row[f'{name}_std'] = round(stds.get(name, 0.0), 6)
    # any extra configured metrics
    for name, value in results.items():
        if name not in row:
            row[name] = round(value, 6)
    return row


def append_csv(row, out_path):
    os.makedirs(osp.dirname(osp.abspath(out_path)), exist_ok=True)
    exists = osp.isfile(out_path)
    with open(out_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--opt', required=True, help='training yml of the model')
    p.add_argument('--ckpt', required=True, help='checkpoint .pth to evaluate')
    p.add_argument('--param-key', default='params_ema',
                   help="state-dict key to load; falls back to 'params' if absent "
                        "(default: params_ema, matching what validation scored)")
    p.add_argument('--dataroot-lq', default=None, help='override val LQ dir')
    p.add_argument('--dataroot-gt', default=None, help='override val GT dir')
    p.add_argument('--dataset-name', default=None, help='label for the val set')
    p.add_argument('--inference-steps', type=int, default=0,
                   help='override NFE at inference (ISB only; 0 = use trained nfe)')
    p.add_argument('--override', action='append', default=None,
                   metavar='KEY=VALUE',
                   help='dotted-path opt override, e.g. network_g.use_eca=false '
                        '(repeatable; value parsed as YAML)')
    p.add_argument('--tag', default=None, help='row label in the output CSV')
    p.add_argument('--save-img', action='store_true',
                   help='save outputs under results/<name>/visualization/ '
                        '(qualitative export)')
    p.add_argument('--out', default=None, help='CSV path to append the summary row')
    args = p.parse_args()

    opt = build_opt(args)
    dataset, loader = build_val_loader(opt, args)
    model = create_model(opt)

    rgb2bgr = opt['val'].get('rgb2bgr', True)
    use_image = opt['val'].get('use_image', False)

    t0 = time.time()
    model.validation(loader, current_iter=0, tb_logger=None,
                     save_img=bool(args.save_img),
                     rgb2bgr=rgb2bgr, use_image=use_image)
    wall_s = time.time() - t0

    row = summarize(model, dataset, args, opt, wall_s)
    print('EVAL_RESULT ' + json.dumps(row, ensure_ascii=False))
    if args.out:
        append_csv(row, args.out)


if __name__ == '__main__':
    main()
