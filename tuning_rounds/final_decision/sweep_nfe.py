#!/usr/bin/env python3
import argparse
import copy
import csv
import os
from os import path as osp

import torch

from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.utils import get_env_info, get_root_logger, set_random_seed
from basicsr.utils.options import dict2str, parse


def build_eval_opt(opt_path, checkpoint_path, nfe, tag):
    opt = parse(opt_path, is_train=False)
    opt['dist'] = False
    opt['rank'] = 0
    opt['world_size'] = 1

    opt['name'] = f'{tag}_nfe{nfe}'
    opt['network_g']['nfe'] = int(nfe)
    opt['path']['pretrain_network_g'] = osp.expanduser(checkpoint_path)
    opt['path']['resume_state'] = None

    results_root = osp.join(opt['path']['root'], 'results', 'nfe_sweep', tag,
                            f'nfe{nfe}')
    opt['path']['results_root'] = results_root
    opt['path']['log'] = results_root
    opt['path']['visualization'] = osp.join(results_root, 'visualization')
    os.makedirs(opt['path']['visualization'], exist_ok=True)

    return opt


def init_logger(log_file):
    logger = get_root_logger(
        logger_name='basicsr',
        log_file=log_file)
    logger.info(get_env_info())
    return logger


def evaluate_once(opt, val_loader, current_iter):
    model = create_model(opt)
    model.validation(
        val_loader,
        current_iter=current_iter,
        tb_logger=None,
        save_img=False,
        rgb2bgr=opt['val'].get('rgb2bgr', True),
        use_image=opt['val'].get('use_image', True))
    metric_results = getattr(model, 'metric_results', {})
    return {
        'psnr': float(metric_results.get('psnr', float('nan'))),
        'ssim': float(metric_results.get('ssim', float('nan'))),
        'lpips': float(metric_results.get('lpips', float('nan')))
    }


def write_csv(rows, out_csv):
    os.makedirs(osp.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['nfe', 'psnr', 'ssim', 'lpips', 'checkpoint'])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description='Sweep validation metrics over NFE.')
    parser.add_argument(
        '--opt',
        default='Options/ISB_ecaformer_full_s19_r6_c_identity_unclamp_det_wd5e4.yml',
        help='Base YAML config.')
    parser.add_argument(
        '--checkpoint',
        default='experiments/ISB_ecaformer_full_s19_r6_c_identity_unclamp_det_wd5e4/best_psnr_20.11_8500.pth',
        help='Checkpoint to evaluate.')
    parser.add_argument(
        '--nfe',
        nargs='+',
        type=int,
        default=[4, 6, 8, 12],
        help='List of NFE values to evaluate.')
    parser.add_argument(
        '--tag',
        default='ISB_ecaformer_full_s19_r6_c_nfe_sweep',
        help='Result tag used under results/nfe_sweep/.')
    parser.add_argument(
        '--out-csv',
        default='tuning_rounds/final_decision/results_r6c_nfe_sweep.csv',
        help='CSV output path.')
    args = parser.parse_args()

    base_opt = parse(args.opt, is_train=False)
    base_opt['dist'] = False
    base_opt['rank'] = 0
    base_opt['world_size'] = 1
    set_random_seed(base_opt.get('manual_seed', 0))
    torch.backends.cudnn.benchmark = True

    results_base = osp.join(
        base_opt['path']['root'], 'results', 'nfe_sweep', args.tag)
    os.makedirs(results_base, exist_ok=True)
    logger = init_logger(osp.join(results_base, 'sweep.log'))
    logger.info('NFE sweep base config:\n%s', dict2str(base_opt))
    logger.info('Checkpoint: %s', osp.expanduser(args.checkpoint))
    logger.info('NFEs: %s', args.nfe)

    val_opt = copy.deepcopy(base_opt['datasets']['val'])
    val_set = create_dataset(val_opt)
    val_loader = create_dataloader(
        val_set,
        val_opt,
        num_gpu=base_opt['num_gpu'],
        dist=False,
        sampler=None,
        seed=base_opt.get('manual_seed', 0))

    rows = []
    for nfe in args.nfe:
        opt = build_eval_opt(args.opt, args.checkpoint, nfe, args.tag)
        logger.info('Evaluating NFE=%d', nfe)
        metrics = evaluate_once(opt, val_loader, current_iter=nfe)
        row = {
            'nfe': int(nfe),
            'psnr': metrics['psnr'],
            'ssim': metrics['ssim'],
            'lpips': metrics['lpips'],
            'checkpoint': osp.expanduser(args.checkpoint),
        }
        rows.append(row)
        logger.info(
            'NFE=%d -> PSNR=%.4f SSIM=%.4f LPIPS=%.4f',
            nfe, row['psnr'], row['ssim'], row['lpips'])

    write_csv(rows, args.out_csv)
    ranked = sorted(rows, key=lambda item: item['psnr'], reverse=True)

    print('=== NFE Sweep Ranking (by PSNR) ===')
    for idx, row in enumerate(ranked, start=1):
        print(
            f'{idx}. nfe={row["nfe"]}  '
            f'PSNR={row["psnr"]:.4f}  '
            f'SSIM={row["ssim"]:.4f}  '
            f'LPIPS={row["lpips"]:.4f}'
        )
    best = ranked[0]
    print('\nRecommended inference setup:')
    print(f'- checkpoint: {osp.expanduser(args.checkpoint)}')
    print(f'- nfe: {best["nfe"]}')
    print(f'- csv: {args.out_csv}')


if __name__ == '__main__':
    main()
