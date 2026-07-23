import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import glob

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import master_only

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial

try :
    from torch.amp import autocast
    from torch.cuda.amp import GradScaler
    load_amp = True
except:
    load_amp = False


class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(
            torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1, 1)).item()

        r_index = torch.randperm(target.size(0)).to(self.device)

        target = lam * target + (1 - lam) * target[r_index, :]
        input_ = lam * input_ + (1 - lam) * input_[r_index, :]

        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments) - 1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_


class ImageCleanModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageCleanModel, self).__init__(opt)

        # define mixed precision
        # Accept use_amp from either the top level or the train section.
        # ImageISBModel hoists train.use_amp to the top level before calling
        # this __init__, but plain-baseline configs instantiate this class
        # directly — without this fallback they silently run fp32 and OOM at
        # the matched batch/patch (fair24k, 2026-07-23).
        self.use_amp = (opt.get('use_amp', False)
                        or opt.get('train', {}).get('use_amp', False)) and load_amp
        self.amp_scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp:
            print('Using Automatic Mixed Precision')
        else:
            print('Not using Automatic Mixed Precision')
                  
        # define network
        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta = self.opt['train']['mixing_augs'].get(
                'mixup_beta', 1.2)
            use_identity = self.opt['train']['mixing_augs'].get(
                'use_identity', False)
            self.mixing_augmentation = Mixing_Augment(
                mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        # self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.use_grad_clip = bool(train_opt.get('use_grad_clip', True))
        self.grad_clip_value = float(train_opt.get('grad_clip_value', 0.01))
        if self.grad_clip_value <= 0:
            raise ValueError(
                f"ImageCleanModel: grad_clip_value={self.grad_clip_value} is invalid. "
                "Expected a value > 0."
            )

        self.ema_decay = train_opt.get('ema_decay', 0)
        # R49 (G1): EMA warmup — early in training the EMA tracks the live net
        # (decay ~= t/(t+10)) and converges to ema_decay. Without it, with a
        # constant 0.999 the EMA net is still ~61% random init at iter 500, so
        # early validation images lag the live net by 1-2K iters.
        self.ema_warmup = bool(train_opt.get('ema_warmup', False))
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            logger.info(
                f'Gradient clipping: enabled={self.use_grad_clip}, '
                f'clip_value={self.grad_clip_value}'
            )
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(
                self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_opt = deepcopy(train_opt['pixel_opt'])
            pixel_type = pixel_opt.pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**pixel_opt).to(self.device)
        else:
            raise ValueError('pixel loss are None.')

        self.cri_perceptual = None
        if train_opt.get('perceptual_opt'):
            perceptual_opt = deepcopy(train_opt['perceptual_opt'])
            perceptual_type = perceptual_opt.pop('type')
            cri_perceptual_cls = getattr(loss_module, perceptual_type)
            self.cri_perceptual = cri_perceptual_cls(**perceptual_opt).to(self.device)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        logger = get_root_logger()

        # R50: optionally train the illumination estimator at a lower LR.
        # x1 (bridge endpoint + residual base) drifts with the estimator; a
        # slower estimator lets the denoiser/AdaLN track it instead of
        # transiently mismatching (the mid-training PSNR-only crash).
        estimator_lr_mult = float(train_opt.get('estimator_lr_mult', 1.0))
        if estimator_lr_mult <= 0:
            raise ValueError(
                f"estimator_lr_mult={estimator_lr_mult} is invalid. "
                "Expected a positive multiplier (typically in (0, 1])."
            )
        base_params = []
        estimator_params = []
        for k, v in self.net_g.named_parameters():
            if not v.requires_grad:
                logger.warning(f'Params {k} will not be optimized.')
                continue
            if estimator_lr_mult != 1.0 and 'estimator.' in k:
                estimator_params.append(v)
            else:
                base_params.append(v)
        if estimator_lr_mult != 1.0 and not estimator_params:
            logger.warning(
                f'estimator_lr_mult={estimator_lr_mult} was set but NO parameter '
                "name contains 'estimator.' — the multiplier is being ignored and "
                'all params train at the base LR. Check the network exposes an '
                '`estimator` submodule.')

        optim_type = train_opt['optim_g'].pop('type')
        optim_kwargs = train_opt['optim_g']
        if estimator_params:
            base_lr = float(optim_kwargs['lr'])
            optim_params = [
                {'params': base_params},
                {'params': estimator_params, 'lr': base_lr * estimator_lr_mult},
            ]
            logger.info(
                f'estimator_lr_mult={estimator_lr_mult}: '
                f'{len(estimator_params)} estimator params at lr '
                f'{base_lr * estimator_lr_mult:.2e}, '
                f'{len(base_params)} params at base lr {base_lr:.2e}.')
        else:
            optim_params = base_params
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(
                optim_params, **optim_kwargs)
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(
                optim_params, **optim_kwargs)
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        with autocast(device_type='cuda', enabled=self.use_amp):
            preds = self.net_g(self.lq)
            if not isinstance(preds, list):
                preds = [preds]

            self.output = preds[-1]

            loss_dict = OrderedDict()
            l_pix = 0.
            l_percep = 0.
            for pred in preds:
                l_pix += self.cri_pix(pred, self.gt)
                if self.cri_perceptual is not None:
                    l_percep += self.cri_perceptual(pred, self.gt)

            l_total = l_pix + l_percep
            loss_dict['l_pix'] = l_pix
            if self.cri_perceptual is not None:
                loss_dict['l_percep'] = l_percep
            loss_dict['l_total'] = l_total

        self.amp_scaler.scale(l_total).backward()
        self.amp_scaler.unscale_(self.optimizer_g) # 在梯度裁剪前先unscale梯度
        # l_pix.backward()

        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(
                self.net_g.parameters(), self.grad_clip_value
            )
        # self.optimizer_g.step()
        self.amp_scaler.step(self.optimizer_g)
        self.amp_scaler.update()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            decay = self.ema_decay
            if self.ema_warmup:
                decay = min(self.ema_decay, (1 + current_iter) / (10 + current_iter))
            self.model_ema(decay=decay)

    def pad_test(self, window_size):
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h -
                                  mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
            self.metric_distributions = {
                metric: []
                for metric in self.opt['val']['metrics'].keys()
            }
        # pbar = tqdm(total=len(dataloader), unit='image')

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0
        raw_pred_min = float('inf')
        raw_pred_max = float('-inf')
        metric_pred_min = float('inf')
        metric_pred_max = float('-inf')
        metric_gt_min = float('inf')
        metric_gt_max = float('-inf')

        # Cache rendered uint8 images for a possible "best" dump from train.py.
        # self.output/self.gt/self.lq are deleted mid-loop, so we must stash the
        # already-rendered sr_img/gt_img here rather than re-deriving them later.
        self._val_visual_cache = {}

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            test()

            visuals = self.get_current_visuals()
            pred_raw = visuals['result']
            pred_metric = torch.clamp(pred_raw, 0.0, 1.0)

            raw_pred_min = min(raw_pred_min, float(pred_raw.min().item()))
            raw_pred_max = max(raw_pred_max, float(pred_raw.max().item()))
            metric_pred_min = min(metric_pred_min, float(pred_metric.min().item()))
            metric_pred_max = max(metric_pred_max, float(pred_metric.max().item()))

            sr_img = tensor2img([pred_metric], rgb2bgr=rgb2bgr)
            gt_metric = None
            gt_img = None
            if 'gt' in visuals:
                gt_metric = torch.clamp(visuals['gt'], 0.0, 1.0)
                metric_gt_min = min(metric_gt_min, float(gt_metric.min().item()))
                metric_gt_max = max(metric_gt_max, float(gt_metric.max().item()))
                gt_img = tensor2img([gt_metric], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            # Cache the rendered images so train.py can dump them to
            # visualization/best_results/ if this validation turns out to be a
            # new best — without re-running inference.
            if self.opt['is_train']:
                self._val_visual_cache[img_name] = (sr_img, gt_img)

            # ----------------------------------------------------------------
            # Disk image saving policy (disk-space safe):
            #   * Training (is_train): memory-only by default. We write to disk
            #     ONLY for the one-time baseline dump (first validation), via the
            #     self._dump_baseline flag set by train.py. New-best dumps are
            #     handled separately by dump_best_visuals() (called from train.py).
            #   * Testing: always save (that's the whole point of a test run).
            # The legacy `save_img` config flag is intentionally ignored during
            # training so a stale `save_img: true` can't refill the disk.
            # ----------------------------------------------------------------
            if self.opt['is_train']:
                should_save = bool(getattr(self, '_dump_baseline', False))
            else:
                should_save = bool(save_img)

            if should_save:
                if self.opt['is_train']:
                    # One-time baseline snapshot of the initial qualitative state.
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             'baseline',
                                             f'{img_name}.png')
                    save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                'baseline',
                                                f'{img_name}_gt.png')
                else:

                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_gt.png')

                imwrite(sr_img, save_img_path)
                if gt_img is not None:
                    imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                if gt_metric is None:
                    raise ValueError(
                        'Validation metrics require ground truth, but `gt` is missing from visuals.'
                    )
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        if metric_type in ('calculate_psnr', 'calculate_ssim'):
                            opt_.setdefault('data_range', 255.0)
                        metric_value = getattr(metric_module, metric_type)(sr_img, gt_img, **opt_)
                        self.metric_results[name] += metric_value
                        self.metric_distributions[name].append(float(metric_value))
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        if metric_type in ('calculate_psnr', 'calculate_ssim'):
                            opt_.setdefault('data_range', 1.0)
                        metric_value = getattr(
                            metric_module, metric_type)(pred_metric, gt_metric, **opt_)
                        self.metric_results[name] += metric_value
                        self.metric_distributions[name].append(float(metric_value))

            cnt += 1

        # The one-time baseline dump (if requested) has now happened; clear the
        # flag so subsequent validations stay memory-only.
        self._dump_baseline = False

        if cnt > 0:
            self._val_audit_stats = {
                'raw_pred_min': raw_pred_min,
                'raw_pred_max': raw_pred_max,
                'metric_pred_min': metric_pred_min,
                'metric_pred_max': metric_pred_max,
                'metric_gt_min': metric_gt_min,
                'metric_gt_max': metric_gt_max,
                'use_image': bool(use_image),
            }
        else:
            self._val_audit_stats = None

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            dist = self.metric_distributions.get(metric, [])
            if dist:
                arr = np.asarray(dist, dtype=np.float32)
                log_str += (
                    f'\t # {metric}: {value:.4f}'
                    f' (min={arr.min():.4f}, max={arr.max():.4f}, std={arr.std():.4f})'
                )
            else:
                log_str += f'\t # {metric}: {value:.4f}'
        audit = getattr(self, '_val_audit_stats', None)
        if audit is not None:
            log_str += (
                '\t # val_output_range: '
                f'raw_min={audit["raw_pred_min"]:.4f}, raw_max={audit["raw_pred_max"]:.4f}, '
                f'metric_min={audit["metric_pred_min"]:.4f}, metric_max={audit["metric_pred_max"]:.4f}, '
                f'gt_min={audit["metric_gt_min"]:.4f}, gt_max={audit["metric_gt_max"]:.4f}, '
                f'use_image={audit["use_image"]}'
            )
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    @master_only
    def dump_best_visuals(self, current_iter):
        """Dump the most recent validation's rendered images as the new best.

        Called from train.py only when a new best PSNR is achieved. Writes the
        images cached during the last nondist_validation() into
        visualization/best_results/, after clearing any previous best images so
        they never accumulate on disk.
        """
        cache = getattr(self, '_val_visual_cache', None)
        if not cache:
            return
        best_dir = osp.join(self.opt['path']['visualization'], 'best_results')
        # Clear prior best visuals so only the current optimum is kept on disk.
        if osp.isdir(best_dir):
            for old_file in glob.glob(osp.join(best_dir, '*.png')):
                try:
                    os.remove(old_file)
                except OSError:
                    pass
        else:
            os.makedirs(best_dir, exist_ok=True)
        for img_name, (sr_img, gt_img) in cache.items():
            imwrite(sr_img, osp.join(best_dir, f'{img_name}.png'))
            if gt_img is not None:
                imwrite(gt_img, osp.join(best_dir, f'{img_name}_gt.png'))
        logger = get_root_logger()
        logger.info(
            f'Best visuals updated at iter {current_iter} '
            f'({len(cache)} images) -> {best_dir}')

    def save(self, epoch, current_iter, **kwargs):
        # Disk-space safe: always overwrite a single `net_g_latest.pth` for the
        # running checkpoint (-1 -> save_network names it 'latest') instead of
        # accumulating per-iteration `net_g_{iter}.pth` files. The true iter is
        # preserved inside the training state (state['iter']). Best-metric
        # checkpoints (save_best) are separate and keep their own filenames.
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              -1,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', -1)
        self.save_training_state(epoch, current_iter, **kwargs)

    def save_best(self, best_metric, param_key='params', metric_key='psnr'):
        if metric_key == 'psnr':
            val = best_metric['psnr']
            cur_iter = best_metric['iter']
            save_filename = f'best_psnr_{val:.2f}_{cur_iter}.pth'
        elif metric_key == 'ssim':
            val = best_metric.get('best_ssim', best_metric.get('ssim', 0))
            cur_iter = best_metric['iter']
            save_filename = f'best_ssim_{val:.4f}_{cur_iter}.pth'
        elif metric_key == 'lpips':
            val = best_metric.get('best_lpips', best_metric.get('lpips', 0))
            cur_iter = best_metric['iter']
            save_filename = f'best_lpips_{val:.4f}_{cur_iter}.pth'
        else:
            val = best_metric.get(metric_key, 0)
            cur_iter = best_metric['iter']
            save_filename = f'best_{metric_key}_{val:.4f}_{cur_iter}.pth'

        exp_root = self.opt['path']['experiments_root']
        save_path = os.path.join(
            self.opt['path']['experiments_root'], save_filename)

        if not os.path.exists(save_path):
            for r_file in glob.glob(f'{exp_root}/best_{metric_key}_*'):
                os.remove(r_file)
            if self.ema_decay > 0 and hasattr(self, 'net_g_ema'):
                # Validation scores net_g_ema; a best checkpoint without the
                # EMA weights cannot reproduce its logged metrics. Mirror
                # save(): store both nets under ['params', 'params_ema'].
                net = [self.net_g, self.net_g_ema]
                param_key = ['params', 'params_ema']
            else:
                net = self.net_g

            net = net if isinstance(net, list) else [net]
            param_key = param_key if isinstance(
                param_key, list) else [param_key]
            assert len(net) == len(
                param_key), 'The lengths of net and param_key should be the same.'

            save_dict = {}
            for net_, param_key_ in zip(net, param_key):
                net_ = self.get_bare_model(net_)
                state_dict = net_.state_dict()
                for key, param in state_dict.items():
                    if key.startswith('module.'):  # remove unnecessary 'module.'
                        key = key[7:]
                    state_dict[key] = param.cpu()
                save_dict[param_key_] = state_dict

            torch.save(save_dict, save_path)
