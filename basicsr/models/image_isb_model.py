"""
ImageISBModel: Training model for RetinexFormer + I2SB (v2)
============================================================

Implements 8-point design requirements:
- x0 prediction loss as configurable primary term (MSE/L1/Charbonnier)
- Pixel L1 with configurable weight (secondary)
- TV loss on illumination map for smoothness
- FP32 enforcement for P40
- Handles (predicted_x0, gt, illu_map) tuple from network
"""

import math
import torch
import torch.nn.functional as F
from collections import OrderedDict, deque

from basicsr.models.image_restoration_model import ImageCleanModel
from basicsr.models import losses as loss_module
from basicsr.models.losses import CharbonnierLoss
from basicsr.utils import get_root_logger


def tv_loss(x):
    """Total Variation loss for spatial smoothness of illumination map."""
    diff_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    diff_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return diff_h + diff_w


class ImageISBModel(ImageCleanModel):
    """
    Training model for RetinexFormer + I2SB.

    Loss design (requirement #5):
    - Primary: configurable x0 loss(predicted_x0, gt)
      (MSE / L1 / Charbonnier)
    - Secondary: L1(predicted_x0, gt) * pixel_weight — pixel-level auxiliary
    - Regularization: TV(illu_map) — illumination smoothness

    Config keys:
    - train.x0_loss_weight: x0 primary loss weight (default 1.0)
    - train.x0_loss_type: mse | l1 | charbonnier (default mse)
    - train.x0_charbonnier_eps: epsilon for charbonnier (default 1e-3)
    - train.pixel_loss_weight: L1 weight (default 0.1)
    - train.tv_loss_weight: TV weight (default 0.01)
    """

    def __init__(self, opt):
        # Allow AMP if specified in train config
        opt['use_amp'] = opt.get('train', {}).get('use_amp', False)
        super(ImageISBModel, self).__init__(opt)
        # Decoupled inference steps (can differ from training nfe)
        self.inference_steps = int(opt['val'].get('inference_steps', 0))

        train_opt = opt.get('train', {})
        self.bridge_weight = float(train_opt.get('bridge_weight', 1.0))
        self.x0_loss_weight = train_opt.get('x0_loss_weight', 1.0)
        self.pixel_loss_weight = train_opt.get('pixel_loss_weight', 0.1)
        self.tv_loss_weight = train_opt.get('tv_loss_weight', 0.01)
        self.color_loss_weight = float(train_opt.get('color_loss_weight', 0.0))
        self.chroma_loss_weight = float(train_opt.get('chroma_loss_weight', 0.0))
        self.green_loss_weight = float(train_opt.get('green_loss_weight', 0.0))
        # R49 (S1): anchor the bridge endpoint x1's per-channel means so the
        # estimator cannot drift the whole bridge distribution mid-training
        # (the PSNR-only crash where SSIM/LPIPS stay fine = global drift).
        # anchor_mode:
        #   'gt'    — legacy R49 target mean_c(gt). INERT on LOLv1: x1 =
        #             lq*(1+sigmoid) <= 2*lq and 2*lq_gray (~0.09) << gt_gray
        #             (~0.49) on 100% of images, so the loss is a constant
        #             floor with a saturating one-sided gradient, not a
        #             restoring force. Kept only for config compatibility.
        #   'x1_lq' — R50 target ratio*mean_c(lq), the midpoint of x1's
        #             reachable range (lq..2*lq for ratio 1.5): a two-sided
        #             restoring force that pins the endpoint scale.
        self.anchor_loss_weight = float(train_opt.get('anchor_loss_weight', 0.0))
        self.anchor_mode = str(train_opt.get('anchor_mode', 'gt')).lower()
        self.anchor_target_ratio = float(train_opt.get('anchor_target_ratio', 1.5))
        # R51: deadzone — zero penalty while |mean_c(x1) - target| is within
        # anchor_deadzone * target (relative band; x1's reachable range is
        # only +-33% around the 1.5*lq midpoint and scales with brightness).
        # Preserves per-image freedom inside the band, restores outside.
        # 0.0 = exact L1 (R50c behavior). Must stay well below 0.33 or the
        # band swallows the whole reachable range (no force at all).
        self.anchor_deadzone = float(train_opt.get('anchor_deadzone', 0.0))
        # R52: the mid-training valley is a PHASE TRANSITION into a
        # higher-PSNR illumination regime (every 22.2 run crashed; every
        # from-iter-0-anchored run capped at ~21.8). anchor_start_iter
        # delays the anchor so the transition can happen, then locks the
        # NEW regime. anchor_mode 'x1_ema' targets the running EMA of the
        # achieved mean(x1)/mean(lq) ratio (tracked from iter 0, detached)
        # instead of a fixed prior — r51a showed even w0.1 of fixed-target
        # drag costs ~0.23 dB post-transition. The EMA is not persisted
        # across resume: it re-warms in ~100 iters (momentum 0.99) and the
        # deadzone tolerates the transient.
        self.anchor_start_iter = int(train_opt.get('anchor_start_iter', 0))
        self.anchor_ema_momentum = float(train_opt.get('anchor_ema_momentum', 0.99))
        self._anchor_ratio_ema = None
        # Warm-update counter: the ratio EMA updates freely before
        # anchor_start_iter, then FREEZES (a tracking EMA would follow the
        # slow post-16K sag it exists to prevent). After a resume the EMA
        # restarts from None — allow 200 warm iters (tau~100 at m=0.99) to
        # re-converge before freezing again; anchored runs sit at
        # equilibrium so the warm target is already correct.
        self._anchor_ratio_warm = 0
        # R53: auto-engage — a fixed anchor_start_iter is fragile because the
        # transition timing is stochastic (observed 5.5K-9K across identical
        # configs; r52a's 12K engage landed AFTER the peak and locked a
        # declined state, r52b's 9K landed at the valley bottom and produced
        # the 22.69 record). 'auto' watches a train-PSNR EMA: valley = drop of
        # anchor_valley_drop dB below the running peak; engage fires when the
        # EMA turns and rises anchor_rise_margin dB off the valley bottom
        # (early recovery — the r52b-winning timing). anchor_start_iter then
        # acts as a HARD CAP (also the resume fallback). anchor_freeze_delay
        # keeps the ratio-EMA target trailing (tau~100) for that many iters
        # after engage before freezing, so the lock lands on the settled
        # regime, not mid-recovery.
        self.anchor_engage_mode = str(train_opt.get('anchor_engage_mode', 'fixed')).lower()
        self.anchor_valley_drop = float(train_opt.get('anchor_valley_drop', 1.0))
        self.anchor_rise_margin = float(train_opt.get('anchor_rise_margin', 0.5))
        self.anchor_min_engage_iter = int(train_opt.get('anchor_min_engage_iter', 4000))
        self.anchor_freeze_delay = int(train_opt.get('anchor_freeze_delay', 0))
        self._anchor_engaged_iter = None
        self._anchor_psnr_ema = None
        self._anchor_psnr_peak = float('-inf')
        self._anchor_valley_bottom = None
        if self.anchor_engage_mode not in ('fixed', 'auto'):
            raise ValueError(
                f"ImageISBModel: anchor_engage_mode='{self.anchor_engage_mode}' "
                "is invalid. Supported values: 'fixed', 'auto'."
            )
        if self.anchor_valley_drop <= 0 or self.anchor_rise_margin <= 0:
            raise ValueError(
                "ImageISBModel: anchor_valley_drop and anchor_rise_margin "
                "must both be > 0."
            )
        if self.anchor_min_engage_iter < 0 or self.anchor_freeze_delay < 0:
            raise ValueError(
                "ImageISBModel: anchor_min_engage_iter and anchor_freeze_delay "
                "must both be >= 0."
            )
        if self.anchor_mode not in ('gt', 'x1_lq', 'x1_ema'):
            raise ValueError(
                f"ImageISBModel: anchor_mode='{self.anchor_mode}' is invalid. "
                "Supported values: 'gt', 'x1_lq', 'x1_ema'."
            )
        if not 1.0 <= self.anchor_target_ratio <= 2.0:
            raise ValueError(
                f"ImageISBModel: anchor_target_ratio={self.anchor_target_ratio} "
                "is invalid. x1 = lq*(1+sigmoid) can only reach [1, 2]*lq."
            )
        if self.anchor_start_iter < 0:
            raise ValueError(
                f"ImageISBModel: anchor_start_iter={self.anchor_start_iter} "
                "is invalid. Expected a value >= 0."
            )
        if not 0.0 < self.anchor_ema_momentum < 1.0:
            raise ValueError(
                f"ImageISBModel: anchor_ema_momentum={self.anchor_ema_momentum} "
                "is invalid. Expected a value in (0, 1)."
            )
        if self.anchor_deadzone < 0:
            raise ValueError(
                f"ImageISBModel: anchor_deadzone={self.anchor_deadzone} is "
                "invalid. Expected a value >= 0."
            )
        if self.anchor_mode == 'x1_lq' and self.anchor_deadzone > 0:
            # Reachable range around ratio*lq is [1, 2]*lq, i.e. a relative
            # half-width of (2-ratio)/ratio above and (ratio-1)/ratio below.
            # A band at/above the smaller half-width means zero force on that
            # side everywhere — a silently inert anchor (the R49b failure).
            max_band = min(2.0 - self.anchor_target_ratio,
                           self.anchor_target_ratio - 1.0) / self.anchor_target_ratio
            if self.anchor_deadzone >= max_band:
                raise ValueError(
                    f"ImageISBModel: anchor_deadzone={self.anchor_deadzone} >= "
                    f"{max_band:.3f} covers x1's whole reachable range on one "
                    f"side (target_ratio={self.anchor_target_ratio}) — the "
                    "anchor would be silently inert. Use a smaller band."
                )
        self.x0_loss_type = str(train_opt.get('x0_loss_type', 'mse')).lower()
        self.x0_charbonnier_eps = float(train_opt.get('x0_charbonnier_eps', 1e-3))
        self.accumulate_steps = int(train_opt.get('accumulate_steps', 1))
        self.use_grad_clip = bool(train_opt.get('use_grad_clip', True))
        self.grad_clip_value = float(train_opt.get('grad_clip_value', 1.0))
        self.strict_output_range = bool(train_opt.get('strict_output_range', True))
        self.loss_on_clamped_output = bool(
            train_opt.get('loss_on_clamped_output', True)
        )
        self.nan_guard = bool(train_opt.get('nan_guard', True))
        self.output_range_log_interval = int(
            train_opt.get('output_range_log_interval', 200)
        )
        self.train_psnr_window = int(train_opt.get('train_psnr_window', 512))
        if self.accumulate_steps < 1:
            raise ValueError(
                f"ImageISBModel: accumulate_steps={self.accumulate_steps} is invalid. "
                "Expected an integer >= 1."
            )
        if self.grad_clip_value <= 0:
            raise ValueError(
                f"ImageISBModel: grad_clip_value={self.grad_clip_value} is invalid. "
                "Expected a value > 0."
            )
        if self.x0_loss_type not in ('mse', 'l1', 'charbonnier'):
            raise ValueError(
                f"ImageISBModel: x0_loss_type='{self.x0_loss_type}' is invalid. "
                "Supported values: 'mse', 'l1', 'charbonnier'."
            )
        if self.x0_charbonnier_eps <= 0:
            raise ValueError(
                f"ImageISBModel: x0_charbonnier_eps={self.x0_charbonnier_eps} is invalid. "
                "Expected a value > 0."
            )

        self._x0_charbonnier = None
        if self.x0_loss_type == 'charbonnier':
            self._x0_charbonnier = CharbonnierLoss(
                eps=self.x0_charbonnier_eps
            ).to(self.device)

        self.cri_perceptual = None
        if train_opt.get('perceptual_opt'):
            perceptual_opt = dict(train_opt['perceptual_opt'])
            perceptual_type = perceptual_opt.pop('type')
            cri_perceptual_cls = getattr(loss_module, perceptual_type)
            self.cri_perceptual = cri_perceptual_cls(**perceptual_opt).to(self.device)

        self.cri_fft = None
        self.fft_loss_weight = float(train_opt.get('fft_loss_weight', 0.0))
        if self.fft_loss_weight > 0 and train_opt.get('fft_opt'):
            fft_opt = dict(train_opt['fft_opt'])
            fft_type = fft_opt.pop('type')
            cri_fft_cls = getattr(loss_module, fft_type)
            self.cri_fft = cri_fft_cls(**fft_opt).to(self.device)

        # Running diagnostics for stability and overfitting analysis.
        self._last_range_warn_iter = -10**9
        self._train_psnr_values = deque(maxlen=max(self.train_psnr_window, 1))
        self._epoch_raw_out_min = float('inf')
        self._epoch_raw_out_max = float('-inf')
        self._epoch_out_min = float('inf')
        self._epoch_out_max = float('-inf')
        self._epoch_raw_out_sum = 0.0
        self._epoch_raw_out_sum_sq = 0.0
        self._epoch_out_sum = 0.0
        self._epoch_out_sum_sq = 0.0
        self._epoch_out_count = 0
        self._gt_range_warned = False
        self._nan_skip_count_epoch = 0
        self._nan_skip_count_total = 0
        self._nan_skip_by_reason_epoch = self._new_nan_reason_counter()
        self._nan_skip_by_reason_total = self._new_nan_reason_counter()

        logger = get_root_logger()
        logger.info(
            f"ImageISBModel v2: bridge_w={self.bridge_weight}, x0_w={self.x0_loss_weight}, "
            f"pixel_w={self.pixel_loss_weight}, tv_w={self.tv_loss_weight}, "
            f"x0_loss_type={self.x0_loss_type}, "
            f"x0_charbonnier_eps={self.x0_charbonnier_eps}, "
            f"accumulate_steps={self.accumulate_steps}, "
            f"grad_clip={self.use_grad_clip}, grad_clip_value={self.grad_clip_value}, "
            f"strict_output_range={self.strict_output_range}, "
            f"loss_on_clamped_output={self.loss_on_clamped_output}, "
            f"nan_guard={self.nan_guard}"
        )

    def _compute_x0_loss(self, pred, gt):
        if self.x0_loss_type == 'mse':
            return F.mse_loss(pred, gt)
        if self.x0_loss_type == 'l1':
            return F.l1_loss(pred, gt)
        # self.x0_loss_type == 'charbonnier'
        return self._x0_charbonnier(pred, gt)

    @staticmethod
    def _new_nan_reason_counter():
        return {
            'output_nonfinite': 0,
            'loss_nonfinite': 0,
            'grad_nonfinite': 0,
            'fallback_loss_nonfinite': 0,
            'unknown': 0
        }

    def _has_nonfinite_grad(self):
        for p in self.net_g.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                return True
        return False

    @staticmethod
    def _has_nonfinite_tensor(x):
        return not torch.isfinite(x).all()

    @staticmethod
    def _tensor_range(x):
        with torch.no_grad():
            return x.detach().amin().item(), x.detach().amax().item()

    def _update_epoch_output_range(self, raw_pred, pred):
        raw_min, raw_max = self._tensor_range(raw_pred)
        out_min, out_max = self._tensor_range(pred)
        self._epoch_raw_out_min = min(self._epoch_raw_out_min, raw_min)
        self._epoch_raw_out_max = max(self._epoch_raw_out_max, raw_max)
        self._epoch_out_min = min(self._epoch_out_min, out_min)
        self._epoch_out_max = max(self._epoch_out_max, out_max)
        with torch.no_grad():
            raw_det = raw_pred.detach()
            pred_det = pred.detach()
            self._epoch_raw_out_sum += float(raw_det.sum().item())
            self._epoch_raw_out_sum_sq += float((raw_det * raw_det).sum().item())
            self._epoch_out_sum += float(pred_det.sum().item())
            self._epoch_out_sum_sq += float((pred_det * pred_det).sum().item())
            self._epoch_out_count += int(pred_det.numel())
        return raw_min, raw_max, out_min, out_max

    def get_epoch_output_range_stats(self, reset=False):
        has_values = self._epoch_raw_out_min != float('inf') and self._epoch_out_count > 0
        if not has_values:
            return None
        count = float(self._epoch_out_count)
        raw_mean = self._epoch_raw_out_sum / count
        raw_var = max(self._epoch_raw_out_sum_sq / count - raw_mean * raw_mean, 0.0)
        out_mean = self._epoch_out_sum / count
        out_var = max(self._epoch_out_sum_sq / count - out_mean * out_mean, 0.0)
        stats = {
            'raw_out_min': self._epoch_raw_out_min,
            'raw_out_max': self._epoch_raw_out_max,
            'raw_out_mean': raw_mean,
            'raw_out_std': math.sqrt(raw_var),
            'out_min': self._epoch_out_min,
            'out_max': self._epoch_out_max,
            'out_mean': out_mean,
            'out_std': math.sqrt(out_var),
            'out_count': self._epoch_out_count
        }
        if reset:
            self._epoch_raw_out_min = float('inf')
            self._epoch_raw_out_max = float('-inf')
            self._epoch_out_min = float('inf')
            self._epoch_out_max = float('-inf')
            self._epoch_raw_out_sum = 0.0
            self._epoch_raw_out_sum_sq = 0.0
            self._epoch_out_sum = 0.0
            self._epoch_out_sum_sq = 0.0
            self._epoch_out_count = 0
        return stats

    def _append_train_psnr(self, pred, gt):
        # pred/gt are expected in [0, 1], compute batch-level PSNR for trend tracking.
        with torch.no_grad():
            mse = F.mse_loss(pred.detach(), gt.detach()).item()
            psnr = -10.0 * math.log10(max(mse, 1e-12))
        self._train_psnr_values.append(psnr)

    def _anchor_active(self, current_iter):
        return (self._anchor_engaged_iter is not None
                and current_iter >= self._anchor_engaged_iter)

    def _update_anchor_engage(self, current_iter):
        """R53: decide when the anchor engages (see __init__ comment)."""
        if self.anchor_loss_weight <= 0 or self._anchor_engaged_iter is not None:
            return
        if self.anchor_engage_mode == 'fixed':
            if current_iter >= self.anchor_start_iter:
                self._anchor_engaged_iter = current_iter
            return
        # auto mode — hard cap (also covers resumes that lost the state)
        if current_iter >= self.anchor_start_iter:
            self._anchor_engaged_iter = current_iter
            get_root_logger().info(
                f'[anchor] engaged at iter {current_iter} (hard cap '
                f'{self.anchor_start_iter}).')
            return
        if not self._train_psnr_values:
            return
        p = self._train_psnr_values[-1]
        self._anchor_psnr_ema = (
            p if self._anchor_psnr_ema is None
            else 0.99 * self._anchor_psnr_ema + 0.01 * p
        )
        ema = self._anchor_psnr_ema
        if current_iter < self.anchor_min_engage_iter:
            self._anchor_psnr_peak = max(self._anchor_psnr_peak, ema)
            return
        if self._anchor_valley_bottom is None:
            self._anchor_psnr_peak = max(self._anchor_psnr_peak, ema)
            if ema <= self._anchor_psnr_peak - self.anchor_valley_drop:
                self._anchor_valley_bottom = ema
                get_root_logger().info(
                    f'[anchor] valley detected at iter {current_iter} '
                    f'(train-PSNR EMA {ema:.2f}, peak {self._anchor_psnr_peak:.2f}).')
        else:
            if ema < self._anchor_valley_bottom:
                self._anchor_valley_bottom = ema
            elif ema >= self._anchor_valley_bottom + self.anchor_rise_margin:
                self._anchor_engaged_iter = current_iter
                get_root_logger().info(
                    f'[anchor] engaged at iter {current_iter} (recovery turn: '
                    f'EMA {ema:.2f}, bottom {self._anchor_valley_bottom:.2f}).')

    def _extra_training_state(self):
        # Persist R52/R53 anchor state so a resume neither re-runs the valley
        # detector from scratch nor re-warms the frozen ratio target.
        if self.anchor_loss_weight <= 0:
            return {}
        extra = {
            'anchor_engaged_iter': self._anchor_engaged_iter,
            'anchor_ratio_warm': self._anchor_ratio_warm,
            'anchor_psnr_ema': self._anchor_psnr_ema,
            'anchor_psnr_peak': self._anchor_psnr_peak,
            'anchor_valley_bottom': self._anchor_valley_bottom,
        }
        if self._anchor_ratio_ema is not None:
            extra['anchor_ratio_ema'] = self._anchor_ratio_ema.detach().cpu()
        return extra

    def _load_extra_training_state(self, extra):
        if not extra:
            return
        self._anchor_engaged_iter = extra.get('anchor_engaged_iter', None)
        self._anchor_ratio_warm = int(extra.get('anchor_ratio_warm', 0))
        self._anchor_psnr_ema = extra.get('anchor_psnr_ema', None)
        self._anchor_psnr_peak = float(
            extra.get('anchor_psnr_peak', float('-inf')))
        self._anchor_valley_bottom = extra.get('anchor_valley_bottom', None)
        ema = extra.get('anchor_ratio_ema', None)
        if ema is not None:
            self._anchor_ratio_ema = ema.to(self.device)
        get_root_logger().info(
            f'[anchor] state restored: engaged_iter={self._anchor_engaged_iter}, '
            f'ratio_warm={self._anchor_ratio_warm}.')

    def _mark_nan_skip(self, reason='unknown'):
        self._nan_skip_count_epoch += 1
        self._nan_skip_count_total += 1
        if reason not in self._nan_skip_by_reason_epoch:
            reason = 'unknown'
        self._nan_skip_by_reason_epoch[reason] += 1
        self._nan_skip_by_reason_total[reason] += 1

    def get_train_psnr_stats(self, reset=False):
        if not self._train_psnr_values:
            return None
        values = torch.tensor(list(self._train_psnr_values), dtype=torch.float32)
        stats = {
            'min': float(values.min().item()),
            'max': float(values.max().item()),
            'mean': float(values.mean().item()),
            'std': float(values.std(unbiased=False).item()),
            'count': int(values.numel())
        }
        if reset:
            self._train_psnr_values.clear()
        return stats

    def get_nan_skip_stats(self, reset=False):
        stats = {
            'epoch_nan_skip': int(self._nan_skip_count_epoch),
            'total_nan_skip': int(self._nan_skip_count_total),
            'epoch_nan_skip_by_reason': dict(self._nan_skip_by_reason_epoch),
            'total_nan_skip_by_reason': dict(self._nan_skip_by_reason_total)
        }
        if reset:
            self._nan_skip_count_epoch = 0
            self._nan_skip_by_reason_epoch = self._new_nan_reason_counter()
        return stats

    def step_learning_rate(self, current_iter):
        # For gradient accumulation, step scheduler only when optimizer steps.
        if current_iter % self.accumulate_steps == 0:
            super().step_learning_rate(current_iter)

    def optimize_parameters(self, current_iter):
        """
        Training step with x0-prediction loss (requirement #5).

        """
        logger = get_root_logger()

        # Start of a new accumulation window.
        if (current_iter - 1) % self.accumulate_steps == 0:
            self.optimizer_g.zero_grad(set_to_none=True)

        # Pass current_iter to the networks (identity_scale warmup, R50
        # gray-world residual decay). net_g_ema renders validation, so it
        # must see the same iter or train/val blends would diverge.
        self.get_bare_model(self.net_g)._current_iter = current_iter
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema._current_iter = current_iter

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            preds = self.net_g(self.lq, self.gt)

        if isinstance(preds, tuple) and len(preds) == 3:
            predicted_x0, gt, illu_map = preds
        else:
            # Fallback for use_sb=False or unexpected output
            self.output = preds if not isinstance(preds, (list, tuple)) else preds[-1]
            if isinstance(self.output, tuple):
                self.output = self.output[0]
            raw_pred = self.output
            pred_for_eval = (
                torch.clamp(raw_pred, 0.0, 1.0)
                if self.strict_output_range else raw_pred
            )
            self.output = pred_for_eval
            gt_safe = torch.clamp(self.gt, 0.0, 1.0)
            self._update_epoch_output_range(raw_pred, pred_for_eval)
            self._append_train_psnr(pred_for_eval, gt_safe)
            pred_for_loss = pred_for_eval if self.loss_on_clamped_output else raw_pred
            loss = F.l1_loss(pred_for_loss, gt_safe)
            if self.nan_guard and self._has_nonfinite_tensor(loss):
                logger.warning(
                    f'Non-finite fallback loss at iter {current_iter}, skipping optimizer step.'
                )
                self._mark_nan_skip('fallback_loss_nonfinite')
                self.optimizer_g.zero_grad(set_to_none=True)
                self.amp_scaler.update()
                self.log_dict = {'l_pix': 0.0}
                return

            self.amp_scaler.scale(loss).backward()
            self.amp_scaler.unscale_(self.optimizer_g)
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.net_g.parameters(),
                    self.grad_clip_value
                )
            if self._has_nonfinite_grad():
                logger.warning(f'Non-finite gradients detected at iter {current_iter}, skipping optimizer step.')
                self._mark_nan_skip('grad_nonfinite')
                self.optimizer_g.zero_grad(set_to_none=True)
                self.amp_scaler.update()
                self.log_dict = {'l_pix': loss.item()}
                return

            self.amp_scaler.step(self.optimizer_g)
            self.amp_scaler.update()

            self.log_dict = {'l_pix': loss.item()}
            return

        raw_predicted_x0 = predicted_x0
        gt = torch.clamp(gt, 0.0, 1.0)
        predicted_x0_eval = (
            torch.clamp(raw_predicted_x0, 0.0, 1.0)
            if self.strict_output_range else raw_predicted_x0
        )
        predicted_x0_for_loss = (
            predicted_x0_eval if self.loss_on_clamped_output else raw_predicted_x0
        )
        self.output = predicted_x0_eval
        if self.nan_guard and self._has_nonfinite_tensor(raw_predicted_x0):
            logger.warning(f'Non-finite model output at iter {current_iter}, skipping optimizer step.')
            self._mark_nan_skip('output_nonfinite')
            self.optimizer_g.zero_grad(set_to_none=True)
            self.amp_scaler.update()
            self.log_dict = {'l_total': 0.0}
            return
        raw_out_min, raw_out_max, out_min, out_max = self._update_epoch_output_range(
            raw_predicted_x0, predicted_x0_eval
        )
        self._append_train_psnr(predicted_x0_eval, gt)
        # R53: adaptive anchor engagement watches the train-PSNR trend.
        self._update_anchor_engage(current_iter)
        if (raw_out_min < 0.0 or raw_out_max > 1.0) and (
            current_iter - self._last_range_warn_iter >= self.output_range_log_interval
        ):
            logger.info(
                f'Raw output range out of [0,1] at iter {current_iter}: '
                f'min={raw_out_min:.4f}, max={raw_out_max:.4f}. '
                f'Clamped range: min={out_min:.4f}, max={out_max:.4f}.'
            )
            self._last_range_warn_iter = current_iter

        loss_dict = OrderedDict()

        # Primary: x0 prediction loss (configurable)
        l_x0 = self._compute_x0_loss(predicted_x0_for_loss, gt)
        loss_dict['l_x0'] = l_x0

        # Secondary: L1 pixel loss (configurable weight)
        l_pix = F.l1_loss(predicted_x0_for_loss, gt)
        loss_dict['l_pix'] = l_pix

        # TV loss on illumination map
        l_tv = tv_loss(illu_map)
        loss_dict['l_tv'] = l_tv

        l_percep = torch.tensor(0.0, device=predicted_x0_for_loss.device)
        if self.cri_perceptual is not None:
            l_percep = self.cri_perceptual(predicted_x0_for_loss, gt)
        loss_dict['l_percep'] = l_percep

        # Color loss: penalize channel-mean difference to preserve saturation
        l_color = torch.tensor(0.0, device=predicted_x0_for_loss.device)
        if self.color_loss_weight > 0:
            pred_mean = predicted_x0_for_loss.mean(dim=(2, 3))  # [b, c]
            gt_mean = gt.mean(dim=(2, 3))  # [b, c]
            l_color = F.l1_loss(pred_mean, gt_mean)
        loss_dict['l_color'] = l_color

        # Chroma loss: penalize per-pixel saturation reduction via channel std
        l_chroma = torch.tensor(0.0, device=predicted_x0_for_loss.device)
        if self.chroma_loss_weight > 0:
            pred_std = predicted_x0_for_loss.std(dim=1)  # [b, h, w]
            gt_std = gt.std(dim=1)  # [b, h, w]
            l_chroma = F.l1_loss(pred_std.mean(dim=(1, 2)), gt_std.mean(dim=(1, 2)))
        loss_dict['l_chroma'] = l_chroma

        # Green loss: penalize green channel excess over (R+B)/2
        l_green = torch.tensor(0.0, device=predicted_x0_for_loss.device)
        if self.green_loss_weight > 0:
            pred_rb_mean = (predicted_x0_for_loss[:, 0:1] + predicted_x0_for_loss[:, 2:3]) / 2
            green_excess = torch.clamp(predicted_x0_for_loss[:, 1:2] - pred_rb_mean, min=0)
            l_green = green_excess.mean()
        loss_dict['l_green'] = l_green

        # Anchor loss (R49/S1): pin per-channel means of the bridge endpoint
        # x1 so the estimator cannot drift the bridge distribution. x1 is
        # reconstructed as x_low * illu_map + x_low, which matches the
        # network's construction when identity_scale=[1,1,1] and
        # pre_denoiser_x1_clamp=false (the R49/R50 configs).
        l_anchor = torch.tensor(0.0, device=predicted_x0_for_loss.device)
        if self.anchor_loss_weight > 0:
            x1_recon = self.lq * illu_map + self.lq
            x1_mean = x1_recon.mean(dim=(2, 3))
            lq_mean = self.lq.mean(dim=(2, 3))
            if self.anchor_mode == 'x1_ema':
                # R52: track the achieved ratio from iter 0 so the target is
                # already calibrated to the current regime when the anchor
                # engages. R53: keep trailing for anchor_freeze_delay iters
                # after engage (lock the settled regime, not mid-recovery),
                # then FREEZE (see __init__ comment).
                with torch.no_grad():
                    ratio = (x1_mean.float() /
                             lq_mean.float().clamp_min(1e-6)).mean(dim=0)
                    engaged = self._anchor_active(current_iter)
                    still_trailing = (
                        self._anchor_engaged_iter is not None
                        and current_iter < (self._anchor_engaged_iter
                                            + self.anchor_freeze_delay)
                    )
                    if self._anchor_ratio_ema is None:
                        self._anchor_ratio_ema = ratio
                        self._anchor_ratio_warm = 1
                    elif (not engaged or still_trailing
                          or self._anchor_ratio_warm < 200):
                        m = self.anchor_ema_momentum
                        self._anchor_ratio_ema = (
                            m * self._anchor_ratio_ema + (1.0 - m) * ratio
                        )
                        self._anchor_ratio_warm += 1
                target = (self._anchor_ratio_ema.unsqueeze(0)
                          * lq_mean.float()).to(x1_mean.dtype)
            elif self.anchor_mode == 'x1_lq':
                # R50: reachable two-sided target = ratio * mean_c(lq).
                target = self.anchor_target_ratio * lq_mean
            else:
                target = gt.mean(dim=(2, 3))  # legacy 'gt' (inert on LOLv1)
            if self._anchor_active(current_iter):
                # R51: deadzone-L1 — no force while |x1_mean - target| is
                # within deadzone*target (RELATIVE band: x1's reachable range
                # is only +-33% around its midpoint and scales with image
                # brightness). deadzone 0.0 reduces to plain L1 (R50c).
                diff = (x1_mean - target.detach()).abs()
                band = self.anchor_deadzone * target.detach()
                l_anchor = torch.clamp(diff - band, min=0.0).mean()
        loss_dict['l_anchor'] = l_anchor

        # FFT loss: frequency domain constraint
        l_fft = torch.tensor(0.0, device=predicted_x0_for_loss.device)
        if self.cri_fft is not None:
            l_fft = self.cri_fft(predicted_x0_for_loss, gt)
        loss_dict['l_fft'] = l_fft

        # Combined loss: bridge_weight * (x0 + pixel) + perceptual + TV + color + chroma + green + anchor + FFT
        l_total = self.bridge_weight * (
            self.x0_loss_weight * l_x0
            + self.pixel_loss_weight * l_pix
        ) + l_percep + self.tv_loss_weight * l_tv + self.color_loss_weight * l_color + self.chroma_loss_weight * l_chroma + self.green_loss_weight * l_green + self.anchor_loss_weight * l_anchor + self.fft_loss_weight * l_fft
        if self.nan_guard and self._has_nonfinite_tensor(l_total):
            logger.warning(
                f'Non-finite total loss at iter {current_iter}, skipping optimizer step. '
                f'l_x0={l_x0.item()}, l_pix={l_pix.item()}, l_tv={l_tv.item()}, l_color={l_color.item()}, l_chroma={l_chroma.item()}'
            )
            self._mark_nan_skip('loss_nonfinite')
            self.optimizer_g.zero_grad(set_to_none=True)
            self.amp_scaler.update()
            self.log_dict = {'l_total': 0.0}
            return

        loss_dict['l_total'] = l_total

        # Gradient accumulation: divide loss so gradients average correctly
        scaled_total = l_total / self.accumulate_steps

        # Backward constructs the gradient sum over multiple un-stepped passes
        self.amp_scaler.scale(scaled_total).backward()

        if current_iter % self.accumulate_steps == 0:
            # Must unscale before clipping gradients in AMP
            self.amp_scaler.unscale_(self.optimizer_g)
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.net_g.parameters(),
                    self.grad_clip_value
                )

            if self._has_nonfinite_grad():
                logger.warning(f'Non-finite gradients detected at iter {current_iter}, skipping optimizer step.')
                self._mark_nan_skip('grad_nonfinite')
                self.optimizer_g.zero_grad(set_to_none=True)
                self.amp_scaler.update()
            else:
                self.amp_scaler.step(self.optimizer_g)
                self.amp_scaler.update()
                # Zero out gradients only after stepping
                self.optimizer_g.zero_grad(set_to_none=True)

        # R50: log every loss component (loss_dict was previously dead code —
        # r49b's inert anchor was invisible because only l_total was logged)
        # plus drift diagnostics for the mid-training crash window: x1
        # per-channel means, the gray-world gains they induce, and the
        # current WB blend.
        log_dict = OrderedDict()
        for k, v in loss_dict.items():
            log_dict[k] = v.item() if torch.is_tensor(v) else float(v)
        with torch.no_grad():
            x1_stat = (self.lq * illu_map + self.lq).float()
            # Per-image stats, matching _gray_world's actual computation —
            # batch-pooled means would hide per-crop clamp saturation.
            ch_mean = x1_stat.mean(dim=(2, 3))                          # [B,3] RGB
            gray = ch_mean.mean(dim=1, keepdim=True)                    # [B,1]
            gains = (gray / ch_mean.clamp_min(1e-6)).clamp(0.5, 2.0)   # [B,3]
            mean_c = ch_mean.mean(dim=0)
            gain_c = gains.mean(dim=0)
            for i, ch in enumerate(('r', 'g', 'b')):
                log_dict[f'x1_mean_{ch}'] = mean_c[i].item()
                log_dict[f'gw_gain_{ch}'] = gain_c[i].item()
            # Fraction of per-image gains pinned at the 0.5/2.0 clamp: high
            # values mean the WB statistic is truncated on extreme crops.
            clamp_frac = ((gains <= 0.5) | (gains >= 2.0)).float().mean()
            log_dict['gw_gain_clamp_frac'] = clamp_frac.item()
        bare_net = self.get_bare_model(self.net_g)
        denoiser = getattr(bare_net, 'denoiser', None)
        if denoiser is not None and getattr(denoiser, 'residual_gray_world', False):
            log_dict['gw_blend'] = float(getattr(denoiser, '_gray_world_blend', 1.0))
        # R52: where the regime actually sits — watch this jump during the
        # phase transition and freeze once the anchor engages.
        if self._anchor_ratio_ema is not None:
            for i, ch in enumerate(('r', 'g', 'b')):
                log_dict[f'anchor_ratio_ema_{ch}'] = self._anchor_ratio_ema[i].item()
        if self.anchor_loss_weight > 0:
            log_dict['anchor_engaged'] = 1.0 if self._anchor_active(current_iter) else 0.0
        self.log_dict = log_dict

        if self.ema_decay > 0:
            decay = self.ema_decay
            if self.ema_warmup:
                # R49 (G1): warmup so early validation reflects the live net;
                # converges to ema_decay after ~10/(1-ema_decay) iters.
                decay = min(self.ema_decay, (1 + current_iter) / (10 + current_iter))
            self.model_ema(decay=decay)

    def nonpad_test(self, img=None):
        """Inference: network returns enhanced image directly in eval mode.

        Supports decoupled inference_steps: if val.inference_steps is set,
        temporarily override the model's nfe for validation.
        """
        if img is None:
            img = self.lq

        # Temporarily override nfe if inference_steps is configured
        orig_nfe = None
        if self.inference_steps > 0:
            net = self.get_bare_model(self.net_g_ema if hasattr(self, 'net_g_ema') else self.net_g)
            orig_nfe = net.nfe
            net.nfe = self.inference_steps

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img)
            if isinstance(pred, (tuple, list)):
                pred = pred[0]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img)
            if isinstance(pred, (tuple, list)):
                pred = pred[0]
            self.output = pred
            self.net_g.train()

        # Restore original nfe
        if orig_nfe is not None:
            net = self.get_bare_model(self.net_g_ema if hasattr(self, 'net_g_ema') else self.net_g)
            net.nfe = orig_nfe

    def feed_train_data(self, data):
        logger = get_root_logger()
        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device)
        gt_min, gt_max = self._tensor_range(self.gt)
        if (gt_min < 0.0 or gt_max > 1.0) and not self._gt_range_warned:
            logger.warning(
                f'GT is out of [0,1] (min={gt_min:.4f}, max={gt_max:.4f}). '
                'Clamping GT for stability. Please verify dataloader normalization.'
            )
            self._gt_range_warned = True
        self.gt = torch.clamp(self.gt, 0.0, 1.0)
        self.lq = torch.clamp(self.lq, 0.0, 1.0)
        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)
