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

        train_opt = opt.get('train', {})
        self.x0_loss_weight = train_opt.get('x0_loss_weight', 1.0)
        self.pixel_loss_weight = train_opt.get('pixel_loss_weight', 0.1)
        self.tv_loss_weight = train_opt.get('tv_loss_weight', 0.01)
        self.color_loss_weight = float(train_opt.get('color_loss_weight', 0.0))
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
            f"ImageISBModel v2: x0_w={self.x0_loss_weight}, "
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
            self.log_dict = {'l_x0': 0.0, 'l_pix': 0.0, 'l_tv': 0.0, 'l_total': 0.0}
            return
        raw_out_min, raw_out_max, out_min, out_max = self._update_epoch_output_range(
            raw_predicted_x0, predicted_x0_eval
        )
        self._append_train_psnr(predicted_x0_eval, gt)
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

        # Color loss: penalize channel-mean difference to preserve saturation
        l_color = torch.tensor(0.0, device=predicted_x0_for_loss.device)
        if self.color_loss_weight > 0:
            pred_mean = predicted_x0_for_loss.mean(dim=(2, 3))  # [b, c]
            gt_mean = gt.mean(dim=(2, 3))  # [b, c]
            l_color = F.l1_loss(pred_mean, gt_mean)
        loss_dict['l_color'] = l_color

        # Combined loss: weighted x0 + pixel + TV + color
        l_total = (
            self.x0_loss_weight * l_x0
            + self.pixel_loss_weight * l_pix
            + self.tv_loss_weight * l_tv
            + self.color_loss_weight * l_color
        )
        if self.nan_guard and self._has_nonfinite_tensor(l_total):
            logger.warning(
                f'Non-finite total loss at iter {current_iter}, skipping optimizer step. '
                f'l_x0={l_x0.item()}, l_pix={l_pix.item()}, l_tv={l_tv.item()}, l_color={l_color.item()}'
            )
            self._mark_nan_skip('loss_nonfinite')
            self.optimizer_g.zero_grad(set_to_none=True)
            self.amp_scaler.update()
            self.log_dict = {'l_x0': 0.0, 'l_pix': 0.0, 'l_tv': 0.0, 'l_color': 0.0, 'l_total': 0.0}
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

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def nonpad_test(self, img=None):
        """Inference: network returns enhanced image directly in eval mode."""
        if img is None:
            img = self.lq

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
