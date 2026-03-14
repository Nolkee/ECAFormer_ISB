"""
ImageISBModel: Training model for RetinexFormer + I2SB (v2)
============================================================

Implements 8-point design requirements:
- x0 prediction MSE as primary loss
- Pixel L1 at 0.1 weight (secondary)
- TV loss on illumination map for smoothness
- FP32 enforcement for P40
- Handles (predicted_x0, gt, illu_map) tuple from network
"""

import torch
import torch.nn.functional as F
from collections import OrderedDict

from basicsr.models.image_restoration_model import ImageCleanModel
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
    - Primary: MSE(predicted_x0, gt) — x0 reconstruction loss
    - Secondary: L1(predicted_x0, gt) * 0.1 — pixel-level auxiliary
    - Regularization: TV(illu_map) — illumination smoothness

    Config keys:
    - train.x0_loss_weight: MSE weight (default 1.0)
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
        self.accumulate_steps = int(train_opt.get('accumulate_steps', 1))
        if self.accumulate_steps < 1:
            raise ValueError(
                f"ImageISBModel: accumulate_steps={self.accumulate_steps} is invalid. "
                "Expected an integer >= 1."
            )

        logger = get_root_logger()
        logger.info(
            f"ImageISBModel v2: x0_w={self.x0_loss_weight}, "
            f"pixel_w={self.pixel_loss_weight}, tv_w={self.tv_loss_weight}, "
            f"accumulate_steps={self.accumulate_steps}"
        )

    def _has_nonfinite_grad(self):
        for p in self.net_g.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                return True
        return False

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
            loss = F.l1_loss(self.output, self.gt)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite fallback loss at iter {current_iter}: {loss.item()}')

            self.amp_scaler.scale(loss).backward()
            self.amp_scaler.unscale_(self.optimizer_g)
            if self.opt['train'].get('use_grad_clip', False):
                torch.nn.utils.clip_grad_norm_(
                    self.net_g.parameters(),
                    self.opt['train'].get('grad_clip_value', 1.0)
                )
            if self._has_nonfinite_grad():
                logger.warning(f'Non-finite gradients detected at iter {current_iter}, skipping optimizer step.')
                self.optimizer_g.zero_grad(set_to_none=True)
                self.amp_scaler.update()
                self.log_dict = {'l_pix': loss.item()}
                return

            self.amp_scaler.step(self.optimizer_g)
            self.amp_scaler.update()

            self.log_dict = {'l_pix': loss.item()}
            return

        self.output = predicted_x0
        if not torch.isfinite(self.output).all():
            raise FloatingPointError(f'Non-finite model output at iter {current_iter}.')
        with torch.no_grad():
            out_min = self.output.detach().amin().item()
            out_max = self.output.detach().amax().item()
        if out_min < -0.05 or out_max > 1.05:
            logger.warning(
                f'Output range out of expected [0,1] at iter {current_iter}: '
                f'min={out_min:.4f}, max={out_max:.4f}')

        loss_dict = OrderedDict()

        # Primary: x0 prediction MSE
        l_x0 = F.mse_loss(predicted_x0, gt)
        loss_dict['l_x0'] = l_x0

        # Secondary: L1 pixel loss (weight 0.1)
        l_pix = F.l1_loss(predicted_x0, gt)
        loss_dict['l_pix'] = l_pix

        # TV loss on illumination map
        l_tv = tv_loss(illu_map)
        loss_dict['l_tv'] = l_tv

        # Combined loss (requirement #5: pixel weight << x0 weight)
        l_total = (
            self.x0_loss_weight * l_x0
            + self.pixel_loss_weight * l_pix
            + self.tv_loss_weight * l_tv
        )
        if not torch.isfinite(l_total):
            raise FloatingPointError(
                f'Non-finite total loss at iter {current_iter}: '
                f'l_x0={l_x0.item()}, l_pix={l_pix.item()}, l_tv={l_tv.item()}')

        loss_dict['l_total'] = l_total

        # Gradient accumulation: divide loss so gradients average correctly
        scaled_total = l_total / self.accumulate_steps

        # Backward constructs the gradient sum over multiple un-stepped passes
        self.amp_scaler.scale(scaled_total).backward()

        if current_iter % self.accumulate_steps == 0:
            # Must unscale before clipping gradients in AMP
            self.amp_scaler.unscale_(self.optimizer_g)
            if self.opt['train'].get('use_grad_clip', False):
                torch.nn.utils.clip_grad_norm_(
                    self.net_g.parameters(),
                    self.opt['train'].get('grad_clip_value', 1.0)
                )

            if self._has_nonfinite_grad():
                logger.warning(f'Non-finite gradients detected at iter {current_iter}, skipping optimizer step.')
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
        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device)
        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)


