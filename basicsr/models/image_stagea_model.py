from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F

from basicsr.models.image_restoration_model import ImageCleanModel, autocast


class ImageStageAModel(ImageCleanModel):
    """
    Stage-A training model for the reference-style generator pretraining.

    This stays in the paired low-light setting and adds lightweight auxiliary
    losses to encourage barycenter/residual disentanglement without introducing
    adversarial training or unpaired data yet.
    """

    def init_training_settings(self):
        super().init_training_settings()
        train_opt = self.opt['train']
        self.detail_loss_weight = float(train_opt.get('detail_loss_weight', 0.2))
        self.bro_loss_weight = float(train_opt.get('bro_loss_weight', 0.05))
        self.irc_loss_weight = float(train_opt.get('irc_loss_weight', 0.05))
        self.irc_temperature = float(train_opt.get('irc_temperature', 0.07))
        self.stagea_log_stats = bool(train_opt.get('stagea_log_stats', True))

        if self.detail_loss_weight < 0 or self.bro_loss_weight < 0 or self.irc_loss_weight < 0:
            raise ValueError('Stage-A loss weights must be >= 0.')
        if self.irc_temperature <= 0:
            raise ValueError('irc_temperature must be > 0.')

    @staticmethod
    def _sobel_kernels(device, dtype):
        sobel_x = torch.tensor(
            [[1.0, 0.0, -1.0],
             [2.0, 0.0, -2.0],
             [1.0, 0.0, -1.0]],
            device=device, dtype=dtype
        ).view(1, 1, 3, 3) / 8.0
        sobel_y = sobel_x.transpose(-1, -2)
        return sobel_x, sobel_y

    def _detail_preserving_loss(self, pred, target):
        pred_gray = pred.mean(dim=1, keepdim=True)
        target_gray = target.mean(dim=1, keepdim=True)
        sobel_x, sobel_y = self._sobel_kernels(pred.device, pred.dtype)
        pred_dx = F.conv2d(pred_gray, sobel_x, padding=1)
        pred_dy = F.conv2d(pred_gray, sobel_y, padding=1)
        target_dx = F.conv2d(target_gray, sobel_x, padding=1)
        target_dy = F.conv2d(target_gray, sobel_y, padding=1)
        return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)

    @staticmethod
    def _bro_loss(bary_feat, residual_feat):
        bary_vec = F.normalize(bary_feat.flatten(2).mean(dim=-1), dim=1)
        residual_vec = F.normalize(residual_feat.flatten(2).mean(dim=-1), dim=1)
        return (bary_vec * residual_vec).sum(dim=1).pow(2).mean()

    def _irc_loss(self, residual_feat, x1, gt):
        batch = residual_feat.shape[0]
        if batch < 2:
            return residual_feat.new_zeros(())

        residual_vec = F.normalize(residual_feat.flatten(2).mean(dim=-1), dim=1)
        target_strength = (gt - x1).abs().mean(dim=(1, 2, 3))

        pairwise_dist = torch.abs(target_strength[:, None] - target_strength[None, :])
        pairwise_dist.fill_diagonal_(float('inf'))
        positive_idx = pairwise_dist.argmin(dim=1)

        logits = residual_vec @ residual_vec.t()
        logits = logits / self.irc_temperature
        logits.fill_diagonal_(-1e9)
        return F.cross_entropy(logits, positive_idx)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        with autocast(device_type='cuda', enabled=self.use_amp):
            outputs = self.net_g(self.lq, return_aux=True)
            pred = outputs['pred']
            self.output = pred

            l_pix = self.cri_pix(pred, self.gt)
            l_dp = self._detail_preserving_loss(pred, self.gt)
            l_bro = self._bro_loss(outputs['bary_feat'], outputs['residual_feat'])
            l_irc = self._irc_loss(outputs['residual_feat'], outputs['x1'], self.gt)
            l_total = (
                l_pix
                + self.detail_loss_weight * l_dp
                + self.bro_loss_weight * l_bro
                + self.irc_loss_weight * l_irc
            )

            loss_dict = OrderedDict()
            loss_dict['l_pix'] = l_pix
            loss_dict['l_dp'] = l_dp
            loss_dict['l_bro'] = l_bro
            loss_dict['l_irc'] = l_irc
            loss_dict['l_total'] = l_total
            if self.stagea_log_stats:
                loss_dict['mask_res_mean'] = outputs['mask_res'].mean()
                loss_dict['bary_norm'] = outputs['bary_feat'].abs().mean()
                loss_dict['residual_norm'] = outputs['residual_feat'].abs().mean()

        self.amp_scaler.scale(l_total).backward()
        self.amp_scaler.unscale_(self.optimizer_g)

        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), self.grad_clip_value)

        self.amp_scaler.step(self.optimizer_g)
        self.amp_scaler.update()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
