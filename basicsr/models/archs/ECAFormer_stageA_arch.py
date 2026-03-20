"""
ECAFormer Stage-A Architecture
==============================

Minimal Stage-A adaptation inspired by the reference three-stage strategy.

Goals of this module:
  1. Start from the pure ECAFormer generator instead of the ISB branch.
  2. Add an explicit barycenter/residual decomposition on the final latent.
  3. Preserve a stable baseline initialization so Stage-A starts close to
     the original ECAFormer behaviour.

This file deliberately does not introduce adversarial training, unpaired SB,
or barycenter transport. Those belong to later stages.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from basicsr.models.archs.ECAFormer_ISB_arch import (
    DMSABlock,
    ShallowDeepConv,
    _best_group_count,
)


class CrossAttenUnetStageA(nn.Module):
    """CrossAttenUnet with explicit barycenter/residual feature decomposition."""

    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=None):
        super().__init__()
        if num_blocks is None:
            num_blocks = [1, 2, 2]

        self.dim = dim
        self.level = level
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                DMSABlock(
                    dim=dim_level,
                    num_blocks=num_blocks[i],
                    dim_head=dim,
                    heads=dim_level // dim,
                ),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
            ]))
            dim_level *= 2

        self.bottleneck = DMSABlock(
            dim=dim_level,
            dim_head=dim,
            heads=dim_level // dim,
            num_blocks=num_blocks[-1],
        )

        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, 2, 2),
                nn.ConvTranspose2d(dim_level, dim_level // 2, 2, 2),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                DMSABlock(
                    dim=dim_level // 2,
                    num_blocks=num_blocks[level - 1 - i],
                    dim_head=dim,
                    heads=(dim_level // 2) // dim,
                ),
            ]))
            dim_level //= 2

        fusion_dim = self.dim * 2
        gn_groups = _best_group_count(fusion_dim, max_groups=8)
        self.fusion_norm = nn.GroupNorm(gn_groups, fusion_dim)
        self.bary_proj = nn.Conv2d(fusion_dim, fusion_dim, 1, 1, bias=True)
        self.residual_proj = nn.Conv2d(fusion_dim, fusion_dim, 1, 1, bias=True)
        self.mask_res_proj = nn.Conv2d(fusion_dim, fusion_dim, 1, 1, bias=True)
        self.mapping = nn.Conv2d(fusion_dim, out_dim, 3, 1, 1, bias=False)
        self._init_stagea_heads()

    def _init_stagea_heads(self):
        # Start close to the vanilla ECAFormer path:
        # barycenter branch copies shared features,
        # residual branch starts near zero,
        # residual mask starts suppressed.
        nn.init.dirac_(self.bary_proj.weight)
        nn.init.constant_(self.bary_proj.bias, 0.0)
        nn.init.constant_(self.residual_proj.weight, 0.0)
        nn.init.constant_(self.residual_proj.bias, 0.0)
        nn.init.constant_(self.mask_res_proj.weight, 0.0)
        nn.init.constant_(self.mask_res_proj.bias, -2.0)

    def forward(self, x, y, return_aux=False):
        x_res = x
        x = self.embedding(x)

        fea_xlist = []
        fea_ylist = []
        for (block, down_x, down_y) in self.encoder_layers:
            x, y = block(x, y)
            fea_xlist.append(x)
            fea_ylist.append(y)
            x = down_x(x)
            y = down_y(y)

        x, y = self.bottleneck(x, y)

        for i, (up_x, up_y, fuse_x, fuse_y, block) in enumerate(self.decoder_layers):
            x = up_x(x)
            y = up_y(y)
            x = fuse_x(torch.cat([x, fea_xlist[self.level - 1 - i]], dim=1))
            y = fuse_y(torch.cat([fea_ylist[self.level - 1 - i], y], dim=1))
            x, y = block(x, y)

        shared_feat = self.fusion_norm(torch.cat([x, y], dim=1))
        bary_feat = self.bary_proj(shared_feat)
        residual_feat = self.residual_proj(shared_feat)
        mask_res = torch.sigmoid(self.mask_res_proj(shared_feat))
        fused_feat = bary_feat + mask_res * residual_feat
        out = self.mapping(fused_feat) + x_res

        if not return_aux:
            return out

        return out, {
            'shared_feat': shared_feat,
            'bary_feat': bary_feat,
            'residual_feat': residual_feat,
            'mask_res': mask_res,
            'fused_feat': fused_feat,
        }


class ECAFormerStageA(nn.Module):
    """
    Stage-A generator based on pure ECAFormer.

    The model returns auxiliary decomposition features during training and a
    plain restored image during evaluation, so it stays compatible with the
    existing validation pipeline.
    """

    def __init__(self, in_channels=3, out_channels=3, n_feat=40,
                 level=2, num_blocks=None, stage=1,
                 sigmoid_illu_map=True, clamp_x1=True,
                 train_output_clamp=False, inference_output_clamp=False,
                 **kwargs):
        super().__init__()
        if num_blocks is None:
            num_blocks = [1, 2, 2]

        self.sigmoid_illu_map = bool(sigmoid_illu_map)
        self.clamp_x1 = bool(clamp_x1)
        self.train_output_clamp = bool(train_output_clamp)
        self.inference_output_clamp = bool(inference_output_clamp)

        self.estimator = ShallowDeepConv(n_feat)
        self.unet = CrossAttenUnetStageA(
            in_dim=in_channels,
            out_dim=out_channels,
            dim=n_feat,
            level=level,
            num_blocks=num_blocks,
        )

    def forward(self, img, return_aux=None):
        if return_aux is None:
            return_aux = self.training

        visual_feat, semantic_feat = self.estimator(img)
        illu_map = torch.sigmoid(semantic_feat) if self.sigmoid_illu_map else semantic_feat
        x1 = img * illu_map + img
        if self.clamp_x1:
            x1 = torch.clamp(x1, 0.0, 1.0)

        pred, aux = self.unet(x1, visual_feat, return_aux=True)
        if self.training and self.train_output_clamp:
            pred = pred.clamp(0.0, 1.0)
        elif (not self.training) and self.inference_output_clamp:
            pred = pred.clamp(0.0, 1.0)

        if not return_aux:
            return pred

        aux.update({
            'pred': pred,
            'illu_map': illu_map,
            'x1': x1,
            'visual_feat': visual_feat,
        })
        return aux
