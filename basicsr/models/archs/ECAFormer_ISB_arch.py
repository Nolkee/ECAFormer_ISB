"""
ECAFormer + I2SB Architecture (AdaLayerNorm)
=============================================

Integrates the ECAFormer backbone with I2SB (Schrödinger Bridge) framework.
Uses Adaptive Layer Normalization (AdaLN) for time-step conditioning,
following the same proven strategy from the RetinexFormer ISB experiments.

Architecture overview:
    1. ShallowDeepConv → (visual_feat, semantic_feat)  [= illu_fea, illu_map]
    2. x1 = img * semantic_feat + img                  [illumination enhancement]
    3. CrossAttenUnet_ISB(x_t, visual_feat, t)          [AdaLN-conditioned denoiser]
    4. output = mapping(features) + x1                  [residual x0 prediction]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from basicsr.models.archs.isb_module import (
    ISBEngine,
    NoiseSchedule,
    SinusoidalTimeEmbedding,
)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    import warnings
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b]", stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# ---------------------------------------------------------------------------
# Base modules (from ECAFormer)
# ---------------------------------------------------------------------------

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class ShallowDeepConv(nn.Module):
    """Feature extractor: produces visual features and semantic (illumination) map."""
    def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        super().__init__()
        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(n_fea_middle, n_fea_middle, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)
        )
        self.depth_conv = nn.Sequential(
            nn.Conv2d(n_fea_middle, n_fea_middle, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(n_fea_middle, n_fea_middle, kernel_size=3, padding=1, bias=True, groups=n_fea_in)
        )

    def forward(self, img):
        input = torch.cat([img, img.mean(dim=1).unsqueeze(1)], dim=1)
        x_1 = self.conv1(input)
        visual_feats = self.depth_conv(x_1)
        semantic_feats = self.conv2(visual_feats)
        return visual_feats, semantic_feats


class DMSA(nn.Module):
    """Dual-stream Multi-head Self-Attention with cross-attention."""
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.fusion_x = nn.Sequential(
            nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=1, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1)
        )
        self.fusion_y = nn.Sequential(
            nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=1, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1)
        )
        self.to_q = nn.Sequential(
            nn.Linear(dim, dim_head * heads, bias=True),
            nn.LeakyReLU(),
            nn.Linear(dim_head * heads, dim_head * heads, bias=False)
        )
        self.to_k = nn.Sequential(
            nn.Linear(dim, dim_head * heads, bias=True),
            nn.LeakyReLU(),
            nn.Linear(dim_head * heads, dim_head * heads, bias=False)
        )
        self.to_v = nn.Sequential(
            nn.Linear(dim, dim_head * heads, bias=True),
            nn.LeakyReLU(),
            nn.Linear(dim_head * heads, dim_head * heads, bias=False)
        )
        self.rescale_x = nn.Parameter(torch.ones(heads, 1, 1))
        self.rescale_y = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj_x = nn.Linear(dim_head * heads, dim, bias=True)
        self.proj_y = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, y_in):
        b, h, w, c = x_in.shape
        fusion_k_x = self.fusion_x(
            torch.cat([x_in, y_in], dim=-1).permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1).reshape(b, h * w, c)
        fusion_k_y = self.fusion_y(
            torch.cat([x_in, y_in], dim=-1).permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1).reshape(b, h * w, c)

        x = x_in.reshape(b, h * w, c)
        y = y_in.reshape(b, h * w, c)
        q_inp_x = self.to_q(x)
        k_inp_x = self.to_k(fusion_k_x)
        v_inp_x = self.to_v(x)
        q_inp_y = self.to_q(y)
        k_inp_y = self.to_k(fusion_k_y)
        v_inp_y = self.to_v(y)

        q_x, k_x, v_x = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                            (q_inp_x, k_inp_x, v_inp_x,))
        q_y, k_y, v_y = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                            (q_inp_y, k_inp_y, v_inp_y,))

        q_x = q_x.transpose(-2, -1)
        k_x = k_x.transpose(-2, -1)
        v_x = v_x.transpose(-2, -1)
        q_y = q_y.transpose(-2, -1)
        k_y = k_y.transpose(-2, -1)
        v_y = v_y.transpose(-2, -1)

        q_x = F.normalize(q_x, dim=-1, p=2)
        k_x = F.normalize(k_x, dim=-1, p=2)
        q_y = F.normalize(q_y, dim=-1, p=2)
        k_y = F.normalize(k_y, dim=-1, p=2)

        attn_x = (k_y @ q_x.transpose(-2, -1))
        attn_x = attn_x * self.rescale_x
        attn_x = attn_x.softmax(dim=-1)
        x = attn_x @ v_x
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c_x = self.proj_x(x).view(b, h, w, c)
        out_p_x = self.pos_emb(
            v_inp_x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1)
        out_x = out_c_x + out_p_x

        attn_y = (k_x @ q_y.transpose(-2, -1))
        attn_y = attn_y * self.rescale_y
        attn_y = attn_y.softmax(dim=-1)
        y = attn_y @ v_y
        y = y.permute(0, 3, 1, 2)
        y = y.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c_y = self.proj_y(y).view(b, h, w, c)
        out_p_y = self.pos_emb(
            v_inp_y.reshape(b, h, w, c).permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1)
        out_y = out_c_y + out_p_y
        return out_x, out_y


class FeedForward(nn.Module):
    def __init__(self, dim, expand=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * expand, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * expand, dim * expand, 3, 1, 1,
                      bias=False, groups=dim * expand),
            GELU(),
            nn.Conv2d(dim * expand, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


# ---------------------------------------------------------------------------
# AdaLayerNorm — time conditioning (same as RetinexFormer ISB)
# ---------------------------------------------------------------------------

class AdaLayerNorm(nn.Module):
    """Adaptive Layer Norm: dynamically adjusts scale/shift based on t_emb."""
    def __init__(self, dim, time_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(time_dim, dim * 2)
        nn.init.constant_(self.proj.weight, 0)
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x, t_emb):
        scale, shift = self.proj(t_emb).chunk(2, dim=-1)
        scale = scale[:, None, None, :]
        shift = shift[:, None, None, :]
        return self.norm(x) * (1 + scale) + shift


# ---------------------------------------------------------------------------
# DMSABlock variants
# ---------------------------------------------------------------------------

class DMSABlock(nn.Module):
    """Original ECAFormer block (no time conditioning)."""
    def __init__(self, dim, dim_head=64, heads=8, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                DMSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, y):
        x = x.permute(0, 2, 3, 1)
        y = y.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x_hat, y_hat = attn(x, y)
            x = x_hat + x
            x = ff(x) + x
        out_x = x.permute(0, 3, 1, 2)
        out_y = y_hat.permute(0, 3, 1, 2)
        return out_x, out_y


class DMSABlock_AdaLN(nn.Module):
    """ECAFormer block with AdaLayerNorm for ISB time conditioning.
    
    AdaLN is applied to the x-stream (main feature) before FeedForward.
    The y-stream (auxiliary visual features) is NOT time-conditioned.
    """
    def __init__(self, dim, time_dim, dim_head=64, heads=8, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                DMSA(dim=dim, dim_head=dim_head, heads=heads),
                FeedForward(dim=dim),
                AdaLayerNorm(dim, time_dim),  # norm before FF on x-stream
            ]))

    def forward(self, x, y, t_emb):
        x = x.permute(0, 2, 3, 1)
        y = y.permute(0, 2, 3, 1)
        for (attn, ff, adaln) in self.blocks:
            x_hat, y_hat = attn(x, y)
            x = x_hat + x
            x_norm = adaln(x, t_emb)      # AdaLN modulation
            x = ff(x_norm) + x
        out_x = x.permute(0, 3, 1, 2)
        out_y = y_hat.permute(0, 3, 1, 2)
        return out_x, out_y


# ---------------------------------------------------------------------------
# CrossAttenUnet_ISB — ECAFormer U-Net with time conditioning
# ---------------------------------------------------------------------------

class CrossAttenUnet_ISB(nn.Module):
    """ECAFormer's CrossAttenUnet adapted for I2SB with time conditioning."""

    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2,
                 num_blocks=None):
        super().__init__()
        if num_blocks is None:
            num_blocks = [1, 2, 2]

        self.dim = dim
        self.level = level

        # Time embedding MLP
        time_dim = dim * 4
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(dim),
            nn.Linear(dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Input embedding
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                DMSABlock_AdaLN(
                    dim=dim_level, time_dim=time_dim,
                    num_blocks=num_blocks[i], dim_head=dim,
                    heads=dim_level // dim),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
            ]))
            dim_level *= 2

        # Bottleneck
        self.bottleneck = DMSABlock_AdaLN(
            dim=dim_level, time_dim=time_dim,
            dim_head=dim, heads=dim_level // dim,
            num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2,
                                   kernel_size=2, padding=0, output_padding=0),
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2,
                                   kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                DMSABlock_AdaLN(
                    dim=dim_level // 2, time_dim=time_dim,
                    num_blocks=num_blocks[level - 1 - i], dim_head=dim,
                    heads=(dim_level // 2) // dim),
            ]))
            dim_level //= 2

        # Output: predict residual, add x1 for final x0 prediction
        self.mapping = nn.Conv2d(self.dim * 2, out_dim, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x_t, x1, visual_fea, t_batch):
        """
        Predict clean x0 from noisy x_t.

        Args:
            x_t:         Noisy interpolated state [b, 3, h, w]
            x1:          Illumination-enhanced input [b, 3, h, w] (for residual)
            visual_fea:  Visual features from ShallowDeepConv [b, dim, h, w]
            t_batch:     Diffusion time [b]

        Returns:
            predicted_x0 = mapping(features) + x1
        """
        t_emb = self.time_embed(t_batch)   # [b, time_dim]
        x = self.embedding(x_t)
        y = visual_fea

        fea_xlist = []
        fea_ylist = []
        for (dmsa_block, fea_down_x, fea_down_y) in self.encoder_layers:
            x, y = dmsa_block(x, y, t_emb)
            fea_xlist.append(x)
            fea_ylist.append(y)
            x = fea_down_x(x)
            y = fea_down_y(y)

        x, y = self.bottleneck(x, y, t_emb)

        for i, (up_x, up_y, fuse_x, fuse_y, dmsa_block) in enumerate(self.decoder_layers):
            x = up_x(x)
            y = up_y(y)
            x = fuse_x(torch.cat([x, fea_xlist[self.level - 1 - i]], dim=1))
            y = fuse_y(torch.cat([fea_ylist[self.level - 1 - i], y], dim=1))
            x, y = dmsa_block(x, y, t_emb)

        out = self.mapping(torch.cat([x, y], dim=1)) + x1
        return out


# ---------------------------------------------------------------------------
# ECAFormerISB — Top-level module (drop-in replacement for RetinexFormerISB)
# ---------------------------------------------------------------------------

class ECAFormerISB(nn.Module):
    """
    ECAFormer with I2SB (Schrödinger Bridge).

    Drop-in replacement for RetinexFormerISB:
      - forward(x_low, x_high=None) returns same format
      - Training: returns (predicted_x0, gt, illu_map)
      - Inference: returns enhanced image
    """

    def __init__(self, in_channels=3, out_channels=3, n_feat=40,
                 level=2, num_blocks=None, nfe=8, sigma_max=0.5,
                 use_checkpoint=True, use_sb=True, self_cond_prob=0.0,
                 cond_type="adaln", stage=1):
        super().__init__()
        if num_blocks is None:
            num_blocks = [1, 2, 2]

        self.n_feat = n_feat
        self.nfe = max(int(nfe), 1)
        self.use_sb = use_sb
        self.self_cond_prob = float(self_cond_prob)
        self.cond_type = str(cond_type).lower()
        if not 0.0 <= self.self_cond_prob <= 1.0:
            raise ValueError(
                f"ECAFormerISB: self_cond_prob={self.self_cond_prob} is invalid. "
                "Expected a value in [0, 1]."
            )

        # Feature extractor (replaces Illumination_Estimator)
        self.estimator = ShallowDeepConv(n_feat)

        if use_sb:
            if self.cond_type == "adaln":
                self.denoiser = CrossAttenUnet_ISB(
                    in_dim=in_channels, out_dim=out_channels,
                    dim=n_feat, level=level, num_blocks=num_blocks,
                )
            elif self.cond_type == "none":
                # No time conditioning — plain CrossAttenUnet wrapped
                # to accept the 4-arg denoiser interface
                self._plain_unet = CrossAttenUnet(
                    in_dim=in_channels, out_dim=out_channels,
                    dim=n_feat, level=level, num_blocks=num_blocks,
                )
                # Wrap so forward(x_t, x1, visual_fea, t) works
                self.denoiser = self._make_nocond_wrapper()
            else:
                raise ValueError(
                    f"ECAFormerISB: cond_type='{self.cond_type}' is invalid. "
                    f"Supported values: 'adaln', 'none'."
                )
            noise_schedule = NoiseSchedule(sigma_max=sigma_max)
            self.isb_engine = ISBEngine(noise_schedule=noise_schedule, nfe=nfe)
        else:
            # Original ECAFormer (no diffusion)
            self.denoiser = CrossAttenUnet(
                in_dim=in_channels, out_dim=out_channels,
                dim=n_feat, level=level, num_blocks=num_blocks,
            )

    def _make_nocond_wrapper(self):
        """Wrap plain CrossAttenUnet to accept (x_t, x1, visual_fea, t) signature."""
        plain = self._plain_unet
        class _NoCond(nn.Module):
            def __init__(self, unet):
                super().__init__()
                self.unet = unet
            def forward(self, x_t, x1, visual_fea, t_batch):
                # Ignore t_batch — no time conditioning
                return self.unet(x_t, visual_fea)
        return _NoCond(plain)

    def forward(self, x_low, x_high=None):
        # ShallowDeepConv → visual features + illumination map
        visual_fea, illu_map = self.estimator(x_low)
        x1 = x_low * illu_map + x_low

        if not self.use_sb:
            output = self.denoiser(x1, visual_fea)
            if x_high is not None and self.training:
                return output, x_high, illu_map
            return output

        if x_high is not None and self.training:
            return self._train_forward(x1, x_high, visual_fea, illu_map)
        else:
            return self._inference_forward(x1, visual_fea)

    def _train_forward(self, x1, x_high, visual_fea, illu_map):
        b = x1.shape[0]
        device = x1.device
        dtype = x1.dtype
        x0 = x_high

        t = torch.rand(b, device=device, dtype=dtype).clamp(0.01, 0.99)
        x_t = self.isb_engine.q_sample(x0, x1, t)

        # Self-conditioning: blend a detached first-pass prediction into cond.
        x1_cond = x1
        if self.self_cond_prob > 0 and torch.rand(1, device=device).item() < self.self_cond_prob:
            with torch.no_grad():
                predicted_x0_sc = self.denoiser(x_t, x1, visual_fea, t).detach()
            x1_cond = 0.5 * x1 + 0.5 * predicted_x0_sc

        predicted_x0 = self.denoiser(x_t, x1_cond, visual_fea, t)
        return predicted_x0, x0, illu_map

    def _inference_forward(self, x1, visual_fea):
        """Inference: single-step (nfe<=1) or multi-step (nfe>1) x0-prediction."""
        if self.nfe <= 1:
            # Single-step: directly predict x0 from x1 at t=1.0
            b = x1.shape[0]
            t_batch = torch.ones(b, device=x1.device, dtype=x1.dtype)
            return self.denoiser(x1, x1, visual_fea, t_batch).clamp(0.0, 1.0)
        else:
            # Multi-step: iterative refinement via ISBEngine
            def _denoise_fn(x_t, cond, t_batch):
                return self.denoiser(x_t, cond, visual_fea, t_batch)
            return self.isb_engine.reverse_sample_fast(
                _denoise_fn, x1, x1, nfe=self.nfe, predict_x0=True
            ).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Original CrossAttenUnet (for use_sb=False baseline)
# ---------------------------------------------------------------------------

class CrossAttenUnet(nn.Module):
    """Original ECAFormer U-Net without time conditioning."""
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=None):
        super().__init__()
        if num_blocks is None:
            num_blocks = [2, 4, 4]
        self.dim = dim
        self.level = level
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                DMSABlock(dim=dim_level, num_blocks=num_blocks[i],
                          dim_head=dim, heads=dim_level // dim),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)
            ]))
            dim_level *= 2
        self.bottleneck = DMSABlock(
            dim=dim_level, dim_head=dim, heads=dim_level // dim,
            num_blocks=num_blocks[-1])
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2,
                                   kernel_size=2, padding=0, output_padding=0),
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2,
                                   kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                DMSABlock(dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i],
                          dim_head=dim, heads=(dim_level // 2) // dim),
            ]))
            dim_level //= 2
        self.mapping = nn.Conv2d(self.dim * 2, out_dim, 3, 1, 1, bias=False)

    def forward(self, x, y):
        x_res = x
        x = self.embedding(x)
        fea_xlist = []
        fea_ylist = []
        for (block, down_x, down_y) in self.encoder_layers:
            x, y = block(x, y)
            fea_ylist.append(y)
            fea_xlist.append(x)
            x = down_x(x)
            y = down_y(y)
        x, y = self.bottleneck(x, y)
        for i, (up_x, up_y, fuse_x, fuse_y, block) in enumerate(self.decoder_layers):
            x = up_x(x)
            y = up_y(y)
            x = fuse_x(torch.cat([x, fea_xlist[self.level - 1 - i]], dim=1))
            y = fuse_y(torch.cat([fea_ylist[self.level - 1 - i], y], dim=1))
            x, y = block(x, y)
        out = self.mapping(torch.cat([x, y], dim=1)) + x_res
        return out


# ---------------------------------------------------------------------------
# ECAFormerBaseline — Pure ECAFormer without ISB (for controlled experiments)
# ---------------------------------------------------------------------------

class ECAFormerBaseline(nn.Module):
    """
    Pure ECAFormer baseline (no ISB / no diffusion).

    Compatible with ImageCleanModel: forward(img) → enhanced_img.
    Uses the same ShallowDeepConv + CrossAttenUnet as ECAFormerISB,
    but without any time conditioning or Schrödinger Bridge.
    """

    def __init__(self, in_channels=3, out_channels=3, n_feat=40,
                 level=2, num_blocks=None, stage=1, **kwargs):
        super().__init__()
        if num_blocks is None:
            num_blocks = [1, 2, 2]

        self.estimator = ShallowDeepConv(n_feat)
        self.unet = CrossAttenUnet(
            in_dim=in_channels, out_dim=out_channels,
            dim=n_feat, level=level, num_blocks=num_blocks,
        )

    def forward(self, img):
        visual_feat, semantic_feat = self.estimator(img)
        x1 = img * semantic_feat + img
        output = self.unet(x1, visual_feat)
        return output
