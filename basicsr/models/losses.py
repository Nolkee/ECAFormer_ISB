import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


class L1Loss(nn.Module):
    """L1 (MAE) loss."""

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: none, mean, sum')
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target):
        return self.loss_weight * F.l1_loss(
            pred, target, reduction=self.reduction)


class MSELoss(nn.Module):
    """MSE (L2) loss."""

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target):
        return self.loss_weight * F.mse_loss(
            pred, target, reduction=self.reduction)


class FFTLoss(nn.Module):
    """Frequency domain loss using FFT.

    Computes L1 loss on both magnitude and phase components
    of the 2D FFT of predicted and target images.

    Args:
        loss_weight: Weight for this loss term.
        use_magnitude: Whether to compute loss on magnitude. Default: True.
        use_phase: Whether to compute loss on phase. Default: True.
    """

    def __init__(self, loss_weight=1.0, use_magnitude=True, use_phase=True):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.use_magnitude = use_magnitude
        self.use_phase = use_phase

    def forward(self, pred, target):
        # pred, target: [b, c, h, w]
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')

        loss = 0.0
        if self.use_magnitude:
            pred_mag = torch.abs(pred_fft)
            target_mag = torch.abs(target_fft)
            loss += F.l1_loss(pred_mag, target_mag)

        if self.use_phase:
            pred_phase = torch.angle(pred_fft)
            target_phase = torch.angle(target_fft)
            loss += F.l1_loss(pred_phase, target_phase)

        return self.loss_weight * loss


class VGGPerceptualLoss(nn.Module):
    """VGG-based perceptual loss using pretrained VGG19 features.

    Computes L1 distance between VGG feature maps of prediction and target.
    Uses relu1_2, relu2_2, relu3_4, relu4_4 layers by default.
    """

    def __init__(self, loss_weight=1.0, layer_weights=None, use_input_norm=True):
        super(VGGPerceptualLoss, self).__init__()
        self.loss_weight = loss_weight
        self.use_input_norm = use_input_norm

        if layer_weights is None:
            # Default: equal weight on 4 layers
            self.layer_weights = {
                'relu1_2': 1.0,
                'relu2_2': 1.0,
                'relu3_4': 1.0,
                'relu4_4': 1.0,
            }
        else:
            self.layer_weights = layer_weights

        # Load pretrained VGG19
        from torchvision.models import vgg19, VGG19_Weights
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features

        # Extract layers
        self.slices = nn.ModuleDict()
        layer_map = {
            'relu1_2': 4,   # after relu1_2
            'relu2_2': 9,   # after relu2_2
            'relu3_4': 18,  # after relu3_4
            'relu4_4': 27,  # after relu4_4
        }

        prev_idx = 0
        for name, idx in sorted(layer_map.items(), key=lambda x: x[1]):
            if name in self.layer_weights:
                self.slices[name] = nn.Sequential(*list(vgg.children())[prev_idx:idx + 1])
                prev_idx = idx + 1

        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False

        # VGG normalization (ImageNet stats)
        if self.use_input_norm:
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        if self.use_input_norm:
            pred = (pred - self.mean) / self.std
            target = (target - self.mean) / self.std

        loss = 0.0
        x_pred = pred
        x_target = target

        for name, slice_module in self.slices.items():
            x_pred = slice_module(x_pred)
            x_target = slice_module(x_target)
            loss += self.layer_weights[name] * F.l1_loss(x_pred, x_target)

        return self.loss_weight * loss
