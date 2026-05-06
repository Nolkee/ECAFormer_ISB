import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss   #把 l1_loss 作为 weighted_loss 的输入
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss   #把 mse_loss 作为 weighted_loss 的输入
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.loss_weight = loss_weight
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return self.loss_weight * loss


_vgg_model_cache = {}


class VGGFeatureExtractor(nn.Module):
    """Extract features from specified VGG-19 layers for perceptual loss.

    Default layers: relu1_1, relu2_1, relu3_1, relu4_1, relu5_1
    (matching the ECAFormer paper recipe).
    """

    def __init__(self, layer_weights=None, use_input_norm=True):
        super().__init__()
        if layer_weights is None:
            layer_weights = {
                '2': 1.0,   # relu1_1
                '7': 1.0,   # relu2_1
                '14': 1.0,  # relu3_1
                '21': 1.0,  # relu4_1
                '28': 1.0,  # relu5_1
            }
        self.layer_weights = {int(k): float(v) for k, v in layer_weights.items()}
        self.use_input_norm = use_input_norm

        # Load pretrained VGG-19, keep only the layers we need
        try:
            from torchvision.models import vgg19, VGG19_Weights
            vgg = vgg19(weights=VGG19_Weights.DEFAULT)
        except ImportError:
            from torchvision.models import vgg19
            vgg = vgg19(pretrained=True)

        max_layer = max(self.layer_weights.keys())
        self.features = nn.Sequential(*list(vgg.features.children())[:max_layer + 1])

        # Freeze all parameters
        for param in self.features.parameters():
            param.requires_grad_(False)

        # ImageNet normalization constants
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std

        features = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.layer_weights:
                features[i] = x
        return features


class VGGPerceptualLoss(nn.Module):
    """VGG-19 perceptual loss (L1 on feature maps).

    Matches the ECAFormer paper: relu1_1 through relu5_1 of VGG-19.
    """

    def __init__(self, loss_weight=1.0, layer_weights=None, use_input_norm=True):
        super().__init__()
        self.loss_weight = loss_weight
        self.extractor = VGGFeatureExtractor(
            layer_weights=layer_weights,
            use_input_norm=use_input_norm,
        )

    def forward(self, pred, target):
        if pred.dim() != 4 or target.dim() != 4:
            raise ValueError(
                'VGGPerceptualLoss expects NCHW tensors, '
                f'got pred={tuple(pred.shape)}, target={tuple(target.shape)}'
            )

        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)

        pred_features = self.extractor(pred)
        target_features = self.extractor(target)

        loss = 0.0
        for idx, weight in self.extractor.layer_weights.items():
            loss += weight * F.l1_loss(pred_features[idx], target_features[idx])

        return self.loss_weight * loss


_lpips_module = None
_lpips_models = {}


def _get_lpips_model(net='alex', device='cpu'):
    global _lpips_module

    if _lpips_module is None:
        try:
            import lpips as _imported_lpips
        except ImportError as exc:
            raise ImportError(
                'lpips package is required for LPIPSPerceptualLoss. '
                'Install it with: pip install lpips') from exc
        _lpips_module = _imported_lpips

    key = (net, device)
    if key not in _lpips_models:
        model = _lpips_module.LPIPS(net=net)
        model = model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        _lpips_models[key] = model
    return _lpips_models[key]


class LPIPSPerceptualLoss(nn.Module):
    """LPIPS perceptual loss for image restoration training."""

    def __init__(self, loss_weight=1.0, net='alex', use_gpu=True):
        super(LPIPSPerceptualLoss, self).__init__()
        self.loss_weight = loss_weight
        self.net = net
        self.use_gpu = use_gpu

    def forward(self, pred, target):
        if pred.dim() != 4 or target.dim() != 4:
            raise ValueError(
                'LPIPSPerceptualLoss expects NCHW tensors, '
                f'got pred={tuple(pred.shape)}, target={tuple(target.shape)}'
            )

        device = pred.device
        if not (self.use_gpu and pred.is_cuda):
            device = torch.device('cpu')

        model = _get_lpips_model(net=self.net, device=str(device))
        pred = torch.clamp(pred, 0.0, 1.0).to(device)
        target = torch.clamp(target, 0.0, 1.0).to(device)
        pred = pred * 2.0 - 1.0
        target = target * 2.0 - 1.0
        loss = model(pred, target).mean()
        return self.loss_weight * loss

# def gradient(input_tensor, direction):
#     smooth_kernel_x = torch.reshape(torch.tensor([[0, 0], [-1, 1]], dtype=torch.float32), [2, 2, 1, 1])
#     smooth_kernel_y = torch.transpose(smooth_kernel_x, 0, 1)
#     if direction == "x":
#         kernel = smooth_kernel_x
#     elif direction == "y":
#         kernel = smooth_kernel_y
#     gradient_orig = torch.abs(torch.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))
#     grad_min = torch.min(gradient_orig)
#     grad_max = torch.max(gradient_orig)
#     grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
#     return grad_norm

# class SmoothLoss(nn.Moudle):
#     """ illumination smoothness"""

#     def __init__(self, loss_weight=0.15, reduction='mean', eps=1e-2):
#         super(SmoothLoss,self).__init__()
#         self.loss_weight = loss_weight
#         self.eps = eps
#         self.reduction = reduction
    
#     def forward(self, illu, img):
#         # illu: b×c×h×w   illumination map
#         # img:  b×c×h×w   input image
#         illu_gradient_x = gradient(illu, "x")
#         img_gradient_x  = gradient(img, "x")
#         x_loss = torch.abs(torch.div(illu_gradient_x, torch.maximum(img_gradient_x, 0.01)))

#         illu_gradient_y = gradient(illu, "y")
#         img_gradient_y  = gradient(img, "y")
#         y_loss = torch.abs(torch.div(illu_gradient_y, torch.maximum(img_gradient_y, 0.01)))

#         loss = torch.mean(x_loss + y_loss) * self.loss_weight

#         return loss

# class MultualLoss(nn.Moudle):
#     """ Multual Consistency"""

#     def __init__(self, loss_weight=0.20, reduction='mean'):
#         super(MultualLoss,self).__init__()

#         self.loss_weight = loss_weight
#         self.reduction = reduction
    

#     def forward(self, illu):
#         # illu: b x c x h x w
#         gradient_x = gradient(illu,"x")
#         gradient_y = gradient(illu,"y")

#         x_loss = gradient_x * torch.exp(-10*gradient_x)
#         y_loss = gradient_y * torch.exp(-10*gradient_y)

#         loss = torch.mean(x_loss+y_loss) * self.loss_weight
#         return loss




