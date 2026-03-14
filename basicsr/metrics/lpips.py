import numpy as np
import torch

from basicsr.metrics.metric_util import reorder_image, to_y_channel

_lpips_module = None
_lpips_models = {}


def _get_lpips_model(net='alex', use_gpu=True):
    global _lpips_module

    if _lpips_module is None:
        try:
            import lpips as _imported_lpips
        except ImportError as exc:
            raise ImportError(
                'lpips package is required for calculate_lpips. '
                'Install it with: pip install lpips') from exc
        _lpips_module = _imported_lpips

    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    key = (net, device)
    if key not in _lpips_models:
        model = _lpips_module.LPIPS(net=net)
        model = model.to(device)
        model.eval()
        _lpips_models[key] = model
    return _lpips_models[key], device


def _to_lpips_tensor(img, input_order='HWC', crop_border=0, test_y_channel=False, device='cpu'):
    if isinstance(img, torch.Tensor):
        if img.dim() == 4:
            img = img[0]
        if img.dim() == 3:
            img = img.detach().cpu().float().numpy().transpose(1, 2, 0)
        elif img.dim() == 2:
            img = img.detach().cpu().float().numpy()
        else:
            raise ValueError(f'Unsupported tensor shape for LPIPS: {tuple(img.shape)}')

    if not isinstance(img, np.ndarray):
        raise TypeError(f'Unsupported image type for LPIPS: {type(img)}')

    img = reorder_image(img, input_order=input_order)
    img = img.astype(np.float32)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img = np.repeat(img[..., None], 3, axis=2)

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=2)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    max_value = 1.0 if img.max() <= 1.0 else 255.0
    img = img / max_value
    img = np.clip(img, 0.0, 1.0)

    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor * 2.0 - 1.0
    return tensor.to(device)


def calculate_lpips(img1,
                    img2,
                    crop_border,
                    input_order='HWC',
                    test_y_channel=False,
                    net='alex',
                    use_gpu=True):
    """Calculate LPIPS perceptual distance.

    Args:
        img1 (ndarray/tensor): Images with range [0, 255] or [0, 1].
        img2 (ndarray/tensor): Images with range [0, 255] or [0, 1].
        crop_border (int): Cropped pixels in each edge of an image.
        input_order (str): Input order 'HWC' or 'CHW'.
        test_y_channel (bool): Whether to convert to Y channel before score.
        net (str): LPIPS backbone. Choices commonly include 'alex', 'vgg', 'squeeze'.
        use_gpu (bool): Whether to use GPU when available.

    Returns:
        float: LPIPS result (lower is better).
    """
    model, device = _get_lpips_model(net=net, use_gpu=use_gpu)

    x = _to_lpips_tensor(
        img1,
        input_order=input_order,
        crop_border=crop_border,
        test_y_channel=test_y_channel,
        device=device)
    y = _to_lpips_tensor(
        img2,
        input_order=input_order,
        crop_border=crop_border,
        test_y_channel=test_y_channel,
        device=device)

    with torch.no_grad():
        score = model(x, y)
    return float(score.item())
