# ECAFormer-ISB

Low-light image enhancement using Image Schrödinger Bridge (ISB) with ECAFormer backbone.

## Quick Start

```bash
# Train R43a (recommended - identity_scale green fix)
python -m basicsr.train --opt Options/ISB_ecaformer_r43a_identity_scale.yml

# Or use tmux pipeline (R42a validation → R43 series)
tmux new -s train "bash train_r42a_then_r43.sh"
```

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for tmux workflow, monitoring, and troubleshooting.

## Current Best

**R38c baseline**: PSNR 22.10 @ 10K iter (SSIM 0.788, LPIPS 0.166)

**R43 series** (expected PSNR 22.3-22.5): Root-cause green tint fix via `identity_scale` in x1 construction.

## Architecture

- **Model**: `basicsr/models/archs/ECAFormer_ISB_arch.py` — ECAFormerISB + ShallowDeepConv estimator
- **Training**: `basicsr/models/image_isb_model.py` — ImageISBModel with bridge loss + pixel/perceptual/color/chroma losses
- **Data**: LOLv1 (485 train / 15 test), LOLv2 Real (~689 train / ~100 test)

## Key Innovation (R43)

```python
# Previous (R38-R42): x_low's green bias injected at 2x weight
x1 = x_low * illu_map + x_low

# R43: Channel-aware identity shortcut
x1 = x_low * illu_map + identity_scale * x_low
# identity_scale = [1.0, 0.92, 1.0] suppresses green at source
```

See [CLAUDE.md](CLAUDE.md) for full project conventions and [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for training workflow.

## Requirements

- Python 3.8+
- PyTorch 1.11+
- CUDA 11.3+
- basicsr (included in repo)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt  # if available
```

## Dataset Setup

```bash
# LOLv1 structure
data/LOLv1/
├── Train/
│   ├── input/   # Low-light images
│   └── target/  # Normal-light images
└── Test/
    ├── input/
    └── target/

# LOLv2 Real structure
data/LOLv2Real/
├── Train/
│   ├── Low/     # Low-light images
│   └── Normal/  # Normal-light images
└── Test/
    ├── Low/
    └── Normal/
```

## Experiments

All configs in `Options/`:
- **R38c**: Baseline (PSNR 22.10)
- **R42 series**: 4 orthogonal green fixes (per-channel residual, green_norm, green_loss, channel_permute)
- **R43 series**: identity_scale root-cause fix (recommended)

Training logs and checkpoints saved to `experiments/<config-name>/`.

## License

This project is based on [BasicSR](https://github.com/XPixelGroup/BasicSR).
