# ECAFormer-ISB

Low-light image enhancement using Image Schrödinger Bridge (ISB) with ECAFormer backbone.

## Quick Start

```bash
# Clone repository
git clone https://github.com/Nolkee/ECAFormer_ISB.git
cd ECAFormer_ISB

# Install dependencies (see docs/QUICKSTART.md for details)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# Install basicsr and other dependencies

# Train stable baseline (recommended)
python -m basicsr.train --opt Options/ISB_ecaformer_r42a_per_ch_res.yml

# Run diagnostic experiments
bash diagnostic_scripts/quick_test_warmup.sh
```

**Full guide**: [docs/QUICKSTART.md](docs/QUICKSTART.md)

## Current Best: R42a (Stable)

**Config**: `Options/ISB_ecaformer_r42a_per_ch_res.yml`  
**Result**: PSNR 21.64 @ 10.5K iter (SSIM 0.788, LPIPS 0.164)  
**Status**: ✅ Stable training, no oscillation

**Why R42a over R38c/R43a**: R38c achieves higher PSNR (22.10) but has unstable training at 3500-6000 iterations. R42a uses per-channel `residual_scale` at denoiser output, avoiding AdaLN conflict.

## Project Structure

```
ECAFormer_ISB/
├── basicsr/                    # Core training framework
│   ├── models/
│   │   ├── archs/ECAFormer_ISB_arch.py  # Main architecture
│   │   └── image_isb_model.py           # Training loop
├── Options/                    # Experiment configs (R11-R43 series)
├── diagnostic_scripts/         # Training stability analysis tools
├── legacy_training_scripts/    # Historical training scripts (R11-R43)
├── tools/                      # Checkpoint diagnosis, inference
├── docs/                       # Documentation
│   ├── QUICKSTART.md          # Installation & training guide
│   └── ARCHITECTURE.md        # Design details & findings
├── CLAUDE.md                   # Project conventions for AI
└── README.md                   # This file
```

## Key Findings (2026-06-08)

**Training instability pattern (R38c, R43a)**: PSNR drops at 3500-6000 iterations

**Root cause**: Early-stage channel imbalance (identity_scale/channel_scale) → AdaLN learns compensatory modulation → Gradient conflict when learning rate transitions

**Solution**: Apply channel correction at denoiser **output** (residual_scale) rather than x1 construction or illumination map

## Architecture

- **Model**: `basicsr/models/archs/ECAFormer_ISB_arch.py` — ECAFormerISB + ShallowDeepConv estimator
- **Training**: `basicsr/models/image_isb_model.py` — ImageISBModel with bridge loss + pixel/perceptual/color/chroma losses
- **Data**: LOLv1 (485 train / 15 test), LOLv2 Real (~689 train / ~100 test)

**Details**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Requirements

- Python 3.8+
- PyTorch 1.11+ with CUDA 11.x/12.x
- 1x GPU (12GB+ VRAM recommended)
- basicsr, lpips, tensorboard, pyyaml

## Documentation

- **[docs/QUICKSTART.md](docs/QUICKSTART.md)** — Installation, training, inference, troubleshooting
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — Design details, stability findings, model variants
- **[diagnostic_scripts/README.md](diagnostic_scripts/README.md)** — Diagnostic framework for training analysis
- **[legacy_training_scripts/README.md](legacy_training_scripts/README.md)** — Experiment history (R11-R43)
- **[CLAUDE.md](CLAUDE.md)** — Project conventions for AI collaboration

## Experiments

**Active research**: R43a-warmup diagnostic (testing identity_scale warmup hypothesis)

**Stable baseline**: R42a with per-channel residual_scale

**Historical**: 33 experiments (R11-R43) archived in `legacy_training_scripts/`

Training logs and checkpoints: `experiments/<config-name>/`

## License

Based on [BasicSR](https://github.com/XPixelGroup/BasicSR).
