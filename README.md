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

# Train current champion
python -m basicsr.train --opt Options/ISB_ecaformer_r48b_illum3ch_bridge_reweight.yml
```

**Full guide**: [docs/QUICKSTART.md](docs/QUICKSTART.md)

## Current Champion: R48b

**Config**: `Options/ISB_ecaformer_r48b_illum3ch_bridge_reweight.yml`
**Result**: PSNR 22.21 @ 11.5K iter (SSIM 0.7959, LPIPS 0.164)
**Key**: 3-channel illumination + per-channel bridge noise `channel_noise_scale=[1.0, 0.8, 1.0]`

**Active research (R51)**: R50 confirmed the drift theory — the reachable x1
anchor (r50c, w0.5) produced the first crash-free run in project history (SSIM
0.8064 record) but capped PSNR at 21.75, while the gray-world decay alone
(r50a) matched the champion incl. a first-ever LPIPS win (0.1603) but still
crashed at 6-7K. R51 sweeps the anchor weight (0.1/0.25) and adds a relative
deadzone (full force only outside +-15% of target) to get both.
See [docs/COLOR_SHIFT_ROOT_CAUSE.md](docs/COLOR_SHIFT_ROOT_CAUSE.md).

## Project Structure

```
ECAFormer_ISB/
├── basicsr/                    # Core training framework
│   ├── models/
│   │   ├── archs/ECAFormer_ISB_arch.py  # Main architecture
│   │   └── image_isb_model.py           # Training loop
├── Options/                    # Experiment configs (R11-R51 series)
├── diagnostic_scripts/         # Training stability analysis tools
├── legacy_training_scripts/    # Historical training scripts (R11-R43)
├── tools/                      # Checkpoint diagnosis, inference
├── docs/                       # Documentation
│   ├── QUICKSTART.md          # Installation & training guide
│   ├── ARCHITECTURE.md        # Design details & findings
│   └── COLOR_SHIFT_ROOT_CAUSE.md  # Verified green-tint root cause
├── CLAUDE.md                   # Project conventions for AI
└── README.md                   # This file
```

## Key Findings (2026-07-10)

**Early green tint (verified root cause)**: the output residual shortcut
`out = mapping + 0.6*x1` makes early outputs a copy of the green-biased
low-light input; EMA cold-start (validation uses `net_g_ema`, ~61% random init
at iter 500) delays visible progress by 1-2K iters. Fixed in R49 via
`residual_gray_world` + `ema_warmup`.

**Mid-training PSNR crash**: PSNR drops 3+ dB while SSIM/LPIPS keep improving =
global brightness/color drift from unanchored illumination scale. Fixed in R49
via `anchor_loss_weight` (pins bridge-endpoint channel means to GT).

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

**Active research**: R51 series (anchor sweep + deadzone, on r50a base)

**Champion**: R48b (PSNR 22.21, SSIM 0.7959) — `channel_noise_scale` on bridge noise

**Historical**: R11-R50 archived in `legacy_training_scripts/` and `Options/`

Training logs and checkpoints: `experiments/<config-name>/` (disk-safe: single
`latest.state` + `net_g_latest.pth`, images only for baseline & best)

## License

Based on [BasicSR](https://github.com/XPixelGroup/BasicSR).
