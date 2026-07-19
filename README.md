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

# Train current champion recipe
python -m basicsr.train --opt Options/ISB_ecaformer_r52b_late_ema_anchor_9k.yml
```

**Full guide**: [docs/QUICKSTART.md](docs/QUICKSTART.md)

## Current Champion: r52b (late EMA anchor)

**Config**: `Options/ISB_ecaformer_r52b_late_ema_anchor_9k.yml`
**Result**: PSNR 22.689 / SSIM 0.8042 / LPIPS 0.1664 — all at one checkpoint (11.5K)
**Key**: let the mid-training phase transition happen, then lock the achieved
illumination regime with a frozen-EMA anchor (`anchor_mode: x1_ema` @ 9K)

**Active research (R53, final pre-AAAI)**: r52b validated the phase-transition
design but sagged -0.76 dB after its peak (the 0.05 deadzone let slow drift
escape) and the transition timing is stochastic (fixed engage iters can land
after the peak, as r52a's did). R53: exact lock (deadzone 0) + auto-engage at
the train-PSNR recovery turn (12K hard cap, state persisted across resume) +
a second seed for mean±std. Generalization config for the paper:
`Options/ISB_ecaformer_r53_lolv2real.yml`.
See [docs/COLOR_SHIFT_ROOT_CAUSE.md](docs/COLOR_SHIFT_ROOT_CAUSE.md).

## Project Structure

```
ECAFormer_ISB/
├── basicsr/                    # Core training framework
│   ├── models/
│   │   ├── archs/ECAFormer_ISB_arch.py  # Main architecture
│   │   └── image_isb_model.py           # Training loop
├── Options/                    # Experiment configs (R11-R53 series)
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

## Key Findings

**Early green tint (verified 2026-07-10)**: the output residual shortcut
`out = mapping + 0.6*x1` makes early outputs a copy of the green-biased
low-light input; EMA cold-start (validation uses `net_g_ema`, ~61% random init
at iter 500) delays visible progress by 1-2K iters. Fixed via a DECAYING
gray-world residual (`gray_world_decay_start/end`) + `ema_warmup`.

**Mid-training PSNR valley = phase transition (verified 2026-07-16)**: the
network migrates to a higher-PSNR illumination regime; blocking it (from-0
anchors) caps PSNR ~21.8, letting it run then LOCKING the achieved regime
(`anchor_mode: x1_ema` + `anchor_start_iter`) produced the 22.689 champion.

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

**Active research**: R53 series (exact lock + auto-engage, on r52b base)

**Champion**: r52b (PSNR 22.689, SSIM 0.8042, LPIPS 0.1664 @ one checkpoint) — late frozen-EMA anchor

**Historical**: R11-R52 archived in `legacy_training_scripts/` and `Options/`

Training logs and checkpoints: `experiments/<config-name>/` (disk-safe: single
`latest.state` + `net_g_latest.pth`, images only for baseline & best)

## License

Based on [BasicSR](https://github.com/XPixelGroup/BasicSR).
