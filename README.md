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

**R53 outcome (2026-07-20)**: r52b stays champion. r53b validated auto-engage
(22.59 @ 14K; the anchor waits for the recovery turn instead of a fixed iter —
this de-lucks the stochastic transition timing and is the recommended default).
r53c (seed 3407) collapsed from init (~10 dB) — the recipe has a bad-basin
sensitivity that we disclose as a limitation.
See [docs/COLOR_SHIFT_ROOT_CAUSE.md](docs/COLOR_SHIFT_ROOT_CAUSE.md).

## Paper prep (AAAI-27) — evidence rules

All paper numbers come from `paper_pack/NOTES.md` (single source of truth) and
are produced by the unified evaluator, which reuses the training validation
code path exactly (no GT-mean anywhere, LPIPS-alex, crop_border 0, best
validation checkpoint, three metrics at one checkpoint):

```bash
# evaluate any checkpoint (reproduces training-val numbers bit-exactly)
python tools/eval_lol.py --opt Options/ISB_ecaformer_r54_repro_r52b.yml \
    --ckpt <best_psnr_*.pth> --param-key params_ema --out paper_pack/metrics/eval.csv
# NFE sweep without retraining:  --inference-steps {2,4,16}
# cross-dataset probe:           --dataroot-lq/--dataroot-gt overrides
```

Contributions under test: (1) illumination-lifted short bridge, 8-step
deterministic sampling; (2) endpoint-drift / phase-transition analysis + late
self-calibrating anchor (auto-engage validated); (3) controlled matched-budget
comparison against the same backbone (`run_paper_p1_queue.sh`: r54 champion
rerun, fair24k baseline pair, LOLv2-Real pair; `run_paper_p2_followup.sh`:
NFE sweep + x1-endpoint ablation).

Known evidence constraints (do not overclaim in docs or comments):
- Best checkpoints saved before 2026-07-20 hold bare (non-EMA) weights and do
  NOT reproduce their logged metrics (validation scores `net_g_ema`; fixed —
  `save_best` now stores `params` + `params_ema`).
- LOL-v2-Real Test overlaps LOL-v1 by 99/100 images (91 pixel-identical to
  v1-Train); LOLv1→LOLv2-Real "cross-dataset" transfer numbers are leaked and
  must not be used as generalization evidence (`tools/scan_overlap.py`).
- Cross-domain claims require the clean probe (LOL-v2-Synthetic, verified
  disjoint). Current measurement: the bridge does not transfer better than its
  regression backbone — report as measured; do not write "proven robust".

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
