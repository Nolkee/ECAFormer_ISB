# Quick Start Guide

Get ECAFormer-ISB running for low-light image enhancement.

## Prerequisites

- Python 3.8+
- CUDA 11.x or 12.x
- 1x GPU (12GB+ VRAM recommended)

## Installation

```bash
# Clone repository
git clone https://github.com/Nolkee/ECAFormer_ISB.git
cd ECAFormer_ISB

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt  # If requirements.txt exists
# Or manually install: basicsr, lpips, tensorboard, pyyaml
```

## Data Preparation

**LOLv1 dataset** (recommended for quick test):
```bash
mkdir -p data/LOLv1
# Place dataset in:
# data/LOLv1/Train/input/  - Low-light training images
# data/LOLv1/Train/target/ - Normal-light training images
# data/LOLv1/Test/input/   - Test low-light images
# data/LOLv1/Test/target/  - Test ground truth
```

Expected structure:
```
data/LOLv1/
├── Train/
│   ├── input/  (485 images)
│   └── target/ (485 images)
└── Test/
    ├── input/  (15 images)
    └── target/ (15 images)
```

## Training

### Recommended: current champion recipe (r52b)

```bash
python -m basicsr.train --opt Options/ISB_ecaformer_r52b_late_ema_anchor_9k.yml
```

**Expected result**: PSNR ~22.7 / SSIM ~0.804 / LPIPS ~0.166 at one checkpoint
(~11.5K). Note: a mid-training PSNR valley around 6-9K is EXPECTED (phase
transition) — do not stop the run; the anchor locks the recovered regime.

### Active research: R53 series (AAAI final round)

```bash
bash train_r53_series.sh   # r53a (dz0) -> r53b (auto-engage) -> r53c (seed 3407)
```

Training auto-resumes from `experiments/<name>/training_states/latest.state` if present.

## Monitoring

**TensorBoard**:
```bash
tensorboard --logdir experiments --port 6006
# Open http://localhost:6006
```

**Real-time progress**:
```bash
bash diagnostic_scripts/monitor_training.sh
```

**Key metrics**:
- PSNR: ~22+ dB target
- SSIM: ~0.79-0.80 target
- LPIPS: ~0.16 target (lower is better)

**Where images are** (disk-safe policy — validation is memory-only by default):
- `experiments/<name>/visualization/baseline/` — first validation, saved once
- `experiments/<name>/visualization/best_results/` — overwritten on each new best PSNR
- Checkpoints: `models/net_g_latest.pth` (running, overwritten), `best_psnr_*.pth` (weights-only, at experiment root)

## Inference

```bash
python ECAFormer_inference.py \
    --opt Options/ISB_ecaformer_r52b_late_ema_anchor_9k.yml \
    --checkpoint experiments/<exp_name>/best_psnr_22.69_11500.pth \
    --input_dir <low_light_images> \
    --output_dir results/
```

## Troubleshooting

### Out of Memory

Reduce batch size:
```yaml
# In config .yml
batch_size_per_gpu: 16  # From default 24
accumulate_steps: 2      # Maintain effective batch size
```

### Early images look green

Expected for configs without the R49/R50 fixes: early output is a copy of the
green-biased input, and validation uses EMA weights that lag the live net.
Use `residual_gray_world: true` + `ema_warmup: true` + the R50 decay window
(`gray_world_decay_start/end` — without it converged outputs desaturate).
See `docs/COLOR_SHIFT_ROOT_CAUSE.md`.

### Mid-training PSNR drop (SSIM/LPIPS unaffected)

Global brightness/color drift: the estimator moves x1 (bridge endpoint) faster
than the denoiser can track. Check `x1_mean_*` / `gw_*` curves in TensorBoard.
This valley is a phase transition into a higher-PSNR regime (R51 verdict) —
suppress it only on perceptual runs (`anchor_mode: x1_lq` w0.5, caps PSNR
~21.8). For PSNR runs let it happen and lock afterwards:
`anchor_start_iter: 12000` + `anchor_mode: x1_ema` (R52). Do NOT use weak
anchors (w<=0.25), `estimator_lr_mult < 1`, or the R49 anchor-to-GT.

### Disk fills up

Should not happen anymore: training keeps one `latest.state` + one
`net_g_latest.pth` per experiment (atomic overwrite), and images are only
written for baseline + best. If an old experiment hogs space, delete its
`training_states/*.state`, `models/net_g_*.pth` and `visualization/` —
best checkpoints at the experiment root are the ones worth keeping.

### Checkpoint analysis

```bash
python tools/diagnose_checkpoint.py \
    --exp_dir experiments/<exp_name> \
    --iters 1000 3000 5000 7000 10000 \
    --output diagnosis.png
```

## Next Steps

- **Architecture details**: `docs/ARCHITECTURE.md`
- **Experiment history**: `legacy_training_scripts/README.md`
- **Diagnostic framework**: `diagnostic_scripts/README.md`
- **Project conventions**: `CLAUDE.md`

## Common Commands

```bash
# Check training progress
tail -f train_<config_name>.log

# Resume training (automatic — just rerun the same command;
# it picks up training_states/latest.state)
python -m basicsr.train --opt <config>.yml
```

---

**Dataset**: LOLv1/v2 Real
**Champion config**: `Options/ISB_ecaformer_r52b_late_ema_anchor_9k.yml`
**Training time**: ~24-48h (24K iter, single GPU; early stop usually triggers earlier)
