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

### Recommended: current champion (R48b)

```bash
python -m basicsr.train --opt Options/ISB_ecaformer_r48b_illum3ch_bridge_reweight.yml
```

**Expected result**: PSNR ~22.2 @ 11.5K iter, SSIM ~0.796

### Active research: R50 series (color restore + crash fixes)

```bash
bash train_r50_series.sh   # r50a (gray-world decay) -> r50b (+estimator_lr 0.3x) -> r50c (+reachable anchor)
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
    --opt Options/ISB_ecaformer_r48b_illum3ch_bridge_reweight.yml \
    --checkpoint experiments/<exp_name>/best_psnr_22.21_11500.pth \
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
Fixes: `estimator_lr_mult: 0.3` (r50b) or `anchor_mode: x1_lq` with weight 0.5
(r50c). The R49 `anchor_loss_weight: 0.05` anchor-to-GT is inert — do not use.

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
**Champion config**: `Options/ISB_ecaformer_r48b_illum3ch_bridge_reweight.yml`
**Training time**: ~24-48h (24K iter, single GPU; early stop usually triggers earlier)
