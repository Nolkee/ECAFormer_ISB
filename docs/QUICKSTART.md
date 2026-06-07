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

### Recommended: Stable baseline (R42a)

```bash
python -m basicsr.train --opt Options/ISB_ecaformer_r42a_per_ch_res.yml
```

**Expected result**: PSNR ~21.6 @ 10.5K iter, stable training

### Diagnostic experiments (R43 series)

```bash
# Quick 8K test (2-3 hours)
bash diagnostic_scripts/quick_test_warmup.sh

# Full diagnostic suite (6-7 days single GPU)
bash diagnostic_scripts/run_all_diagnostic_experiments.sh

# Multi-GPU parallel (if available)
bash diagnostic_scripts/run_parallel_experiments.sh
```

See `diagnostic_scripts/README.md` for details.

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
- PSNR: ~21-22 dB target
- SSIM: ~0.79-0.80 target
- LPIPS: ~0.15-0.16 target (lower is better)

## Inference

```bash
python ECAFormer_inference.py \
    --opt Options/ISB_ecaformer_r42a_per_ch_res.yml \
    --checkpoint experiments/<exp_name>/models/net_g_10500.pth \
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

### Training unstable (PSNR drops at 3500-6000 iter)

Use R42a config (residual_scale at output) or R43a-warmup (identity_scale with warmup).

Avoid: R38c, R43a without warmup.

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
tail -f experiments/<exp_name>/train.log

# Find best checkpoint
grep "Best metric" experiments/<exp_name>/train.log

# Resume training
python -m basicsr.train --opt <config>.yml --resume

# Multi-GPU training
CUDA_VISIBLE_DEVICES=0,1 python -m basicsr.train --opt <config>.yml
```

---

**Dataset**: LOLv1/v2 Real  
**Best config**: `Options/ISB_ecaformer_r42a_per_ch_res.yml`  
**Training time**: ~48h (24K iter, single GPU)
