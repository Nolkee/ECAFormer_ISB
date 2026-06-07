# ECAFormer-ISB Architecture

Low-light image enhancement using Image Schrödinger Bridge (ISB) with ECAFormer backbone.

## Overview

**Core Method**: Retinex theory + diffusion bridge
- Input: Low-light image → Output: Enhanced image
- Two-stage: Illumination estimation + ISB denoising

## Key Components

### 1. Architecture Stack

```
x_low (input) 
  ↓
[ShallowDeepConv estimator]
  ↓ (visual_fea, illu_map)
x1 = x_low * illu_map + identity_scale * x_low
  ↓
[CrossAttenUnet_ISB denoiser with AdaLN]
  ↓ (diffusion bridge, 8 NFE)
x0 (output)
```

**Files:**
- `basicsr/models/archs/ECAFormer_ISB_arch.py` - Main architecture
- `basicsr/models/image_isb_model.py` - Training loop

### 2. Critical Design Choices

#### Channel-Aware Scaling

**Problem**: Bayer sensor green bias (RGGB 2:1:1) → x_low inherently green-tinted

**Solutions tested** (R38-R43 series):

| Approach | Where Applied | Result |
|----------|--------------|--------|
| `channel_scale=[1,0.95,1]` | Illumination map | PSNR 22.10, **unstable** 3500-6000 iter |
| `identity_scale=[1,0.92,1]` | x1 construction | PSNR 21.57, **unstable** 3500-6000 iter |
| `residual_scale=[0.6,0.5,0.6]` | Denoiser output | PSNR 21.64, **stable** ✅ |

**Root Cause (diagnosed 2026-06-08)**:
- Early-stage channel imbalance (identity_scale/channel_scale) → AdaLN learns compensatory modulation
- At 3500-6000 iter: Learning rate drops + AdaLN-locked compensation → gradient conflict → PSNR oscillation
- Late-stage correction (residual_scale) avoids this conflict

**Current best**: R42a with per-channel residual_scale at output

#### Identity Scale Warmup (R43a diagnostic)

**Implementation** (2026-06-08):
```python
# Gradual transition [1,1,1] → [1,0.92,1] over 5K iter
current_scale = (1-α) * [1,1,1] + α * [1,0.92,1]
x1 = x_low * illu_map + current_scale * x_low
```

**Config**: `identity_scale_warmup_iters: 5000`

**Expected outcome**: Prevent AdaLN conflict, achieve PSNR > 22.0 with stable training

### 3. Training Configuration

**Stable baseline (R42a)**:
```yaml
residual_scale_init: [0.6, 0.5, 0.6]  # Green 17% less
channel_scale_init: [1.0, 0.95, 1.0]
illumination_channels: 1
use_out_norm: true
output_activation: identity
train_output_clamp: true
```

**Loss components**:
- Bridge loss (x_t prediction)
- x0 loss (0.4 weight)
- Pixel L1 (0.6 weight)
- VGG perceptual (0.1 weight)
- Color loss (0.1 weight)
- Chroma loss (0.05 weight)
- TV loss on illu_map (0.002 weight)

**Optimizer**: AdamW, lr 6e-5, cosine annealing 24K iter

## Data Flow

1. **Input**: LOLv1 (485 train / 15 test) or LOLv2 Real (~689 train / ~100 test)
2. **Estimator** (ShallowDeepConv): x_low → (visual_fea, illu_map)
3. **x1 construction**: Retinex formula with optional identity_scale
4. **Bridge sampling**: t ~ U(0.01, 1.0), x_t = interpolate(x0, bridge_base, t)
5. **Denoiser** (CrossAttenUnet_ISB): Predicts x_0 from x_t, conditioned on t via AdaLN
6. **Output**: Clamped to [0,1]

## Model Variants

| Name | Key Difference | PSNR @ 10K | Stability |
|------|----------------|-----------|-----------|
| R38c | channel_scale green suppress | 22.10 | ❌ Oscillates 3500-6000 |
| R42a | residual_scale per-channel | 21.64 | ✅ Stable |
| R43a | identity_scale (no warmup) | 21.57 | ❌ Oscillates 3500-6000 |
| R43a-warmup | identity_scale + 5K warmup | TBD | Testing hypothesis |

## Known Issues & Mitigations

### Training Instability (3500-6000 iter)

**Symptom**: PSNR drops from ~20 to 17-18, then recovers with oscillation

**Root cause**: Early x1 channel perturbation → AdaLN overcompensation → gradient conflict

**Mitigation**:
- ✅ Use residual_scale at output (R42a)
- 🔬 Testing: identity_scale warmup (R43a-warmup)
- ❌ Avoid: Non-warmup identity_scale or aggressive channel_scale

### Forbidden Configurations

- ❌ `output_activation: sigmoid` - Kills dynamic range (R39 confirmed)
- ❌ `use_out_norm: 'post'` - GroupNorm(1,3) at output causes training plateau ~13 PSNR (R41a/d)
- ❌ `illumination_channels: 3` without stabilization - Training crash at ~8K iter (R41c)
- ❌ `channel_scale < 0.90` - PSNR loss exceeds green fix benefit (R40)

## Inference

**Script**: `ECAFormer_inference.py`

**Config**: Use corresponding Options/ISB_ecaformer_*.yml

**Output**: Enhanced images saved to `results/<config_name>/`

## Diagnostic Tools

**Location**: `diagnostic_scripts/`

**Purpose**: Analyze training instability, extract AdaLN evolution, compare stable vs unstable configs

**Key tool**: `tools/diagnose_checkpoint.py` - Checkpoint parameter extraction and visualization

See `diagnostic_scripts/README.md` for usage.

---

**Last updated**: 2026-06-08  
**Current best config**: R42a (`Options/ISB_ecaformer_r42a_per_ch_res.yml`)  
**Active research**: R43a warmup hypothesis (diagnostic framework)
