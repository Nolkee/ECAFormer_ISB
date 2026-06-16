# ECAFormer-ISB

Low-light image enhancement using Image Schrödinger Bridge (ISB) with ECAFormer backbone. LOLv1/v2 datasets.

## Key Files

- **Model**: `basicsr/models/archs/ECAFormer_ISB_arch.py` — ECAFormerISB (denoiser) + ShallowDeepConv (estimator)
- **Training**: `basicsr/models/image_isb_model.py` — ImageISBModel with bridge loss + pixel/perceptual/color/chroma losses
- **Config**: `Options/ISB_ecaformer_r*.yml` — YAML configs, `model_type: ImageISBModel`
- **Data**: `data/LOLv1/` (Train/input, Train/target), `data/LOLv2Real/` (Train/Low, Train/Normal)
- **Diagnostic tools**: `diagnostic_scripts/` — Training stability analysis, checkpoint diagnosis
- **Legacy scripts**: `legacy_training_scripts/` — Historical experiments R11-R43

## Current Stable Best: R38c

**Config**: `Options/ISB_ecaformer_r38c_chscale_chroma.yml`  
**Result**: PSNR 22.10 @ 10K (SSIM 0.7882, LPIPS 0.1660), best SSIM/LPIPS @ 14K (0.7918/0.1572)
**Key params**: `residual_scale=0.6`, `channel_scale=[1.0, 0.95, 1.0]`, `chroma_loss=0.05`, `illumination_channels=1`
**Status**: ✅ Training complete, but shows instability at 5.5K (解耦相变谷底)

## Training Stability Issue (3500-6000 iter)

**Affected configs**: R38c, R43a (without warmup)  
**Symptom**: PSNR drops from ~20 to 17-18, then recovers with oscillation  
**Root cause** (diagnosed 2026-06-08): Early x1 channel imbalance → AdaLN overcompensation → gradient conflict at LR transition phase

**Where channel correction applied matters**:
- ❌ x1 construction (`identity_scale`) or illumination map (`channel_scale`) → Triggers AdaLN conflict
- ✅ Denoiser output (`residual_scale`) → Stable, no conflict

**Mitigation**: Use R42a, or R43a-warmup with `identity_scale_warmup_iters: 5000`

## 🔥 Color Shift Root Cause (2026-06-17)

**Problem**: Current ISB uses `illumination_channels=1` (single-channel illumination map) then manually expands with `channel_scale=[1.0, 0.95, 1.0]`. This:
1. **Forces RGB to share one illumination value**
2. Network CANNOT learn per-channel light differences
3. Manual green suppression is a patch, not a fix

**Original ECAFormer** used `illumination_channels=3` → each RGB channel had independent learned illumination → NO color issues

**Solution**: R47 series returns to 3ch illumination (architecturally correct fix)

## R42a: Most Stable Alternative

**Config**: `Options/ISB_ecaformer_r42a_per_ch_res.yml`  
**Result**: PSNR 21.64 @ 10.5K (SSIM 0.7888, LPIPS 0.1644), best SSIM/LPIPS @ 14K (0.7984/0.1568)
**Key**: `residual_scale=[0.6, 0.5, 0.6]` — per-channel at denoiser output  
**Status**: ✅ Stable training, no oscillation
**Tradeoff**: -0.46 dB PSNR vs R38c, but better perceptual quality (SSIM/LPIPS)

## Config Conventions

- `total_iter: 24000`, `batch_size_per_gpu: 24`, `gt_size: 128`, `lr: 6e-5`
- `val_freq: 500`, `save_checkpoint_freq: 500`, `print_freq: 200`
- `early_stop_patience_val: 8`, `use_amp: true`, `grad_clip_value: 0.02`
- All losses: Charbonnier pixel + VGG perceptual + color + chroma + TV

## Training Commands

**Stable baseline**:
```bash
python -m basicsr.train --opt Options/ISB_ecaformer_r42a_per_ch_res.yml
```

**Diagnostic experiments**:
```bash
bash diagnostic_scripts/quick_test_warmup.sh           # 8K quick test
bash diagnostic_scripts/run_all_diagnostic_experiments.sh  # Full suite
```

**Historical scripts**: See `legacy_training_scripts/`

## Forbidden Configurations

- ❌ `output_activation: sigmoid` — Kills dynamic range (R39 confirmed)
- ❌ `channel_scale < 0.90` — PSNR loss > green fix benefit (R40 confirmed)
- ❌ `use_out_norm: 'post'` — GroupNorm(1,3) at output causes plateau ~13 PSNR (R41a/d)
- ❌ `illumination_channels: 3` without stabilization — Training crash at ~8K (R41c)
- ❌ `identity_scale` or aggressive `channel_scale` without warmup — 3500-6000 instability
- ❌ `green_norm` only in training mode — Train/val data mismatch, 泛化失败 (R45 confirmed)

## Rules

- Do NOT add BGR/RGB conversions — data pipeline handles bgr2rgb correctly
- Do NOT amend git commits unless explicitly requested — create new commits
- Do NOT add emojis unless user requests
- Do NOT create markdown docs unless user requests
- Use `residual_scale` for channel correction (not `identity_scale` or `channel_scale`)
- Default to 24K iters for ablation, longer only for confirmed winners
- When implementing warmup: Pass `current_iter` from training loop to network forward
- **Backward compatibility**: `base_model.py` auto-fills missing `identity_scale` keys with [1,1,1] for old checkpoints
- **LOLv2Real paths**: Use `Train/Low` and `Train/Normal` (not `input`/`target`)
- **NaN guard**: `nan_guard: true` in config skips optimizer steps on non-finite gradients (expected behavior)
- **Smart image saving**: Validation only saves images at early stage (first 3), near best, and final stage (last 2) to reduce disk usage

## Deep Dive Documentation

- **Color shift root cause**: `docs/COLOR_SHIFT_ROOT_CAUSE.md` — Why illumination_channels=1 causes color bias
- **Architecture & design choices**: `docs/ARCHITECTURE.md`
- **Quick start & troubleshooting**: `docs/QUICKSTART.md`
- **Diagnostic framework**: `diagnostic_scripts/README.md`
- **Experiment history**: `legacy_training_scripts/README.md`
