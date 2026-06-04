# ECAFormer-ISB

Low-light image enhancement using Image Schrodinger Bridge (ISB) with ECAFormer backbone. Trained on LOLv1/v2 datasets.

## Key Architecture

- **Model**: `basicsr/models/archs/ECAFormer_ISB_arch.py` — ECAFormerISB (denoiser) + ShallowDeepConv (estimator)
- **Training**: `basicsr/models/image_isb_model.py` — ImageISBModel with bridge loss + pixel/perceptual/color/chroma losses
- **Config**: `Options/ISB_ecaformer_r*.yml` — YAML configs, `model_type: ImageISBModel`
- **Data**: `data/LOLv1/` (485 train / 15 test), `data/LOLv2Real/` (~689 train / ~100 test)

## Current Best: R38c

PSNR 22.10 @ iter 10K (SSIM 0.7882, LPIPS 0.1660). Best SSIM/LPIPS @ iter 14K (0.7918/0.1572). Config: `Options/ISB_ecaformer_r38c_chscale_chroma.yml`

Key params: `channel_scale_init: [1.0, 0.95, 1.0]`, `residual_scale_init: 0.6`, `illumination_channels: 1`, `use_out_norm: true`, `output_activation: identity`, `train_output_clamp: true`

## R43 Series: Root-Cause Green Fix (2026-06-04)

**Core Innovation**: `identity_scale` in x1 construction — `x1 = x_low * illu_map + identity_scale * x_low`

Previous fixes (R38-R42) targeted downstream (channel_scale, residual_scale, losses), but x_low's green bias was injected at 2x weight (illumination term + identity term). R43 directly suppresses green in the identity shortcut.

- **R43a**: `identity_scale=[1.0, 0.92, 1.0]`, neutral residual. Expected PSNR 22.3-22.5
- **R43b**: `identity_scale=[1.0, 0.90, 1.0]` + per-channel residual. Double suppression
- **R43c**: R43a + green_loss. Triple defense

Launch: `bash train_r43_series.sh` or `bash auto_train_all.sh r43`

## Config Conventions

- `total_iter: 24000`, `batch_size_per_gpu: 24`, `gt_size: 128`, `lr: 6e-5`
- `val_freq: 500`, `save_checkpoint_freq: 500`, `print_freq: 200`
- `early_stop_patience_val: 8`, `use_amp: true`, `grad_clip_value: 0.02`
- All losses: Charbonnier pixel + VGG perceptual + color + chroma + TV

## Training

```bash
python -m basicsr.train --opt Options/ISB_ecaformer_r38c_chscale_chroma.yml
# Or use shell scripts: bash train_r41_series.sh
```

## Green Tint Root Cause (diagnosed 2026-06-03)

`x1 = x_low * (1 + illu_map)` preserves x_low's Bayer green bias. Residual shortcut injects it into output. `channel_scale` operates on illumination (wrong target). See memory `project_green_tint_direction.md` for full analysis.

## Experiment Naming

R{number}{letter}: R38a, R38b, R38c etc. Same number = same series, letters = variants within series.

## Rules

- Do NOT add BGR/RGB conversions — data pipeline already handles bgr2rgb correctly
- Do NOT use sigmoid output activation — kills dynamic range (R39 confirmed)
- Do NOT tune channel_scale below 0.90 — hurts PSNR more than it helps green (R40 confirmed)
- Do NOT use `use_out_norm: 'post'` — GroupNorm(1,3) at output breaks training, PSNR plateaus ~13 (R41a/d confirmed)
- Do NOT use `illumination_channels: 3` without stabilization — too much freedom early, causes training crash at ~8K (R41c confirmed)
- `zero_init_mapping_bias: true` is safe — PSNR 21.72, best SSIM/LPIPS of all configs (R41b)
- R43 architecture change: `identity_scale` in x1 construction targets green bias at injection point (推理时生效)
- Default to 24K iters for ablation, longer only for confirmed winners
