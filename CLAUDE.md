# ECAFormer-ISB

Low-light image enhancement using Image Schrödinger Bridge (ISB) with ECAFormer backbone. LOLv1/v2 datasets.

## Key Files

- **Model**: `basicsr/models/archs/ECAFormer_ISB_arch.py` — ECAFormerISB (denoiser) + ShallowDeepConv (estimator)
- **Training**: `basicsr/models/image_isb_model.py` — ImageISBModel with bridge loss + pixel/perceptual/color/chroma/anchor losses
- **Config**: `Options/ISB_ecaformer_r*.yml` — YAML configs, `model_type: ImageISBModel`
- **Data**: `data/LOLv1/` (Train/input, Train/target), `data/LOLv2Real/` (Train/Low, Train/Normal)
- **Diagnostic tools**: `diagnostic_scripts/` — Training stability analysis, checkpoint diagnosis
- **Legacy scripts**: `legacy_training_scripts/` — Historical experiments R11-R43

## Current Champion: R48b

**Config**: `Options/ISB_ecaformer_r48b_illum3ch_bridge_reweight.yml`
**Result**: PSNR 22.21 @ 11.5K (best SSIM 0.7959 @ 15.5K, LPIPS 0.1635)
**Key params**: `illumination_channels=3`, `channel_noise_scale=[1.0, 0.8, 1.0]`, `residual_scale=0.6`, `color_loss=0.2`

**Active research**: R49 series (`train_r49_series.sh`) — fixes early green tint
(`residual_gray_world` + `ema_warmup`) and the mid-training PSNR crash
(`anchor_loss_weight`). Root-cause analysis: `docs/COLOR_SHIFT_ROOT_CAUSE.md`.

## Config Conventions

- `total_iter: 24000`, `batch_size_per_gpu: 24`, `gt_size: 128`, `lr: 6e-5`
- `val_freq: 500` (fine-grained metric tracking), `save_checkpoint_freq: 2000`, `print_freq: 1000`
- `save_img: false` — selective image saving is handled in code, not by this flag
- `early_stop_patience_val: 8`, `use_amp: true`, `grad_clip_value: 0.02`
- All losses: Charbonnier pixel + VGG perceptual + color + chroma + TV (+ anchor in R49)

## Training Commands

```bash
python -m basicsr.train --opt Options/ISB_ecaformer_r48b_illum3ch_bridge_reweight.yml  # champion
bash train_r49_series.sh   # active research (r49a/b/c)
```

Auto-resume: rerun the same command; it picks up `training_states/latest.state`
(falls back to legacy numeric `{iter}.state`).

## Forbidden Configurations

- ❌ `output_activation: sigmoid` — Kills dynamic range (R39 confirmed)
- ❌ `channel_scale < 0.90` — PSNR loss > green fix benefit (R40 confirmed)
- ❌ `use_out_norm: 'post'` — GroupNorm(1,3) at output causes plateau ~13 PSNR (R41a/d)
- ❌ `identity_scale` in any form — unstable everywhere tried (R43/R44/R48c), abandoned
- ❌ `green_norm` only in training mode — Train/val data mismatch, 泛化失败 (R45 confirmed)
- ❌ Do NOT revisit the "illumination_channels=1 causes green tint" theory — disproven
  by R47/R48 (3ch still green early). Verified root cause is the residual passthrough
  + EMA cold-start; see `docs/COLOR_SHIFT_ROOT_CAUSE.md`

## Rules

- Do NOT add BGR/RGB conversions — data pipeline handles bgr2rgb correctly
- Do NOT amend git commits unless explicitly requested — create new commits
- Do NOT add emojis unless user requests
- Do NOT create markdown docs unless user requests
- Channel-color fixes that touch x1 or the illumination map destabilize training;
  fix color at the bridge noise (`channel_noise_scale`) or residual (`residual_gray_world`) instead
- Mechanisms applied to inputs/residuals MUST behave identically in train and inference (R45 lesson)
- Default to 24K iters for ablation, longer only for confirmed winners
- When implementing warmup: Pass `current_iter` from training loop to network forward
- **Backward compatibility**: `base_model.py` auto-fills missing `identity_scale` keys with [1,1,1] for old checkpoints
- **LOLv2Real paths**: Use `Train/Low` and `Train/Normal` (not `input`/`target`)
- **NaN guard**: `nan_guard: true` in config skips optimizer steps on non-finite gradients (expected behavior)
- **Disk-space safety** (added 2026-06-23 after /dev/sda2 filled up):
  - Training states: single overwrite-only `training_states/latest.state` (atomic tmp+replace). Legacy numeric `{iter}.state` files still resumable.
  - Running weights: single overwrite-only `models/net_g_latest.pth`. Best checkpoints (`best_psnr_*.pth` etc., weights-only) unchanged.
  - Validation images: memory-only metrics by default. Disk writes ONLY for one-time `visualization/baseline/` dump (first validation) and `visualization/best_results/` (overwritten on each new best PSNR). The `save_img` config flag is ignored during training.

## Deep Dive Documentation

- **Green tint & crash root cause**: `docs/COLOR_SHIFT_ROOT_CAUSE.md` — verified mechanism + R49 fixes
- **Architecture & design choices**: `docs/ARCHITECTURE.md`
- **Quick start & troubleshooting**: `docs/QUICKSTART.md`
- **Diagnostic framework**: `diagnostic_scripts/README.md`
- **Experiment history**: `legacy_training_scripts/README.md`
