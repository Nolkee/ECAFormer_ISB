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

**Active research**: R51 series (`train_r51_series.sh`) — anchor-strength sweep.
R50 verdict (2026-07-15): r50a (gw decay) fixed desaturation and matched the
champion (22.21/0.7964/**0.1603** — first LPIPS win) but still crashed 6-7K;
r50c (anchor x1_lq w0.5) ELIMINATED the crash (first no-valley curve; SSIM
0.8064 record) but capped PSNR at 21.75; r50b (slow estimator) REFUTED. Anchor
strength = clean one-variable tradeoff -> R51 sweeps w 0.1/0.25 and adds a
relative deadzone (w0.5 force only outside +-15% of target).
Root-cause analysis: `docs/COLOR_SHIFT_ROOT_CAUSE.md`.

## Config Conventions

- `total_iter: 24000`, `batch_size_per_gpu: 24`, `gt_size: 128`, `lr: 6e-5`
- `val_freq: 500` (fine-grained metric tracking), `save_checkpoint_freq: 2000`, `print_freq: 1000`
- `save_img: false` — selective image saving is handled in code, not by this flag
- `early_stop_patience_val: 8`, `use_amp: true`, `grad_clip_value: 0.02`
- All losses: Charbonnier pixel + VGG perceptual + color + chroma + TV (+ x1_lq anchor since R50c)

## Training Commands

```bash
python -m basicsr.train --opt Options/ISB_ecaformer_r48b_illum3ch_bridge_reweight.yml  # champion
bash train_r51_series.sh   # active research (r51a/b/c: anchor w0.1/w0.25/w0.5+deadzone)
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
- ❌ `anchor_mode: gt` (anchor x1 means to GT means) — mathematically INERT on LOLv1:
  x1 = lq·(1+sigmoid) ≤ 2·lq and 2·lq_gray (~0.09) << gt_gray (~0.49) on 100% of
  images, so the loss is a saturated constant, not a restoring force (R49b confirmed:
  identical crash with and without it). Use `anchor_mode: x1_lq` instead
- ❌ Permanent (unscheduled) `residual_gray_world` — desaturates converged outputs and
  amplifies estimator drift via detached reciprocal gains (R49 confirmed). Always pair
  with `gray_world_decay_start/end`
- ❌ `estimator_lr_mult < 1` (slowing the estimator to stop the drift crash) — REFUTED
  by r50b: the transient got longer and deeper (peak 20.83, early-stopped). The drift
  needs a restoring force (anchor), not a slower clock

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
- **R50 mechanism keys**: `gray_world_decay_start/end` live under `network_g` (arch
  kwargs); `estimator_lr_mult`, `anchor_mode`, `anchor_target_ratio`, `anchor_deadzone` (R51:
  relative band, must be < (2-ratio)/ratio) live under `train`.
  A key in the wrong section is silently ignored — the R50 code validates its own keys
  (bad decay window / non-positive lr mult / unknown anchor_mode raise at init)
- **Training logs**: since R50, `log_dict` carries every loss component plus drift
  diagnostics (`x1_mean_r/g/b`, `gw_gain_r/g/b`, `gw_gain_clamp_frac`, `gw_blend`) to
  TensorBoard — check these curves first when diagnosing a PSNR-only crash window
- **`_current_iter` plumbing**: `optimize_parameters` sets it on both bare `net_g` and
  `net_g_ema` every step; the arch propagates it to the denoiser. Deployment inference
  (attr unset = -1) uses the TERMINAL schedule state — scheduled mechanisms must
  converge to their final value before `total_iter`
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
