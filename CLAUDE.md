# ECAFormer-ISB

Low-light image enhancement using Image Schrödinger Bridge (ISB) with ECAFormer backbone. LOLv1/v2 datasets.

## Key Files

- **Model**: `basicsr/models/archs/ECAFormer_ISB_arch.py` — ECAFormerISB (denoiser) + ShallowDeepConv (estimator)
- **Training**: `basicsr/models/image_isb_model.py` — ImageISBModel with bridge loss + pixel/perceptual/color/chroma/anchor losses
- **Config**: `Options/ISB_ecaformer_r*.yml` — YAML configs, `model_type: ImageISBModel`
- **Data**: `data/LOLv1/` (Train/input, Train/target), `data/LOLv2Real/` (Train/Low, Train/Normal)
- **Diagnostic tools**: `diagnostic_scripts/` — Training stability analysis, checkpoint diagnosis
- **Legacy scripts**: `legacy_training_scripts/` — Historical experiments R11-R43

## Current Champion: r52b (late EMA anchor)

**Config**: `Options/ISB_ecaformer_r52b_late_ema_anchor_9k.yml`
**Result**: PSNR 22.689 / SSIM 0.8042 / LPIPS 0.1664 — all at ONE checkpoint (11.5K)
**Recipe**: R48b base + gray-world decay 1500-3500 + ema_warmup + x1_ema anchor
(w0.5, deadzone 0.05, engage @ 9K, frozen-EMA target). Beat previous bests on
PSNR (+0.48 dB) and SSIM simultaneously. AAAI headline model; the saved
`best_psnr_*.pth` is CONFIRMED (2026-07-19) to be the 11.5K checkpoint.

**Active research**: R53 series (`train_r53_series.sh`) — final pre-AAAI round.
R52 verdict (2026-07-19): the phase-transition model delivered — r52b is the
new champion — but (a) the 0.05 deadzone let a -0.76 dB post-peak sag escape
(r50c/r51c tight pins never sagged) and (b) transition timing is stochastic
(5.5K vs 7.5K on identical configs), so r52a's fixed 12K engage locked in a
post-peak decline. R53: deadzone 0 + patience 12 (hold the peak), auto-engage
at the train-PSNR recovery turn with 12K cap + engage-state persistence
(de-luck the timing), seed 3407 arm (second draw + mean±std for reviewers).
Generalization run for the paper: `Options/ISB_ecaformer_r53_lolv2real.yml`.
Root-cause analysis: `docs/COLOR_SHIFT_ROOT_CAUSE.md`.

## Config Conventions

- `total_iter: 24000`, `batch_size_per_gpu: 24`, `gt_size: 128`, `lr: 6e-5`
- `val_freq: 500` (fine-grained metric tracking), `save_checkpoint_freq: 2000`, `print_freq: 1000`
- `save_img: false` — selective image saving is handled in code, not by this flag
- `early_stop_patience_val: 8`, `use_amp: true`, `grad_clip_value: 0.02`
- All losses: Charbonnier pixel + VGG perceptual + color + chroma + TV (+ x1_lq anchor since R50c)

## Training Commands

```bash
python -m basicsr.train --opt Options/ISB_ecaformer_r52b_late_ema_anchor_9k.yml  # champion recipe
bash train_r53_series.sh   # active research (r53a dz0 / r53b auto-engage / r53c seed 3407)
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
- ❌ Anchor weights <= 0.25 (x1_lq) — fail BOTH ways (R51): the valley still happens
  and the fixed target drags post-transition PSNR. Stability needs w~0.5-class force
- ❌ From-iter-0 anchor on a PSNR-chasing run — blocks the phase transition, caps PSNR
  ~21.8 (r50c/r51c). Use `anchor_start_iter` + `anchor_mode: x1_ema` (R52) instead;
  from-0 anchoring is only for perceptual-champion runs (SSIM/LPIPS)
- ❌ `zero_init_mapping_bias` — REFUTED twice: worst valley unanchored (r49c) AND
  strictly worse on the anchored base (r52c: 21.18/0.786 vs r51c 21.83/0.8011)

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
  relative band, must be < (2-ratio)/ratio), `anchor_start_iter`, `anchor_ema_momentum`
  (R52: x1_ema tracks from iter 0, freezes at engage), `anchor_engage_mode` +
  `anchor_valley_drop/rise_margin/min_engage_iter/freeze_delay` (R53 auto-engage;
  in auto mode anchor_start_iter is the hard cap) live under `train`.
  R53 persists engage state + frozen ratio in `latest.state` (`model_extra` via
  the `_extra_training_state` hook in base_model) — legacy states resume fine.
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
  - Training states: overwrite-only `training_states/latest.state` (atomic tmp+replace) PLUS `training_states/best.state` snapshotted at every new best PSNR (added 2026-07-21 after the r52b peak-EMA loss). Legacy numeric `{iter}.state` files still resumable.
  - Running weights: single overwrite-only `models/net_g_latest.pth`. Best checkpoints (`best_psnr_*.pth` etc.) store BOTH `params` and `params_ema` since 2026-07-21 — validation scores net_g_ema, so pre-fix best files (bare `params` only) do NOT reproduce their logged metrics.
  - Validation images: memory-only metrics by default. Disk writes ONLY for one-time `visualization/baseline/` dump (first validation) and `visualization/best_results/` (overwritten on each new best PSNR). The `save_img` config flag is ignored during training.

## Deep Dive Documentation

- **Green tint & crash root cause**: `docs/COLOR_SHIFT_ROOT_CAUSE.md` — verified mechanism + R49 fixes
- **Architecture & design choices**: `docs/ARCHITECTURE.md`
- **Quick start & troubleshooting**: `docs/QUICKSTART.md`
- **Diagnostic framework**: `diagnostic_scripts/README.md`
- **Experiment history**: `legacy_training_scripts/README.md`
