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
[ShallowDeepConv estimator]  → (visual_fea, illu_map)
  ↓
x1 = x_low * illu_map + identity_scale * x_low   (Retinex; identity_scale=[1,1,1] by default)
  ↓
[CrossAttenUnet_ISB denoiser with AdaLN]  → 8-NFE diffusion bridge
  ↓
out = mapping(features) + residual_scale * WB(x1)   (WB = gray-world if residual_gray_world)
  ↓
x0 (output, clamped to [0,1])
```

**Files:**
- `basicsr/models/archs/ECAFormer_ISB_arch.py` — Main architecture (ECAFormerISB + CrossAttenUnet_ISB + ShallowDeepConv)
- `basicsr/models/archs/isb_module.py` — ISBEngine (q_sample, reverse_sample_fast)
- `basicsr/models/image_isb_model.py` — Training loop (ImageISBModel), losses
- `basicsr/models/image_restoration_model.py` — ImageCleanModel base (EMA, save, validation)

### 2. Color Fidelity — the central design problem

LOLv1 low-light input is green-biased (Bayer 2G:1R:1B). Because the output has a
residual shortcut `out = mapping(features) + 0.6*x1`, and at init `mapping≈0`,
the early-training output is ≈`0.9*x_low` — a copy of the green input. This is
the verified root cause of early green tint (see `docs/COLOR_SHIFT_ROOT_CAUSE.md`),
NOT the illumination channel count (that theory was disproven by R47/R48).

**Correction strategies tried** (where the channel fix is applied matters):

| Approach | Where | Result |
|----------|-------|--------|
| `channel_scale=[1,0.95,1]` | Illumination map | R38c: PSNR 22.10, still green early, mid crash |
| `identity_scale=[1,0.92,1]` | x1 construction | Unstable everywhere tried (R43/R44/R48c) — abandoned |
| `residual_scale=[0.6,0.5,0.6]` | Denoiser output | R42a: PSNR 21.64, stable but below champion |
| `channel_noise_scale=[1,0.8,1]` | Bridge noise | **R48b champion: PSNR 22.21 / SSIM 0.7959** |
| `residual_gray_world` | Residual shortcut | R49: kills early green, but permanent WB desaturates + destabilizes |
| `gray_world_decay_start/end` | Residual shortcut | R50: WB only while needed (blend 1->0 over 1500-3500) |

### 3. R49 mechanisms (green-tint root fixes) and the R50 corrections

Built on the R48b champion. R49 verdict (2026-07-14): green tint fixed, but two
regressions — see `docs/COLOR_SHIFT_ROOT_CAUSE.md` for the full mechanism.

- **`ema_warmup: true`** — EMA decay `min(ema_decay, (1+t)/(10+t))` so early
  validation (rendered from `net_g_ema`) reflects the live net instead of ~61%
  random init at iter 500. Converges to `ema_decay`. KEPT in R50.
- **`residual_gray_world: true`** — per-image gray-world white balance on the
  residual shortcut only. Gains detached, clamped [0.5, 2.0], identical in
  train and inference. Fixed the green tint BUT permanent WB desaturated
  converged outputs and amplified drift. R50 adds **`gray_world_decay_start:
  1500` / `gray_world_decay_end: 3500`** (network_g keys): the WB blend decays
  linearly to 0, restoring the residual's natural color cast after the
  green-risk phase. Deployment inference uses the terminal blend (0).
- **`anchor_loss_weight` with `anchor_mode: gt` (R49) — INERT**: x1 =
  lq*(1+sigmoid) <= 2*lq, and 2*lq_gray (~0.09) << gt_gray (~0.49) on 100% of
  LOLv1, so the loss was a saturated constant (r49b crashed identically to
  r49a). R50 replaces it with **`anchor_mode: x1_lq`** + `anchor_target_ratio:
  1.5` + weight 0.5: pins mean_c(x1) to the reachable midpoint 1.5*mean_c(lq),
  a two-sided restoring force.
- **`estimator_lr_mult: 0.3`** (R50, train key) — separate AdamW param group
  for the estimator at 0.3x LR (warmup/cosine scale per group). Slows x1 drift
  at the source so the denoiser/AdaLN can track it (crash fix A; the anchor is
  fix B — r50b vs r50c compare them head-to-head).
- **Observability** (R50): `log_dict` now carries every loss component plus
  `x1_mean_r/g/b`, `gw_gain_r/g/b`, `gw_gain_clamp_frac`, `gw_blend` — the
  drift is directly visible in TensorBoard during any crash window.

### 4. Training Configuration

**Champion (R48b)**:
```yaml
illumination_channels: 3
residual_scale_init: 0.6
channel_noise_scale: [1.0, 0.8, 1.0]
channel_scale_init: [1.0, 1.0, 1.0]
use_out_norm: true
output_activation: identity
```

**Loss components** (`image_isb_model.py`):
- Bridge loss = x0_loss (0.4) + pixel L1 (0.6), scaled by bridge_weight 1.0
- VGG perceptual (0.1), color (channel-mean L1, 0.2), chroma (channel-std L1, 0.05)
- TV on illu_map (0.002)
- anchor (R52: `anchor_mode: x1_ema` + `anchor_start_iter` — engage post-transition, frozen-EMA target; `x1_lq` for from-0 perceptual runs), green/fft (off by default)

**Optimizer**: AdamW, lr 6e-5, cosine annealing 24K iter. Single param group
by default; `estimator_lr_mult` (R50b: 0.3) splits the estimator into a second
group at a lower LR (warmup and cosine scale per group).

## Data Flow (training)

1. **Input**: LOLv1 (485 train / 15 test) or LOLv2 Real (~689 train / ~100 test)
2. **Estimator**: x_low → (visual_fea, illu_map)
3. **x1 construction**: `x_low * illu_map + identity_scale * x_low`
4. **Bridge sampling**: t ~ U(0.01, 1.0), `x_t = (1-t)*x0 + t*x1 + sigma_t*eps`
   (eps scaled per-channel by `channel_noise_scale`)
5. **Denoiser**: predicts x0 from x_t, conditioned on t via AdaLN
6. **Output**: `mapping(features) + residual_scale * WB(x1)`, clamped to [0,1]

## Model Variants (LOLv1 test)

| Name | Key Difference | Best PSNR | Notes |
|------|----------------|-----------|-------|
| R38c | channel_scale, illum=1 | 22.10 @ 10K | Green early, mid crash |
| R42a | residual_scale per-ch | 21.64 @ 10.5K | Stable, below champion |
| R48b | channel_noise_scale, illum=3 | **22.21 @ 11.5K** | **Champion** (SSIM 0.7959) |
| R49a | + gray-world/ema_warmup | 22.20 @ 9K | Green FIXED; desaturated, crash earlier (6.5K) |
| R49b | + anchor 0.05 (gt) | 21.94 @ 8.5K | Anchor inert — crash identical to R49a |
| R49c | + zero_init_mapping_bias | 22.07 @ 9.5K | Worst valley (17.5 @ 6.5K) |
| R50a | gw decay 1500-3500 | 22.21 @ 16K | Desat FIXED, LPIPS 0.1603 (first win); crash remains 6-7K |
| R50b | + estimator_lr 0.3x | 20.83 @ 4.5K | REFUTED — longer/deeper transient, early-stopped |
| R50c | + anchor x1_lq w0.5 | 21.75 @ 14.5K | NO CRASH (first ever); SSIM 0.8064 record; PSNR capped |
| R51a | anchor w0.1 | 21.98 @ 16.5K | Still crashes (-2.9 dB); fixed-target drag post-transition |
| R51b | anchor w0.25 | 21.61 @ 12.5K | Crashes EARLIER; strictly worst — no middle sweet spot |
| R51c | anchor w0.5 + deadzone 0.15 | 21.83 @ 16K | No valley; SSIM 0.8011/LPIPS 0.1639 @ 20K = perceptual champ |
| R52a/b/c | late x1_ema anchor 12K/9K / +zero-bias | pending | Let the transition happen, then lock the new regime |

## Checkpoint & Disk Policy

Training is disk-space safe (single-file overwrite, atomic writes):
- Training state → one `training_states/latest.state` (overwritten each save)
- Running weights → one `models/net_g_latest.pth`
- Best weights → `best_{psnr,ssim,lpips}_*.pth` (weights only, no optimizer)
- Validation images → memory-only metrics; disk writes only for `visualization/baseline/`
  (first validation, once) and `visualization/best_results/` (overwritten on each new best PSNR)

Auto-resume prefers `latest.state`, falling back to legacy numeric `{iter}.state`.
See `basicsr/models/base_model.py` and `image_restoration_model.py`.

## Forbidden Configurations

- ❌ `output_activation: sigmoid` — Kills dynamic range (R39)
- ❌ `use_out_norm: 'post'` — GroupNorm(1,3) at output plateaus ~13 PSNR (R41a/d)
- ❌ `channel_scale < 0.90` — PSNR loss exceeds green fix benefit (R40)
- ❌ `identity_scale` (any variant) — unstable everywhere tried (R43/R44/R48c)
- ❌ `green_norm` in training mode only — train/val mismatch (R45)
- ❌ `anchor_mode: gt` — target unreachable (sigmoid caps x1 at 2*lq << gt), loss is a saturated constant (R49b)
- ❌ permanent `residual_gray_world` (no decay) — desaturates + destabilizes (R49); pair with `gray_world_decay_start/end`

## Inference

**Script**: `ECAFormer_inference.py` · **Output**: `results/<config_name>/`

## Diagnostic Tools

`diagnostic_scripts/` — training instability analysis.
`tools/diagnose_checkpoint.py` — checkpoint parameter extraction and visualization.

---

**Last updated**: 2026-07-16
**Champion config**: R48b (`Options/ISB_ecaformer_r48b_illum3ch_bridge_reweight.yml`)
**Active research**: R52 series (late self-calibrating anchor — phase-transition model)
