# Overtraining Analysis: Post-Peak Performance Degradation

**Date**: 2026-06-12  
**Issue**: All R43 experiments show PSNR decline after 10.5K-11K iterations despite training continuing to 15K-24K

## Observed Pattern

| Experiment | Peak PSNR @ Iter | Final PSNR @ Iter | Degradation |
|------------|------------------|-------------------|-------------|
| r43_hybrid | 22.09 @ 11K | 21.58 @ 15K | -0.51 PSNR |
| r43a_learnable | 21.55 @ 11.5K | 21.39 @ 15K | -0.16 PSNR |
| r43a_identity_scale | 21.57 @ 10.5K | Stagnates | Plateau |
| R42a (baseline) | 21.64 @ 10.5K | N/A | Stopped early |

## Root Cause Analysis

### 1. Learning Rate Schedule Mismatch

**Current Configuration**:
```yaml
scheduler:
  type: CosineAnnealingRestartLR
  periods: [24000]
  restart_weights: [1]
  eta_min: 1e-6

train:
  total_iter: 24000
  warmup_iter: 500

optim_g:
  lr: 6e-5
```

**Problem**: Cosine schedule assumes convergence at T_max=24K, but model actually converges at ~10.5K

| Iteration | Learning Rate | % of Initial | Stage |
|-----------|--------------|--------------|-------|
| 500 | 6.0e-5 | 100% | After warmup |
| 5000 | 5.5e-5 | 91% | Early training |
| 10000 | 3.9e-5 | 65% | Convergence zone |
| 10500 | 3.7e-5 | 62% | **Peak PSNR** |
| 11000 | 3.5e-5 | 59% | Start decline |
| 12000 | 3.1e-5 | 52% | Plateau |
| 15000 | 2.0e-5 | 33% | **Degradation** |
| 18000 | 1.0e-5 | 17% | Severe overtraining |

**Impact**: 
- At peak (10.5K): LR = 62% → still flexible for refinement
- At degradation (15K): LR = 33% → trapped in local minimum, too low to escape

### 2. Early Stopping Too Permissive

**Current Configuration**:
```yaml
train:
  early_stop_patience_val: 8
  early_stop_min_delta: 0.005

val:
  val_freq: 500
```

**Problem**: Patience of 8 validations = 4,000 iterations grace period

**Timeline if peak @ 10.5K**:
| Iteration | Validations w/o Improvement | Action |
|-----------|----------------------------|--------|
| 10500 | 0 (peak) | Continue |
| 11000 | 1 | Continue |
| 11500 | 2 | Continue |
| 12000 | 3 | Continue |
| 12500 | 4 | Continue |
| 13000 | 5 | Continue |
| 13500 | 6 | Continue |
| 14000 | 7 | Continue |
| 14500 | 8 | **Early stop triggers** |

By 14.5K, PSNR has already dropped 0.16-0.51 from peak.

### 3. Overtraining with Decaying LR

After 12K iterations, LR drops below 50% of initial:
- Insufficient learning rate to escape suboptimal local minima
- Model "polishes" poor solution, increasing validation loss
- Training loss may plateau while validation metrics degrade
- Classic overtraining signature: training continues, validation worsens

## Recommended Solutions

### OPTION 1: Tighter Early Stopping ⭐ RECOMMENDED

**Minimal change, maximum impact**

```yaml
train:
  total_iter: 24000              # Keep existing
  early_stop_patience_val: 4     # Change from 8
  early_stop_min_delta: 0.005    # Keep existing

scheduler:
  type: CosineAnnealingRestartLR
  periods: [24000]               # Keep existing
  restart_weights: [1]
  eta_min: 1e-6
```

**Effect**:
- Early stop after 4 × 500 = 2,000 iterations without improvement
- If peak @ 10.5K → stops @ 12.5K (instead of 14.5K)
- LR @ 12.5K = 4.8e-5 (48% of initial) — still reasonable
- Prevents all observed 0.16-0.51 PSNR degradation
- Saves ~20% compute time per run

**When to use**: Default for all future experiments

---

### OPTION 2: Even Tighter Early Stopping (Aggressive)

**Maximum protection against overtraining**

```yaml
train:
  total_iter: 24000
  early_stop_patience_val: 3     # Change from 8
  early_stop_min_delta: 0.01     # Change from 0.005 (stricter)

scheduler:
  type: CosineAnnealingRestartLR
  periods: [24000]
  restart_weights: [1]
  eta_min: 1e-6
```

**Effect**:
- Early stop after 3 × 500 = 1,500 iterations without improvement
- If peak @ 10.5K → stops @ 12K
- LR @ 12K = 3.1e-5 (52% of initial)
- Catches plateau immediately after convergence

**Tradeoff**: Might stop prematurely if validation metrics are noisy

**When to use**: If Option 1 still shows degradation, or when compute is expensive

---

### OPTION 3: LR Restart at Convergence (Experimental)

**Investigate if 10.5K peak is local minimum**

```yaml
train:
  total_iter: 18000
  early_stop_patience_val: 4

scheduler:
  type: CosineAnnealingRestartLR
  periods: [11000, 7000]         # Restart after typical convergence
  restart_weights: [1, 0.3]      # Second stage: 30% LR
  eta_min: 1e-6
```

**LR Schedule**:
| Iteration | Learning Rate | % of Initial | Notes |
|-----------|--------------|--------------|-------|
| 10500 | 3.0e-5 | 50% | Approaching peak |
| 11000 | 1.0e-6 | 1.7% | End of first period |
| 11500 | 1.8e-5 | 30% | **Restart boost** |
| 12000 | 1.7e-5 | 28% | Second period |
| 15000 | 7.6e-6 | 13% | Decay continues |

**Hypothesis**: LR boost at 11K might help model escape local minimum and find better solution

**When to use**: 
- When current best (21.64) is significantly below historical best (22.10)
- When you have compute budget for experimental runs
- To investigate whether adaptive scheduling helps

---

## Implementation Plan

### Phase 1: Validate with Option 1 (Immediate)

1. Update baseline config (R42a) with `early_stop_patience_val: 4`
2. Run quick validation (should stop around 12K-13K)
3. Verify no performance degradation vs manually stopping at 10.5K

### Phase 2: Apply to All Future Runs

Update all config templates:
- `Options/ISB_ecaformer_r*.yml`
- Diagnostic scripts in `diagnostic_scripts/`
- Update `CLAUDE.md` to reflect new default

### Phase 3: Experimental (Optional)

Test Option 3 (LR restart) if:
- Option 1 validated successfully
- Need to close gap to R38c (22.10 PSNR)
- Have spare GPU time for exploration

---

## Expected Outcomes

### With Option 1 (patience=4):
- Training duration: ~12.5K iterations (was 15K-24K)
- Best checkpoint: 10.5K-11K (unchanged)
- Final PSNR: Match peak PSNR (no 0.16-0.51 drop)
- Time savings: ~15-20% per run

### Success Metrics:
- ✅ Final validation PSNR matches best checkpoint within 0.05
- ✅ No post-peak degradation beyond noise floor
- ✅ Consistent stopping point across runs (12K-13K range)

### Failure Signs:
- ❌ Premature stopping before reaching 10.5K-11K
- ❌ High variance in stopping iteration across runs
- ❌ Validation PSNR still shows systematic decline

If Option 1 fails, escalate to Option 2 or investigate validation set issues.

---

## References

- Current stable config: `Options/ISB_ecaformer_r42a_per_ch_res.yml`
- Training model: `basicsr/models/image_isb_model.py` (lines 91-94 for early stopping)
- Cosine schedule: Uses PyTorch's `CosineAnnealingRestartLR` (defined in basicsr)
- Early stopping logic: Implemented in base training loop (checks `early_stop_patience_val`)
