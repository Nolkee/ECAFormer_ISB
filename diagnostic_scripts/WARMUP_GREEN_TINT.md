# Early-Stage Green Tint During Warmup

## Problem

Training starts with green-tinted output for first ~500-1000 iterations.

## Root Cause

**Warmup period lacks green suppression:**

```python
# Iteration 0-500 (alpha ≈ 0.0-0.1):
identity_scale ≈ [1.0, 1.0, 1.0]  # No green suppression yet (warmup start)
residual_scale = 0.6               # Uniform across channels (R43a-warmup config)

# Data flow:
x_low (green-biased) → x1 = x_low * illu_map + [1,1,1] * x_low → denoiser → output + 0.6 * x1
                                      ↑                                           ↑
                               Preserves green                            Uniform scale, no correction
```

**Result**: Green bias from x_low flows through entire pipeline uncorrected.

## Solution: Dual-Stage Green Suppression (Hybrid Config)

### R43a-warmup-hybrid Strategy

Combine two orthogonal green fixes:

1. **Upstream (gradual)**: identity_scale warmup [1,1,1] → [1,0.92,1]
   - Prevents AdaLN conflict (primary goal)
   - Takes 5000 iterations to reach full suppression

2. **Downstream (immediate)**: residual_scale = [0.6, 0.5, 0.6]
   - Per-channel correction at denoiser output
   - Active from iteration 0
   - Compensates for warmup period's lack of upstream suppression

### Config Comparison

| Config | identity_scale | residual_scale | Early Green? | 3500-6000 Stable? |
|--------|---------------|----------------|--------------|-------------------|
| R42a | [1,1,1] (fixed) | [0.6, 0.5, 0.6] | ❌ No | ✅ Yes |
| R43a-warmup | [1,1,1]→[1,0.92,1] | 0.6 (uniform) | ⚠️ Yes (0-1K iter) | 🔬 Testing |
| **R43a-warmup-hybrid** | [1,1,1]→[1,0.92,1] | **[0.6, 0.5, 0.6]** | ✅ No | 🔬 Testing |

### Usage

**If you observe green tint in first 1K iterations:**

```bash
# Switch to hybrid config
python -m basicsr.train --opt Options/ISB_ecaformer_r43a_warmup_hybrid.yml
```

**Config file**: `Options/ISB_ecaformer_r43a_warmup_hybrid.yml`

Key change:
```yaml
residual_scale_init: [0.6, 0.5, 0.6]  # Was: 0.6
```

## Design Trade-off

**Why not always use hybrid?**

- R43a-warmup (identity_scale only) tests whether warmup **alone** can solve instability
- Hybrid config adds R42a's known fix, making it harder to isolate warmup's contribution

**Recommendation**:
1. **Research**: Use R43a-warmup to test pure warmup hypothesis
2. **Production**: Use R43a-warmup-hybrid for best quality throughout training

## Expected Behavior

### R43a-warmup (identity_scale only)
- Iter 0-500: Green tint present (expected)
- Iter 500-1000: Green gradually reduces
- Iter 1000-5000: Continued green reduction as identity_scale → [1,0.92,1]
- Iter 5000+: Full green suppression

### R43a-warmup-hybrid (dual-stage)
- Iter 0+: No green tint (residual_scale corrects immediately)
- Iter 0-5000: Warmup still active, preventing AdaLN conflict
- Iter 5000+: Both corrections active

---

**Files**:
- Standard: `Options/ISB_ecaformer_r43a_warmup.yml`
- Hybrid: `Options/ISB_ecaformer_r43a_warmup_hybrid.yml`
- Safe (clamp enabled): `Options/ISB_ecaformer_r43a_warmup_safe.yml`
