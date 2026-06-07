#!/bin/bash
# R32c-fix + R33 + R34 + R35 Series: Complete Ablation Pipeline
#
# Phase 1: R32c fix (OOM fix: batch_size 24->12)
# Phase 2: R33 (green tint + gray desaturation fix)
# Phase 3: R34 (capacity + residual scale)
# Phase 4: R35 (LR restart + 250K long training)
#
# R33 ablation target: green tint + gray output
#   R33a: illumination_channels=1 (single-channel illumination)
#   R33b: use_out_norm='post' (GroupNorm on RGB, not features)
#   R33c: color_loss_weight=0.2 (stronger channel balance)
#   R33d: combined (illum_ch=1 + post_norm + color=0.2)
#
# R34 ablation target: model capacity + residual path
#   R34a: n_feat=96 (50% wider)
#   R34b: residual_scale_init=0.5 (reduce x1 shortcut)
#   R34c: learnable_residual_scale=true
#   R34d: n_feat=96 + residual_scale=0.5
#
# R35: Best config from R33/R34 + LR restart + 250K

set -e

# Phase 1: R32c fix
echo "=========================================="
echo "Phase 1: R32c fix (batch_size=12)"
echo "=========================================="
python -m basicsr.train --opt Options/ISB_ecaformer_r32c_gt192.yml 2>&1 | tee train_r32c_fix.log

# Phase 2: R33 series (green/gray fix)
R33_CONFIGS=(
    "Options/ISB_ecaformer_r33a_illum1ch.yml"
    "Options/ISB_ecaformer_r33b_postnorm.yml"
    "Options/ISB_ecaformer_r33c_color02.yml"
    "Options/ISB_ecaformer_r33d_combined.yml"
)

echo ""
echo "=========================================="
echo "Phase 2: R33 series (green/gray fix)"
echo "=========================================="
for cfg in "${R33_CONFIGS[@]}"; do
    name=$(basename "$cfg" .yml)
    echo "------------------------------------------"
    echo "Training: $name"
    echo "Config:   $cfg"
    echo "Time:     $(date)"
    echo "------------------------------------------"

    python -m basicsr.train --opt "$cfg" 2>&1 | tee "train_${name}.log"

    echo "Finished: $name at $(date)"
    echo ""
done

# Phase 3: R34 series (capacity + residual)
R34_CONFIGS=(
    "Options/ISB_ecaformer_r34a_nfeat96.yml"
    "Options/ISB_ecaformer_r34b_residual05.yml"
    "Options/ISB_ecaformer_r34c_learnable.yml"
    "Options/ISB_ecaformer_r34d_nfeat96_res05.yml"
)

echo ""
echo "=========================================="
echo "Phase 3: R34 series (capacity + residual)"
echo "=========================================="
for cfg in "${R34_CONFIGS[@]}"; do
    name=$(basename "$cfg" .yml)
    echo "------------------------------------------"
    echo "Training: $name"
    echo "Config:   $cfg"
    echo "Time:     $(date)"
    echo "------------------------------------------"

    python -m basicsr.train --opt "$cfg" 2>&1 | tee "train_${name}.log"

    echo "Finished: $name at $(date)"
    echo ""
done

# Phase 4: R35 (LR restart + 250K)
echo ""
echo "=========================================="
echo "Phase 4: R35a (LR restart + 250K)"
echo "=========================================="
python -m basicsr.train --opt Options/ISB_ecaformer_r35a_restart250k.yml 2>&1 | tee train_r35a_restart.log

echo ""
echo "=========================================="
echo "All experiments completed."
echo "=========================================="
