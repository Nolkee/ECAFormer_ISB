#!/bin/bash
# R30 + R31 Series: Fix green tint + dim output
#
# R30 (based on R28a: eca_gamma=2, lr=2e-4):
#   r30a: + color_loss_weight=0.1 (fix green tint)
#   r30b: + use_out_norm=false (fix dim/gray output)
#   r30c: Combined (color_loss + no out_norm + lr=1e-4)
#
# R31 (based on R29a: eca_gamma=1.5, lr=6e-5, stable curves):
#   r31a: + color_loss_weight=0.1 (fix green tint)
#   r31b: + use_out_norm=false (fix dim/gray output)
#   r31c: Combined (color_loss + no out_norm)
#
# Compare:
#   R28a: PSNR=21.01, SSIM=0.797, LPIPS=0.164
#   R29a: PSNR=20.94, SSIM=0.789, LPIPS=0.175

set -e

CONFIGS=(
    # R30 series (R28a base)
    "Options/ISB_ecaformer_r30a_color_loss.yml"
    "Options/ISB_ecaformer_r30b_no_outnorm.yml"
    "Options/ISB_ecaformer_r30c_combined.yml"
    # R31 series (R29a base)
    "Options/ISB_ecaformer_r31a_color_loss.yml"
    "Options/ISB_ecaformer_r31b_no_outnorm.yml"
    "Options/ISB_ecaformer_r31c_combined.yml"
)

for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .yml)
    echo "=========================================="
    echo "Training: $name"
    echo "Config:   $cfg"
    echo "Time:     $(date)"
    echo "=========================================="

    python -m basicsr.train --opt "$cfg" 2>&1 | tee "train_${name}.log"

    echo "Finished: $name at $(date)"
    echo ""
done

echo "R30+R31 ablation completed."
