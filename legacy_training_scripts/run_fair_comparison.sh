#!/bin/bash
# Fair comparison: ECAFormer baseline vs ISB
# LOLv1 only, paper-aligned recipe, sequential training
set -e

cd "$(dirname "$0")"

CONFIGS=(
    "Options/ECAFormer_baseline_lolv1_fair_paper.yml"
    "Options/ISB_ecaformer_lolv1_fair_paper.yml"
)

for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .yml)
    echo "=========================================="
    echo "Training: $name"
    echo "Config:   $cfg"
    echo "Start:    $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    python -m basicsr.train --opt "$cfg"
    echo ""
    echo "Done: $name | End: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
done

echo "=========================================="
echo "All fair comparison runs completed."
echo "=========================================="
