#!/bin/bash
# Paper P1 GPU queue (2026-07-20) — runs after the zero-shot probes finish.
#
# 1. r54_repro_r52b      — recover a reproducible champion ckpt (fixed save_best
#                          now stores params+params_ema; the original r52b best
#                          file holds bare weights that score 21.04, not 22.69)
# 2. baseline_lolv1_fair24k  — matched-budget control (n80/128/24/24K): decides
#                          the "concedes fidelity" vs "comparable" narrative and
#                          tests the patch-256 LPIPS confound
# 3. r53_lolv2real       — ours on LOLv2-Real (r53b auto-engage recipe)
# 4. baseline_lolv2real_fair24k — paired control on LOLv2-Real
#
# Launch: nohup bash run_paper_p1_queue.sh > p1_queue.log 2>&1 &
set -e
set -o pipefail
cd "$(dirname "$0")"

PY=$HOME/anaconda3/envs/Retinexformer/bin/python

CONFIGS=(
  "Options/ISB_ecaformer_r54_repro_r52b.yml"
  "Options/ECAFormer_baseline_lolv1_fair24k.yml"
  "Options/ISB_ecaformer_r53_lolv2real.yml"
  "Options/ECAFormer_baseline_lolv2real_fair24k.yml"
)

for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .yml)
    echo "=========================================="
    echo "Training: $name"
    echo "Start:    $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    "$PY" -m basicsr.train --opt "$cfg" 2>&1 | tee "train_${name}.log"
    echo "Finished: $name at $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
done

echo "P1 queue complete: $(date '+%Y-%m-%d %H:%M:%S')"
