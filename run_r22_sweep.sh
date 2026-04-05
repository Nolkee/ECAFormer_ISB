#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODE="${1:-single}"

# Round 22 sweep: fix desaturation on r20d baseline
# r22a: GroupNorm + color_loss=0.1          — loss-level fix
# r22b: InstanceNorm (no color loss)        — structural fix
# r22c: InstanceNorm + color_loss=0.1       — structural + loss fix
# r22d: GroupNorm + Charbonnier + clamped   — loss-level alternative

bash run_train_sequence.sh "$MODE" \
  "Options/ISB_ecaformer_full_s19_r22a_color_loss.yml" \
  "Options/ISB_ecaformer_full_s19_r22b_instancenorm.yml" \
  "Options/ISB_ecaformer_full_s19_r22c_instnorm_color.yml" \
  "Options/ISB_ecaformer_full_s19_r22d_clamped_charb.yml"
