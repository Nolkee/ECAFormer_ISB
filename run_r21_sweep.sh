#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODE="${1:-single}"

# Round 21 sweep: brightness recovery on top of r20d (out_norm=true) baseline
# r21a: residual_scale=1.3          — direct brightness boost
# r21b: learnable residual + outnorm — network finds optimal scale
# r21c: 20k iterations              — longer runway
# r21d: x0_loss_weight=0.6          — stronger GT matching

bash run_train_sequence.sh "$MODE" \
  "Options/ISB_ecaformer_full_s19_r21a_resscale13.yml" \
  "Options/ISB_ecaformer_full_s19_r21b_learnable_outnorm.yml" \
  "Options/ISB_ecaformer_full_s19_r21c_longer.yml" \
  "Options/ISB_ecaformer_full_s19_r21d_x0w06.yml"
