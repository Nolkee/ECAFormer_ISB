#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODE="${1:-single}"

# Round 20 sweep: residual scale + output normalization ablation
# r20a: residual_scale=0.5, no out_norm
# r20b: residual_scale=0.7, no out_norm
# r20c: learnable residual scale, no out_norm
# r20d: residual_scale=1.0, WITH out_norm (r6c structure revert)

bash run_train_sequence.sh "$MODE" \
  "Options/ISB_ecaformer_full_s19_r20a_resscale05.yml" \
  "Options/ISB_ecaformer_full_s19_r20b_resscale07.yml" \
  "Options/ISB_ecaformer_full_s19_r20c_learnable_resscale.yml" \
  "Options/ISB_ecaformer_full_s19_r20d_outnorm.yml"
