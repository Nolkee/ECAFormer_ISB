#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODE="${1:-single}"

# Round 25 sweep: push saturation further on r24c baseline
# r25a: gt_size=192, batch=16    — larger patches for global color context
# r25b: n_feat=48, batch=16      — wider network for more color capacity
# r25c: residual_scale=0.8+bias  — let mapping dominate over desaturated x1

bash run_train_sequence.sh "$MODE" \
  "Options/ISB_ecaformer_full_s19_r25a_gt192.yml" \
  "Options/ISB_ecaformer_full_s19_r25b_nfeat48.yml" \
  "Options/ISB_ecaformer_full_s19_r25c_resscale08_bias.yml"
