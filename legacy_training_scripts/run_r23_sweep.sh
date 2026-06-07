#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODE="${1:-single}"

# Round 23 sweep: fix desaturation without losing r20d's PSNR/SSIM gains
# r23a: GroupNorm(2 groups) — less cross-channel mixing
# r23b: post-mapping norm on 3ch RGB — norm doesn't touch feature space
# r23c: loss_on_clamped_output=true — single variable isolation from r22d

bash run_train_sequence.sh "$MODE" \
  "Options/ISB_ecaformer_full_s19_r23a_group2.yml" \
  "Options/ISB_ecaformer_full_s19_r23b_postnorm.yml" \
  "Options/ISB_ecaformer_full_s19_r23c_clamped_only.yml"
