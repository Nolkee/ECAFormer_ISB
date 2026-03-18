#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-single}"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

bash run_train_sequence.sh "$MODE" \
  Options/ISB_ecaformer_full_s19_r7_a_sigma007.yml \
  Options/ISB_ecaformer_full_s19_r7_b_sigma007_charb.yml \
  Options/ISB_ecaformer_full_s19_r7_c_sigma007_charb_x055.yml
