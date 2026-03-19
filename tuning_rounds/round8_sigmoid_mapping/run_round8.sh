#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-single}"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

bash run_train_sequence.sh "$MODE" \
  Options/ISB_ecaformer_full_s19_r8_a_sigmoid_base.yml \
  Options/ISB_ecaformer_full_s19_r8_b_sigmoid_charb.yml \
  Options/ISB_ecaformer_full_s19_r8_c_sigmoid_charb_sigma007_lowlr.yml
