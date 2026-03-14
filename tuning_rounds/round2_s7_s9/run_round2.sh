#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-single}"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
bash run_train_sequence.sh "$MODE" \
  tuning_rounds/round2_s7_s9/configs/ISB_ecaformer_screen_S7.yml \
  tuning_rounds/round2_s7_s9/configs/ISB_ecaformer_screen_S8.yml \
  tuning_rounds/round2_s7_s9/configs/ISB_ecaformer_screen_S9.yml
