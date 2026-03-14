#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-single}"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
bash run_train_sequence.sh "$MODE" \
  tuning_rounds/round1_s1_s6/configs/ISB_ecaformer_screen_S1.yml \
  tuning_rounds/round1_s1_s6/configs/ISB_ecaformer_screen_S2.yml \
  tuning_rounds/round1_s1_s6/configs/ISB_ecaformer_screen_S3.yml \
  tuning_rounds/round1_s1_s6/configs/ISB_ecaformer_screen_S4.yml \
  tuning_rounds/round1_s1_s6/configs/ISB_ecaformer_screen_S5.yml \
  tuning_rounds/round1_s1_s6/configs/ISB_ecaformer_screen_S6.yml
