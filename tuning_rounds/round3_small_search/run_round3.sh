#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-single}"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
bash run_train_sequence.sh "$MODE" \
  tuning_rounds/round3_small_search/configs/ISB_ecaformer_screen_S10.yml \
  tuning_rounds/round3_small_search/configs/ISB_ecaformer_screen_S11.yml \
  tuning_rounds/round3_small_search/configs/ISB_ecaformer_screen_S12.yml \
  tuning_rounds/round3_small_search/configs/ISB_ecaformer_screen_S13.yml \
  tuning_rounds/round3_small_search/configs/ISB_ecaformer_screen_S14.yml \
  tuning_rounds/round3_small_search/configs/ISB_ecaformer_screen_S15.yml \
  tuning_rounds/round3_small_search/configs/ISB_ecaformer_screen_S16.yml
