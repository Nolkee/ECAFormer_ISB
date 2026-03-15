#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-single}"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

bash run_train_sequence.sh "$MODE" \
  tuning_rounds/round4_three_groups/configs/ISB_ecaformer_screen_S17.yml \
  tuning_rounds/round4_three_groups/configs/ISB_ecaformer_screen_S18.yml \
  tuning_rounds/round4_three_groups/configs/ISB_ecaformer_screen_S19.yml
