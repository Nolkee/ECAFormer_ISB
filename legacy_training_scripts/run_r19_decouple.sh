#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODE="${1:-single}"

bash run_train_sequence.sh "$MODE" \
  "Options/ISB_ecaformer_full_s19_r18b_w040_p060_lr6e5.yml" \
  "Options/ISB_ecaformer_full_s19_r19a_decouple_bridge.yml"
