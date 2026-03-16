#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-single}"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

bash run_train_sequence.sh "$MODE" \
  Options/ISB_ecaformer_full_s19_adamw_wd1e4.yml \
  Options/ISB_ecaformer_full_s19_adamw_wd3e4.yml \
  Options/ISB_ecaformer_full_s19_adamw_wd1e4_lr5e5.yml
