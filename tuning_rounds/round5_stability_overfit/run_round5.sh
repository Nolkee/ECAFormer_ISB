#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-single}"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

bash run_train_sequence.sh "$MODE" \
  Options/ISB_ecaformer_full_s19_r5_a_wd3e4_tvlow.yml \
  Options/ISB_ecaformer_full_s19_r5_b_identity_softloss.yml \
  Options/ISB_ecaformer_full_s19_r5_c_noamp.yml
