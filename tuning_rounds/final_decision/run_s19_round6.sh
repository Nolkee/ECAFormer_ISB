#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-single}"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

bash run_train_sequence.sh "$MODE" \
  Options/ISB_ecaformer_full_s19_r6_a_identity_unclamp_det.yml \
  Options/ISB_ecaformer_full_s19_r6_b_identity_unclamp_det_lowlr.yml \
  Options/ISB_ecaformer_full_s19_r6_c_identity_unclamp_det_wd5e4.yml
