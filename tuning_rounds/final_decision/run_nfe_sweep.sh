#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-single}"
OPT_PATH="${2:-Options/ISB_ecaformer_full_s19_r6_c_identity_unclamp_det_wd5e4.yml}"
CKPT_PATH="${3:-experiments/ISB_ecaformer_full_s19_r6_c_identity_unclamp_det_wd5e4/best_psnr_20.11_8500.pth}"
OUT_CSV="${4:-tuning_rounds/final_decision/results_r6c_nfe_sweep.csv}"

if [[ "$MODE" != "single" ]]; then
  echo "Only single mode is supported for NFE sweep."
  echo "Usage: bash tuning_rounds/final_decision/run_nfe_sweep.sh single [opt] [checkpoint] [out_csv]"
  exit 1
fi

python3 tuning_rounds/final_decision/sweep_nfe.py \
  --opt "$OPT_PATH" \
  --checkpoint "$CKPT_PATH" \
  --nfe 4 6 8 12 \
  --tag "ISB_ecaformer_full_s19_r6_c_nfe_sweep" \
  --out-csv "$OUT_CSV"
