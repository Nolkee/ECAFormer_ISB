#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-single}"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

bash run_train_sequence.sh "$MODE" \
  Options/ISB_ecaformer_full_s19_r10_a_x1noclamp.yml \
  Options/ISB_ecaformer_full_s19_r10_b_noillusigmoid.yml \
  Options/ISB_ecaformer_full_s19_r10_c_noillusigmoid_x1noclamp.yml
