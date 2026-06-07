#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODE="${1:-single}"

# Round 26 sweep: target remaining saturation gap via illumination/x1 path
# r26a: illumination_map_activation=identity  — remove illumination sigmoid compression
# r26b: pre_denoiser_x1_clamp=false          — keep unclipped x1 before denoiser
# r26c: both above                            — least-constrained illumination path

bash run_train_sequence.sh "$MODE" \
  "Options/ISB_ecaformer_full_s19_r26a_noillusigmoid.yml" \
  "Options/ISB_ecaformer_full_s19_r26b_x1noclamp.yml" \
  "Options/ISB_ecaformer_full_s19_r26c_noillusigmoid_x1noclamp.yml"
