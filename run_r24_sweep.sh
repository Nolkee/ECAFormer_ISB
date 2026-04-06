#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODE="${1:-single}"

# Round 24 sweep: fix desaturation via data + architecture
# r24a: no mixup                — pure GT colors, no blending
# r24b: mapping bias=true       — Conv can shift output mean away from gray
# r24c: no mixup + mapping bias — both fixes combined

bash run_train_sequence.sh "$MODE" \
  "Options/ISB_ecaformer_full_s19_r24a_no_mixup.yml" \
  "Options/ISB_ecaformer_full_s19_r24b_mapping_bias.yml" \
  "Options/ISB_ecaformer_full_s19_r24c_no_mixup_bias.yml"
