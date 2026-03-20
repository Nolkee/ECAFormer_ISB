#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-single}"

cd ~/Projects/ECAFormer_ISB
bash run_train_sequence.sh "$MODE" Options/ECAFormer_stageA_reference.yml
