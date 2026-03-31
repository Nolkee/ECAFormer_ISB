#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

MODE="${1:-single}"
CFG="Options/ISB_ecaformer_full_s19_r17_r6c_grayillu_no_outnorm.yml"

run_single() {
  echo "[RUN][single] $CFG"
  PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" python -m basicsr.train --opt "$CFG"
}

run_multi() {
  local gpu_ids="${GPU_IDS:-0,1,2,3}"
  local port="${MASTER_PORT:-4321}"
  echo "[RUN][multi] $CFG (GPU_IDS=${gpu_ids}, PORT=${port})"
  bash train_multigpu.sh "$CFG" "$gpu_ids" "$port"
}

echo "=========================================="
echo "启动 Round 17 训练"
echo "MODE: $MODE"
echo "=========================================="

if [[ ! -f "$CFG" ]]; then
  echo "[ERROR] Config not found: $CFG"
  exit 1
fi

case "$MODE" in
  single)
    run_single
    ;;
  multi)
    run_multi
    ;;
  *)
    echo "[ERROR] MODE must be 'single' or 'multi', got: $MODE"
    exit 1
    ;;
esac

echo "[DONE] $(basename "$CFG")"
