#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

MODE="${1:-single}"

CONFIGS=(
  "Options/ISB_ecaformer_full_s19_r18a_w035_p065_lr6e5.yml"
  "Options/ISB_ecaformer_full_s19_r18b_w040_p060_lr6e5.yml"
  "Options/ISB_ecaformer_full_s19_r18c_w045_p055_lr6e5.yml"
  "Options/ISB_ecaformer_full_s19_r18d_w040_p060_lr5e5.yml"
)

run_single() {
  local cfg="$1"
  echo "[RUN][single] $cfg"
  PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" python -m basicsr.train --opt "$cfg"
}

run_multi() {
  local cfg="$1"
  local idx="$2"
  local gpu_ids="${GPU_IDS:-0,1,2,3}"
  local port_base="${MASTER_PORT_BASE:-4321}"
  local port=$((port_base + idx))
  echo "[RUN][multi] $cfg (GPU_IDS=${gpu_ids}, PORT=${port})"
  bash train_multigpu.sh "$cfg" "$gpu_ids" "$port"
}

echo "=========================================="
echo "启动 Round 18 focused sweep"
echo "MODE: $MODE"
echo "=========================================="

for i in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$i]}"
  if [[ ! -f "$cfg" ]]; then
    echo "[ERROR] Config not found: $cfg"
    exit 1
  fi

  echo ""
  echo "[$((i + 1))/${#CONFIGS[@]}] 启动 $(basename "$cfg" .yml)..."

  case "$MODE" in
    single)
      run_single "$cfg"
      ;;
    multi)
      run_multi "$cfg" "$i"
      ;;
    *)
      echo "[ERROR] MODE must be 'single' or 'multi', got: $MODE"
      exit 1
      ;;
  esac

  echo "[DONE] $(basename "$cfg")"
done

echo ""
echo "=========================================="
echo "Round 18 focused sweep 所有训练完成！"
echo "=========================================="
