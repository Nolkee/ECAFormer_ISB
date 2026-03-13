#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <single|multi> <config1.yml> [config2.yml ...]"
  echo "Example(single): $0 single Options/ECAFormer_baseline_fair.yml Options/ISB_ecaformer_full.yml"
  echo "Example(multi):  $0 multi Options/ECAFormer_baseline_fair.yml Options/ISB_ecaformer_full.yml"
  echo "Environment vars for multi: GPU_IDS=0,1,2,3 MASTER_PORT_BASE=4321"
  exit 1
fi

MODE="$1"
shift
CONFIGS=("$@")

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-4321}"

run_single() {
  local cfg="$1"
  echo "[RUN][single] $cfg"
  PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" python -m basicsr.train --opt "$cfg"
}

run_multi() {
  local cfg="$1"
  local idx="$2"
  local port=$((MASTER_PORT_BASE + idx))
  echo "[RUN][multi] $cfg (GPU_IDS=${GPU_IDS}, PORT=${port})"
  bash train_multigpu.sh "$cfg" "$GPU_IDS" "$port"
}

for i in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$i]}"
  if [[ ! -f "$cfg" ]]; then
    echo "[ERROR] Config not found: $cfg"
    exit 1
  fi

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

done

echo "[DONE] All configs finished sequentially."
