#!/bin/bash
# One-click automatic training launcher for all R42/R43 experiments
#
# Usage:
#   bash auto_train_all.sh          # Run all experiments sequentially
#   bash auto_train_all.sh r43      # Run only R43 series
#   bash auto_train_all.sh r42      # Run only R42 series
#   bash auto_train_all.sh parallel # Run R42a and R43a in parallel (requires 2 GPUs)

set -e

R42_CONFIGS=(
  "Options/ISB_ecaformer_r42a_per_ch_res.yml"
  "Options/ISB_ecaformer_r42b_green_norm.yml"
  "Options/ISB_ecaformer_r42c_green_loss.yml"
  "Options/ISB_ecaformer_r42d_ch_permute.yml"
  "Options/ISB_ecaformer_r42e_all_green_fixes.yml"
)

R43_CONFIGS=(
  "Options/ISB_ecaformer_r43a_identity_scale.yml"
  "Options/ISB_ecaformer_r43b_double_suppress.yml"
  "Options/ISB_ecaformer_r43c_identity_green_loss.yml"
)

function train_config() {
    cfg=$1
    name=$(basename "$cfg" .yml)
    echo "=========================================="
    echo "Training: $name"
    echo "Config:   $cfg"
    echo "Started:  $(date)"
    echo "=========================================="

    python -m basicsr.train --opt "$cfg" 2>&1 | tee "train_${name}.log"

    echo "Finished: $name at $(date)"
    echo ""
}

function train_series() {
    series_name=$1
    shift
    configs=("$@")

    echo "=========================================="
    echo "Starting $series_name series"
    echo "Total configs: ${#configs[@]}"
    echo "=========================================="

    for cfg in "${configs[@]}"; do
        train_config "$cfg"
    done

    echo "=========================================="
    echo "$series_name series completed at $(date)"
    echo "=========================================="
}

function run_parallel() {
    echo "=========================================="
    echo "Parallel training: R42a + R43a"
    echo "Requires: 2 GPUs (CUDA_VISIBLE_DEVICES will be set automatically)"
    echo "=========================================="

    # R42a on GPU 0
    CUDA_VISIBLE_DEVICES=0 python -m basicsr.train --opt Options/ISB_ecaformer_r42a_per_ch_res.yml \
        2>&1 | tee train_r42a.log &
    pid_r42a=$!

    # R43a on GPU 1
    CUDA_VISIBLE_DEVICES=1 python -m basicsr.train --opt Options/ISB_ecaformer_r43a_identity_scale.yml \
        2>&1 | tee train_r43a.log &
    pid_r43a=$!

    echo "R42a (PID: $pid_r42a) on GPU 0"
    echo "R43a (PID: $pid_r43a) on GPU 1"
    echo "Waiting for both to complete..."

    wait $pid_r42a
    echo "R42a completed at $(date)"

    wait $pid_r43a
    echo "R43a completed at $(date)"

    echo "=========================================="
    echo "Parallel training completed."
    echo "=========================================="
}

# Main logic
MODE="${1:-all}"

case "$MODE" in
    r42)
        train_series "R42" "${R42_CONFIGS[@]}"
        ;;
    r43)
        train_series "R43" "${R43_CONFIGS[@]}"
        ;;
    parallel)
        run_parallel
        ;;
    all)
        echo "=========================================="
        echo "Auto-training: R42 + R43 series"
        echo "Mode: Sequential"
        echo "Started: $(date)"
        echo "=========================================="
        train_series "R42" "${R42_CONFIGS[@]}"
        train_series "R43" "${R43_CONFIGS[@]}"
        echo "=========================================="
        echo "All experiments completed at $(date)"
        echo "=========================================="
        ;;
    *)
        echo "Usage: bash auto_train_all.sh [all|r42|r43|parallel]"
        echo "  all      - Run all R42 + R43 experiments sequentially (default)"
        echo "  r42      - Run only R42 series (5 configs)"
        echo "  r43      - Run only R43 series (3 configs)"
        echo "  parallel - Run R42a + R43a in parallel (requires 2 GPUs)"
        exit 1
        ;;
esac
