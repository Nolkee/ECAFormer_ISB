#!/bin/bash
# Sequential training: R42a (validation) → R43 series (main experiments)
#
# Usage in tmux:
#   tmux new -s train
#   bash train_r42a_then_r43.sh
#   # Ctrl+B, D to detach
#   # tmux attach -t train to reattach

set -e

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

function log_section() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "Time: $(date)"
    echo "=========================================="
}

function train_with_log() {
    config=$1
    name=$(basename "$config" .yml)
    log_file="${LOG_DIR}/train_${name}_${TIMESTAMP}.log"

    log_section "Training: $name"
    echo "Config: $config"
    echo "Log: $log_file"

    # Train and tee to both stdout and log file
    python -m basicsr.train --opt "$config" 2>&1 | tee "$log_file"

    # Extract best metrics
    if [ -f "$log_file" ]; then
        echo ""
        echo "=========================================="
        echo "Training Summary: $name"
        echo "=========================================="
        grep -E "(Best metric|Validation)" "$log_file" | tail -5 || echo "No metrics found yet"
        echo ""
    fi
}

function check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "Warning: nvidia-smi not found. Cannot check GPU status."
        return
    fi

    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
    echo ""
}

# Main execution
log_section "R42a → R43 Sequential Training Pipeline"
echo "Session: tmux session '$(tmux display-message -p '#S' 2>/dev/null || echo 'not in tmux')'"
echo "Working directory: $(pwd)"
echo "Log directory: $LOG_DIR"
echo ""

check_gpu

# Phase 1: R42a (validation baseline)
log_section "Phase 1: R42a (Per-channel Residual Scale)"
echo "Purpose: Validate that reducing green residual helps"
echo "Expected: PSNR 22.0-22.2, improvement over R38c baseline"
echo ""
train_with_log "Options/ISB_ecaformer_r42a_per_ch_res.yml"

# Brief pause to check results
sleep 5

# Phase 2: R43 series (main experiments)
log_section "Phase 2: R43 Series (Identity Scale Root Fix)"
echo "Purpose: Test identity_scale green suppression"
echo "Expected: PSNR 22.3-22.5, SSIM 0.795+, LPIPS <0.16"
echo ""

R43_CONFIGS=(
    "Options/ISB_ecaformer_r43a_identity_scale.yml"
    "Options/ISB_ecaformer_r43b_double_suppress.yml"
    "Options/ISB_ecaformer_r43c_identity_green_loss.yml"
)

for cfg in "${R43_CONFIGS[@]}"; do
    train_with_log "$cfg"
    sleep 5
done

# Final summary
log_section "All Training Completed"
echo "Logs saved in: $LOG_DIR"
echo ""
echo "Quick comparison (Best PSNR):"
for log in ${LOG_DIR}/train_*_${TIMESTAMP}.log; do
    if [ -f "$log" ]; then
        name=$(basename "$log" .log)
        best=$(grep "Best metric" "$log" | tail -1 || echo "Not found")
        echo "$name: $best"
    fi
done
echo ""
echo "Compare to R38c baseline: PSNR 22.10, SSIM 0.788, LPIPS 0.166"
echo "=========================================="
