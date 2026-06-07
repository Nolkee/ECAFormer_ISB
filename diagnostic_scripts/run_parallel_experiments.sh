#!/bin/bash
# 并行训练脚本 - 如果有多GPU可同时运行多个实验
# Usage: bash run_parallel_experiments.sh

echo "========================================"
echo "并行诊断实验训练脚本"
echo "========================================"
echo ""

# 检测可用GPU数量
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "检测到 $GPU_COUNT 个GPU"
else
    echo "警告: 无法检测GPU，假设单GPU"
    GPU_COUNT=1
fi

if [ $GPU_COUNT -lt 2 ]; then
    echo "只有1个GPU，建议使用 run_all_diagnostic_experiments.sh 顺序执行"
    echo "是否仍要在后台并行运行? (可能导致显存不足)"
    read -p "继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="diagnostic_logs_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo ""
echo "日志目录: $LOG_DIR"
echo ""

# 启动并行训练
echo "启动并行训练..."
echo "----------------------------------------"

# GPU 0: r43a_warmup (最关键)
CUDA_VISIBLE_DEVICES=0 python -m basicsr.train \
    --opt Options/ISB_ecaformer_r43a_warmup.yml \
    > "$LOG_DIR/train_r43a_warmup.log" 2>&1 &
PID1=$!
echo "✓ r43a_warmup 在GPU 0启动 (PID: $PID1)"

# 如果有多GPU，启动其他实验
if [ $GPU_COUNT -ge 2 ]; then
    CUDA_VISIBLE_DEVICES=1 python -m basicsr.train \
        --opt Options/ISB_ecaformer_r43_hybrid.yml \
        > "$LOG_DIR/train_r43_hybrid.log" 2>&1 &
    PID2=$!
    echo "✓ r43_hybrid 在GPU 1启动 (PID: $PID2)"
fi

if [ $GPU_COUNT -ge 3 ]; then
    CUDA_VISIBLE_DEVICES=2 python -m basicsr.train \
        --opt Options/ISB_ecaformer_r43a_learnable.yml \
        > "$LOG_DIR/train_r43a_learnable.log" 2>&1 &
    PID3=$!
    echo "✓ r43a_learnable 在GPU 2启动 (PID: $PID3)"
fi

echo ""
echo "所有训练已在后台启动"
echo "----------------------------------------"
echo ""
echo "监控训练进度:"
echo "  tail -f $LOG_DIR/train_r43a_warmup.log"
echo "  tail -f $LOG_DIR/train_r43_hybrid.log"
echo "  tail -f $LOG_DIR/train_r43a_learnable.log"
echo ""
echo "查看进程:"
echo "  ps aux | grep basicsr.train"
echo ""
echo "等待所有训练完成..."
echo ""

# 等待所有进程
wait $PID1
echo "✓ r43a_warmup 完成"

if [ ! -z "$PID2" ]; then
    wait $PID2
    echo "✓ r43_hybrid 完成"
fi

if [ ! -z "$PID3" ]; then
    wait $PID3
    echo "✓ r43a_learnable 完成"
fi

echo ""
echo "========================================"
echo "所有实验完成！"
echo "========================================"
echo ""
echo "结果总结:"
for log in "$LOG_DIR"/train_*.log; do
    exp_name=$(basename "$log" .log | sed 's/train_//')
    echo ""
    echo "[$exp_name]"
    grep "Best metric" "$log" | tail -1
done
