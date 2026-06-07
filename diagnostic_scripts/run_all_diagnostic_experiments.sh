#!/bin/bash
# 批量训练脚本 - 按顺序运行所有诊断实验
# Usage: bash run_all_diagnostic_experiments.sh

set -e

echo "========================================"
echo "诊断实验批量训练脚本"
echo "========================================"
echo ""
echo "实验序列:"
echo "  1. r43a_warmup (关键测试)"
echo "  2. r43_hybrid (备选方案1)"
echo "  3. r43a_learnable (备选方案2)"
echo ""
echo "预计总时间: ~6-7天 (单GPU)"
echo ""

# 确认是否继续
read -p "是否开始批量训练? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="diagnostic_logs_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "日志将保存到: $LOG_DIR"
echo ""

# 实验1: Warmup (24K iter)
echo "========================================"
echo "[1/3] 开始训练 r43a_warmup"
echo "========================================"
python -m basicsr.train --opt Options/ISB_ecaformer_r43a_warmup.yml 2>&1 | tee "$LOG_DIR/train_r43a_warmup.log"

echo ""
echo "✓ r43a_warmup 训练完成"
echo ""

# 检查PSNR结果
echo "提取 r43a_warmup 最佳PSNR..."
grep "Best metric" "$LOG_DIR/train_r43a_warmup.log" | tail -5
echo ""

# 询问是否继续
read -p "r43a_warmup 是否成功? 是否继续运行其他实验? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "训练终止。如果r43a_warmup成功，可能不需要运行其他实验。"
    exit 0
fi

# 实验2: Hybrid (24K iter)
echo "========================================"
echo "[2/3] 开始训练 r43_hybrid"
echo "========================================"
python -m basicsr.train --opt Options/ISB_ecaformer_r43_hybrid.yml 2>&1 | tee "$LOG_DIR/train_r43_hybrid.log"

echo ""
echo "✓ r43_hybrid 训练完成"
echo ""

# 提取结果
echo "提取 r43_hybrid 最佳PSNR..."
grep "Best metric" "$LOG_DIR/train_r43_hybrid.log" | tail -5
echo ""

# 实验3: Learnable (24K iter)
echo "========================================"
echo "[3/3] 开始训练 r43a_learnable"
echo "========================================"
python -m basicsr.train --opt Options/ISB_ecaformer_r43a_learnable.yml 2>&1 | tee "$LOG_DIR/train_r43a_learnable.log"

echo ""
echo "✓ r43a_learnable 训练完成"
echo ""

# 提取结果
echo "提取 r43a_learnable 最佳PSNR..."
grep "Best metric" "$LOG_DIR/train_r43a_learnable.log" | tail -5
echo ""

# 总结
echo "========================================"
echo "所有实验完成！"
echo "========================================"
echo ""
echo "结果总结:"
echo "----------------------------------------"
for exp in r43a_warmup r43_hybrid r43a_learnable; do
    echo ""
    echo "[$exp]"
    if [ -f "$LOG_DIR/train_${exp}.log" ]; then
        grep "Best metric" "$LOG_DIR/train_${exp}.log" | tail -1
    fi
done
echo ""
echo "详细日志: $LOG_DIR/"
echo ""
echo "下一步:"
echo "  1. 运行 checkpoint 分析: python tools/diagnose_checkpoint.py"
echo "  2. 对比训练曲线: tensorboard --logdir experiments"
echo "  3. 根据决策树判断根因"
