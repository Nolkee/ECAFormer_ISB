#!/bin/bash
# 连续训练脚本 - Round 11 & 12

set -e

# 激活conda环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate RetInexformer

echo "=========================================="
echo "开始连续训练实验"
echo "=========================================="

# Round 11: 中等激进方案
echo ""
echo "[1/2] 启动 Round 11 训练..."
echo "配置: x0=0.55, lr=1e-4, iter=16000"
python basicsr/train.py -opt Options/ISB_ecaformer_full_s19_r11_x0boost.yml

echo ""
echo "Round 11 完成！"
echo ""

# Round 12: 激进方案
echo "[2/2] 启动 Round 12 训练..."
echo "配置: x0=0.7, lr=1.2e-4, iter=16000"
python basicsr/train.py -opt Options/ISB_ecaformer_full_s19_r12_aggressive.yml

echo ""
echo "=========================================="
echo "所有训练完成！"
echo "=========================================="
