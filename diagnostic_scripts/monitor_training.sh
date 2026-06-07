#!/bin/bash
# 监控训练进度脚本
# Usage: bash monitor_training.sh

echo "========================================"
echo "训练监控工具"
echo "========================================"
echo ""

# 查找最近的实验目录
EXPERIMENTS=$(ls -dt experiments/ISB_ecaformer_r43* 2>/dev/null | head -3)

if [ -z "$EXPERIMENTS" ]; then
    echo "未找到正在运行的实验"
    exit 1
fi

echo "正在监控的实验:"
echo "$EXPERIMENTS"
echo ""

# 函数: 提取最新PSNR
get_latest_psnr() {
    local exp_dir=$1
    local log_file="$exp_dir/train.log"
    if [ -f "$log_file" ]; then
        grep "Best metric" "$log_file" | tail -1 | awk '{print $NF}'
    else
        echo "N/A"
    fi
}

# 函数: 获取当前iter
get_current_iter() {
    local exp_dir=$1
    local log_file="$exp_dir/train.log"
    if [ -f "$log_file" ]; then
        grep -oP 'Iter:\s*\K\d+' "$log_file" | tail -1
    else
        echo "0"
    fi
}

# 循环监控
while true; do
    clear
    echo "========================================"
    echo "训练监控 - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    echo ""

    for exp in $EXPERIMENTS; do
        exp_name=$(basename "$exp")
        current_iter=$(get_current_iter "$exp")
        latest_psnr=$(get_latest_psnr "$exp")

        echo "[$exp_name]"
        echo "  当前 iter: $current_iter / 24000"
        echo "  最佳 PSNR: $latest_psnr"

        # 检查是否在不稳定窗口
        if [ "$current_iter" -ge 3500 ] && [ "$current_iter" -le 6000 ]; then
            echo "  ⚠️  正在通过不稳定窗口 (3500-6000)"
        fi

        # 检查训练进度
        if [ -f "$exp/train.log" ]; then
            tail -3 "$exp/train.log" | grep -E "(Iter:|Best metric)" | head -2
        fi
        echo ""
    done

    echo "----------------------------------------"
    echo "按 Ctrl+C 退出监控"
    echo "刷新间隔: 30秒"

    sleep 30
done
