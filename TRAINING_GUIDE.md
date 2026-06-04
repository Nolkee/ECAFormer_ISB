# ECAFormer-ISB Training Guide (tmux)

## Quick Start (推荐)

```bash
# 在 tmux 中启动顺序训练 R42a → R43 系列
tmux new -s train
bash train_r42a_then_r43.sh

# 分离会话 (Ctrl+B, 然后按 D)
# 或直接运行：tmux new -s train "bash train_r42a_then_r43.sh"

# 重新连接查看进度
tmux attach -t train

# 查看实时日志
tail -f logs/train_ISB_ecaformer_r43a_identity_scale_*.log
```

## 训练流程

### Phase 1: R42a 验证 (~12 小时)
- 配置: `Options/ISB_ecaformer_r42a_per_ch_res.yml`
- 目标: 验证 per-channel residual_scale 是否有效
- 预期: PSNR 22.0-22.2

### Phase 2: R43 系列 (~36 小时)
1. **R43a** (identity_scale=[1.0, 0.92, 1.0])
   - 最有希望的配置
   - 预期: PSNR 22.3-22.5

2. **R43b** (identity_scale=[1.0, 0.90, 1.0] + per-channel residual)
   - 双重绿色抑制
   - 预期: 绿色最少，PSNR 可能略低

3. **R43c** (R43a + green_loss)
   - 三重防御
   - 预期: 平衡绿色和 PSNR

## tmux 常用命令

```bash
# 创建新会话
tmux new -s train

# 列出所有会话
tmux ls

# 连接到会话
tmux attach -t train

# 分离会话 (在 tmux 内)
Ctrl+B, D

# 杀死会话 (完成后清理)
tmux kill-session -t train

# 查看会话中的窗口
Ctrl+B, W

# 滚动查看历史 (在 tmux 内)
Ctrl+B, [    # 然后用 ↑↓ 或 PgUp/PgDn
# 按 q 退出滚动模式
```

## 监控训练

### 方式 1: 实时日志
```bash
# 在另一个终端监控最新日志
tail -f logs/train_ISB_ecaformer_r43a_identity_scale_*.log

# 或使用 tmux 分屏
Ctrl+B, %    # 垂直分屏
Ctrl+B, "    # 水平分屏
Ctrl+B, 方向键  # 切换面板
```

### 方式 2: TensorBoard
```bash
# 在新 tmux 窗口启动 TensorBoard
tmux new-window -t train -n tensorboard
tensorboard --logdir experiments --port 6006 --bind_all

# 浏览器访问: http://localhost:6006
```

### 方式 3: 快速检查
```bash
# 查看最新 PSNR
grep -h "Best metric" logs/*.log | tail -5

# 查看当前迭代
grep -h "iter:" logs/*.log | tail -1

# 检查 GPU 使用
watch -n 5 nvidia-smi
```

## 训练完成后

```bash
# 比较所有配置的最佳结果
bash train_r42a_then_r43.sh  # 会自动在最后输出总结

# 或手动提取
for log in logs/train_*.log; do
    echo "=== $(basename $log) ==="
    grep "Best metric" "$log" | tail -1
done
```

## 故障处理

### 训练中断
```bash
# 检查最后的 checkpoint
ls -lth experiments/ISB_ecaformer_r43a_identity_scale/models/

# 从 checkpoint 恢复 (修改配置)
# 在 .yml 中设置: resume_state: experiments/.../training_states/XXX.state
python -m basicsr.train --opt Options/ISB_ecaformer_r43a_identity_scale.yml
```

### OOM (显存不足)
```bash
# 减小 batch_size (在配置文件中)
batch_size_per_gpu: 24 → 16 或 12

# 或减少 accumulate_steps
accumulate_steps: 1 → 2  # 等效 batch size 翻倍，显存减半
```

### NaN Loss
```bash
# 检查日志中的 NaN 警告
grep -i "nan\|inf" logs/*.log

# 如果频繁出现，降低学习率
lr: 6e-5 → 3e-5
```

## 时间估算

| 配置 | 迭代数 | 单 GPU (V100) | 双 GPU |
|------|--------|--------------|--------|
| R42a | 24K | ~12h | ~6h |
| R43a | 24K | ~12h | ~6h |
| R43b | 24K | ~12h | ~6h |
| R43c | 24K | ~12h | ~6h |
| **总计** | | **~48h** | **~24h** |

## 并行训练 (如果有多 GPU)

```bash
# 编辑 train_r42a_then_r43.sh，在 Phase 2 改为:
CUDA_VISIBLE_DEVICES=0 train_with_log "${R43_CONFIGS[0]}" &
CUDA_VISIBLE_DEVICES=1 train_with_log "${R43_CONFIGS[1]}" &
wait
```

或使用专用脚本:
```bash
bash auto_train_all.sh parallel  # R42a + R43a 并行 (需 2 GPU)
```
