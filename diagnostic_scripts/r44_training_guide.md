# R44 Training Guide

## 配置文件说明

基于多智能体分析结果，创建了两个新配置：

### 1. ISB_ecaformer_r43_hybrid_earlystop.yml
**快速验证版** - 仅修复过训练问题

**核心改动**：
- `early_stop_patience_val: 8 → 4`（2000 iters容忍期）

**目标**：
- 验证"过训练假设" - 在12.5K停止而非14.5K
- 保住r43_hybrid的峰值22.09 PSNR
- 节省15-20%训练时间（~6-10小时）

**启动命令**：
```bash
python -m basicsr.train --opt Options/ISB_ecaformer_r43_hybrid_earlystop.yml
```

**预期时间线**：
- 3500-6000 iter: PSNR降至16.7（接受这个不稳定性）
- 6000-11000 iter: 恢复并攀升至22.09
- 11000 iter: 达到峰值
- ~12500 iter: 早停触发（vs 原版14500 iter）

---

### 2. ISB_ecaformer_r44_optimized.yml
**完整优化版** - 多维度改进

**关键创新**：

#### A. 架构容量提升
```yaml
level: 3                    # 2→3 (添加640维瓶颈层)
num_blocks: [2, 4, 6, 6]    # [1,2,2]→[2,4,6,6] (约2倍DMSA块)
```
**预期收益**：+0.4-0.6 PSNR

#### B. "可控混乱"策略
```yaml
identity_scale_init: [1.0, 0.95, 1.0]  # 比r43的0.96温和
residual_scale_init: [0.6, 0.5, 0.6]   # 保持强下游校正
learnable_identity_scale: false         # 固定→制造有益梯度冲突
```
**理论依据**：固定双阶段校正在3500-6000制造梯度张力，迫使网络逃离尖锐局部最小值（类似SAM优化器效应）

#### C. 自适应训练调度
```yaml
total_iter: 18000                    # 24000→18000 (避免过训练)
scheduler:
  periods: [6000, 12000]             # 在6K重启LR（不稳定后）
early_stop_patience_val: 4           # 8→4
grad_clip_value: 0.015               # 0.02→0.015 (更紧)
weight_decay: 0.08                   # 0.05→0.08 (更强正则化)
```

#### D. 渐进式损失重平衡
```yaml
perceptual_loss_weight: 0.15         # 0.1→0.15 (更强特征引导)
fft_loss_weight: 0.05                # 0.0→0.05 (启用频域损失)
```

**启动命令**：
```bash
python -m basicsr.train --opt Options/ISB_ecaformer_r44_optimized.yml
```

**预期结果**：
- PSNR: **22.5-22.8** @ 10-12K iterations
- 训练时间：30-36小时（单GPU）
- 内存占用：~增加20%（因为level=3）

---

## 训练策略对比

| 维度 | R42a (基线) | R43_hybrid | R43_hybrid_earlystop | R44_optimized |
|------|------------|-----------|---------------------|---------------|
| **峰值PSNR** | 21.64 | 22.09 | 22.09 (预期) | **22.5-22.8** (预期) |
| **训练稳定性** | ✅ 平滑 | ❌ 3500-6000崩盘 | ❌ 同左 | ⚠️ 可控不稳定 |
| **训练时间** | 48h | 48h | **36-40h** | **30-36h** |
| **架构容量** | level=2, [1,2,2] | 同左 | 同左 | **level=3, [2,4,6,6]** |
| **过训练问题** | 有 | 严重(-0.51) | **已修复** | **已修复** |
| **策略** | 仅下游校正 | 双阶段固定 | 同左+早停 | 双阶段+深架构+多损失 |

---

## 监控指标

### 关键检查点

**500 iter**：
- PSNR应在10.2-10.3
- SSIM ~0.042
- LPIPS ~0.83

**3500 iter**（不稳定开始）：
- PSNR开始下降（19.8 → 17-18区间）
- 这是**预期行为**，不要panic
- 检查`nan_guard`是否触发（应该不会）

**6000 iter**（恢复开始）：
- PSNR应回升至17-18
- AdaLN层的scale/shift参数方差会减小

**11000 iter**（峰值窗口）：
- R43_hybrid_earlystop: 应达到22.09
- R44_optimized: 应达到22.5+

**12500 iter**：
- 早停应触发（patience=4, 每500 iter验证 = 4次）

### 实时监控命令

```bash
# 监控训练日志
tail -f experiments/ISB_ecaformer_r44_optimized/train_ISB_ecaformer_r44_optimized_*.log | grep "psnr"

# 监控GPU显存
watch -n 1 nvidia-smi

# TensorBoard（如果启用）
tensorboard --logdir experiments/ISB_ecaformer_r44_optimized/tb_logger/
```

---

## 异常情况处理

### 问题1：内存不足（R44因level=3可能触发）

**症状**：`CUDA out of memory`

**解决方案A**：减小batch size
```yaml
batch_size_per_gpu: 24 → 16  # 或12
```

**解决方案B**：启用更激进的checkpointing（已启用）
```yaml
use_checkpoint: true  # 已设置
```

**解决方案C**：回退到level=2
```yaml
level: 3 → 2
num_blocks: [2, 4, 6] → [2, 4, 6]  # 移除最后一层
```

### 问题2：不稳定期NaN

**症状**：`nan_guard`在3500-6000频繁触发

**解决方案**：
```yaml
grad_clip_value: 0.015 → 0.01  # 更紧梯度裁剪
weight_decay: 0.08 → 0.10      # 更强正则化
```

### 问题3：PSNR不收敛

**症状**：10K后仍低于21.5

**诊断步骤**：
1. 检查数据路径是否正确
2. 检查learning rate是否正常衰减
3. 对比R42a基线（应该稳定达到21.64）

---

## 实验时间线

### 第1天（快速验证）
```bash
# 上午：启动早停修复版
python -m basicsr.train --opt Options/ISB_ecaformer_r43_hybrid_earlystop.yml

# 下午：监控到6K iter（确认不稳定期通过）
# 预期：明天上午12.5K早停
```

### 第2天（启动R44）
```bash
# 上午：检查早停版结果，启动R44
python -m basicsr.train --opt Options/ISB_ecaformer_r44_optimized.yml

# 预期：48小时后完成（第4天上午）
```

### 第4天（结果分析）
- 对比R43_hybrid_earlystop vs R44_optimized
- 如果R44达到22.5+，进入下一阶段（架构深度改进）
- 如果增益不足，调整损失权重

---

## 下一步架构改进（如果R44成功）

基于智能体分析的"架构瓶颈"报告，按优先级：

### Phase 1：增强特征提取器（预期+0.3-0.5 PSNR）
替换ShallowDeepConv为PyramidEstimator（多尺度金字塔）

### Phase 2：多尺度DMSA（预期+0.3-0.5 PSNR）
在DMSA块中添加[1, 2, 4]尺度的注意力

### Phase 3：可学习桥接路径（预期+0.2-0.4 PSNR）
将线性插值`(1-t)·x0 + t·x1`替换为三次样条控制点

**累积潜力**：23.5+ PSNR

---

## 引用分析报告

详细理论依据见：
- `diagnostic_scripts/OVERTRAINING_ANALYSIS.md` - 过训练根因分析
- 智能体报告"instability-paradox" - 有益不稳定性机制
- 智能体报告"architecture-bottleneck" - 架构天花板突破方案
