# 诊断实施完成报告

## 已完成的工作

### 1. 代码修改

#### ✅ ECAFormer_ISB_arch.py (identity_scale warmup)
- **位置**: `basicsr/models/archs/ECAFormer_ISB_arch.py`
- **修改**:
  - Line 656-675: 添加 `identity_scale_warmup_iters` 参数支持
  - 创建 `identity_scale_start` 和 `identity_scale_target` buffers
  - Line 758-768: 在forward中实现warmup逻辑
  - 公式: `current_scale = (1-α) * [1,1,1] + α * [1,0.92,1]`, α = min(1, iter/5000)

#### ✅ image_isb_model.py (current_iter传递)
- **位置**: `basicsr/models/image_isb_model.py`
- **修改**: Line 291添加 `self.net_g._current_iter = current_iter`
- **作用**: 让网络知道当前训练iteration，用于warmup计算

### 2. 配置文件

#### ✅ ISB_ecaformer_r43a_warmup.yml
- **实验目的**: 测试identity_scale warmup是否消除3500-6000不稳定
- **关键参数**:
  - `identity_scale_init: [1.0, 0.92, 1.0]`
  - `identity_scale_warmup_iters: 5000` (5K iter从[1,1,1]渐变到目标值)
  - 其他参数与r43a相同
- **预期**: 如果假设正确，PSNR曲线平滑，无震荡

#### ✅ ISB_ecaformer_r43_hybrid.yml
- **实验目的**: 测试双阶段通道校正是否协同
- **关键参数**:
  - `identity_scale_init: [1.0, 0.96, 1.0]` (温和上游压制)
  - `residual_scale_init: [0.6, 0.5, 0.6]` (r42a的下游压制)
- **预期**: 如果机制正交，PSNR > 22.3

#### ✅ ISB_ecaformer_r43a_learnable.yml
- **实验目的**: 判断learnable identity_scale是否能自适应解决冲突
- **关键参数**:
  - `identity_scale_init: [1.0, 0.92, 1.0]`
  - `learnable_identity_scale: true`
- **诊断点**: 监控identity_scale.grad在3500-6000的magnitude

### 3. 诊断工具

#### ✅ tools/diagnose_checkpoint.py
- **功能**:
  - 从checkpoint提取AdaLN参数norm evolution
  - 提取channel scale参数历史
  - 可视化3500-6000窗口的参数divergence
- **用法**:
  ```bash
  # 单实验分析
  python tools/diagnose_checkpoint.py \
    --exp_dir experiments/ISB_ecaformer_r43a_identity_scale \
    --iters 1000 3000 5000 7000 10000 \
    --output diagnosis_r43a.png
  
  # 对比分析
  python tools/diagnose_checkpoint.py \
    --exp_dirs experiments/ISB_ecaformer_r42a_per_ch_res \
                 experiments/ISB_ecaformer_r43a_identity_scale \
    --labels "r42a(stable)" "r43a(unstable)" \
    --output r42a_vs_r43a.png
  ```

#### ✅ quick_test_warmup.sh
- **功能**: 快速8K iter测试warmup效果
- **用法**: `bash quick_test_warmup.sh`
- **输出**: `train_warmup_8k.log`

## 下一步行动

### 阶段1：快速验证 (立即执行)

```bash
# 1. 运行8K快速测试
bash quick_test_warmup.sh

# 2. 观察PSNR @ 5K-8K iter是否稳定
tail -f train_warmup_8k.log | grep "Best metric"

# 3. 查看TensorBoard
tensorboard --logdir experiments/ISB_ecaformer_r43a_warmup_8k/tb_logger --port 6006
```

**判断标准**:
- ✅ 成功: PSNR在5K-8K持续上升，无回落 → 延长至24K
- ❌ 失败: 仍有震荡 → 尝试实验2/3

### 阶段2：完整训练 (如果快速测试成功)

```bash
# Warmup方案完整训练
python -m basicsr.train --opt Options/ISB_ecaformer_r43a_warmup.yml
```

**成功标准**:
- PSNR @ 10K > 22.0 (超过r42a的21.64)
- 无3500-6000窗口的PSNR回落
- 最终PSNR > 22.3

### 阶段3：诊断分析 (训练后)

```bash
# 分析checkpoint演化
python tools/diagnose_checkpoint.py \
  --exp_dirs experiments/ISB_ecaformer_r42a_per_ch_res \
               experiments/ISB_ecaformer_r43a_warmup \
  --labels "r42a(stable)" "r43a_warmup(fixed)" \
  --iters 1000 3000 5000 7000 10000 \
  --output warmup_fix_verification.png
```

**预期结果**:
- r43a_warmup的AdaLN norm growth与r42a相近
- 不出现r43a原版的AdaLN norm突然跳变

## 根因决策树

根据实验结果，按以下逻辑定位根因：

```
Q1: warmup是否消除不稳定？
├─ 是 → 根因 = 早期x1扰动与AdaLN初始化冲突
│         解决方案: 默认启用warmup
├─ 否 → 进入Q2

Q2: hybrid是否稳定？
├─ 是 → 根因 = identity_scale力度过猛
│         解决方案: 温和的identity_scale + residual_scale
├─ 否 → 进入Q3

Q3: learnable的gradient在3500-6000是否飙升？
├─ 是 → 根因 = 优化器与损失冲突
│         解决方案: 固定identity_scale或独立优化器
├─ 否 → 其他原因（LR schedule/数据分布/损失组件）
```

## 关键文件清单

### 修改的文件
- ✅ `basicsr/models/archs/ECAFormer_ISB_arch.py`
- ✅ `basicsr/models/image_isb_model.py`

### 新增的文件
- ✅ `Options/ISB_ecaformer_r43a_warmup.yml`
- ✅ `Options/ISB_ecaformer_r43_hybrid.yml`
- ✅ `Options/ISB_ecaformer_r43a_learnable.yml`
- ✅ `tools/diagnose_checkpoint.py`
- ✅ `quick_test_warmup.sh`
- ✅ `DIAGNOSTIC_README.md` (本文件)

## 预期诊断信号

### 如果假设正确（AdaLN冲突）:

**Checkpoint分析显示**:
- AdaLN norm在1K→5K增长2-3倍（r43a原版）
- r42a的AdaLN norm稳定增长
- r43a_warmup的AdaLN norm与r42a相似

**训练曲线**:
- r43a原版: PSNR在3500-6000震荡
- r43a_warmup: PSNR平滑上升

### 如果假设错误:

**其他可能原因**:
- LR schedule: 改变schedule改变不稳定时机
- 数据分布: 改变seed改变不稳定位置
- 损失组件: color/chroma loss干扰

## 实施时间估算

- ✅ 第1天: 代码修改和配置 (已完成)
- 第2天: 快速8K测试 (2-3小时GPU时间)
- 第3-5天: 完整24K训练 (如果测试成功，~48小时GPU)
- 第6天: Checkpoint分析和结果总结
