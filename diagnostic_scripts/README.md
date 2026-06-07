# 诊断脚本集合

此文件夹包含用于诊断r43a/r38c训练不稳定性的所有脚本和工具。

## 目录结构

```
diagnostic_scripts/
├── README.md                           # 本文件
├── DIAGNOSTIC_README.md                # 完整诊断方案文档
├── quick_test_warmup.sh                # 快速8K测试
├── run_all_diagnostic_experiments.sh   # 顺序批量训练
├── run_parallel_experiments.sh         # 并行训练（多GPU）
├── monitor_training.sh                 # 实时监控训练进度
└── verify_implementation.sh            # 验证实施完整性
```

## 快速开始

### 1. 验证实施

```bash
bash diagnostic_scripts/verify_implementation.sh
```

检查所有代码修改和配置文件是否就绪。

### 2. 快速测试（推荐先做）

```bash
bash diagnostic_scripts/quick_test_warmup.sh
```

运行8K iter快速测试，验证warmup假设（2-3小时）。

### 3. 完整训练

**单GPU顺序训练：**
```bash
bash diagnostic_scripts/run_all_diagnostic_experiments.sh
```

**多GPU并行训练：**
```bash
bash diagnostic_scripts/run_parallel_experiments.sh
```

### 4. 监控训练

```bash
bash diagnostic_scripts/monitor_training.sh
```

实时显示训练进度，自动高亮3500-6000不稳定窗口。

## 脚本说明

### quick_test_warmup.sh
- **用途**: 快速验证warmup机制是否有效
- **时长**: 8K iter，约2-3小时
- **输出**: `train_warmup_8k.log`
- **判断标准**: PSNR在5K-8K持续上升 → warmup有效

### run_all_diagnostic_experiments.sh
- **用途**: 顺序运行所有3个诊断实验
- **时长**: 约6-7天（单GPU，24K iter × 3）
- **实验序列**:
  1. r43a_warmup（关键测试）
  2. r43_hybrid（备选方案1）
  3. r43a_learnable（备选方案2）
- **特点**: 可在r43a_warmup成功后提前终止

### run_parallel_experiments.sh
- **用途**: 多GPU并行训练，加速实验
- **要求**: 至少2个GPU
- **时长**: 约2-3天（3GPU并行）
- **GPU分配**:
  - GPU 0: r43a_warmup
  - GPU 1: r43_hybrid
  - GPU 2: r43a_learnable

### monitor_training.sh
- **用途**: 实时监控训练进度
- **功能**:
  - 显示当前iteration
  - 显示最佳PSNR
  - 高亮3500-6000不稳定窗口
  - 自动刷新（30秒间隔）

### verify_implementation.sh
- **用途**: 验证所有修改是否正确
- **检查项**:
  - 代码语法
  - 配置文件存在性
  - 关键修改点

## 实验配置

所有实验配置位于 `Options/` 目录：

- **ISB_ecaformer_r43a_warmup.yml**: identity_scale warmup测试
- **ISB_ecaformer_r43_hybrid.yml**: 双阶段通道校正
- **ISB_ecaformer_r43a_learnable.yml**: 可学习identity_scale

## 诊断工具

### tools/diagnose_checkpoint.py

分析checkpoint演化，提取AdaLN参数和通道scale历史。

**单实验分析：**
```bash
python tools/diagnose_checkpoint.py \
  --exp_dir experiments/ISB_ecaformer_r43a_warmup \
  --iters 1000 3000 5000 7000 10000 \
  --output diagnosis_r43a_warmup.png
```

**对比分析：**
```bash
python tools/diagnose_checkpoint.py \
  --exp_dirs experiments/ISB_ecaformer_r42a_per_ch_res \
               experiments/ISB_ecaformer_r43a_warmup \
  --labels "r42a(stable)" "r43a_warmup" \
  --output r42a_vs_r43a_warmup.png
```

## 根因决策树

根据实验结果判断不稳定性根因：

```
Q1: warmup是否消除不稳定？
├─ 是 → 根因 = 早期x1扰动与AdaLN初始化冲突
│         解决方案: 默认启用warmup
├─ 否 → 进入Q2

Q2: hybrid是否稳定？
├─ 是 → 根因 = identity_scale力度过猛
│         解决方案: 温和identity_scale + residual_scale
├─ 否 → 进入Q3

Q3: learnable的gradient在3500-6000是否飙升？
├─ 是 → 根因 = 优化器与损失冲突
│         解决方案: 固定identity_scale或独立优化器
├─ 否 → 其他原因（LR schedule/数据分布/损失组件）
```

## 预期结果

### 如果假设正确（AdaLN冲突）

**r43a_warmup成功标志：**
- PSNR @ 10K > 22.0（超过r42a的21.64）
- 无3500-6000窗口的PSNR回落
- 训练曲线平滑

**Checkpoint分析显示：**
- r43a_warmup的AdaLN norm与r42a相似（平滑增长）
- 不出现r43a原版的AdaLN norm突然跳变

### 如果假设错误

其他可能原因：
- LR schedule干扰
- 数据分布敏感
- 损失组件冲突

## TensorBoard可视化

```bash
tensorboard --logdir experiments --port 6006
```

在浏览器打开 http://localhost:6006 查看：
- PSNR/SSIM/LPIPS曲线
- Loss组件演化
- 学习率变化

## 故障排查

### GPU显存不足
减小batch_size或使用梯度累积：
```yaml
batch_size_per_gpu: 16  # 从24降到16
accumulate_steps: 2      # 从1改为2
```

### 训练卡住
检查数据加载：
```bash
ps aux | grep python
nvidia-smi
```

### 日志丢失
所有脚本都使用 `tee` 保存日志，检查 `diagnostic_logs_*/` 目录。

## 联系信息

问题反馈：
- GitHub Issues: https://github.com/your-repo/issues
- 参考文档: DIAGNOSTIC_README.md

## 许可证

MIT License
