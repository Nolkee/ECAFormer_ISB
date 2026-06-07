# 历史训练脚本归档

此文件夹包含ECAFormer_ISB项目的历史训练脚本，用于归档和参考。

## 脚本分类

### 早期实验系列（R11-R26）
这些是项目早期的探索性实验脚本：

- `run_r11_r12.sh` - R11/R12实验
- `run_r14_r16.sh` - R14-R16调参
- `run_r17.sh`, `run_r17_fixes.sh` - R17系列
- `run_r18.sh`, `run_r18_sweep.sh` - R18参数扫描
- `run_r19_decouple.sh` - R19解耦实验
- `run_r20_sweep.sh` ~ `run_r26_sweep.sh` - R20-R26 sweep系列

### 中期优化系列（R27-R37）
架构优化和损失函数调整：

- `train_r27_series.sh` - R27系列
- `train_r28b_r29.sh` - R28b/R29
- `train_r29_series.sh` - R29系列
- `train_r30_r31_series.sh` - R30/R31
- `train_r32_series.sh`, `train_r32c_r33_r34_r35.sh` - R32-R35
- `train_r36_series.sh` - R36系列
- `train_r37_series.sh` - R37系列

### 绿色色调修复系列（R38-R43）
针对绿色色调问题的系统性实验：

**R38系列：channel_scale调整**
- `train_r38_series.sh` - R38a/b/c实验
- 最佳结果：R38c PSNR 22.10（但训练不稳定）

**R39-R41系列：架构探索**
- `train_r39_series.sh` - R39: sigmoid activation测试（失败）
- `train_r40_series.sh` - R40: 更激进的channel_scale（失败）
- `train_r41_series.sh` - R41: GroupNorm和zero_init测试（部分成功）

**R42系列：per-channel residual_scale**
- `train_r42_series.sh` - R42a: 首个稳定的绿色修复方案
- 结果：PSNR 21.64，训练平稳，无震荡

**R43系列：identity_scale实验**
- `train_r43_series.sh` - R43a/b/c: identity_scale方案
- `train_r42a_then_r43.sh` - 顺序训练R42a→R43
- 问题：R43a在3500-6000 iter出现不稳定（与R38c相同）

### 通用训练脚本

- `auto_train_all.sh` - 自动批量训练脚本
- `run_train_sequence.sh` - 训练序列管理
- `run_fair_comparison.sh` - 公平对比实验
- `train_lolv2real.sh` - LOLv2 Real数据集训练
- `train_multigpu.sh` - 多GPU训练脚本

## 重要发现（来自历史实验）

### 1. 绿色色调问题的演进

**问题根源**：
- Bayer sensor的绿色通道权重（2:1:1 RGGB）
- x_low输入带有固有绿色偏差
- 残差连接将绿色偏差注入输出

**尝试过的方案**：
1. **R38c**: `channel_scale=[1.0, 0.95, 1.0]` - PSNR 22.10，但不稳定
2. **R39**: sigmoid激活 - 动态范围受损，失败
3. **R40**: 更激进的channel_scale=[1.0, 0.90, 1.0] - PSNR下降更多
4. **R41**: 架构修改（GroupNorm/zero_init） - 混合结果
5. **R42a**: `residual_scale=[0.6, 0.5, 0.6]` - PSNR 21.64，**稳定训练**✓
6. **R43a**: `identity_scale=[1.0, 0.92, 1.0]` - 复现R38c不稳定问题

### 2. 训练不稳定性模式

**现象**：
- R38c和R43a都在3500-6000 iter出现PSNR回落
- PSNR从20掉到17-18，然后恢复但带震荡

**区别**：
- R42a（稳定）：per-channel缩放在denoiser**输出**
- R43a（不稳定）：per-channel缩放在x1**构造**
- R38c（不稳定）：per-channel缩放在**illumination map**

**假设**（见诊断方案）：
早期x1通道不平衡 → AdaLN过度补偿 → 梯度冲突

### 3. 最佳实践总结

**推荐架构**：
```yaml
residual_scale_init: [0.6, 0.5, 0.6]  # R42a方案
channel_scale_init: [1.0, 0.95, 1.0]
illumination_channels: 1
use_out_norm: true
output_activation: identity
```

**避免的配置**：
- ❌ `output_activation: sigmoid` - 动态范围受限
- ❌ `use_out_norm: 'post'` - GroupNorm(1,3)导致训练崩溃
- ❌ `illumination_channels: 3` 不加稳定措施 - 训练crash
- ❌ identity_scale不加warmup - 3500-6000不稳定

## 当前最佳配置

**生产环境推荐**：
- 配置：`Options/ISB_ecaformer_r42a_per_ch_res.yml`
- PSNR @ 10.5K: 21.64
- 特点：训练稳定，无震荡

**实验中（诊断框架）**：
- 目标：修复R43a不稳定性，突破22+ PSNR
- 方法：identity_scale warmup（见`diagnostic_scripts/`）

## 如何使用历史脚本

这些脚本保留用于：
1. **参考历史实验设置**
2. **复现早期结果**
3. **理解实验演进路径**

**注意**：
- 部分脚本引用的配置文件可能已过时
- 运行前请检查Options/目录中是否有对应的.yml文件
- 建议查看脚本内容理解实验设计，而非直接运行

## 当前活跃脚本

最新的训练脚本位于：
- **诊断实验**: `diagnostic_scripts/` 目录
- **当前推荐**: 使用诊断脚本套件进行系统化实验

## 实验编号说明

- R11-R26: 早期探索
- R27-R37: 架构优化
- R38-R43: 绿色色调系统性修复
- R44+: （未来）诊断框架验证后的优化方案

---

**归档日期**: 2026-06-08  
**最后活跃版本**: R43系列  
**下一步**: 诊断R43a不稳定性根因
