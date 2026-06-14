# R45 Green Tint Root Cause Analysis & Solution

## 问题根源（所有实验忽略的关键事实）

**x_low 本身就有绿色偏差**（Bayer 传感器 2×2 阵列中有 2 个绿色像素，只有 1 个红色和 1 个蓝色）

### 当前 x1 构造的致命缺陷

```python
x1 = x_low * illu_map + x_low
   = x_low * (1 + illu_map)
```

**`+ x_low` 这一项无条件注入了 x_low 的绿色偏差**，无论 illu_map 如何调整都无法消除。

### 为什么所有修复都失败了

| 方法 | 作用位置 | 失败原因 |
|------|----------|----------|
| `channel_scale` | illu_map | 只影响 `x_low * illu_map`，**不影响 `+ x_low`** |
| `residual_scale` (scalar) | 输出残差 | 等比例缩放，无法针对性压制绿色 |
| `residual_scale` (per-ch) R42a | 输出残差 | **x1 已经发绿**，denoiser 被迫过度补偿 → PSNR -0.46 dB |
| `identity_scale` R43 | x1 构造 | 能修正但引发 AdaLN 梯度冲突 → 训练崩溃（3.5K-6K PSNR 暴跌） |
| `green_loss` R42c | 训练 loss | 软约束，模型可以选择忽略 |

### R38c 为什么是 22.10 但依然发绿？

R38c 让 denoiser 的 `mapping` 层通过训练学习了"事后颜色校正"——这是被迫的补偿策略，不是根本解决。

## 根本解决方案：R45 系列

### R45a: green_norm only（最干净的测试）

```python
# 在 forward 最开始，修正 x_low
r_mean = x_low[:, 0].mean()
g_mean = x_low[:, 1].mean()
x_low[:, 1] -= (g_mean - r_mean)  # 让绿色均值等于红色
x_low[:, 1] = clamp(x_low[:, 1], 0, 1)

# 然后再构造 x1，这时 x_low 已经没有绿色偏差了
x1 = x_low * illu_map + x_low
```

**与 R38c 的唯一区别：**
- R38c: `channel_scale = [1.0, 0.95, 1.0]`（压制 illu_map 的绿色 5%）
- R45a: `green_norm = true`（源头修正 x_low）+ `channel_scale = [1.0, 1.0, 1.0]`（无压制）

**假设：** R45a 应该达到或超过 R38c 的 22.10 PSNR，且**完全不发绿**。

### R45b: green_norm + channel_scale（双重校正）

如果 green_norm 的均值校正还不够（可能存在 per-pixel 方差导致的绿色残留），channel_scale 可以进一步微调。

### R45c: green_norm + green_loss（源头修正 + 梯度引导）

green_loss 告诉模型"不要学习绿色补偿"，因为 green_norm 已经修正了输入。

## 预期结果

| Config | 预期 PSNR | 预期绿色 | 训练稳定性 |
|--------|-----------|----------|------------|
| R45a | **22.2-22.4** | ✅ 完全消除 | ✅ 平滑 |
| R45b | 22.1-22.3 | ✅ 消除 | ✅ 平滑 |
| R45c | 22.3-22.5 | ✅ 消除 | ✅ 平滑 |

**R45a 是最关键的验证**：如果它成功，证明绿色问题的根源就是 x_low 本身，所有后续修复都是"治标不治本"。
