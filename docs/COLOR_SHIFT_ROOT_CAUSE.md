# 深度分析：为什么 ECAFormer_ISB 早期训练发绿？

> 验证结论（2026-07-10）。此前的 `illumination_channels=1` 理论已被 R47/R48 实验推翻：改成 3 通道光照后早期依然发绿且中期崩溃，说明通道数不是根因。

## 真正的根因：残差捷径把绿色输入直接拷贝到早期输出

### 输出装配公式

`basicsr/models/archs/ECAFormer_ISB_arch.py`（`CrossAttenUnet_ISB.forward` 输出段）：

```python
x1_res = self._gray_world(x1) if self.residual_gray_world else x1
residual = self.residual_scale * x1_res          # 0.6 · x1
out = self.mapping(features) + residual          # identity 激活，无压缩
```

### 早期输出退化为输入的拷贝

训练初期各组件的初始状态：

1. **estimator**（ShallowDeepConv）随机初始化 → conv 输出 ≈ 0 → `illu_map = sigmoid(0) ≈ 0.5`
2. `x1 = x_low * illu_map + x_low ≈ 1.5 · x_low`
3. **mapping** 之前有 `out_norm`（GroupNorm）把特征归一化 → mapping 初始输出 ≈ 0
4. 所以 **`out ≈ 0.6 × 1.5 · x_low = 0.9 · x_low`** —— 一张略微增亮的低光图

LOLv1 低光输入本身绿色占优（Bayer 2G:1R:1B 传感器绿通道 SNR 更高），于是这张"输入拷贝"就是绿的。8 步反向采样不改变结论（`isb_module.py` 最后一步直接输出 predicted_x0）。

### EMA 冷启动放大了观感

验证出图用 `net_g_ema`（`image_isb_model.py` 的 `nonpad_test`），而 EMA：

- 从随机初始化拷贝开始（`image_restoration_model.py`）
- 旧代码用恒定 `decay=0.999`、无 warmup

数学后果：iter 500 时 `0.999^500 ≈ 61%` 权重仍是随机初始化。**你在 baseline 看到的图是一个"六成还是随机"的网络画的**，比 live 网络的真实学习进度落后 1-2K iters。

### 为什么 R19 是"灰"而现在是"绿"

R19 配置用 `use_out_norm: false` —— 特征不归一化就过 mapping conv，初始 mapping 输出是大幅随机噪声，**噪声糊掩盖了绿色拷贝，看起来是灰**。现在 `use_out_norm: group` 让 mapping 初始输出干净地趋近 0，绿色拷贝清晰显影。差异不在残差捷径有无（R19 也有），在 out_norm 让捷径"露了出来"。

## 中期 PSNR 崩溃：同源的全局漂移

所有配置在中段（1ch 约 4-6K，3ch 约 7-9.5K）都出现 **PSNR 掉 3+ dB 但 SSIM/LPIPS 不受影响**的现象。只有 PSNR 崩 = 全局亮度/颜色漂移，结构没坏。

机制：illu_map/x1 **没有任何尺度锚定损失**。x1 同时是残差来源、bridge 端点、denoiser 条件 —— estimator 一更新，整个 `x_t = (1-t)·x0 + t·x1 + noise` 分布跟着漂，denoiser + AdaLN 被迫追赶，短暂失配即崩。通道数只改变崩溃时间，不改变本质。

## 解决方案（R49 系列，2026-07-10）

三个正交机制，全部实现并可配置：

| 机制 | 配置项 | 代码位置 | 作用 |
|---|---|---|---|
| **G1 EMA warmup** | `ema_warmup: true` | `image_restoration_model.py` / `image_isb_model.py` | `decay = min(ema_decay, (1+t)/(10+t))`，早期 EMA 紧跟 live 网络，收敛回 0.999，不影响终点精度 |
| **G2 灰世界残差** | `residual_gray_world: true` | `ECAFormer_ISB_arch.py` `_gray_world` | 逐图灰世界白平衡只作用于残差项，gain detach、限幅 [0.5, 2.0]、训练/推理一致（吸取 R45 教训），早期输出变中性 |
| **S1 亮度锚定损失** | `anchor_loss_weight: 0.05` | `image_isb_model.py` | `L1(mean_c(x1), mean_c(gt))` 钉住 bridge 端点通道均值，消除全局漂移崩溃 |

`x1_recon = x_low * illu_map + x_low` 与网络构造一致，仅在 `identity_scale=[1,1,1]` 且 `pre_denoiser_x1_clamp: false` 时精确（R49 配置满足）。

## 被推翻/放弃的方向（勿重试）

- **illumination_channels=1 是根因** —— 推翻。3 通道（R47/R48）早期同样发绿。
- **channel_scale / identity_scale / green_norm** —— 都是调"拷贝内部通道比例"，治标不治本；identity_scale 还在各处（R43/R44/R48c）触发不稳定。
- **AdaLN 梯度冲突是崩溃根因** —— 修正为"illu_map 无锚定导致的全局漂移"（SSIM/LPIPS 不崩是判据）。
