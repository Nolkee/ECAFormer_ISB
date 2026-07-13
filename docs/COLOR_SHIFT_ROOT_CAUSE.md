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

## R49 实验裁决（2026-07-14）：绿修好了，但引入两个回归

**成功**：iter-500 baseline 不再发绿（G1+G2 按预期工作）。

**回归 1 — 收敛输出去饱和**：永久灰世界把残差的三通道空间均值强制拉平，
残差永远携带零逐图色偏 DC；mapping 在 GroupNorm 之后，唯一图像无关 DC 来源是
数据集常数的 conv bias，于是网络用"数据集平均色偏"近似每张图 → 彩色物体发灰。
chroma loss（0.05，全图单标量 std）比像素项弱 ~65 倍，无力抵抗。

**回归 2 — 崩溃提前（5.5-7K，R48b 是 7-9.5K）**：detach 的倒数增益
`gain = gray/ch_mean`（ch_mean 仅 ~0.05-0.1）把 estimator 的微小均值漂移放大成
残差基底大幅摆动；暗通道有效残差权重最高 1.2。`ema_warmup` 在崩溃窗口平滑常数
612-779（R48b 为 1000），让谷底在验证曲线上显得更早更深（观测放大器）。

**S1 锚定完全失效（关键发现）**：illu_map 过 sigmoid ⇒ `x1 = lq·(1+σ) ≤ 2·lq`，
而 LOLv1 实测 `2·lq_gray≈0.09 ≪ gt_gray≈0.49`（100% 图片不可达）——损失是
~0.42 的常数底，梯度单向且经 sigmoid 饱和，**不是恢复力**。r49a（无锚）与
r49b（锚 0.05）崩溃曲线几乎一致即为实验证据。当时不可见的原因：loss_dict 是
死代码，日志只有 l_total（R50 已修复，全部分量+漂移诊断进 TensorBoard）。

## 解决方案 v2（R50 系列，2026-07-14）

| 机制 | 配置项 | 作用 |
|---|---|---|
| **灰世界衰减调度** | `gray_world_decay_start: 1500` + `gray_world_decay_end: 3500`（network_g 段） | blend λ 从 1 线性降到 0：绿风险期全量 WB，3.5K 后残差恢复真实色彩先验（去饱和消失、倒数增益放大器退场）。同 iter 下 train/val 一致（`_current_iter` 同步到 net_g 与 net_g_ema）；部署推理（无 iter 上下文）自动用终态 λ=0 |
| **estimator 慢学习率** | `estimator_lr_mult: 0.3`（train 段） | estimator 单独参数组按 0.3× 基础 LR（warmup/cosine 按组缩放），从源头减慢 x1 漂移速度，denoiser 能跟上 → 不崩 |
| **可达锚定** | `anchor_loss_weight: 0.5` + `anchor_mode: x1_lq` + `anchor_target_ratio: 1.5`（train 段） | 钉 `mean_c(x1) → 1.5·mean_c(lq)`——x1 可达域 [lq, 2lq] 的中点，双向恢复力；权重 0.5 占总损失 ~10%（R49 的 0.05 只占 ~2% 且被裁剪淹没） |

消融：r50a（仅调度）验证去饱和消失+崩溃退回 R48b 时间线；r50b（+慢 LR）与
r50c（+可达锚）对照哪种稳定器根治 5.5-9.5K 谷底与 9K 后阴跌。

## 被推翻/放弃的方向（勿重试）

- **illumination_channels=1 是根因** —— 推翻。3 通道（R47/R48）早期同样发绿。
- **channel_scale / identity_scale / green_norm** —— 都是调"拷贝内部通道比例"，治标不治本；identity_scale 还在各处（R43/R44/R48c）触发不稳定。
- **AdaLN 梯度冲突是崩溃根因** —— 修正为"illu_map 无锚定导致的全局漂移"（SSIM/LPIPS 不崩是判据）。
- **锚定 x1 通道均值到 GT（`anchor_mode: gt`）** —— sigmoid 上限使目标 100% 不可达，损失为饱和常数（R49b 实证零效果）。锚定目标必须落在 x1 的可达域内。
- **永久（无调度）灰世界残差** —— 去饱和 + 崩溃提前（R49 实证）。必须配 `gray_world_decay_start/end`。
