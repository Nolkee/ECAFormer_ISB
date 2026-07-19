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

## R50 实验裁决（2026-07-15）：漂移理论被 r50c 直接证实

- **r50a（仅衰减调度）**：去饱和消失，PSNR 22.21 / SSIM 0.7964 / LPIPS 0.1603
  —— 首个三项均不输 R48b 的配置（LPIPS 首次胜出）。崩溃仍在 6-7K（-2.4 dB）。
- **r50b（慢 estimator 0.3×）**：被推翻 —— 峰值仅 20.83，瞬态更长更深，~8.5K
  被 early-stop。崩溃需要的是恢复力，不是更慢的漂移。
- **r50c（可达锚 w0.5）**：**全程无谷底（项目史上第一条）**，SSIM 0.8064 历史
  最佳；代价 PSNR 封顶 21.75 —— 精确钉死通道均值剥夺了逐图照明自由度。

结论（当时）：锚定强度是单变量权衡 → R51 扫描 w 0.1/0.25 + 相对死区 0.15。

## R51 裁决（2026-07-16）：权衡模型被推翻，改立"相变"模型

- r51a（w0.1）：**仍崩溃**（-2.9 dB @ 6-8K），峰值 21.98——轻锚不买稳定，
  固定目标 1.5·lq 在相变后还持续拖拽（-0.23 dB vs r50a）。
- r51b（w0.25）：崩得**更早**（5-5.5K），全场最差 21.61——中间不存在甜点。
- r51c（w0.5+死区0.15）：无谷底 ✅ 但 PSNR 只到 21.83——把 ±15% 均值自由还给
  网络也没解锁 PSNR，说明 21.8 天花板不是"钉太死"，是**锚定阻止了某个过程**。

**相变模型**：把全部实验并置——每一个到过 22.2 的模型都崩过（R48b、r50a），
每一个从 iter 0 上锚的模型都停在 21.75-21.83。谷底是网络迁移到高 PSNR 照明域
的**相变**：放行它 → PSNR 22.2 / SSIM ~0.796；阻止它 → 留在初始域，
SSIM 0.80+ / PSNR ≤21.83。稳定性近似二值（w0.5 级力才挡得住），弱锚两头落空。

## 解决方案 v3（R52 系列，2026-07-16）：迟到的自校准锚

| 机制 | 配置项 | 作用 |
|---|---|---|
| **延迟启用** | `anchor_start_iter: 12000`（r52a）/ `9000`（r52b） | 相变（6-8K）+ 平台确立后才上锚——放行相变，锁定新域 |
| **自校准目标** | `anchor_mode: x1_ema` + `anchor_ema_momentum: 0.99` | 目标 = 实际达成的 mean(x1)/mean(lq) 比率的 EMA（iter 0 起跟踪、**engage 时冻结**，否则会跟着阴跌走）；锚"现在的自己"而非固定先验（吸取 r51a 教训）。resume 后重热 200 iters 再冻结 |
| **小死区** | `anchor_deadzone: 0.05` | 只吸收 batch 噪声，不放过域漂移 |
| **感知冲榜** | r52c = r51c + `zero_init_mapping_bias` | 假设 r49c 的失败源于无锚相变；锚稳后 R41b 的感知增益（史上 LPIPS 0.1537）应可安全收割 |

## R52 裁决（2026-07-19）：相变模型兑现，r52b 登顶

- **r52b（engage 9K）@ 11.5K：22.689 / 0.8042 / 0.1664 —— 新总冠军**，单
  checkpoint 同时刷新 PSNR（+0.48 dB）与 SSIM。engage 恰好落在谷底回升段。
- r52a（engage 12K）：相变时机随机（同配置谷底 5.5K vs 7.5K），12K 落在峰值
  之后 → 锁住了已阴跌的状态（峰 22.37 @ 9.5K，锁定后 ~22.13）。
- 冻结锚没止住峰后阴跌：r52b 12.5K→15.5K 掉 -0.76 dB——0.05 死区允许 ±5% 游走，
  阴跌从带内溜走（对照：r50c/r51c 紧锁全程无阴跌）。
- r52c（zero_init_mapping_bias @ 锚定基座）：再次被推翻（21.18/0.786），方向关闭。

## 解决方案 v4（R53 系列，2026-07-19，投稿前最后一轮）

| 机制 | 配置项 | 作用 |
|---|---|---|
| **精确锁** | `anchor_deadzone: 0` + `early_stop_patience_val: 12` | 关死区堵住阴跌逃逸通道；放宽耐心在锁定域里收割 SSIM/LPIPS |
| **自动 engage** | `anchor_engage_mode: auto`（`valley_drop 1.0` / `rise_margin 0.5` / `min_engage 4000` / cap=anchor_start_iter 12000） | 用 train-PSNR EMA 检测谷底回升拐点自动上锁——把 r52b 的运气变成机制；三条观测时间线仿真触发于 7.1K/9.0K/7.2K |
| **延迟冻结** | `anchor_freeze_delay: 1500` | engage 后目标再跟随 1.5K iters 才冻结，锁"稳定后的域"而非恢复中途 |
| **状态持久化** | `latest.state` 的 `model_extra` | engage 状态+冻结比率跨 resume 保留（base_model `_extra_training_state` 钩子），resume 不再失忆 |

当前资产：**总冠军 r52b @ 11.5K（22.689/0.8042/0.1664）**；感知冠军 r51c @ 20K。
论文泛化表：`Options/ISB_ecaformer_r53_lolv2real.yml`（r53b 配方 + LOLv2Real）。

## 被推翻/放弃的方向（勿重试）

- **illumination_channels=1 是根因** —— 推翻。3 通道（R47/R48）早期同样发绿。
- **channel_scale / identity_scale / green_norm** —— 都是调"拷贝内部通道比例"，治标不治本；identity_scale 还在各处（R43/R44/R48c）触发不稳定。
- **AdaLN 梯度冲突是崩溃根因** —— 修正为"illu_map 无锚定导致的全局漂移"（SSIM/LPIPS 不崩是判据）。
- **锚定 x1 通道均值到 GT（`anchor_mode: gt`）** —— sigmoid 上限使目标 100% 不可达，损失为饱和常数（R49b 实证零效果）。锚定目标必须落在 x1 的可达域内。
- **永久（无调度）灰世界残差** —— 去饱和 + 崩溃提前（R49 实证）。必须配 `gray_world_decay_start/end`。
- **弱锚定（w ≤ 0.25）** —— 挡不住相变还拖累恢复后的域（R51 实证）。
- **zero_init_mapping_bias** —— 两度被推翻：无锚基座最深谷底（r49c）、锚定基座全面变差（r52c）。
- **"锚定强度是连续权衡"** —— 推翻。稳定性近似二值；PSNR 天花板来自被阻止的相变，不是被钉死的均值（r51c 死区实验为证）。
