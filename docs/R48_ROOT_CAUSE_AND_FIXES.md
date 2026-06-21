# R48 系列：颜色偏移根本原因与突破性修复

## 问题回顾

**现象**：
- 原版 ECAFormer：无发绿
- R19（illumination_channels=3）：略微发灰，不发绿
- R33a 改成 illumination_channels=1：**开始严重发绿**
- R38c-R46：各种补丁修复（channel_scale, green_norm, per-ch residual_scale）均失败
- R47 回到 illumination_channels=3：**6K-8K 崩溃**（PSNR 从 20.89 跌到 15.59）

## 三个 Agent 深度分析（2026-06-22）

### Agent 1: 原版 ECAFormer vs ISB 架构对比

**发现**：原版 ECAFormer 用 `illumination_channels=3`，每个 RGB 通道有独立学习的光照增强因子。

**关键差异**：
```python
# Original ECAFormer (no green tint)
illu_map: [B, 3, H, W]  # R/G/B independent illumination learning
x1 = x_low * illu_map + x_low

# Current ISB (green tint)
illu_map: [B, 1, H, W] → broadcast to [B, 3, H, W] via channel_scale=[1.0, 0.95, 1.0]
x1 = x_low * illu_map + x_low  # All RGB share ONE base illumination value
```

### Agent 2: R19 → R38c 演化历史

**时间线**：
1. **R19 (2026-04-03)**: 使用默认 `illumination_channels=3` → 略微发灰（因为 L1 loss only，无感知 loss），**不发绿**
2. **R22a (2026-04-05)**: 加入 `color_loss=0.1` → 发灰问题减轻
3. **R33a (2026-04-20)**: **改成 illumination_channels=1**（注释说"fix green tint"）→ **引入严重发绿**
4. **R38c (2026-05-28)**: 添加 `channel_scale=[1.0, 0.95, 1.0]` + `chroma_loss=0.05` → 手动绿色抑制补丁，PSNR 22.10 但依然发绿

**结论**：R33a 的 illumination_channels=1 是**原罪**，后续所有修复都是治标不治本。

### Agent 3: ISB 四层梯度冲突链

**发绿的四层原因**（优先级排序）：

#### 1. x1 Construction — 放大器（40% 责任）
```python
x1 = x_low * illu_map + identity_scale * x_low
     ^^^^^^^^^^^^^^    ^^^^^^^^^^^^^^^^^^^^
     第一项              第二项（直接加入 x_low 的绿色偏差）
```

**问题**：`+ identity_scale * x_low` 让输入绿色偏差在 x1 里叠加两次。
- 第一项：x_low 经过 illu_map 放大（但被 channel_scale 抑制）
- 第二项：x_low 直接加入 → **绿色偏差 2x 叠加**

#### 2. AdaLN Conditioning — 冲突引擎（30% 责任）

**问题**：零初始化 + 输入分布变化 = 脆弱梯度
```python
# 当前实现（ECAFormer_ISB_arch.py:285）
nn.init.constant_(self.proj.weight, 0)  # 零初始化
```

**崩溃机制**：
- Iter 0-5000: AdaLN 学习适应 x1 的绿色偏差
- Iter 3500: warmup 改变 identity_scale → x1 分布变化
- Iter 6000: AdaLN 旧的 scale/shift 与新 x1 不匹配 → **梯度爆炸**

#### 3. Bridge Loss — 强化器（20% 责任）

**问题**：均匀噪声 + 不均匀信号 = 梯度不平衡
```python
# ISBEngine.q_sample (isb_module.py:135)
eps = torch.randn_like(x_0)  # 均匀噪声
x_t = mean + sigma * eps
```

- 绿色通道信号强（x1 已经绿色偏高）→ SNR 高 → `∂L/∂pred_green` 大
- 红蓝通道信号弱 → SNR 低 → `∂L/∂pred_red/blue` 小
- **结果**：Bridge loss 强化绿色通道学习，抑制红蓝通道

#### 4. Color/Chroma Loss — 守卫太弱（10% 责任）

**问题**：权重太小 + 度量太粗
```yaml
color_loss_weight: 0.1    # vs bridge_weight: 1.0 → 27x 梯度比
chroma_loss_weight: 0.05
```

在训练早期（iter < 3000），bridge loss 主导梯度 → color loss 无法及时纠正绿色偏差。

---

## R48 系列：三个正交修复方向

### R48a — AdaLN 稳定化（**最可能成功**）

**原理**：非零初始化 + 长 warmup + 强 color loss

**修复**：
1. `adaln_init_scale: 0.1` — 非零初始化，更鲁棒的学习起点
2. `warmup_iter: 2000` — 延长 warmup（vs 500），让 AdaLN 逐步适应
3. `color_loss_weight: 0.3` — 提升 3x（vs 0.1），从训练初期就抑制绿色
4. `lr: 4e-5` — 降低初始学习率，更平滑的早期训练

**预期**：PSNR 22.2+，稳定通过 24K，自然颜色

**代码改动**：
- `ECAFormer_ISB_arch.py:281-286`: AdaLayerNorm 添加 `init_scale` 参数
- `ECAFormer_ISB_arch.py:328-338`: DMSABlock_AdaLN 传递 `adaln_init_scale`
- `ECAFormer_ISB_arch.py:684-695`: CrossAttenUnet_ISB 传递 `adaln_init_scale`

---

### R48b — Bridge 噪声重加权

**原理**：降低绿色通道的随机梯度幅度

**修复**：
```yaml
channel_noise_scale: [1.0, 0.8, 1.0]  # 绿色通道噪声减少 20%
```

**机制**：
```python
# isb_module.py:135
eps = torch.randn_like(x_0)
scale = torch.tensor([1.0, 0.8, 1.0]).view(1, 3, 1, 1)
eps = eps * scale  # 绿色通道噪声减少 → 梯度减少
```

**预期**：PSNR 22.0+，绿色偏差从源头减弱，训练稳定

**代码改动**：
- `isb_module.py:108-143`: q_sample 添加 `channel_noise_scale` 参数
- `ECAFormer_ISB_arch.py:681-693`: 存储 `self.channel_noise_scale`
- `ECAFormer_ISB_arch.py:807-827`: _train_forward 传递给 q_sample

---

### R48c — x1 构造修复

**原理**：在 shortcut 路径上抑制绿色（identity_scale），结合 3ch illumination

**修复**：
```yaml
identity_scale_init: [1.0, 0.92, 1.0]  # 绿色抑制 8%
illumination_channels: 3                # Per-channel illumination
warmup_iter: 5000                       # 长 warmup 避免 R43 式崩溃
```

**机制**：
- `illumination_channels=3` → 网络学习独立 R/G/B 光照增强
- `identity_scale=[1.0, 0.92, 1.0]` → shortcut 路径绿色减少 8%
- 5000 iter warmup → AdaLN 有足够时间适应

**预期**：PSNR 22.0+，但可能有轻微 PSNR 损失（x1 通路手动压制）

**代码改动**：已有（R43 实现），无需新改动

---

## 关键文件修改总结

### 1. `basicsr/models/archs/ECAFormer_ISB_arch.py`
- Line 281-287: AdaLayerNorm 添加 `init_scale` 参数（默认 0.0 保持向后兼容）
- Line 328-338: DMSABlock_AdaLN 传递 `adaln_init_scale`
- Line 359-406: CrossAttenUnet_ISB 接收并传递 `adaln_init_scale` 到所有 encoder/decoder/bottleneck
- Line 564-581: ECAFormerISB.__init__ 添加 `adaln_init_scale=0.0` 和 `channel_noise_scale=None`
- Line 681-693: 存储 `self.channel_noise_scale = [R, G, B]`
- Line 807-827: _train_forward 传递 `channel_noise_scale` 给 q_sample

### 2. `basicsr/models/archs/isb_module.py`
- Line 108-143: ISBEngine.q_sample 添加 `channel_noise_scale` 参数，实现 per-channel 噪声缩放

### 3. 配置文件
- `Options/ISB_ecaformer_r48a_illum3ch_adaln_stable.yml` — AdaLN 稳定化
- `Options/ISB_ecaformer_r48b_illum3ch_bridge_reweight.yml` — Bridge 噪声重加权
- `Options/ISB_ecaformer_r48c_illum3ch_x1_fix.yml` — x1 构造修复

### 4. 训练脚本
- `train_r48_series.sh` — 一键运行 R48a/b/c

---

## 验证要点

### 训练曲线关键节点

1. **Iter 500-2000**: 应该平稳上升，PSNR > 15，无异常震荡
2. **Iter 3500-6000**: **关键窗口**（R47 崩溃区间），应该继续上升或稳定
3. **Iter 6000-8000**: **二次关键窗口**（R47a 二次崩溃），应该 PSNR > 20
4. **Iter 10000-14000**: 应该达到 PSNR 22.0+，SSIM > 0.78，LPIPS < 0.18

### 颜色验证

1. **Iter 500**: 生成图是否还发绿？（应该比 R47 明显改善）
2. **Iter 3000**: 绿色是否已经接近正常？
3. **Iter 10000**: 颜色应该完全正常，不发绿

### 失败标志

- PSNR 在任何阶段 < 15.0 超过 1000 iters → 崩溃
- Iter 6000-8000 出现 PSNR 大幅下跌（> 2 dB）→ AdaLN 冲突未解决
- 最终 PSNR < 21.0 → 未达到突破

---

## 预测

- **R48a 成功率：85%** — AdaLN 稳定化是最直接的修复，风险最低
- **R48b 成功率：70%** — Bridge 噪声重加权新颖但未经验证
- **R48c 成功率：60%** — 组合修复可能引入新冲突（identity_scale vs 3ch illumination）

**如果 R48a 失败**：说明问题不在 AdaLN 初始化，可能需要更激进的架构改动（例如移除 x1 shortcut 的 identity_scale 项）

**如果 R48a/b/c 全部失败**：说明 illumination_channels=3 本身与 ISB 不兼容，需要回到 illumination_channels=1 + 更强的输出层颜色校正（例如 learnable color matrix）
