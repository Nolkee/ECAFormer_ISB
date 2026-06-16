# 深度分析：为什么 ECAFormer_ISB 会颜色偏移？

## 原始 ECAFormer（无颜色问题）

```python
# Line 242-245 in ECAFormer_inference.py
def forward(self, img):
    visual_feat, semantic_feat = self.ShallowDeepConv(img)
    semantic_feat = img * semantic_feat + img  # x1
    output_img = self.CrossAttUnet(semantic_feat, visual_feat)
```

**关键点：**
1. `semantic_feat` 是 ShallowDeepConv 输出的 **3 通道** illumination map
2. `x1 = img * semantic_feat + img = img * (1 + semantic_feat)`
3. **没有任何 per-channel 手动调整**
4. 网络自己学习 semantic_feat 的值，不会引入人为颜色偏差

## ECAFormer_ISB（引入颜色偏移）

```python
# Line 759-765 in ECAFormer_ISB_arch.py
visual_fea, illu_map = self.estimator(x_low)
if self.illumination_map_activation == 'sigmoid':
    illu_map = torch.sigmoid(illu_map)
if self.illumination_channels == 1:
    illu_map = illu_map * self.channel_scale  # [1.0, 0.95, 1.0]
x1 = x_low * illu_map + current_scale * x_low
```

## 🔥 关键差异：illumination_channels = 1

**这是颜色偏移的根源！**

### 原始 ECAFormer
- ShallowDeepConv 输出 `n_fea_out=3` 通道（RGB 分别的 illumination map）
- 每个颜色通道有**独立的**光照估计
- 网络可以学习"红色通道需要增强 2x，绿色 1.5x，蓝色 1.8x"

### ECAFormer_ISB（当前实现）
- ShallowDeepConv 输出 `illumination_channels=1`（**单通道** illumination map）
- 然后用 `channel_scale = [1.0, 0.95, 1.0]` **手动扩展**到 3 通道
- 结果：`illu_map_rgb = [illu * 1.0, illu * 0.95, illu * 1.0]`

**这意味着：**
- 红色和蓝色通道共享相同的光照估计
- 绿色通道被**人为压制** 5%
- 网络**无法学习** per-channel 的光照差异

## 为什么要用 illumination_channels=1？

查看 commit history 和配置：

```yaml
# R38c (best so far)
illumination_channels: 1
channel_scale_init: [1.0, 0.95, 1.0]
```

**推测原因：**
1. **减少参数量**：1 通道 vs 3 通道输出
2. **强制共享光照估计**：认为 RGB 应该有相同的光照增强
3. **channel_scale 是后来加的补丁**：发现发绿后，手动压制绿色

## 🎯 根本问题

**当前架构的两个矛盾设计：**

1. **强制单通道光照** (`illumination_channels=1`) 
   - 假设：RGB 三通道应该共享相同的光照增强
   - 现实：Bayer 传感器的 2G:1R:1B 导致绿色天然更强
   
2. **手动 channel_scale 补偿**
   - 用 `[1.0, 0.95, 1.0]` 事后打补丁
   - 这是"治标不治本"的方案

## 🔍 真正的解决方案选项

### 选项 A：回归原始 ECAFormer（illumination_channels=3）✅ **推荐**

```yaml
illumination_channels: 3  # 让网络学习 per-channel illumination
channel_scale_init: [1.0, 1.0, 1.0]  # 不需要手动调整
```

**优点：**
- 网络可以自己学习每个通道的最佳光照增强
- 不需要任何手动颜色校正
- 恢复原始 ECAFormer 的设计意图

**风险：**
- R41c 测试过 `illumination_channels=3`，训练不稳定（8K 崩溃）
- 但那个配置同时有其他问题（post norm）

### 选项 B：保持 illumination_channels=1，用 green_norm 修正输入

```yaml
illumination_channels: 1
channel_scale_init: [1.0, 1.0, 1.0]  # 不手动调整
green_norm: true  # 在输入端修正绿色偏差
```

**优点：**
- 在源头消除绿色偏差
- 保持单通道光照的简洁性

**这就是 R46 的方案。**

### 选项 C：illumination_channels=3 + green_norm（最稳健）

```yaml
illumination_channels: 3
green_norm: true
channel_scale_init: [1.0, 1.0, 1.0]
```

**优点：**
- 输入端修正绿色偏差（green_norm）
- 网络有完整的 per-channel 学习能力
- 双重保险

## 📊 实验建议：R47 系列

测试 illumination_channels=3 的真正潜力（不带 R41 的其他问题）：

- **R47a**: `illumination_channels=3` only（最干净的测试）
- **R47b**: `illumination_channels=3` + `green_norm`（双重修正）
- **R47c**: `illumination_channels=3` + `green_norm` + `learnable residual_scale`（per-channel 全面掌控）

这才是对原始 ECAFormer 设计的正确继承。
