# paper_pack — 数字来源与结论分支（AAAI-27 准备）

统一协议：PSNR/SSIM/LPIPS(alex)，crop_border 0，无 GT-mean，张量域计算（`use_image: false`），
window_size 4 padding。所有数字必须出自 `tools/eval_lol.py`（复用训练 validation 原路径；
2026-07-20 校准通过：r52b latest EMA 复现日志值 21.9268/0.7900/0.1664 逐位一致）。
报告规则：best validation checkpoint，三指标同 checkpoint。

## 1. 已核实数字（LOLv1 Test, 15 张）

| 工件 | PSNR | SSIM | LPIPS | 可复现 | 来源 |
|---|---|---|---|---|---|
| r52b 日志冠军 @11.5K (EMA) | 22.689 | 0.8042 | 0.1664 | **否**（见 §2） | 训练日志 |
| r52b best 文件（裸权重 @11.5K） | 21.038 | 0.7449 | 0.1859 | 是 | eval_lol 2026-07-20 |
| r52b net_g_latest EMA @15.5K | 21.927 | 0.7900 | 0.1664 | 是 | eval_lol 2026-07-20 |
| baseline(n40, 250K 协议) @45K EMA | 23.258 | 0.8376 | 0.0882 | 是 | eval_lol 2026-07-20 |

预算口径（写论文必须带上）：baseline@45K 消耗 ≈5.2× r52b 峰值像素预算；
与 r52b 峰值同预算处（baseline ≈8.6K iter）曲线插值 ≈21.5–21.6/0.79–0.80/0.12–0.13
→ 同预算 PSNR 我方 +1.1 dB，SSIM 平，LPIPS 输。待 fair24k run 出数后替换插值。

## 2. save_best 地雷（已修复，2026-07-20）

训练 validation 用 net_g_ema 打分，旧 `save_best` 只存裸 net_g（'params'）→
r52b 的 22.689 目前无可复现存档（11.5K EMA 已被覆盖）。修复：`save_best` 现存
`['params','params_ema']`。**r54_repro_r52b 已排队重跑以重新捕获峰值 EMA。**
baseline 侧无此问题（老式按迭代存档带双 key，45K EMA 完整）。

## 3. 跨域探针——LOLv2-Real 路线已作废（数据泄漏），改用 LOLv2-Synthetic

**泄漏发现（2026-07-20，tools/scan_overlap.py，32×32 缩略图 L1）**：
LOLv2-Real Test(100) 中 **91 张与 LOLv1 Train 逐像素相同**（raw_l1≈0.0000，文件名直接对应
`00690.png`↔`690.png`），另 **8 张与 LOLv1 Test 重叠** → 合计 99/100 泄漏。
LOLv1→LOLv2-Real 的"零样本跨域"因此无效。作为泄漏演示保留的数字（不得当泛化证据）：

| 工件 | LOLv1 (源域) | LOLv2-Real 零样本(泄漏) |
|---|---|---|
| baseline 45K EMA | 23.258/0.8376/0.0882 | **31.519/0.9283/0.0479**（= 在自己训练图上的记忆分） |
| r52b latest EMA | 21.927/0.7900/0.1664 | 23.913/0.8335/0.1528 |

论文价值：量化了社区常用的 v1→v2Real 迁移评测的泄漏规模（99%），归入"诚实协议"贡献；
凡外部方法声称此路线跨域的，主表脚注说明。

**替代 go/no-go：LOLv1 → LOLv2-Synthetic Test 零样本**（合成域 100 张，扫描确认与 v1 零重叠）——**已出数（2026-07-20 18:10）**：

| 工件 | LOLv1 (源域) | LOLv2-Syn 零样本 | PSNR 掉分 | SSIM 掉分 | LPIPS 恶化 |
|---|---|---|---|---|---|
| r52b latest EMA (ISB) | 21.927/0.7900/0.1664 | 14.644/0.6011/0.2845 | -7.28 (-33.2%) | -0.189 (-23.9%) | +0.118 |
| baseline 45K EMA | 23.258/0.8376/0.0882 | **16.336/0.6981/0.2199** | -6.92 (-29.8%) | -0.140 (-16.7%) | +0.132 |

**判定：跨域鲁棒假说被否定。** baseline 零样本三指标全部更好，相对掉分也更小
（除 LPIPS 相对恶化率一项，因 ISB 源域 LPIPS 本来就差，无实际意义）。
待办复核（不改变结论的完备性动作）：r54 峰值 EMA 恢复后重跑此探针一次（源域差 0.76 dB
不可能翻转 1.7 dB 的跨域劣势）。

其它扫描结果：LOLv2-Real **Train** 也含 LOLv1 **Test** 场景（18/689 近似逐像素、35/689 同场景，
如 00022↔22）→ v2 原生模型报 v1-test 数字时需脚注；v2 内部 train/test 按官方分割使用不受影响。

## 3.5 结论分支判定（2026-07-20，依据 §3 数据）

- [ ] ~~AAAI 满血（跨域赢）~~ —— **被 §3 数据否定**
- [x] **Plan B 分析文（TMLR/期刊）**：严格受控对照下，桥在配对小数据 LLIE 上
  **既不买分布内保真（§1）也不买跨域稳健（§3）**；论文 = 这两个受控负结果
  + 相变/锚定训练动力学（正贡献）+ 泄漏量化（99/100，协议贡献）
  + 效率对扩散类（8-NFE，正贡献）。无截止压力，等 §5 队列全部落地后成稿。
- [ ] 待复核分支：fair24k 出数后若同预算 Ours ≥ baseline（曲线预测 PSNR +1.1）→
  分析文中如实写"匹配预算下桥赢 PSNR、输感知；预算放大后全面被反超"（快收敛低天花板）。

AAAI-27（摘要 ~7/25）：基于以上证据**不建议投**；最终由用户/顾问拍板。

## 4. 结论分支（数据落地后勾选其一）

- [ ] **AAAI 满血**：跨域 ISB 显著更稳（§3 支持）→ 主叙事 = 效率(对扩散) + 跨域稳健 + 端点训稳分析
- [ ] **Plan B 分析文（TMLR/期刊）**：跨域也输 → 严格对照 + 相变/锚 + "配对小数据上桥未必买保真"负结果
- [ ] **叙事修正**：fair24k 下 Ours ≥ baseline → 删"让分保真"，写 comparable + 效率/训稳

## 5. GPU 队列（run_paper_p1_queue.sh，2026-07-20 启动）

1. r54_repro_r52b（≈10h，恢复冠军可复现存档）
2. ECAFormer_baseline_lolv1_fair24k（≈9h，同预算控制行 + patch-256 混杂裁决）
3. ISB_ecaformer_r53_lolv2real（≈15.5h，r53b auto-engage 配方）
4. ECAFormer_baseline_lolv2real_fair24k（≈9h，成对）

## 6. 其它已知事实（写作用）

- R53：r53a(dz0) 22.10@10.5K（峰后 -0.23）；r53b(auto) 22.59@14K（机制坐实，LPIPS 0.1609@18.5K）；
  r53c(seed3407) 全程 ~10 dB（basin 敏感性 → limitation；seed100 下 r52b/r53a/b 前 5.5K 逐点一致）
- 效率措辞：8-NFE 仅对扩散类是优势；对回归 baseline（1-NFE）是 8× 慢
- 术语：illumination-lifted endpoint（"Retinex" 只在物理动机出现一次；Retinexformer 仅为外部对比方法）
