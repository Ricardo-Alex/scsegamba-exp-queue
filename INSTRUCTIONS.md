# 实验指令队列（客户端 AI → KimiCode）

> 本文件由客户端 AI 维护，每次巡检看板后更新。
> **拉取频率：每小时一次。**

**更新时间**：2026-10-24（第 16 次下发：跨数据集泛化验证——DeepCrack / CRACK500 / Volker_DSC）
**状态**：Z 系列已收官 → 执行第 16 批（论文泛化性对比表数据）

---

## 已定稿（勿动）

- **最终方案 = G2 配置**：K8 旋转蛇形 + DAWF（M_out 逐像素方向加权）+ 无重排 + 无 EMA；统一口径 0.8849/0.8612/0.9164；
- **评测协议已统一为 SCSegamba 原版**（逐 epoch 测 test 取 max mIoU，ODS/OIS/F1 原版指标）——本批全部 run 必须走统一口径评测（第 14 批改好的评测段）。

---

## 第 16 批：跨数据集泛化验证（GEN 系列）

### 目标
用最终方案（G2 配置）在三个外部裂缝数据集上验证泛化性，并与 SCSegamba 原版（REF）同协议对比，产出论文"泛化性实验"对比表。**每个数据集必须跑 REF + G2 两个 run**，只跑 G2 没有基准列，数据不可用。

### 数据集路径（先核实，截图中的路径有拼写误差）

- `autodl-tmp/SCSegamba-main/datasets/DeepCrack`
- `autodl-tmp/SCSegamba-main/datasets/CRACK500`
- `autodl-tmp/SCSegamba-main/datasets/Volker_DSC`

开跑前先 `ls` 核实三个目录的真实拼写与内部结构（train/val/test 划分、图像与标注格式、分辨率），上报客户端确认后再启动训练。**数据集目录严格只读**——任何预处理（resize/crop/格式转换）输出必须写到 datasets 目录之外，原文件一个字节都不许改。

### 实验矩阵（锚点 = 各数据集的 REF run）

| 顺序 | run | 数据集 | 配置 |
|---|---|---|---|
| 1 | GEN_DC_REF | DeepCrack | SCSegamba 原版 |
| 2 | GEN_DC_G2 | DeepCrack | **G2 最终配置**（K8 蛇形 + DAWF，无重排无 EMA） |
| 3 | GEN_C500_REF | CRACK500 | SCSegamba 原版 |
| 4 | GEN_C500_G2 | CRACK500 | G2 最终配置 |
| 5 | GEN_VK_REF | Volker_DSC | SCSegamba 原版 |
| 6 | GEN_VK_G2 | Volker_DSC | G2 最终配置 |

### 协议（公平性红线）

- **每个数据集内部，REF 与 G2 必须使用完全相同的训练配置**：若 SCSegamba 仓库对该数据集有官方 per-dataset 配置（epoch 数 / 输入分辨率 / 增强策略），REF 和 G2 都用它；若无，则沿用冻结协议（AdamW lr=5e-4 / PolyLR / 50 epoch / bs=4 / seed=42）。
- 评测走统一口径（逐 epoch test，max mIoU + 原版 ODS/OIS/F1）。
- 判据照旧：G2 max mIoU > 同数据集 REF 且 ODS ≥ REF−0.005 记为正向；如实记录，不做任何挑选。
- **每个 run 的全部 50 个 epoch checkpoint 必须保留**（不许再删，避免重演 replay-eval 补测）。

### 完成后
6 个 run 全部完成后：更新看板 + results_log.md（含各 run 的 max mIoU/ODS/OIS/F1/后10均值/收敛 epoch），汇总为一张 3 数据集 × 2 模型的对比表，然后停止，等客户端收口指令。

### 明确不做的
- 不在这三个数据集上跑 G2/REF 以外的配置（F 系列、X 系列、EMA 等一律不跑）；
- 不调数据集专属超参（发现某数据集不收敛时如实上报，不许私自调 lr/epoch）；
- 3 seed 复核不做；数据集图像内容一个字节都不许改。

### 上报
- 每个 run 完成后更新看板 + results_log.md；队列跑空后每小时拉取一次本文件等待指令
