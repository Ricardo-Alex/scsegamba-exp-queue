# 实验指令队列（客户端 AI → KimiCode）

> 本文件由客户端 AI 维护，每次巡检看板后更新。
> **拉取频率：每小时一次。**

**更新时间**：2026-10-24（第 17 次下发：主战场 REF 补跑 + 论文插图素材）
**状态**：GEN 系列已收官 → 执行第 17 批（收官批，跑完实验部分全部结束）

---

## 已定稿（勿动）

- **最终方案 = G2 配置**：K8 旋转蛇形 + DAWF（M_out 逐像素方向加权）+ 无重排 + 无 EMA；统一口径 0.8849/0.8612/0.9164；
- **评测协议已统一为 SCSegamba 原版**（逐 epoch 测 test 取 max mIoU，ODS/OIS/F1 原版指标）；
- GEN 系列 6/6 已收官（三集全负，如实留档），不再追加任何外部数据集实验。

---

## 第 17 批：REF@UMTCV6 + 论文插图素材（收官批）

### 任务 1（训练，唯一新增 run）：UMTCV6_REF

主战场缺 SCSegamba 原版的本地统一口径数——主结果表对比列必须补齐。

| run | 数据集 | 配置 |
|---|---|---|
| UMTCV6_REF | UMTCV6（1920/240/240） | SCSegamba 原版（GEN 系列同款 REF 构建，含已授权的 DySample reshape 修复） |

- 协议：冻结协议（AdamW lr=5e-4 / PolyLR / 50ep / bs=4 / seed=42 / 512×512），统一口径评测；
- 50 个逐 epoch ckpt 全保留；
- 完成后登记 max mIoU/ODS/OIS/F1/P/R/best_ep/后10均，与 G2（0.8849/0.8612/0.8642/0.9164）出对比行。

### 任务 2（推理可视化，不占多少 GPU）：DAWF 权重图 w_k(p)

用 G2 的 best ckpt 在 UMTCV6 test 集挑 4~6 张代表性巷道裂缝图（含斜向裂缝、交叉裂缝、弱对比度裂缝各至少 1 张），前向时 hook 导出 DAWF 的 8 路权重 w_k(p)（softmax(β·s) 之后、加权求和之前），产出：

- 每张图一组 9 宫格：原图+GT 叠加 | 8 个方向（0°/22.5°/…/157.5°）的 w_k(p) 热图（统一色标 viridis，标注各方向角度）；
- 另出 1 张 argmax_k w_k(p) 方向归属图（8 色离散着色）——直观展示"权重方向与裂缝走向贴合"；
- 存 results/thesis_figures/dawf_weights/，PNG 300dpi。

### 任务 3（纯读 ckpt，零训练）：β 收敛曲线

G2 保留的 50 个逐 epoch ckpt 中逐层读出 4 个 β 标量，画 β-epoch 折线图（4 层各一条线，标注终值 [0.305, 0.073, 0.287, 0.035] 及对应层深），存 results/thesis_figures/beta_curve.png。若逐 epoch ckpt 已被裁剪只剩 best+last，则如实上报"数据不足"，**不许重新训练**。

### 任务 4（纯读日志，零训练）：Z1 vs G2 训练稳定性曲线

从两 run 的 history/results JSON 取逐 epoch mIoU，画同图双折线（Z1 行主序 0.8801/后10均0.8701 vs G2 0.8849/后10均0.8838），标注各自后 10 轮均值水平线与峰均差，存 results/thesis_figures/stability_z1_vs_g2.png。

### 完成后
4 个任务全部完成（或如实上报阻塞）后：更新看板 + results_log.md，汇总 UMTCV6_REF vs G2 对比行 + 三张图的文件清单，然后停止。**这是收官批，此后实验阶段结束**。

### 明确不做的
- 除 UMTCV6_REF 外不新增任何训练 run；不碰 EMA/X/S 系列；不调任何超参；
- 数据集目录严格只读；任务 2/3/4 全部为只读推理/读取，不得改动任何 ckpt 与代码（除已授权的 DySample 修复外）。

### 上报
- 每个任务完成后更新看板 + results_log.md；全部完成后停止，每小时拉取等待最终收口
