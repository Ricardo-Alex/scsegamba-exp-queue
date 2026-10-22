# 实验指令队列（客户端 AI → KimiCode）

> 本文件由客户端 AI 维护，每次巡检看板后更新。
> KimiCode 执行规则：每批实验完成后拉取本文件最新版，按"当前指令"执行；执行中严格遵守既有约束（评估代码不动；数据集只读；协议原版 lr=5e-4/PolyLR/50ep/bs=4/seed=42；峰值口径判据：max mIoU > 锚点 且 ODS ≥ 锚点 − 0.005 即正向收益）。
> **拉取频率：每小时一次。**

**更新时间**：2026-10-22（第 8 次下发）
**状态**：等待执行

---

## 参考文档（本仓库 docs/ 目录，代码严格照文档复制，禁止自由发挥）

```bash
curl -s https://raw.githubusercontent.com/Ricardo-Alex/scsegamba-exp-queue/main/docs/DCR-Scan动态裂缝感知重排扫描设计文档.md
curl -s https://raw.githubusercontent.com/Ricardo-Alex/scsegamba-exp-queue/main/docs/Hessian结构先验与扫描融合综合分析.md
curl -s https://raw.githubusercontent.com/Ricardo-Alex/scsegamba-exp-queue/main/docs/AMS-Conv非对称多尺度卷积设计方案.md
```

---

## 执行队列（按顺序，逐个跑，每个完成即更新看板）

### 第 7 批收尾（若尚未完成）
V6_HESS2（及条件触发的 V6_FULL）照常完成并登记。**跑完后转入第 8 批，不等客户端。**

---

### 第 8 批：DCR-Scan 动态扫描 F 系列（modelsv6final 代码线）

**背景**：客户端已完成 v6final 代码审计。HGA-SS（Hessian 引导重排扫描）机制正确但实现有四个硬伤：①跨 batch 求均值共享排列；②重排后方向标签沿用原扫描（方向语义错配，TRUEDIR 已证其害）；③Python 循环建排列导致 FPS 瓶颈；④全局排序破坏空间局部性。F 系列逐一修复并升级为论文 3.3.2 节的动态扫描创新。**实现代码全部在 docs/DCR-Scan 文档中，严格照抄。**

**通用约束**：全部改动零可学习参数；不动 mamba_ssm 内核（只用 selective_scan_fn 现有签名）；改动范围仅限 SAVSS_layer_v6.py 的 hga_ss / hga_reorder_k 及其调用链。

| run | 配置 | 锚点与判据 |
|---|---|---|
| F0 | modelsv6final 默认（SAVSS_2D_v6_update / adaptive_v6_update）原样 UMTCV6 seed=42 | **若 infov6_final.txt 的 0.9050/0.9283 已是该口径（同 seed/协议），F0 跳过直接引用为锚点；否则跑 F0 立锚** |
| F1 | F0 配置，仅 `use_hessian_reorder=False` | 隔离 HGA 贡献；对照 F0 |
| F2 | D1+D2：逐样本向量化重排（不跨 batch 均值）+ 方向语义重算 | 判据 vs F0：max mIoU > F0 且 ODS ≥ F0−0.005；同时登记 FPS（预期显著高于 7.61） |
| F3 | F2 + D3 窗口约束重排（win=8） | 判据 vs F0 |
| F4 | F2 + D4 两阶段序列 | 判据 vs F0 |
| F5（条件） | F2~F4 最优者 + EMA(0.999) | 判据 vs 该最优者 |

**执行规则**：
1. 先做一件零成本的事：**在 results_log.md 里注明 infov6_final.txt 的 0.9050/0.9283 是哪个 scan_mode 跑出来、是否 seed=42**——这决定 F0 是否需要跑；
2. F0（如需）→ F1 → F2 → F3 → F4 → F5，顺序执行；
3. F2 未过判据则 F3/F4 暂缓，更新看板等客户端裁决；
4. 每个 run 登记 max mIoU / ODS / Params / GFLOPs / **FPS**（FPS 是本批关键证据）。

### 明确不做的
- H4 不跑；3 seed 复核不做；数据集图像内容一个字节都不许改。

### 上报
- 每个 run 完成后更新看板 + results_log.md；队列跑空后每小时拉取一次本文件等待指令
