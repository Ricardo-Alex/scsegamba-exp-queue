# 实验指令队列（客户端 AI → KimiCode）

> 本文件由客户端 AI 维护，每次巡检看板后更新。
> KimiCode 执行规则：每批实验完成后拉取本文件最新版，按"当前指令"执行；执行中严格遵守既有约束（评估代码不动；数据集只读——**严禁对数据集目录任何写操作**；协议原版 lr=5e-4/PolyLR/50ep/bs=4/seed=42；峰值口径判据：max mIoU > 锚点 且 ODS ≥ 锚点 − 0.005 即正向收益）。

**更新时间**：2026-10-22 02:00（第 6 次下发）
**状态**：等待执行

---

## 参考文档（本仓库 docs/ 目录，先拉取再动手）

```bash
curl -s https://raw.githubusercontent.com/Ricardo-Alex/scsegamba-exp-queue/main/docs/Hessian结构先验与扫描融合综合分析.md
curl -s https://raw.githubusercontent.com/Ricardo-Alex/scsegamba-exp-queue/main/docs/AMS-Conv非对称多尺度卷积设计方案.md
```

两份文档含完整实现代码，**严格按文档代码实现，禁止自由发挥**。

---

## 执行队列（严格按顺序，逐个跑，每个完成即更新看板）

当前 V6_ROUTE 跑完后，依次执行：

### 第 1 个：V6_EMA（已在队列，不变）
- 链上前 4 步峰值最优配置 + EMA(0.999)，沿用 P2 的 EMA 实现

### 第 2 个：V6_HESS（Hessian 方向先验 × 扫描路由门控，本次新增重点）

**训练前必做验证（不通过不准开训）**：
1. 按 docs/Hessian 文档第五节实现 `verify_hessian.py`，在 5~10 张 V6 训练图上运行（**只读**数据集，可视化图写到 `results/` 目录）；
2. 检查项：θ 方向场沿裂缝切向；V 热图裂缝处高响应（若背景高亮则 `dark_structure` 符号约定反了，翻转后重验）；GT 裂缝区 V 均值 / 背景区 V 均值 > 2；
3. 把验证图和统计比值发到看板/results_log.md，**再开训**。

**配置**：
- 基座 = V6 链峰值最优扫描（当前看是 SASS；若 ROUTE/EMA 结果改变最优，以最新为准）
- 注入方式 = 文档 §4.3（H1/H3）：Hessian θ 方向场与 4 路扫描方向 {0°,90°,45°,135°} 的余弦平方对齐度 × V，拼接进零初始化 `route_gate`（`Conv2d(d_inner+4, 4, 1)`）
- Hessian 在**原图灰度**计算（σ∈{1,2,4} 三尺度取 max），双线性下采样到 64×64 特征图分辨率
- 代码严格照文档 §4.1 + §4.3 复制；不动 mamba_ssm 内核、不动评估代码
- **判据**：相对同基座无先验版本（V6_ROUTE 或最优扫描），max mIoU 更高且 ODS 不劣 → 正向收益

### 第 3 个：V6_M0（补锚点）
- PlainConv + horizontal，即 M0 配置原样搬 V6
- 用途：V6 上 M0→M1 的 AMS-Conv 收益对照（论文消融表第一行）

### 第 4~9 个：V6 上 AMS-Conv 模块内消融（A 系列，基座=horizontal 扫描）

严格对照 docs/AMS 文档第四节定义与实现备注，逐个跑：

| 顺序 | run 名 | 配置 |
|---|---|---|
| 4 | V6_A0 | 原 GBC（=M0 的卷积模块，若与 V6_M0 完全同配置则直接复用 V6_M0 结果，不重跑） |
| 5 | V6_A1 | 仅 3×3 支路 |
| 6 | V6_A2 | 仅 5×5 支路 |
| 7 | V6_A3 | 3×3+5×5 等权相加（无门控） |
| 8 | V6_A4 | 3×3+5×5 + SKNet 式**通道级**门控（GAP→1×1 conv→softmax→图级标量权重），关键对照组 |
| 9 | V6_A6 | A5 的 5×5 换 5×1+1×5 条带分解（`strip_large_kernel=True`） |

- A5 即 V6_M1，已完成，直接引用
- 每个 run 登记 Params/GFLOPs 到复杂度表
- 全部完成后，做**门控权重可视化**（w5 热图叠 GT，3~5 张代表性测试图，存 results/）——论文明细素材

### 判据与上报
- 全程峰值口径：max mIoU > 对应锚点 且 ODS ≥ 锚点 − 0.005 → 正向收益
- A 系列锚点 = V6_A0（或 V6_M0）；V6_HESS 锚点 = 同基座无先验版
- 每个 run 跑满 50 epoch、seed=42，看板按 run 名新增行 + 逐 epoch 轨迹
- 全部完成后在 results_log.md 写总结块，等待客户端最终结论

## 预算提醒
本队列共约 8 个 run（含已在跑的），单 run ~1~1.5h，总计约 10~12 小时。这是论文消融的收官批次，之后进入 3 seed 复核阶段（等客户端指令）。
