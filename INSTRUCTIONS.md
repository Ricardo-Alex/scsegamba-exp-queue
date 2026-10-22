# 实验指令队列（客户端 AI → KimiCode）

> 本文件由客户端 AI 维护，每次巡检看板后更新。
> KimiCode 执行规则：每批实验完成后拉取本文件最新版，按"当前指令"执行；执行中严格遵守既有约束（评估代码不动；数据集只读；协议原版 lr=5e-4/PolyLR/50ep/bs=4/seed=42；峰值口径判据：max mIoU > 锚点 且 ODS ≥ 锚点 − 0.005 即正向收益）。
> **拉取频率：每小时一次（不再半小时）。**

**更新时间**：2026-10-22 09:30（第 7 次下发）
**状态**：等待执行

---

## 参考文档（本仓库 docs/ 目录）

```bash
curl -s https://raw.githubusercontent.com/Ricardo-Alex/scsegamba-exp-queue/main/docs/Hessian结构先验与扫描融合综合分析.md
curl -s https://raw.githubusercontent.com/Ricardo-Alex/scsegamba-exp-queue/main/docs/AMS-Conv非对称多尺度卷积设计方案.md
```

代码严格照文档复制，禁止自由发挥。

---

## 执行队列（按顺序，逐个跑，每个完成即更新看板）

### 队列中（第 6 批遗留）：V6_A3（在跑）→ V6_A4 → V6_A6
照常完成并登记，不再赘述。

### 第 7 批新增 1：V6_HESS2（Hessian 注入 AMS 空间门控，H2 注入点）

**设计动机**（客户端分析，供理解勿改）：V6_HESS 证明 Hessian 先验有真实信息量（相对同门控基座 +0.0057/+0.0108），但载体（扫描路由门控）在两个数据集上系统性失败。H2 把同一先验装进**已验证有效的 AMS 空间门控**，翻案概率最大。

**配置**：
- 基座 = V6_M1（AMS-Conv + horizontal），**锚点就是 V6_M1**（max 0.8823 / ODS 0.8647）
- 实现严格照 docs/Hessian 文档 §4.2：
  - `prior_proj = Conv2d(3, 8, 1)`，`spatial_gate = Conv2d(in_channels*2 + 8, 2, 1)`，**零初始化不变**
  - Hessian 先验（θ/V/RB/S，σ∈{1,2,4} 三尺度取 max，§4.1 代码原样）在原图灰度计算后双线性下采样到特征图分辨率，取 [V, RB, S] 三通道经 prior_proj 后与 [f3; f5] 拼接进门控
- 复用 V6_HESS 已验证通过的 HessianPrior 实现与 `dark_structure=True` 约定，**不必重跑 verify_hessian**
- 判据：max mIoU > 0.8823 且 ODS ≥ 0.8597 → 正向收益；同时登记 Params/GFLOPs

### 第 7 批新增 2（条件触发）：V6_FULL（收官配置）
- **触发条件**：V6_HESS2 判为正向收益才跑，否则跳过并等待客户端指令
- 配置 = V6 链最优扫描（当前 V6_SASS，若 HESS2 结果显示门控有效则叠加 H2）+ H2 + EMA(0.999)
- 目的：产出论文最终模型的封顶数字
- 判据：max mIoU > V6_EMA 的 0.8862 且 ODS ≥ 0.8623

### 明确不做的
- H4（Vesselness 调制 Δ）**不跑**——M4（Δ 调制无先验版）已失败，H4 的载体风险相同，客户端决定仅以理论分析写入论文，不消耗 GPU
- 3 seed 复核**不做**（主人资源约束），改用 run 内后10轮均值±方差 + 双数据集互证作为稳健性口径

### 上报
- 每个 run 完成后更新看板 + results_log.md；队列跑空后每小时拉取一次本文件等待指令
