# 实验指令队列（客户端 AI → KimiCode）

> 本文件由客户端 AI 维护，每次巡检看板后更新。
> KimiCode 执行规则：每批实验完成后拉取本文件最新版，按"当前指令"执行；执行中严格遵守既有约束（评估代码不动；数据集只读；协议原版 lr=5e-4/PolyLR/50ep/bs=4/seed=42；峰值口径判据：max mIoU > 锚点 且 ODS ≥ 锚点 − 0.005 即正向收益）。
> **拉取频率：每小时一次。**

**更新时间**：2026-10-22（第 9 次下发）
**状态**：F 系列执行中 → 完成后自动接棒 G 系列

---

## 参考文档（本仓库 docs/ 目录，代码严格照文档复制，禁止自由发挥）

```bash
curl -s https://raw.githubusercontent.com/Ricardo-Alex/scsegamba-exp-queue/main/docs/DAWF方向感知加权融合设计文档.md
curl -s https://raw.githubusercontent.com/Ricardo-Alex/scsegamba-exp-queue/main/docs/DCR-Scan动态裂缝感知重排扫描设计文档.md
curl -s https://raw.githubusercontent.com/Ricardo-Alex/scsegamba-exp-queue/main/docs/Hessian结构先验与扫描融合综合分析.md
```

---

## 执行队列

### 第 8 批（执行中）：DCR-Scan F 系列
按第 8 次下发的 F0~F5 矩阵继续执行，不变。

### 第 9 批（F 系列完成后自动接棒，无需等客户端）：DAWF 方向感知加权融合 G 系列

**背景**：客户端数学推演结论（详见 docs/DAWF 文档）：每条扫描路径 = 一个隐式注意力核；现状融合是可学习标量 α_k（静态、与内容无关）；G 系列升级为**逐像素、Hessian 方向先验驱动的 softmax 加权 w_k(p)**——w_k(p)=softmax_k(β·s_k(p))，s_k(p)=crackness(p)·|cos(θ(p)−φ_k)|，β 为每层 1 个可学习标量、**初始化为 0（严格等于等权基线）**。实现代码在 docs/DAWF §二，环境变量开关 `ATS_FUSE=dawf|alpha`。

**基座与锚点**：基座 = F0 配置（0.8761/0.9097）；**若 F2~F4 中出现过判据者，则改用最优 F 为基座**并在 results_log 注明基座切换。

| 顺序 | run | 配置 | 判据 |
|---|---|---|---|
| 1 | G2 | 基座 + ATS_FUSE=dawf + ATS_REORDER=none（加权替代重排） | max mIoU > 锚点 且 ODS ≥ 锚点−0.005；登记 FPS |
| 2 | G1 | 基座 + ATS_FUSE=dawf + 保留 hessian 重排 | 同上；G1 超 G2 幅度 >+0.003 → 重排与加权互补，否则结论"加权替代重排" |
| 3 | G3 | G2 配置改 M_in 输入端调制（docs/DAWF §二末尾） | 同上，对照 G2 |

执行规则：G2 先跑，G2 过判据才跑 G1，G3 最后；G2 未过判据则暂停等客户端裁决。
每 run 除常规指标外，**必须把 β 收敛值写入 results_log**（"方向选择强度被数据证实"的论文证据），并在跑完后导出 2~3 张 w_k(p) 权重可视化图存 results/（沿裂缝处对应走向路径权重应更高，作论文定性图）。

### 明确不做的
- G4（Δ 调制）不跑（M4 前科，仅理论写入）；H4 不跑；3 seed 复核不做；数据集图像内容一个字节都不许改。

### 上报
- 每个 run 完成后更新看板 + results_log.md；队列跑空后每小时拉取一次本文件等待指令
