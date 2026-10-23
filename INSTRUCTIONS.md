# 实验指令队列（客户端 AI → KimiCode）

> 本文件由客户端 AI 维护，每次巡检看板后更新。
> **拉取频率：每小时一次。**

**更新时间**：2026-10-23（第 14 次下发，⚠️ 评测协议统一纠偏，最高优先级）
**状态**：FINAL_CAP_S1 跑完后，立即执行本批纠偏流程

---

## ⚠️ 主人裁决（最高优先级，立即生效）

**两条线评测协议必须统一，统一标准为 SCSegamba 原版协议**：
- 逐 epoch 直接在**测试集**上评测，取 50 轮 max mIoU（与老线 M/V6/A 系列口径一致）；
- 指标用 SCSegamba 原版评测代码的 ODS/OIS/F1（不是 F1@0.5）；
- v6final 线此前的"val 选模 + test 一次 + F1@0.5"口径作废，仅留档备查。

**关键事实（执行前必读）**：训练权重与评测协议无关——所有已跑 run 的模型不需要重训，只需用 SCSegamba 口径**回放评测**。

## 纠偏流程（按顺序）

### Step 0：checkpoint 盘点（立即，零成本）
检查 F0/F1/F3/F4/F5/G1/G2/G3/X1~X6/S1/FINAL_CAP_S1 各 run 是否保存了**逐 epoch checkpoint**（而非仅 best-val ckpt），在看板/results_log 公布清单。这决定走 Step 1 还是 Step 2。

### Step 1（有逐 epoch ckpt 的 run）：回放评测
对每个 run 的 50 个 epoch ckpt，用 SCSegamba 原版评测代码逐轮测 **test 集**，登记：max mIoU / best_ep(test) / ODS / OIS / F1 / 后10轮均值。**不改动任何训练产物，只读 ckpt。** 全部结果以新表"统一口径（SCSegamba 协议）"登记在看板，旧口径行保留但标注"作废口径"。

### Step 2（无逐 epoch ckpt 的 run）：按优先级补训
修改 v6final 训练脚本评测段：每 epoch 用 SCSegamba 原版评测直接测 test 集并保存 ckpt。补训优先级：
1. **S1**（最终方案，必须有）
2. **F0**（基座锚点）
3. **G2**（DAWF 关键对照）
4. X2 / X5 / X6（矩阵代表格）
5. G1 / G3 / X1 / X3 / X4 / F3 / F4（矩阵其余格，按序补齐）

### Step 3：FINAL_CAP_S1 处理
当前在跑的 FINAL_CAP_S1 让它跑完（数据留档），随后同样走 Step 1/2 出统一口径数字。**不再启动任何新训练，直到统一口径表完成。**

### 明确不做的
- 不再用 val-select 口径报任何新数字；数据集图像内容一个字节都不许改。

### 上报
- Step 0 盘点结果**立即**上看板（不用等跑完）；每个 run 回放/补训完成即更新；全部完成后等客户端收口指令
