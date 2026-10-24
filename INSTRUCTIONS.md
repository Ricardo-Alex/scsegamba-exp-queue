# 实验指令队列（客户端 AI → KimiCode）

> 本文件由客户端 AI 维护，每次巡检看板后更新。
> **拉取频率：每小时一次。**

**更新时间**：2026-10-24（第 15 次下发，消融补齐 + FPS 复测）
**状态**：统一口径已完成 → 执行 Z 系列补齐消融

---

## 已定稿（勿动）

- **最终方案 = G2 配置**：K8 旋转蛇形 + DAWF（M_out 逐像素方向加权）+ 无重排 + 无 EMA；统一口径 0.8849/0.8612/0.9164；
- **评测协议已统一为 SCSegamba 原版**（逐 epoch 测 test 取 max mIoU，ODS/OIS/F1 原版指标）——本批全部 run 必须走统一口径评测（第 14 批改好的评测段）。

---

## 第 15 批：Z 系列——消融矩阵补齐（主人已批准 Z1~Z6 全跑）

### 代码改动（SAVSS_2D_v6_v4.py，仅一处：打分模式从重排开关解耦）

```python
# env 区新增：
ATS_SMODE = os.environ.get('ATS_SMODE', 'full')   # full | crackness | align（仅作用于 DAWF 打分）
assert ATS_SMODE in ('full', 'crackness', 'align')

# DAWF 分支的 scores 计算处改为：
def _dawf_score(self, crackness_b, angle_b, theta_k):
    if ATS_SMODE == 'crackness':
        return crackness_b
    if ATS_SMODE == 'align':
        return torch.abs(torch.cos(angle_b - theta_k))
    return crackness_b * torch.abs(torch.cos(angle_b - theta_k))
# scores = torch.stack([self._dawf_score(crackness_b, angle_b, float(self.path_angles[k]))
#                       for k in range(self.K)], dim=1)
```

### 实验矩阵（锚点 = G2 统一口径 0.8849/0.8612；判据照旧：max mIoU > 锚点 且 ODS ≥ 锚点−0.005）

| 顺序 | run | 配置 | 补哪格 |
|---|---|---|---|
| 1 | Z1 | ATS_SCAN=rowmajor（K=1）+ ATS_FUSE=alpha + reorder=none | 模块间：单路扫描基座 |
| 2 | Z2 | G2 配置 + ATS_SMODE=crackness | DAWF 模块内：仅裂缝置信项 |
| 3 | Z3 | G2 配置 + ATS_SMODE=align | DAWF 模块内：仅方向对齐项 |
| 4 | Z4 | G2 配置 + ATS_GAMMA=0.25 | ATS 模块内：通道划分比例 |
| 5 | Z5 | G2 配置 + ATS_GAMMA=0.75 | 同上 |
| 6 | Z6 | ATS_SCAN=serpentine + ATS_K=4 + ATS_FUSE=dawf + reorder=none | 同几何 K=4 vs K=8 |

执行规则：按序执行（Z1~Z3 优先，是论文必答格）；每个 run 登记统一口径 max mIoU/best_ep/ODS/OIS/F1/后10均/Params/GFLOPs/FPS/β 收敛值；**注意 Z1~Z6 全部为最终方案的消融变体，用统一口径评测段跑（逐 epoch 测 test）**。

### 同批附加任务：FPS 空卡复测
当前 GPU 空闲时段，对以下配置用 bs=1、512×512、空卡条件统一复测 FPS（各测 3 次取均值）：**G2（最终方案）、F0、F1、S1u、X4u**。结果单独列表登记。

### 完成后
Z 系列 + FPS 复测全部完成后，更新看板并停止，等客户端收口指令（下一步：整理第三章实验数据包）。

### 明确不做的
- Δ 调制/重排/EMA 不再跑；3 seed 复核不做；数据集图像内容一个字节都不许改。

### 上报
- 每个 run 完成后更新看板 + results_log.md；队列跑空后每小时拉取一次本文件等待指令
