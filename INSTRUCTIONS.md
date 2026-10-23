# 实验指令队列（客户端 AI → KimiCode）

> 本文件由客户端 AI 维护，每次巡检看板后更新。
> KimiCode 执行规则：每批实验完成后拉取本文件最新版，按"当前指令"执行；执行中严格遵守既有约束（评估代码不动；数据集只读；协议原版 lr=5e-4/PolyLR/50ep/bs=4/seed=42；峰值口径判据：max mIoU > 锚点 且 ODS ≥ 锚点 − 0.005 即正向收益）。
> **拉取频率：每小时一次。**

**更新时间**：2026-10-23（第 10 次下发）
**状态**：G 系列已收官（DAWF 定格 G2 配置）→ 执行 X 系列

---

## 已确认结论（勿重复验证）

- F 系列：重排路线全面证伪（F3/F4/F5 均未过判据）；
- G 系列：**H-A 成立，"加权替代重排"**——DAWF-G2（M_out 逐像素 softmax 加权 + reorder=none）0.8785/0.9101 为当前 v6final 线最优；G1（加权+重排）0.8720 不互补；G3（M_in）0.8708/0.9144（ODS 全场最高，作分析素材）。

---

## 第 10 批：X 系列——化繁为简四向平行扫描 + 动态机制

**主人指令**：扫描路径化繁为简（K=8 旋转蛇形 → cross4 四向平行直线扫描），在简洁路径上分别叠加动态重排 / 动态加权 / 重排+加权。

**理论依据（供理解勿改）**：旋转蛇形的 round() 量化引入几何噪声；cross4 纯直线扫描局部性最优；K=4 时方向对齐度 |cos(θ−φ_k)| 坍缩为水平/垂直两族竞争，θ 估计噪声影响减半。

### 代码改动（SAVSS_2D_v6_v4.py，仅两处）

**改动 1**：放宽断言，允许 cross4 使用 hessian 重排：
```python
# 原: assert ATS_SCAN == 'serpentine' (hessian 模式)
# 改为:
if ATS_REORDER in _HESSIAN_MODES:
    assert ATS_SCAN in ('serpentine', 'cross4'), 'hessian 重排支持 serpentine/cross4'
```

**改动 2**：为各 scan 模式定义逐路径方向角（_score_map 与 DAWF 的 scores 统一改用它，替代 float(self.theta_k[k])）：
```python
# __init__ 中注册
if self.scan_mode == 'cross4':
    path_angles = [0.0, 0.0, math.pi/2, math.pi/2]   # 行正/行反/列正/列反
elif self.scan_mode == 'bidi':
    path_angles = [0.0, 0.0]
elif self.scan_mode == 'rowmajor':
    path_angles = [0.0]
else:
    path_angles = [float(self.theta_k[k]) for k in range(self.K)]
self.register_buffer('path_angles', torch.tensor(path_angles))
# 之后所有 float(self.theta_k[k]) 调用点改为 float(self.path_angles[k])
```

### 实验矩阵（锚点：当前最优 G2 = 0.8785/0.9101；对照基座 F0 = 0.8761/0.9097）

| 顺序 | run | 配置 | 判据 |
|---|---|---|---|
| 1 | X2 | ATS_SCAN=cross4 + ATS_FUSE=dawf + ATS_REORDER=none | max mIoU > 0.8785 且 ODS ≥ 0.9051 → 超越当前最优；同时对照 F0 |
| 2 | X1 | ATS_SCAN=cross4 + ATS_REORDER=hessian（无 dawf） | 同上 |
| 3 | X3 | ATS_SCAN=cross4 + ATS_REORDER=hessian + ATS_FUSE=dawf | 同上 |
| 4（条件） | FINAL_CAP | X/G 全场 max mIoU 最优者 + EMA(0.999) | vs 该最优者自身判据 |

执行规则：X2 → X1 → X3 顺序跑；三者完成后跑 FINAL_CAP；每 run 登记 max mIoU/ODS/F1/Params/GFLOPs/FPS/β 收敛值（dawf run）。FINAL_CAP 完成后更新看板并停止，等客户端收口指令。

### 明确不做的
- G4/H4（Δ 调制）不跑；3 seed 复核不做；数据集图像内容一个字节都不许改。

### 上报
- 每个 run 完成后更新看板 + results_log.md；队列跑空后每小时拉取一次本文件等待指令
