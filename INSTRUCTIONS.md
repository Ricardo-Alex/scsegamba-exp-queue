# 实验指令队列（客户端 AI → KimiCode）

> 本文件由客户端 AI 维护，每次巡检看板后更新。
> KimiCode 执行规则：每批实验完成后拉取本文件最新版，按"当前指令"执行；执行中严格遵守既有约束（评估代码不动；数据集只读；协议原版 lr=5e-4/PolyLR/50ep/bs=4/seed=42；峰值口径判据：max mIoU > 锚点 且 ODS ≥ 锚点 − 0.005 即正向收益）。
> **拉取频率：每小时一次。**

**更新时间**：2026-10-23（第 12 次下发，追加 X6-uni4）
**状态**：X1 训练中 → X3 → X4 → X5 → **X6（新增）** → FINAL_CAP

---

## 已确认结论（勿重复验证）

- 重排路线证伪（F 系列）；**H-A 成立：加权替代重排**（G 系列）；DAWF-G2（0.8785/0.9101）为当前最优；
- X2（cross4+DAWF）0.8753/0.9140 ✗——cross4 为 0°/90° 两轴双向，缺对角朝向。

---

## 队列：X 系列（化繁为简平行扫描 × 动态机制）

### 已完成/在跑（不变）
X2 ✅ → X1 🔄 → X3 排队 → X4（diag4+dawf）排队 → X5（straight8+dawf）排队

### 本次新增：X6 = uni4 + DAWF（主人原始意图：四朝向并行单扫）

**语义**：0°、90°、45°、135° 四个朝向各扫一次（单向），K=4。检验"K=4 预算下方向覆盖 vs 双向性孰重要"，且 DAWF 四路权重互不相同（区别于 cross4 的两族竞争）。

**代码改动（SAVSS_2D_v6_v4.py，改动 1~3 若已完成则跳过；本次为改动 4）**：

```python
# assert 行追加 'uni4'：
assert ATS_SCAN in ('serpentine','rowmajor','bidi','cross4','diag4','straight8','uni4')

# K 分支：uni4 归入 K=4
elif self.scan_mode in ('cross4', 'diag4', 'uni4'): self.K = 4

# path_angles 新增一支（注意：四路各不相同，顺序与 orders 严格对应）：
elif self.scan_mode == 'uni4':
    path_angles = [0.0, math.pi/2, math.pi/4, 3*math.pi/4]

# _get_base_orders 新增一支（全部复用现成原语，禁止新写几何函数）：
elif self.scan_mode == 'uni4':
    rm   = self._build_rowmajor(H, W)                      # 0°
    cm   = self._build_colmajor(H, W)                      # 90°
    d45  = self._build_rotated_serpentine(H, W, math.pi/4) # 45°
    d135 = self._build_rotated_serpentine(H, W, 3*math.pi/4)
    orders = [rm, cm, d45, d135]
```

### 更新后的执行矩阵（锚点 G2 = 0.8785/0.9101；超越线：max mIoU > 0.8785 且 ODS ≥ 0.9051）

| 顺序 | run | 配置 | 状态 |
|---|---|---|---|
| 1 | X2 | cross4+dawf | ✅ 0.8753/0.9140 ✗ |
| 2 | X1 | cross4+hessian 重排 | 🔄 |
| 3 | X3 | cross4+重排+dawf | 排队 |
| 4 | X4 | diag4+dawf | 排队 |
| 5 | X5 | straight8+dawf | 排队 |
| 6 | **X6** | **uni4+dawf**（ATS_SCAN=uni4, ATS_FUSE=dawf, ATS_REORDER=none） | **新增** |
| 7（条件） | FINAL_CAP | 全场 max mIoU 最优者 + EMA(0.999) | 收官 |

执行规则：按序执行；每 run 登记 max mIoU/ODS/F1/Params/GFLOPs/FPS/β 收敛值。FINAL_CAP 完成后更新看板并停止，等客户端收口指令。

### 明确不做的
- Δ 调制不跑；3 seed 复核不做；数据集图像内容一个字节都不许改。

### 上报
- 每个 run 完成后更新看板 + results_log.md；队列跑空后每小时拉取一次本文件等待指令
