# 实验指令队列（客户端 AI → KimiCode）

> 本文件由客户端 AI 维护，每次巡检看板后更新。
> KimiCode 执行规则：每批实验完成后拉取本文件最新版，按"当前指令"执行；执行中严格遵守既有约束（评估代码不动；数据集只读；协议原版 lr=5e-4/PolyLR/50ep/bs=4/seed=42；峰值口径判据：max mIoU > 锚点 且 ODS ≥ 锚点 − 0.005 即正向收益）。
> **拉取频率：每小时一次。**

**更新时间**：2026-10-23（第 11 次下发，追加 X4/X5）
**状态**：X1 训练中 → X3 → **X4 → X5（新增）** → FINAL_CAP

---

## 已确认结论（勿重复验证）

- F 系列：重排路线证伪；G 系列：**H-A 成立，加权替代重排**，DAWF-G2（0.8785/0.9101）为当前 v6final 线最优；
- X2（cross4+DAWF）0.8753/0.9140 ✗ mIoU 未超 G2——**客户端归因：cross4 只有 0°/90° 两轴，缺 45°/135° 对角覆盖**，X4/X5 即为此补测。

---

## 第 10~11 批合并队列：X 系列（化繁为简平行扫描 × 动态机制）

### 代码改动（SAVSS_2D_v6_v4.py；改动 1/2 为第 10 批已下发内容，若已完成则跳过）

**改动 1**：放宽断言，允许 cross4 使用 hessian 重排（略，见第 10 批）。
**改动 2**：逐路径方向角 path_angles（略，见第 10 批）。

**改动 3（本次新增）**：注册两个新 scan 模式 diag4 / straight8：

```python
# assert 行改为：
assert ATS_SCAN in ('serpentine', 'rowmajor', 'bidi', 'cross4', 'diag4', 'straight8')

# __init__ 的 K 分支改为：
if self.scan_mode == 'rowmajor':   self.K = 1
elif self.scan_mode == 'bidi':     self.K = 2
elif self.scan_mode in ('cross4', 'diag4'): self.K = 4
elif self.scan_mode == 'straight8':         self.K = 8
else:                              self.K = ATS_K   # serpentine

# path_angles 新增两支：
elif self.scan_mode == 'diag4':
    path_angles = [math.pi/4, math.pi/4, 3*math.pi/4, 3*math.pi/4]
elif self.scan_mode == 'straight8':
    path_angles = [0., 0., math.pi/2, math.pi/2,
                   math.pi/4, math.pi/4, 3*math.pi/4, 3*math.pi/4]

# _get_base_orders 新增两支（复用现成原语，禁止新写几何函数）：
elif self.scan_mode == 'diag4':
    d45  = self._build_rotated_serpentine(H, W, math.pi/4)
    d135 = self._build_rotated_serpentine(H, W, 3*math.pi/4)
    orders = [d45, d45[::-1], d135, d135[::-1]]
elif self.scan_mode == 'straight8':
    rm   = self._build_rowmajor(H, W)
    cm   = self._build_colmajor(H, W)
    d45  = self._build_rotated_serpentine(H, W, math.pi/4)
    d135 = self._build_rotated_serpentine(H, W, 3*math.pi/4)
    orders = [rm, rm[::-1], cm, cm[::-1], d45, d45[::-1], d135, d135[::-1]]
```

### 实验矩阵（锚点：当前最优 G2 = 0.8785/0.9101；超越线：max mIoU > 0.8785 且 ODS ≥ 0.9051）

| 顺序 | run | 配置 | 语义 |
|---|---|---|---|
| 1 | X2 | cross4 + dawf + reorder=none | ✅ 已完成（0.8753/0.9140 ✗） |
| 2 | X1 | cross4 + hessian 重排 | 🔄 训练中 |
| 3 | X3 | cross4 + hessian 重排 + dawf | 排队 |
| 4 | **X4** | **diag4 + dawf + reorder=none** | 纯对角 4 路——检验"对角缺失是不是 X2 的缺口" |
| 5 | **X5** | **straight8 + dawf + reorder=none** | 8 方向全覆盖 + 纯直线几何——候选最优解 |
| 6（条件） | FINAL_CAP | X/G 全场 max mIoU 最优者 + EMA(0.999) | 收官 |

执行规则：X1 → X3 → X4 → X5 → FINAL_CAP 顺序执行；每 run 登记 max mIoU/ODS/F1/Params/GFLOPs/FPS/β 收敛值。FINAL_CAP 完成后更新看板并停止，等客户端收口指令。

### 明确不做的
- G4/H4（Δ 调制）不跑；3 seed 复核不做；数据集图像内容一个字节都不许改。

### 上报
- 每个 run 完成后更新看板 + results_log.md；队列跑空后每小时拉取一次本文件等待指令
