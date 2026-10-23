# 实验指令队列（客户端 AI → KimiCode）

> 本文件由客户端 AI 维护，每次巡检看板后更新。
> KimiCode 执行规则：每批实验完成后拉取本文件最新版，按"当前指令"执行；执行中严格遵守既有约束（评估代码不动；数据集只读；协议原版 lr=5e-4/PolyLR/50ep/bs=4/seed=42；峰值口径判据：max mIoU > 锚点 且 ODS ≥ 锚点 − 0.005 即正向收益）。
> **拉取频率：每小时一次。**

**更新时间**：2026-10-23（第 13 次下发，插入 S1）
**状态**：X6 排队 → **S1（新增）** → FINAL_CAP（以最强形态收官）

---

## 已确认结论（勿重复验证）

- 重排路线证伪（F 系列、X1/X3）；H-A 成立：加权替代重排；当前最优 **G2 = K8 蛇形 + DAWF（0.8785/0.9101）**；
- X4/X5 证伪"化繁为简"：蛇形局部连续性不可替代（同 8 方向下蛇形比直线 +0.0054）。

---

## 队列

### X6（排队中，不变）：uni4 + DAWF
四朝向单扫，补齐 2×2 矩阵。完成即登记。

### 本次新增：S1 = G2 配置 + 多尺度平滑 Hessian 打分（插在 X6 之后、FINAL_CAP 之前）

**动机（供理解勿改）**：G2 的 DAWF 打分来自无平滑 3×3 单尺度差分，噪声大；verify_hessian 已实测多尺度平滑 vesselness 的裂缝/背景对比度 9.83 倍。打分 s_k(p) 是对齐度的含噪估计，softmax 对噪声敏感——降噪即提质。**零新增参数**（打分在 @torch.no_grad 下）。

**配置**：ATS_SCAN=serpentine（K8 默认）+ ATS_FUSE=dawf + ATS_REORDER=none + **ATS_SCORE=ms**。

**代码改动（SAVSS_2D_v6_v4.py，新增 env 开关 + 替换 _hessian_analysis 内部实现）**：

```python
# 文件头 env 区新增：
ATS_SCORE = os.environ.get('ATS_SCORE', 'ss')   # ss=单尺度(现状默认) | ms=多尺度
assert ATS_SCORE in ('ss', 'ms')

# __init__ 新增：可分离高斯核缓存
self._gauss_cache = {}

def _gaussian_blur(self, img, sigma):
    """img: (B,1,H,W)。可分离高斯平滑，σ∈{1,2,4}。"""
    if sigma not in self._gauss_cache:
        r = max(1, int(3 * sigma))
        x = torch.arange(-r, r + 1, dtype=torch.float32, device=img.device)
        g = torch.exp(-x * x / (2 * sigma * sigma)); g = g / g.sum()
        self._gauss_cache[sigma] = (g, r)
    g, r = self._gauss_cache[sigma]
    img = F.conv2d(img, g.view(1, 1, 1, -1), padding=(0, r))
    img = F.conv2d(img, g.view(1, 1, -1, 1), padding=(r, 0))
    return img

def _hessian_analysis(self, x_2d):
    """ms 模式：σ∈{1,2,4} 逐尺度计算 crackness/angle，crackness 取跨尺度 max，
    angle 取 argmax 尺度对应值（gather）。"""
    F_mean = x_2d.mean(dim=1, keepdim=True)                     # (B,1,H,W)
    if ATS_SCORE == 'ss':
        blurred = [F_mean]
    else:
        blurred = [self._gaussian_blur(F_mean, s) for s in (1.0, 2.0, 4.0)]
    cs, as_ = [], []
    for b in blurred:
        Fxx = F.conv2d(b, self.K_xx, padding=1)
        Fyy = F.conv2d(b, self.K_yy, padding=1)
        Fxy = F.conv2d(b, self.K_xy, padding=1)
        th, dh = (Fxx + Fyy) / 2.0, (Fxx - Fyy) / 2.0
        disc = torch.sqrt(dh * dh + Fxy * Fxy + 1e-8)
        a1 = (th + disc).abs(); a2 = (th - disc).abs()
        c = ((a1 - a2) / (a1 + a2 + 1e-8)).clamp(0., 1.)
        a = (0.5 * torch.atan2(2.0 * Fxy, Fxx - Fyy + 1e-8) + math.pi / 2.0) % math.pi
        cs.append(c); as_.append(a)
    if ATS_SCORE == 'ss':
        return cs[0].squeeze(1), as_[0].squeeze(1)
    cstack = torch.stack(cs, dim=1)                             # (B,S,1,H,W)
    cmax, amax = cstack.max(dim=1)                              # (B,1,H,W)
    astack = torch.stack(as_, dim=1)
    angle = astack.gather(1, amax.unsqueeze(1)).squeeze(1)      # (B,1,H,W)
    return cmax.squeeze(1), angle.squeeze(1)
```

**判据**：锚点 G2（0.8785/0.9101）：max mIoU > 0.8785 且 ODS ≥ 0.9051 → 正向；登记 β 收敛值（预期比 G2 更大——打分降噪后方向选择应更自信，写进 results_log 作论文证据）。S1 未过判据不影响 FINAL_CAP 执行（FINAL_CAP 以届时全场最优为基座）。

### FINAL_CAP（条件 run，不变）：X/G/S 全场 max mIoU 最优者 + EMA(0.999)
S1 过判据则 FINAL_CAP = S1+EMA；否则 = G2+EMA。完成后更新看板并停止，等客户端收口指令。

### 明确不做的
- Δ 调制不跑；3 seed 复核不做；数据集图像内容一个字节都不许改。

### 上报
- 每个 run 完成后更新看板 + results_log.md；队列跑空后每小时拉取一次本文件等待指令
