# DCR-Scan：动态裂缝感知重排扫描 —— 设计文档与实验方案

> 基座代码：modelsv6final（SAVSS_layer_v6.py，含 HGA-SS）；目标：修复 HGA-SS 四个硬伤并升级为论文 3.3.2 节的动态扫描创新
> 约束：零可学习参数改动、不动 mamba_ssm 内核（只用 selective_scan_fn 现有签名）、评估代码不动、数据集只读

---

## 一、现有 HGA-SS 机制与四个硬伤（代码实证）

机制（`SAVSS_2D_v6.hga_ss`，`SAVSS_2D_v6_update.hga_reorder_k` 同构）：
1. `feat_2d.mean(dim=(0,1))` → batch+通道均值灰度图；
2. 3×3 有限差分核 Ixx/Iyy/Ixy → 闭式特征分解 → crackness（各向异性比）与 crack_angle；
3. 每路 importance = crackness × |cos(crack_angle − θ_k)|；
4. `argsort(importance, descending=True)` 对该路 token 全局重排。

| # | 硬伤 | 代码位置 | 后果 |
|---|---|---|---|
| F1 | 跨 batch 求均值，同 batch 共享排列 | `mean(dim=(0,1))` | 非逐图自适应 |
| F2 | 重排后方向标签沿用原几何扫描（`new_dirs.append(base_d)`） | hga_ss / hga_reorder_k | direction_Bs 注入语义噪声（TRUEDIR 已实证错配比不用更糟） |
| F3 | Python 循环构建排列（`for k in sorted_idx.tolist()`，L=4096×K×4层×每forward） | 同上 | FPS 瓶颈（实测 6.67~7.61 FPS） |
| F4 | 全局排序破坏空间局部性 | argsort 全局进行 | SSM 指数衰减记忆作用在混乱邻域上 |

数学框架：SSM 中 token s 对 t 的贡献 ∝ exp(−λ·序列距离)，与空间距离无关。扫描排列定义有效上下文核，三准则：**局部性保持、语义连贯、方向一致**。HGA 抓住了语义连贯、牺牲了另外两条；DCR-Scan 的目标是三准则同时满足。

---

## 二、DCR-Scan 四个创新件（全部零参数）

### D1 逐样本向量化重排（修 F1+F3）

```python
@torch.no_grad()
def hessian_importance_per_image(self, feat_2d):
    """feat_2d: (B,C,H,W) → 逐图 crackness/angle，仅跨通道均值。"""
    img = feat_2d.mean(dim=1, keepdim=True)                     # (B,1,H,W)
    Ixx = F.conv2d(img, self.kernel_xx, padding=1)
    Iyy = F.conv2d(img, self.kernel_yy, padding=1)
    Ixy = F.conv2d(img, self.kernel_xy, padding=1)
    trace, diff = Ixx + Iyy, Ixx - Iyy
    disc = torch.sqrt(diff * diff / 4.0 + Ixy * Ixy + 1e-12)
    a1 = (trace / 2.0 + disc).abs()
    a2 = (trace / 2.0 - disc).abs()
    crackness = ((a1 - a2) / (a1 + a2 + 1e-8)).clamp(0., 1.)
    angle = (0.5 * torch.atan2(2.0 * Ixy, diff) + math.pi / 2.0) % math.pi
    return crackness, angle                                     # 均 (B,1,H,W)

@torch.no_grad()
def reorder_per_image(self, feat_2d, base_orders):
    """返回 new_orders (B,K,L)、new_invs (B,K,L)。向量化，无 Python token 循环。"""
    B = feat_2d.shape[0]
    crackness, angle = self.hessian_importance_per_image(feat_2d)
    device = feat_2d.device
    L = base_orders[0].shape[0] if torch.is_tensor(base_orders[0]) else len(base_orders[0])
    base = torch.as_tensor(base_orders, device=device)          # (K,L)
    K = base.shape[0]
    imp = crackness.flatten(2)                                  # (B,1,L)
    orders, invs = [], []
    for k in range(K):                                          # K 很小(4/8)，可循环
        theta_k = self.scan_angles[k % len(self.scan_angles)]
        align = torch.abs(torch.cos(angle - theta_k)).flatten(2)
        importance = (imp * align).squeeze(1)                   # (B,L)
        imp_along = importance[:, base[k]]                      # 沿基础序的重要性
        sidx = torch.argsort(imp_along, dim=1, descending=True)
        new_o = base[k].unsqueeze(0).expand(B, -1).gather(1, sidx)
        new_inv = torch.empty_like(new_o)
        new_inv.scatter_(1, new_o, torch.arange(L, device=device).unsqueeze(0).expand(B, -1))
        orders.append(new_o); invs.append(new_inv)
    return torch.stack(orders, 1), torch.stack(invs, 1)         # (B,K,L), (B,K,L)
```

forward 中用 `torch.gather` 替代下标索引：

```python
# x_conv: (B,L,C)；new_orders[:,k]: (B,L)
x_dir = torch.gather(x_conv, 1, new_orders[:, k].unsqueeze(-1).expand(B, L, C_in))
y_k = selective_scan_fn(x_dir.permute(0,2,1).contiguous(), ...)   # (B,C,L)
y_k = torch.gather(y_k.permute(0,2,1), 1, new_invs[:, k].unsqueeze(-1).expand(B, L, C_in))
```

### D2 方向语义重算（修 F2）

```python
@torch.no_grad()
def recompute_dirs(self, new_orders, H, W):
    """new_orders: (B,L) → 方向标签 (B,L)：0=begin,1=右,2=左,3=上,4=下。"""
    idx = new_orders
    i, j = idx // W, idx % W
    di = i[:, 1:] - i[:, :-1]; dj = j[:, 1:] - j[:, :-1]        # (B,L-1)
    d = torch.zeros_like(di)
    d[(dj == 1)] = 1; d[(dj == -1)] = 2; d[(di == -1)] = 3; d[(di == 1)] = 4
    diag = (d == 0)                                             # 对角/跳步按主导轴量化
    dom_j = diag & (di.abs() <= dj.abs()); dom_i = diag & (di.abs() > dj.abs())
    d[dom_j] = torch.where(dj[dom_j] > 0, 1, 2)
    d[dom_i] = torch.where(di[dom_i] > 0, 4, 3)
    return torch.cat([torch.zeros_like(d[:, :1]), d], dim=1)    # 首位 begin=0
```

direction_Bs 变为逐图：(B,L,d_state) → 整理为 (B,d_state,L) 与 selective_scan_fn 的 B 参数兼容（与 Bm 相加后传入）。

### D3 窗口约束重排（修 F4）

```python
@torch.no_grad()
def window_reorder(self, importance, H, W, win=8):
    """importance: (B,L)。窗口蛇形推进 + 窗口内 importance 降序。返回 (B,L)。"""
    B = importance.shape[0]; gw = W // win; gh = H // win
    imp2d = importance.view(B, gh, win, gw, win)
    win_order = []
    for r in range(gh):
        cols = range(gw) if r % 2 == 0 else range(gw - 1, -1, -1)
        for c in cols: win_order.append((r, c))
    blocks = []
    base_idx = (torch.arange(win, device=importance.device)[:, None] * win
                + torch.arange(win, device=importance.device)).flatten()
    for (r, c) in win_order:
        blk = imp2d[:, r, :, c, :].reshape(B, win * win)
        sidx = torch.argsort(blk, dim=1, descending=True)
        lin = base_idx.unsqueeze(0).expand(B, -1).gather(1, sidx)
        gi = r * win + lin // win; gj = c * win + lin % win
        blocks.append(gi * W + gj)
    return torch.cat(blocks, dim=1)                             # (B,L)
```

配套：D3 排列生成后同样走 D2 方向重算。

### D4 两阶段序列（创新卖点件）

```python
@torch.no_grad()
def two_phase_order(self, importance, base_snake_order):
    """相位1=top L//2 by importance（降序）；相位2=其余按基础蛇形序。
    相位2 首 token 方向标签强制 0(begin)。返回 (B,L)。"""
    L = importance.shape[1]; k = L // 2
    sidx = torch.argsort(importance, dim=1, descending=True)
    phase1 = sidx[:, :k]
    in_p1 = torch.zeros_like(importance, dtype=torch.bool)
    in_p1.scatter_(1, phase1, True)
    base = torch.as_tensor(base_snake_order, device=importance.device)
    keep = ~in_p1[:, base]                                      # 基础序下未进相位1的位置
    # 逐行抽取保持基础相对序（定长 k，可用 gather 技巧；直接 for B 循环仅 4 次，可接受）
    rows = [base[keep[b]] for b in range(importance.shape[0])]
    phase2 = torch.stack(rows, dim=0)
    return torch.cat([phase1, phase2], dim=1)
```

---

## 三、实验矩阵（modelsv6final 代码线，UMTCV6，seed=42，峰值口径）

> 判据：max mIoU > 锚点 且 ODS ≥ 锚点 − 0.005 → 正向收益；登记 Params/GFLOPs/FPS。

| run | 配置 | 目的 |
|---|---|---|
| F0 | 锚点确认：modelsv6final 默认（SAVSS_2D_v6_update / adaptive_v6_update）原样跑 UMTCV6 seed=42 | 确认 0.9050/0.9283 口径归属；若 KimiCode 确认 infov6_final 已是该口径，F0 可跳过直接引用 |
| F1 | F0 − HGA（use_hessian_reorder=False） | 隔离 HGA 本身贡献（正负未知，必须先测） |
| F2 | D1+D2：逐样本向量化重排 + 方向语义重算 | 修 F1/F2/F3 |
| F3 | F2 + D3 窗口约束重排（win=8） | 修 F4 |
| F4 | F2 + D4 两阶段序列 | 创新卖点件 |
| F5（条件） | F2~F4 最优者 + EMA(0.999) | 封顶 |

执行规则：F0/F1 先行对照；F2 过了才跑 F3/F4；每步 50 epoch，看板即时更新。先跑哪一个以客户端 INSTRUCTIONS.md 为准。
