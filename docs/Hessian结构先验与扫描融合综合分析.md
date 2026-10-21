# Hessian 结构先验 × 四向扫描 × AMS-Conv：综合分析与落地方案

> 承接文档：《扫描方式创新四方案与改进代码.md》《AMS-Conv非对称多尺度卷积设计方案.md》《KimiCode执行指导手册.md》
> 基线：SCSegamba（arXiv:2503.01113）；目标章节：3.3.1 AMS-Conv、3.3.2 ATS 扫描、3.3.3 空间门控融合

---

## 一、文献核查结论（新颖性红线）

| 已有工作 | 做了什么 | 与本方案的关系 |
|---|---|---|
| Frangi 滤波器（MICCAI 1998 系列） | Hessian 特征值 → vesselness 管状结构增强 | 理论源头，必引 |
| Frangi-Net（arXiv:1907.06194） | 把 Frangi 每步实现为网络层：卷积核初始化为高斯二阶导、可学习 β/c，全网络仅 6525 参数 | **工程可行性先例**：证明 Hessian 可微实现成立 |
| Deep Vesselness（MICCAI 2019, Springer） | 用 DNN 学习"特征值→vesselness"的映射，替代手工公式 | 可选升级路径（先固定公式，后消融可学习版） |
| 地铁隧道裂缝提取（ICJE 2026 等） | Frangi 仅作**预处理**：增强线状结构后再进 YOLO | 说明裂缝-血管同构已被工程界认可，但**没人把 Hessian 放进深度模型内部** |
| 3D 混凝土裂缝分割（arXiv:2112.09493） | Hessian percolation 等经典方法，性能不及学习方法 | 佐证纯手工方法已被深度学习超越——我们的定位是"先验注入"而非"先验替代" |
| **Hessian × Mamba 扫描/路由** | **未检索到任何先例** | 本方案的独占创新空间 |

**结论：Hessian 先验引导的各向异性扫描与路由门控，是一个"组件都有出处、组合没人做过"的理想毕业论文创新——每个零件都有文献背书（暴雷风险低），组合方式独占（新颖性达标）。**

---

## 二、为什么 Hessian 先验和 SCSegamba 是"天作之合"

### 2.1 裂缝与血管的形态同构

二者都是**低对比度背景上的细长暗色管状结构**——这正是 Frangi vesselness 的假设前提。地下巷道图像（低照度、宽度变化剧烈、背景噪声重）比视网膜图像更难，因此更需要强结构先验。

### 2.2 Hessian 能提供什么（2D 情形）

对图像 $I$ 在尺度 $\sigma$ 下的 Hessian 矩阵 $H_\sigma = \nabla^2 G_\sigma * I$，2×2 对称矩阵有**闭式特征分解**（无需 `torch.linalg.eigh`，数值稳定且天然可微）：

$$\lambda_{\pm} = \frac{H_{xx}+H_{yy}}{2} \pm \sqrt{\left(\frac{H_{xx}-H_{yy}}{2}\right)^2 + H_{xy}^2}$$

记 $|\lambda_1| \le |\lambda_2|$，得到四个先验量：

| 先验量 | 公式 | 物理意义 | 对应改进点 |
|---|---|---|---|
| 主方向 $\theta$ | $\theta = \frac{1}{2}\mathrm{atan2}(2H_{xy},\; H_{xx}-H_{yy})$，裂缝走向 = 垂直于 $\lambda_2$ 特征向量 | 局部细长结构的切向 | **→ S1 路由门控、S4 Δ 调制** |
| Vesselness $V$ | $V=\exp(-\frac{R_B^2}{2\beta^2})(1-\exp(-\frac{S^2}{2c^2}))$（暗结构需 $\lambda_2>0$，裂缝为暗线，符号约定需在实现时验证） | "这里有多像裂缝"的置信度 | **→ AMS 门控、S4 Δ 调制** |
| Blobness $R_B=|\lambda_1|/|\lambda_2|$ | 比值小 ⇒ 细长；比值大 ⇒ 斑块 | 抑制块状噪声（巷道壁面污渍） | → 门控辅助特征 |
| Structureness $S=\sqrt{\lambda_1^2+\lambda_2^2}$ | 小 ⇒ 平坦背景 | 区分结构与无结构区域 | → 门控辅助特征 |

### 2.3 与现有四个扫描方案 + AMS-Conv 的对接矩阵

关键洞察：**我们之前设计的每个创新点都"缺一个可靠的先验输入"，Hessian 恰好全补上了。**

| 创新点 | 原设计的先验来源 | 换成 Hessian 后的收益 | 风险 |
|---|---|---|---|
| S1 扫描路由注意力 | 纯内容特征（或 Sobel 一阶梯度） | 二阶+多尺度，低照度巷道图下方向估计更稳；θ 与 4 条扫描路径方向可直接算对齐度 | **低，主推** |
| AMS-Conv 空间门控 | 纯特征拼接 [f3; f5] | V 图告诉门控"哪里有裂缝"，S/R_B 告诉门控"尺度/形态置信度" | **低，主推** |
| S2 Hilbert 混合扫描 | 无（固定排列） | 不变；Hessian 不影响排列生成 | — |
| S4 Δ 调制 | 可学习缩放（无先验） | V 高（确信的裂缝段）→ Δ 小、长记忆；背景 → Δ 大、快遗忘 | 中，加分项 |
| 3.3.3 空间门控融合 | PAF 原样 | V 图可作为跨分支融合的引导图 | 低，可选 |

**叙事升级**：引入 Hessian 后，第三章的三节被同一条主线贯穿——**"二阶结构先验驱动"**：3.3.1 卷积尺度选择由 Hessian 形态置信度引导，3.3.2 扫描路由由 Hessian 主方向引导，3.3.3 融合由 Hessian 显著性引导。章名"方向感知状态空间模型"名副其实，答辩逻辑闭环。

---

## 三、推荐落地组合与风险矩阵

| 编号 | 组合 | 新增参数 | 风险 | 预期收益 | 定位 |
|---|---|---|---|---|---|
| H1 | Hessian 方向场 → S1 路由门控 | ~2K | **低** | +0.2~0.5%（叠加在 S1 上） | 主创新 |
| H2 | Vesselness/形态量 → AMS 门控输入 | ~1K | **低** | +0.1~0.3% | 主创新（3.3.1 增强） |
| H3 | 方向-路径对齐度逐 token 门控（利用 `direction_Bs` 已有的方向标记 d1~d4） | ~1K | 低-中 | 视实现 | S1 的增强变体 |
| H4 | Vesselness → S4 Δ 调制 | ~0.5K | **中** | 不确定 | 加分项，最后做 |
| H5 | β/c 可学习化（Frangi-Net 式） | 2×n_scales | 低 | 消融素材 | 消融对比项 |

**防爆雷设计（与此前方案同一哲学）：**
1. Hessian 模块**零可学习参数**（固定高斯二阶导卷积核 + 闭式公式），不引入任何训练不稳定因素；
2. 所有注入点都在**零初始化的门控**之后——Hessian 先验即使全错（极端噪声图），门控初始状态仍退化为"均匀路由/均值融合"的基线行为，网络可在训练中学会忽略先验；
3. Hessian 在**原图尺度计算**后双线性下采样到 64×64，物理意义明确，且与特征学习解耦。

**已知风险与对策：**
- 巷道壁面纹理/水渍会产生虚假 ridge 响应 → V 公式中的 $R_B$（细长度）和 $S$（结构度）项本身就是为此设计的；且先验只作用于门控，不直接决定输出；
- 暗色裂缝的符号约定（Frangi 原式针对亮血管，裂缝是暗线）：实现时取 $V$ 的暗结构变体（$\lambda_2 > 0$ 时置 0 的条件反转），**此约定必须先用 5~10 张图可视化验证再进训练**（见第五节验证脚本）；
- 多尺度 $\sigma$ 选择：裂缝宽度 1~10px，建议 $\sigma \in \{1, 2, 4\}$ 三尺度取 max 响应（Frangi 标准做法）。

---

## 四、完整实现代码

### 4.1 Hessian 结构先验模块（零参数，可直接加入项目）

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_2nd_kernels(sigma, ksize=None):
    """生成高斯二阶导数卷积核 Gxx, Gxy, Gyy（Frangi-Net 同款构造）。"""
    if ksize is None:
        ksize = 2 * int(math.ceil(3 * sigma)) + 1
    ax = torch.arange(ksize, dtype=torch.float32) - ksize // 2
    xx, yy = torch.meshgrid(ax, ax, indexing='xy')
    r2 = (xx ** 2 + yy ** 2) / (sigma ** 2)
    g = torch.exp(-r2 / 2) / (2 * math.pi * sigma ** 2)
    gxx = g * (xx ** 2 - sigma ** 2) / sigma ** 4
    gyy = g * (yy ** 2 - sigma ** 2) / sigma ** 4
    gxy = g * (xx * yy) / sigma ** 4
    # 零均值化（消除直流响应）
    gxx = gxx - gxx.mean()
    gyy = gyy - gyy.mean()
    gxy = gxy - gxy.mean()
    return gxx, gxy, gyy


class HessianPrior(nn.Module):
    """多尺度 Hessian 结构先验：输出主方向 theta、vesselness V、blobness RB、structureness S。
    零可学习参数；输入可为原图灰度 (B,1,H,W) 或特征均值图。"""

    def __init__(self, sigmas=(1.0, 2.0, 4.0), beta=0.5, c=None, dark_structure=True):
        super().__init__()
        self.sigmas = sigmas
        self.beta = beta
        self.c = c  # None -> 按 Frangi 惯例取 max(S)/2
        self.dark_structure = dark_structure
        for i, s in enumerate(sigmas):
            gxx, gxy, gyy = gaussian_2nd_kernels(s)
            self.register_buffer(f'kxx_{i}', gxx[None, None])
            self.register_buffer(f'kxy_{i}', gxy[None, None])
            self.register_buffer(f'kyy_{i}', gyy[None, None])

    def single_scale(self, x, i):
        kxx, kxy, kyy = getattr(self, f'kxx_{i}'), getattr(self, f'kxy_{i}'), getattr(self, f'kyy_{i}')
        pad = kxx.shape[-1] // 2
        Hxx = F.conv2d(x, kxx, padding=pad)
        Hxy = F.conv2d(x, kxy, padding=pad)
        Hyy = F.conv2d(x, kyy, padding=pad)
        # 2x2 对称矩阵闭式特征分解（数值稳定、可微）
        tr = (Hxx + Hyy) / 2
        det_sqrt = torch.sqrt(((Hxx - Hyy) / 2) ** 2 + Hxy ** 2 + 1e-12)
        lam1 = tr - det_sqrt   # |lam1| <= |lam2| 不保证成立时用幅值排序:
        lam2 = tr + det_sqrt
        swap = (lam1.abs() > lam2.abs())
        l1 = torch.where(swap, lam2, lam1)
        l2 = torch.where(swap, lam1, lam2)
        # 主方向: lam2 特征向量方向（垂直于裂缝走向），切向 = theta + pi/2
        theta = 0.5 * torch.atan2(2 * Hxy, Hxx - Hyy + 1e-12)
        RB = (l1.abs() / (l2.abs() + 1e-12))
        S = torch.sqrt(l1 ** 2 + l2 ** 2 + 1e-12)
        # 暗结构（裂缝）: 法向曲率 lam2 > 0 时为暗线 -> 亮结构的排除条件反转
        cond = l2 > 0 if self.dark_structure else l2 < 0
        c_val = self.c if self.c is not None else S.max() / 2 + 1e-12
        V = torch.where(cond,
                        torch.exp(-RB ** 2 / (2 * self.beta ** 2)) * (1 - torch.exp(-S ** 2 / (2 * c_val ** 2))),
                        torch.zeros_like(S))
        return theta, V, RB, S

    def forward(self, x):
        """x: (B,1,H,W)。返回 (theta, V, RB, S)，均为多尺度取 V 最大处合成。"""
        thetas, Vs, RBs, Ss = [], [], [], []
        for i in range(len(self.sigmas)):
            t, v, rb, s = self.single_scale(x, i)
            thetas.append(t); Vs.append(v); RBs.append(rb); Ss.append(s)
        V = torch.stack(Vs, 1).max(1).values          # 多尺度最大响应
        idx = torch.stack(Vs, 1).argmax(1, keepdim=True)
        theta = torch.gather(torch.stack(thetas, 1), 1, idx).squeeze(1)
        RB = torch.gather(torch.stack(RBs, 1), 1, idx).squeeze(1)
        S = torch.gather(torch.stack(Ss, 1), 1, idx).squeeze(1)
        return theta, V, RB, S
```

### 4.2 注入点一：AMS-Conv 门控（H2，改 `AMSGBC.forward`）

```python
# __init__ 新增:
self.prior_proj = nn.Conv2d(3, 8, kernel_size=1)   # [V, RB, S] -> 8 通道先验嵌入
self.spatial_gate = nn.Conv2d(in_channels * 2 + 8, 2, kernel_size=1, bias=True)
nn.init.zeros_(self.spatial_gate.weight); nn.init.zeros_(self.spatial_gate.bias)  # 零初始化不变

# forward 中（prior 已下采样到特征图分辨率）:
prior = self.prior_proj(torch.cat([V, RB, S], dim=1))           # (B,8,h,w)
w = torch.softmax(self.spatial_gate(torch.cat([f3, f5, prior], dim=1)), dim=1)
```

### 4.3 注入点二：扫描路由门控（H1/H3，改 `SAVSS_2D`）

利用 SASS 已有的逐步方向标记 `d1~d4`（取值 0~4，代码中已生成），计算**逐 token 的方向对齐度**：

```python
# __init__ 新增:
self.route_gate = nn.Conv2d(self.d_inner + 4, self.n_directions, kernel_size=1, bias=True)
nn.init.zeros_(self.route_gate.weight); nn.init.zeros_(self.route_gate.bias)

# forward 融合段:
feat = x_conv.permute(0, 2, 1).reshape(batch_size, E, H, W)     # (B,E,H,W)
# 方向对齐度: theta 与 4 条路径主方向 {0°, 90°, 45°, 135°} 的余弦平方
route_angles = torch.tensor([0., math.pi/2, math.pi/4, 3*math.pi/4], device=theta.device)
align = torch.stack([torch.cos(theta - a) ** 2 for a in route_angles], dim=1)   # (B,4,H,W)
align = align * V.unsqueeze(1)                                   # 只有"像裂缝"的位置方向才可信
g = torch.softmax(self.route_gate(torch.cat([feat, align], 1)), dim=1)  # (B,4,H,W)
g = g.reshape(batch_size, self.n_directions, L, 1)
y = (torch.stack(y_scan, 1) * g).sum(1) * 4.0 * self.act(z)
```

### 4.4 注入点三：Δ 调制（H4，加分项，最后做）

```python
# 每条路径的 dt 乘以由 V 调制的逐 token 缩放: 裂缝处记忆长、背景处遗忘快
v_tok = V.reshape(batch_size, 1, L)                              # (B,1,L)
dt_r = dt * (1.0 + self.dt_alpha[r] * v_tok)                     # dt_alpha 零初始化 -> 起点=基线
```

---

## 五、验证与实验设计（衔接 KimiCode 手册，新增 E7–E9）

### 5.1 先验正确性验证（训练前必做，防止符号约定错误）

```python
# verify_hessian.py：在 5~10 张巷道裂缝图上运行
# 1) 叠加显示 theta 方向场与 GT 裂缝骨架 -> 方向应沿裂缝切向
# 2) 显示 V 热图 -> 裂缝处应高响应；若背景高亮则 dark_structure 符号约定反了，翻转后重验
# 3) 统计: 裂缝 GT 区域内的 V 均值 / 背景区域 V 均值，比值应 > 2，否则调 sigma 范围
```

### 5.2 新增实验配置（追加到原 E0–E6 矩阵之后）

| 步 | 配置 | 说明 |
|---|---|---|
| E7 | E3 + H2（AMS 门控注入 V/RB/S） | 验证 Hessian 对卷积尺度的增益 |
| E8 | E3 + H1（路由门控注入方向对齐度） | 验证 Hessian 对扫描路由的增益 |
| E9 | E8 + H4（V 调制 Δ） | 加分项 |
| E10 | 消融：Sobel 先验 vs Hessian 先验（同注入点） | **关键对照**，证明二阶多尺度先验优于一阶 |

### 5.3 论文素材（低成本高说服力）

1. 方向场 θ 与扫描路径权重的对齐可视化（S1 门控图 + θ 叠加）
2. V 热图随训练 epoch 的演化（先验不变、门控学会利用先验）
3. E10 的量化对照表：一句话写明"Hessian 相对 Sobel 的提升来自多尺度与二阶曲率，在低照度巷道图像上差距更显著"

---

## 六、资料链接总表（新增部分）

| 用途 | 文献 | 链接 |
|---|---|---|
| Hessian 可微化先例 | Frangi-Net | https://arxiv.org/abs/1907.06194 |
| 特征值→vesselness 的 DNN 学习 | Deep Vesselness (MICCAI 2019) | https://link.springer.com/chapter/10.1007/978-3-030-31321-0_41 |
| 裂缝-血管同构的工程佐证 | 地铁隧道裂缝 Frangi 增强 | ICJE 12(5), DOI:10.6919/ICJE.202605_12(5).0009 |
| 经典 Hessian 裂缝方法的局限 | 3D 混凝土裂缝方法对比 | https://arxiv.org/abs/2112.09493 |
| 多尺度 vesselness 公式辨析 | LS-CF (MDPI J. Imaging 2022) | https://www.mdpi.com/2313-433X/8/10/291 |
| 广义 Frangi 裂缝提取 | Generalized Frangi crack extraction | https://inria.hal.science/hal-05709697 |
| 其余基线/扫描文献 | 见《扫描方式创新四方案与改进代码.md》第八节 | — |
