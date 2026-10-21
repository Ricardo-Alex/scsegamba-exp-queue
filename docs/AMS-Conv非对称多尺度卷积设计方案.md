# AMS-Conv 非对称多尺度卷积设计方案

> 对应论文第 3.3.1 节：在 SCSegamba 的 GBC（门控瓶颈卷积）基础上做 3×3 / 5×5 双支路非对称卷积创新
> 设计目标：暴雷风险小、提升可预期、轻量性不破、消融故事完整

---

## 一、设计依据：为什么改 GBC 是最划算的切入点

### 1.1 GBC 的现状（已核实代码）

`GBC.py` 的结构：

```
输入 x (B, C, H, W)
 ├─ 主路: block1(BottConv 3×3) → block2(BottConv 3×3)  → x1
 ├─ 门路: block3(BottConv 1×1)                          → x2
 └─ m = x1 ⊙ x2 → block4(BottConv 1×1) → +x（残差）
```

BottConv = pointwise 降维(C→C/8) → depthwise k×k → pointwise 升维(C/8→C)，即低秩瓶颈卷积。

关键事实：

1. **GBC 是 SCSegamba 消融中贡献最大的单一组件**：加入 GBC 带来 F1 +1.57%、mIoU +1.42%（论文 Table 4）。改它天花板最高。
2. **主路两个卷积都是 3×3**：感受野固定，对巷道裂缝"粗细不一"的形态（发丝裂缝 ~1px 到宽裂缝 >10px）只能用单一尺度响应。
3. **瓶颈结构让加支路几乎免费**：depthwise 卷积发生在 C/8=32 通道的低秩子空间里，3×3→5×5 每个支路只多 512 个参数。
4. GBC 在 SAVSS_Layer 前置用 2 次、MFS 分割头里用 1 次——**改一处类定义，全网三处同时升级**。

### 1.2 新颖性红线（检索结论）

| 相关工作 | 做法 | 与 AMS-Conv 的区别 |
|---|---|---|
| SKNet (CVPR 2019) | 3×3 + 膨胀3×3（≈5×5）双支路，**GAP 全局池化 → 通道级 softmax** 融合 | 我们是**逐像素空间门控**；且支路在瓶颈低秩子空间内、嵌入门控卷积骨架 |
| LSKNet (TIP/RS 2023) | 大核分解 + 空间选择，遥感检测，骨干网络级模块 | 我们不做大核（5×5 足够），模块嵌入 GBC 门控框架内做**形态感知**而非目标上下文感知 |
| ACNet (ICCV 2019) | 3×3 = 3×3+1×3+3×1 结构重参数化 | 我们不做重参数化，非对称体现在**双尺度支路 + 可选条带分解** |
| MixerCSeg DEGConv | 1×k/k×1 条带 EdgeConv + HoG 方向门控 | 我们用空间门控做**尺度选择**而非方向选择，错位不撞车 |

**结论：裂缝分割 + 瓶颈结构内双尺度支路 + 逐像素空间门控，没人做过。差异化辩护词现成：SKNet 的通道级选择假设"一张图一个最优尺度"，但裂缝图像中同一幅图内裂缝宽度沿走向剧烈变化（地下巷道图像尤其如此），必须逐像素选择尺度。**

---

## 二、AMS-Conv 结构设计

### 2.1 总体结构（替换 GBC 的 block1，其余骨架不动）

```
                        ┌─ 小尺度支路: BottConv(dw 3×3) → GN → ReLU → f3
 输入 x ────────────────┤
                        └─ 大尺度支路: BottConv(dw 5×5) → GN → ReLU → f5
                                        │
        空间门控: [f3; f5] → Conv1×1(2ch) → Softmax(dim=1) → (w3, w5) ∈ R^{B,2,H,W}
                                        │
              融合: f = w3 ⊙ f3 + w5 ⊙ f5
                                        │
              主路第二级: block2(BottConv 3×3) → GN → ReLU
                                        │
   保留原 GBC 门控: m = f' ⊙ block3(x)  →  block4  →  + x（残差）
```

**"非对称"的三层含义（写论文时用足）：**
1. **尺度非对称**：3×3 与 5×5 支路的感受野不对称，分别匹配细裂缝纹理与宽裂缝形态；
2. **支路角色非对称**：小核支路保细节、大核支路提供上下文，二者不是对称复制；
3. **（可选加分项）核形状非对称**：大核支路的 depthwise 5×5 分解为 5×1 + 1×5 条带卷积，与第 3.3.2 节 ATS 各向异性扫描形成"卷积-扫描"的方向感知呼应（第 3 章章名就是"方向感知"，这个呼应能让整章逻辑闭环）。

### 2.2 公式（论文可直接改写使用）

设输入特征 $x \in \mathbb{R}^{C\times H\times W}$，瓶颈比 $r=8$：

$$f_3 = \delta(\mathrm{GN}(\mathrm{BottConv}_{3\times3}(x))), \quad f_5 = \delta(\mathrm{GN}(\mathrm{BottConv}_{5\times5}(x)))$$

$$[w_3, w_5] = \mathrm{Softmax}_{c}(\mathrm{Conv}_{1\times1}([f_3; f_5])), \quad w_3, w_5 \in \mathbb{R}^{1\times H\times W}$$

$$f = w_3 \odot f_3 + w_5 \odot f_5$$

$$m = \mathrm{block2}(f) \odot \mathrm{block3}(x)$$

$$y = \mathrm{block4}(m) + x$$

其中 $\delta$ 为 ReLU，$\odot$ 为 Hadamard 积，Softmax 沿支路维度（2 维）归一化。

### 2.3 三个防爆雷设计（这部分最重要）

1. **门控零初始化 → 起点即基线**。`Conv1×1` 的权重和偏置全部置零初始化，则初始 $w_3=w_5=0.5$，融合结果严格等于两支路均值——一个合理的强起点，训练初期不会比原 GBC 差，不存在"加了门控训崩"的风险。
2. **残差与门控骨架原封不动**。GBC 已验证有效的 Hadamard 门控（block3）和残差连接完全保留，我们只替换"主路第一级卷积"这一个零件。即使 AMS 支路设计无效，网络仍可退化到接近原 GBC 的行为。
3. **轻量性账目可预期**（C=256, r=8）：3×3 支路 ≈16.7K 参数，5×5 支路 ≈17.2K（深度卷积只多 512），门控 ≈1.0K。单处 AMS-Conv 新增约 18K 参数，全网替换约 +54K，**总量从 2.80M 涨到 ~2.86M（+2%），轻量卖点不动**。FLOPs 方面 depthwise 5×5 在 32 通道、64×64 特征图上仅增加约 0.4 MFLOPs/处，可忽略。

---

## 三、可直接落地的 PyTorch 实现

在原 `GBC.py` 基础上新增 `AMSGBC` 类，与 `GBC` 接口完全对齐（构造参数都是 `(in_channels, norm_type)`），**一行替换即可切换**：

```python
import torch
import torch.nn as nn

class BottConv(nn.Module):  # 与原实现一致
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)
        self.depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, groups=mid_channels, bias=False)
        self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
    def forward(self, x):
        return self.pointwise_2(self.depthwise(self.pointwise_1(x)))

def get_norm_layer(norm_type, channels, num_groups):
    return nn.GroupNorm(num_groups=num_groups, num_channels=channels)

class AMSGBC(nn.Module):
    """非对称多尺度门控瓶颈卷积 (AMS-Conv enhanced GBC)"""
    def __init__(self, in_channels, norm_type='GN', strip_large_kernel=False):
        super().__init__()
        mid = in_channels // 8

        # ---- 非对称双支路（替换原 block1）----
        self.branch_s = nn.Sequential(                       # 小尺度支路 3×3
            BottConv(in_channels, in_channels, mid, 3, 1, 1),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )
        if strip_large_kernel:                               # 可选：5×5 → 5×1+1×5 条带分解
            self.branch_l = nn.Sequential(
                BottConv(in_channels, in_channels, mid, (5, 1), 1, (2, 0)),
                BottConv(in_channels, in_channels, mid, (1, 5), 1, (0, 2)),
                get_norm_layer(norm_type, in_channels, in_channels // 16),
                nn.ReLU()
            )
        else:
            self.branch_l = nn.Sequential(                   # 大尺度支路 5×5
                BottConv(in_channels, in_channels, mid, 5, 1, 2),
                get_norm_layer(norm_type, in_channels, in_channels // 16),
                nn.ReLU()
            )

        # ---- 逐像素空间门控（零初始化，起点 = 两支路均值）----
        self.spatial_gate = nn.Conv2d(in_channels * 2, 2, kernel_size=1, bias=True)
        nn.init.zeros_(self.spatial_gate.weight)
        nn.init.zeros_(self.spatial_gate.bias)

        # ---- 以下与原 GBC 完全一致 ----
        self.block2 = nn.Sequential(
            BottConv(in_channels, in_channels, mid, 3, 1, 1),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            BottConv(in_channels, in_channels, mid, 1, 1, 0),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )
        self.block4 = nn.Sequential(
            BottConv(in_channels, in_channels, mid, 1, 1, 0),
            get_norm_layer(norm_type, in_channels, 16),
            nn.ReLU()
        )

    def forward(self, x):
        residual = x
        f3 = self.branch_s(x)
        f5 = self.branch_l(x)
        w = torch.softmax(self.spatial_gate(torch.cat([f3, f5], dim=1)), dim=1)  # (B,2,H,W)
        f = w[:, 0:1] * f3 + w[:, 1:2] * f5
        m = self.block2(f) * self.block3(x)
        return self.block4(m) + residual
```

**替换方式**：`SAVSS_layer.py` 中 `self.GBC_C = GBC(embed_dims)` → `self.GBC_C = AMSGBC(embed_dims)`；`decoder.py` / `MFS.py` 中的 GBC 同理。注意 `SAVSS_2D` 内部的 `BottConv`（conv2d）不要动，那是 SS2D 的一部分。

---

## 四、消融实验设计（A 系列定义，执行时严格对照此表）

| 编号 | 配置 | 验证目的 |
|---|---|---|
| A0 | 原 GBC（基线） | 锚点 |
| A1 | 仅 3×3 支路（=去掉 5×5 支路和门控） | 应≈A0， sanity check |
| A2 | 仅 5×5 支路 | 验证单一尺度不足 |
| A3 | 3×3+5×5 直接相加（无门控） | 证明"空间门控"是有效增量而非堆支路 |
| A4 | 3×3+5×5 + SKNet 式通道门控（GAP+softmax） | **关键对照**：证明逐像素空间门控 > 通道级选择 |
| A5 | 3×3+5×5 + 空间门控（**本文 AMS-Conv**） | 完整版 |
| A6 | A5 的 5×5 换成 5×1+1×5 条带分解 | 验证非对称条带分解（加分项，同时进一步降参） |

### A 系列实现备注（执行用）

- A1：`AMSGBC` 中删掉 `branch_l` 与 `spatial_gate`，`f = f3` 直接进 block2；
- A2：同理只留 `branch_l`，`f = f5`；
- A3：保留双支路，删掉 `spatial_gate`，`f = 0.5 * (f3 + f5)`（等权，等价于门控永久停在初始状态）；
- A4：通道门控版——`s = GAP(cat([f3,f5]))` → `Conv1×1(2C→2)` → softmax → 两个**标量**权重广播乘 f3/f5（对照 SKNet 的 select 机制，权重是图级而非逐像素）；
- A5：完整 AMSGBC；
- A6：构造时 `strip_large_kernel=True`；
- A0~A6 全部只动 GBC 类定义，扫描、评估、协议一律冻结。

附加两张低成本高说服力的分析图：
1. **门控权重可视化**：叠加显示 w5 热图与 GT，预期宽裂缝区域 w5 高、细裂缝区域 w3 高——直接支撑"尺度自适应"动机；
2. **按裂缝宽度分组的指标统计**：把测试集按 GT 平均宽度分细/中/粗三档分别算 F1，预期 AMS-Conv 在"粗"档提升最大——把提升归因讲死，答辩无忧。

---

## 五、风险矩阵与预期收益

| 风险项 | 概率 | 影响 | 缓解措施 |
|---|---|---|---|
| 门控训练不稳 | 极低（零初始化保底） | 中 | 已结构性消除；必要时 gate 输出加 LayerScale |
| 提升不显著（<0.3%） | 低 | 高 | 退路1：条带分解版（A6）通常对细长结构更有效；退路2：把卖点转向"同参数下宽度鲁棒性"+宽度分组分析，负结果也能写成分析章节 |
| 参数/FLOPs 超轻量叙事 | 极低 | 低 | 新增 ~2% 参数，报告中直接给账目表 |
| 数据集方差淹没提升 | **中（最大隐患）** | 高 | 固定 seed=42 + 3 次重复取均值±方差；巷道自建数据集若样本少，务必报置信区间 |

**预期收益依据**：GBC 单组件在原消融中贡献 F1 +1.57%；AMS-Conv 是对其主路的正交增强，且巷道场景（低照度、裂缝宽度变化大）比公开数据集更吃多尺度红利。合理预期 **F1/mIoU +0.5%~1.0%**，叠加第 3.3.2 节 ATS 扫描创新后，第三章整体相对 SCSegamba 基线 +1.0% 以上是可防守的目标。

---

## 六、参考文献线索

- SKNet (CVPR 2019, arXiv:1903.06586)：双尺度支路 + 通道选择，A4 对照组来源
- LSKNet (arXiv:2403.11735)：空间选择优于通道选择的证据，直接引用于动机段
- ACNet (ICCV 2019)：非对称卷积概念源头
- RepLKNet (CVPR 2022) / UniRepLKNet (CVPR 2024)：大核有效性与 depthwise 大核低开销的依据
- SCSegamba (CVPR 2025)：GBC 基线结构与其消融数据（+1.57% F1）
