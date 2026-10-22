# DAWF：方向感知加权融合 —— 数学推导与实验方案

> 基座代码：modelsv6final / SAVSS_2D_v6_v4.py；对应论文 3.3.3 空间门控融合机制
> 核心思想：扫描路径融合从"静态标量 α_k"升级为"逐像素、Hessian 方向先验驱动的 softmax 加权 w_k(p)"

---

## 一、数学原理推导

### 1.1 每条扫描路径是一个隐式注意力核

按 MixerCSeg（arXiv:2603.01361）的隐式注意力展开，路径 k（排列 π_k）的 SSM 输出：

y^k_i = Σ_{j≺_k i} α^k_{i,j} x_j,  α^k_{i,j} = C_i^T (Π_{u=j+1..i} Ā_u) B̃_j

排列 π_k 决定核 α^k 的支撑域。K 条路径 = K 个不同的隐式注意力核。

### 1.2 融合的三个粒度

- 等权：y_i = (1/K)Σ_k Σ_j α^k_{i,j} x_j（均匀核混合）
- 现状 v4：y_i = Σ_k α_k Σ_j α^k_{i,j} x_j（α_k 可学习标量，静态、与内容无关）
- DAWF：y_i = Σ_k w_k(p_i) Σ_j α^k_{i,j} x_j（逐查询位置核混合）

w_k(p) = softmax_k(β·s_k(p)),  s_k(p) = C(p)·|cos(θ(p)−φ_k)|

C(p)=crackness，θ(p)=裂缝切向（复用现有 `_hessian_analysis` / `_score_map`）。

### 1.3 物理正确性

在走向 θ(p)≈φ_k 的裂缝像素处，路径 k 的核支撑域沿裂缝纵向延伸；w_k(p) 大 ⇔ 该像素表示优先从裂缝纵向取上下文、压制横向背景——有效上下文协方差各向异性化、与局部结构张量对齐。

### 1.4 恒等包含性（安全性）

β=0 ⇒ w_k≡1/K ⇒ y≡等权基线。DAWF 严格包含基线为特例，初始化即基线。

### 1.5 三个注入点（不同数学对象）

| 方案 | 形式 | 权重作用位置 | 风险 |
|---|---|---|---|
| M_out 输出端（推荐） | y_i=Σ_k w_k(p_i)·y^k_i | 查询位置 i（"听谁的"） | 最低 |
| M_in 输入端 | x^k_j=w_k(p_j)·x_j（⇔ 缩放 B̃_j） | 源位置 j（"谁来说"） | 中 |
| M_Δ 动力学端 | Δ^k_j=Δ_j(1+λ·s_k(p_j)) | 衰减率 Ā（记忆长度） | 最高（M4 前科，不跑） |

### 1.6 为什么三次门控失败（M3/R1/V6_ROUTE）而 DAWF 不会

| 失败病灶 | DAWF 对策 |
|---|---|
| 零初始化 softmax 梯度对称、卡在均匀态 | 权重由先验直接给出，仅 1 个温度标量 β 可学，β=0 严格等于基线 |
| 门控需从零学习几何信息 | 方向信息由 Hessian 先验注入，学习负担降为调温度 |
| 逐通道粒度与空间任务错配 | 逐像素粒度 |
| 外部佐证 | MixerCSeg Table 6：先验方向嵌入 > 学习方向预测 |

### 1.7 加权与重排的关系（决定实验优先级）

- 重排改核 α^k→α̃^k（裂缝 token 序列相邻化），代价：破坏局部性、FPS 下降；
- 加权不动核，逐像素选核：局部性保留、开销≈0；
- F1 实验证据：重排路线在当前代码线仅 +0.0082，微弱。
- 假设：H-A 替代（加权单独即可）/ H-B 叠加（互补）/ H-C（加权依赖重排）。

---

## 二、实现代码（改 SAVSS_2D_v6_v4.py forward 第 371-373 行区域）

```python
# __init__ 新增（每层 1 个标量）：
self.beta = nn.Parameter(torch.zeros(1))          # init=0 → 严格等权基线

# forward 中，替代 y_global 的求和：
if ATS_FUSE == 'dawf':
    scores = torch.stack([
        self._score_map(crackness_b, angle_b, float(self.theta_k[k]))
        for k in range(self.K)], dim=1)                          # (B,K,L)
    w = torch.softmax(self.beta * scores, dim=1)                 # (B,K,L)
    y_stack = torch.stack(ys_global, dim=0)                      # (K,B,n_g,L)
    y_global = (y_stack * w.permute(1, 0, 2).unsqueeze(2)).sum(dim=0)
else:  # 原有 α_k 路径
    y_stack = torch.stack(ys_global, dim=0)
    y_global = (y_stack * self.alpha_global.view(-1,1,1,1)).sum(dim=0)
```

环境变量开关：`ATS_FUSE = dawf | alpha`（默认 alpha 保持现状）。
M_in 变体（G3）：在 `_gather_tokens(x_global, order_idx)` 后，对 x_reord 乘 w[:, k] 沿 order_idx gather 后的值。

---

## 三、实验矩阵（G 系列，F 系列完成后接棒；锚点 = F0 0.8761/0.9097，若 F2~F4 有更优者则以最优 F 为基座并在 results_log 注明）

| run | 配置 | 判据 |
|---|---|---|
| G2（最优先） | 基座 + ATS_FUSE=dawf + ATS_REORDER=none（加权替代重排） | max mIoU > 锚点 且 ODS ≥ 锚点−0.005；登记 FPS（预期回升） |
| G1 | 基座 + ATS_FUSE=dawf + 保留 hessian 重排（测 H-B 叠加） | 同上 |
| G3 | G2 配置但改 M_in 输入端调制 | 同上，对照 G2 |
| G4 | 仅理论写入论文，不跑 | — |

执行规则：G2 先跑；G2 过判据再跑 G1；G1 显著优于 G2（>+0.003）说明重排与加权互补，否则结论为"加权替代重排"；G3 最后。每 run 登记 max mIoU/ODS/F1/Params/GFLOPs/FPS + β 收敛值（写入 results_log，作为论文证据）。

---

## 四、论文叙事接口（3.3.3 节）

等权 → 静态 α_k → 逐像素方向感知 w_k(p) 的三级递进；隐式注意力核混合表述；恒等包含性保证；MixerCSeg Table 6 外部佐证；β 收敛值 + w_k(p) 权重图可视化（沿裂缝处对应走向路径权重高）作定性证据。
