# 真正的洛伦兹 Backbone 设计工作计划

**项目代号**：Lorentzian Cognition Architecture (LCA)
**理论依托**：*Realizability and the Origin of Causality* (Y.Y.N. Li)
**起草日期**：2026-05
**最近更新**：2026-05-02
**状态**：Phase 1 基本完成，Phase 2 部分推进；写论文中

---

## 当前进展快照（2026-05-02）

**物理理论侧** ✅ 已完成
- *Realizability and the Origin of Causality* (Y.Y.N. Li) finalize 完毕
- 提交版本（trimmed C）保存在 `Realizability_trimmed.tex`
- 模拟审稿小组（Noether/Einstein/Feynman/von Neumann）一致通过
- 待投：*Foundations of Physics*

**ML 实现侧** ◐ Phase 1 基本完成，Phase 2 部分进行
- v11–v15c 共 25 个实验脚本完成（见 `lorentz_step2_*.py`）
- v15c task diversity battery：4 tasks × 2 backbones × 3 seeds = 24 trainings 完成
- v14c calibration battery：3 configs × 3 seeds × 8 OOD configs = 72 evaluations 完成
- 论文 draft v0 完成（`paper_draft_v0.md`，~7,620 词），三处主章节已根据 v14c/v15c 数据修订

**核心实证发现**
1. **跨因果结构 universality**：18/18 in-distribution direction-match across {AR(1), Granger x→y, do() intervention} × {Lorentzian, Euclidean} × 3 seeds，0 inverted
2. **跨 backbone 等价性**：Lorentzian 和 Euclidean 在所有 4 task 上 mq_AUC 差距 ≤ 0.02，但 |m_q| scale 差 5-15×（4-5 vs 22-58）
3. **Calibration 测试反直觉结果**：把 Lorentzian 的 |m_q| 强行抬到 ~50，sign-split 完全塌缩（OOD 0/24 strong split, AUC<chance）。即 Lorentzian 不会"过渡到 Euclidean 模式"——SplitNorm 耦合 ‖h_t‖,‖h_s‖≈√64 留下的几何空间不允许同时大尺度和清晰类时-类空分离
4. **markov2 边界案例**：当两类都 causally structured（深度差异），Lorentzian 1/3 direction-match (vs Euclidean 3/3)；mq_AUC 仍 ≈0.93 信息保留；0 inverted——是 informative limitation，不是反例

**未做（Phase 2 剩余）**
- ✗ 支柱 5（洛伦兹注意力）尚未实现——目前所有实验都是 MLP backbone
- ✗ 真实数据集（图像/文本/物理）尚未触及——只跑过合成时序
- ✗ 与 hyperbolic embedding 的直接对比未做

---

## 文件清单

**论文**
- `Realizability_trimmed.tex` — 物理论文 finalize 版本（待投 *Foundations of Physics*）
- `paper_draft_v0.md` — ML 论文 draft v0（~7,620 词，§6/§7/§8 已根据 v14c+v15c 修订）

**实验脚本（按时间顺序）**
- `lorentz_step1_splitnorm_v1-v3.py` — 支柱 1+3 早期原型
- `lorentz_step2_causal_seq_v1-v10.py` — 支柱 4 早期迭代
- `lorentz_step2_v11_AR03.py` — 第一个稳定 baseline（AR_RHO=0.3）
- `lorentz_step2_v12_battery.py` — multi-config 测试
- `lorentz_step2_v13_ood.py` — OOD shift battery
- `lorentz_step2_v14c_calibration.py` — calibration test（最终版，3 configs × 3 seeds × 8 OOD = 72 evals）
- `lorentz_step2_v15c_tasks.py` — task diversity battery（最终版，4 tasks × 2 backbones × 3 seeds = 24 trainings）

**最新实验结果**
- v14c 完整 GPU run 结论：calibration 把 Lorentzian |m_q| 抬到 50 后 sign-split 完全塌缩（OOD 0/24 strong split），Euclidean 自然达到 ~50。证明两 backbone 是 distinct geometric regimes
- v15c 完整 GPU run 结论：indep_vs_dep tasks 18/18 direction-match，markov2 4/6 direction-match（0 inverted），pooled 22/24 match 0 inverted

---

## 0. 项目定位

### 这是什么

在神经网络架构层面构造一个**真正的不定度规（Lorentzian）表征空间**，使得"因果结构"不是被学习出来的，而是几何先验直接给定的。这是 *Realizability* 物理理论框架在 AI 表征几何上的工程化落地。

### 这不是什么

- 不是 hyperbolic embedding 的变种（hyperbolic 只允许类时向量，禁止类空）
- 不是在 Euclidean attention 上加号差点积（号差只是 score 的修饰，表征仍是欧氏）
- 不是 world model（不做视觉/3D 生成，不做轨迹模拟）
- 不是 LLM 替代品（架构层工作，不是产品）

### 与现有工作的边界

| 现有工作 | 几何 | 类空向量 | 与本项目区别 |
|----------|------|----------|--------------|
| Nickel & Kiela hyperbolic | 双曲（常负曲率） | 禁止 | 全类时，无因果结构 |
| Lorentz embeddings (Nickel 2018) | 双曲（Lorentz model） | 禁止 | 同上 |
| 现有 F3 attention | 欧氏 + 号差 score | 不区分 | 只在 score 一处用号差 |
| Geometric DL (Bronstein) | 群等变性 | N/A | 关注对称群，不关注因果 |
| **本项目 LCA** | **不定度规 (-,+,+,+)** | **允许** | **真洛伦兹空间** |

---

## 1. 失败案例分析（为什么现有原型不够）

在动手之前，必须明确**已经做过的尝试为什么不算"真洛伦兹"**。这部分决定了我们要新做什么。

### 1.1 F3 模型（Layer1Model with mode='f3'）

```python
# 现状
score = cat([-st, ss]) * scale  # ✓ score 端有号差
embedding 经过 LayerNorm        # ✗ 表征是 L2 归一化的欧氏向量
inner(x, x) 永远 ≥ 0             # ✗ 没有类空向量
```

**问题**：表征空间本质是欧氏，号差只是 attention score 函数的一个修饰。整个 hidden state 流过 LayerNorm 后丢失了号差结构。

### 1.2 LorentzRiemannianLayer1Model

```python
# 现状
MB.project(x) 强制 -x_0² + Σx_i² = -1
```

**问题**：所有点被锁死在双曲面上，`inner(x,x) ≡ -1`。这是双曲空间，不是不定度规空间。**它通过禁止类空向量来回避因果结构问题，等于把要研究的对象删掉了**。

### 1.3 MinkowskiLN

```python
# 现状
mq = (s²).sum() - (t²).sum()
x_normed = x / sqrt(|mq| + eps)  # 取了绝对值
```

**问题**：`|mq|` 抹掉了号差信息。归一化后的向量已经无法区分类时/类空。

---

## 2. 五个技术支柱（核心工作）

每个支柱对应你之前列的五条必要条件。每条都给出**问题陈述、候选方案、验证标准、预计工作量**。

### 支柱 1：不定度规表征向量 ✅ 已实现（方案 A）

**目标**：构造一个 hidden state 空间，使 `inner(x, x) = -x_0² + Σx_i²` 在训练过程中可正可负可零，并且这个号差信息一路保留到输出层。

**已选方案：A — Split-Norm 表征**
- hidden 维度切分为 `(D_t, D_s) = (64, 64)`
- 时间分量和空间分量独立 RMSNorm（各自 √D 缩放），不联合归一化
- 输出 m_q = ‖h_s‖² − ‖h_t‖² 自然有正负

**实现位置**：`lorentz_step1_splitnorm.py` v1-v3，最终版集成进 `lorentz_step2_v11_AR03.py` 后所有版本

**实证结果**：
- m_q 分布在分类任务上呈双峰，class 0 (dependent) → m_q < 0，class 1 (independent) → m_q > 0
- |m_q| 自然落在 4-8 范围（不需要正则化），SplitNorm 耦合 ‖h_t‖,‖h_s‖ ≈ √64 ≈ 8 是这个 scale 的几何来源

---

### 支柱 2：洛伦兹距离与相似度 ◐ 部分实现

**目标**：定义两个样本之间的"距离"，能够区分类时（因果可达）和类空（因果隔离）关系。

**当前实现**：通过 m_q 本身作为类别可分性指标使用：
- mq_AUC 衡量类间 m_q 分布的可分性（v15c 上 4 task 都达到 0.88-1.00）
- 没有显式实现样本-样本的洛伦兹间隔距离矩阵

**为什么部分**：实验任务都是分类任务，每个样本独立的 m_q 标量足以衡量类别归属。如果做对比学习或检索任务，则需要回到下面方案 A/B/C 中选择具体形式。

**待选方案（如未来扩展到 pairwise tasks）**：
- 方案 A：时空间隔 `s²(x,y) = -(x_0-y_0)² + Σ(x_i-y_i)²`
- 方案 B：洛伦兹内积 `<x,y>_M`，保留符号
- 方案 C：因果带权距离（类空记为不可达）

---

### 支柱 3：洛伦兹层归一化 ✅ 已实现（方案 A）

**目标**：替换 LayerNorm，使归一化后保留号差信息。

**已选方案：A — Time-Space Split RMSNorm**

```python
def lorentz_rms_norm(x, gamma_t, gamma_s, beta):
    t, s = x[..., :D_t], x[..., D_t:]
    rms_t = sqrt((t**2).mean(-1, keepdim=True) + eps)
    rms_s = sqrt((s**2).mean(-1, keepdim=True) + eps)
    return cat([gamma_t * t / rms_t, gamma_s * s / rms_s], -1) + beta
```

**为什么不选 B/C**：
- 方案 B（Indefinite Mahalanobis）数值不稳——m_q ≈ 0 时除零；类空向量除以 +√|mq|，类时向量除以 −√|mq|，号差被翻转
- 方案 C（不归一化）尝试过，训练不稳定，loss 震荡严重

**实证结果**：
- 100+ epoch 稳定训练，无 NaN
- m_q 号差在归一化前后保持
- 实现位置：`SplitRMSNorm` 类，所有 v11+ 文件中

---

### 支柱 4：因果保持的残差连接 ✅ 已实现（方案 D 软形式 + 可调 gate）

**问题数学描述**：
- 设 x 类时（mq(x) < 0），f(x) 类时
- 但 mq(x + f(x)) 不一定 < 0
- 例：x = (1, 0)（未来类时），f(x) = (-1, 2)（过去类时但空间分量大）
- x + f(x) = (0, 2)，mq = 4 > 0（变成类空！）

**已选方案：D 的软形式 + spacelike_residual 标量 gate**

```python
# 残差不做硬投影，但通过 spacelike_residual ∈ [0, 1] 控制类空分量回流
# spacelike_residual = 0 → Lorentzian backbone (强先验，|m_q| 小)
# spacelike_residual = 1 → Euclidean backbone (无先验，|m_q| 大)
out = layernorm(x + alpha * f(x))
```

**为什么不选 A/B/C**：
- 方案 A（Causal Cone Projection）：硬投影在 m_q ≈ 0 附近梯度跳变，训练困难
- 方案 B（Adaptive Residual Scaling）：solve_max_alpha 不可微
- 方案 C（Highway-Style Gating）：增加参数，不直接体现"几何先验"
- 方案 D（Soft Causal Loss）：v11 早期试过显式 `λ * relu(mq(out))` 损失，但 SplitNorm 已经把 |m_q| 自然约束到 ~5-8，不需要额外 loss term

**实证发现**：
- spacelike_residual 同时给出 Lorentzian (=0) 和 Euclidean (=1) 两个 baseline，是 paper 的核心对照
- v14c calibration 测试证明：强行抬 |m_q| 不会让 Lorentzian 平滑过渡到 Euclidean——sign-split 完全塌缩。即两个 backbone 是 distinct geometric regimes，不是连续 family

---

### 支柱 5：洛伦兹注意力机制 ✗ 未开始

**当前状态**：所有 v11–v15c 实验都是 MLP backbone，没有触及 attention。这是 Phase 2 剩余最大块工作。

**问题分析**（保留原设计文档）：
- 标准 softmax 假设大 score = 高相关性
- 但在洛伦兹意义下，**负的 score（类时）才是因果可达，正的 score（类空）是因果隔离**
- 直接用 softmax 会让模型认为类空配对最相关，反了

**候选方案 A：Sign-Inverted Softmax**
```python
score = -<Q, K>_M / sqrt(d)  # 翻号，让类时 score 变正
attn = softmax(score, dim=-1)
```

**候选方案 B：Gated by Causality**
```python
score = <Q, K>_M / sqrt(d)
causal_mask = (score < 0).float()  # 只允许类时配对参与注意力
attn = softmax(score - 1e9 * (1 - causal_mask), dim=-1)
```

**候选方案 C：Energy-Based Attention**
```python
# 把 score 换成能量 E(Q, K) = |<Q, K>_M|
# 类时和类空都参与，但权重 = exp(-E / τ)
energy = abs(<Q, K>_M)
attn = softmax(-energy / tau, dim=-1)
```

**候选方案 D：Two-Channel Attention**
- 类时 attention 和类空 attention 分两个通道独立计算
- 输出拼接

**优先级**：中。论文 v0 不依赖 attention（MLP 已够强证据）。如果走 NeurIPS workshop 主路径，先发 MLP 结果；attention block 留作 follow-up。

**何时启动**：等 paper v0 投稿后，或如果审稿人要求"图像/文本任务"才需要

---

## 3. 配套基础设施

### 3.1 因果对比学习目标

光有几何不够，还需要训练目标利用这个几何。

**设计**：
```python
def causal_contrastive_loss(x, y, causal_label):
    """
    causal_label: 1=因果可达（应类时间隔），0=因果隔离（应类空间隔）
    """
    s2 = lorentz_interval(x, y)
    # 因果可达对应该 s2 < 0，因果隔离对应该 s2 > 0
    loss_causal = relu(s2 + margin) * causal_label
    loss_acausal = relu(-s2 + margin) * (1 - causal_label)
    return loss_causal.mean() + loss_acausal.mean()
```

**数据需求**：需要一个带因果可达性标签的数据集。可以从已知因果结构的合成数据开始（DAG 上的时序数据）。

### 3.2 数值稳定性工具

不定度规训练几乎肯定不稳定。需要：

- **梯度监控器**：实时追踪 mq、||grad||、loss 的分布，一旦偏离立即告警
- **类时性追踪器**：每 N 步记录 hidden state 类时比例
- **自适应优化器**：可能需要为时间分量和空间分量分配不同学习率
- **NaN 检测器**：mq 接近零时 `1/sqrt(|mq|)` 会爆炸，必须有检测和回退

### 3.3 可视化套件

- 时空图：把 hidden state 投影到 (t, x) 二维平面
- 光锥图：以某个参考点为顶点画光锥，显示其他点的因果关系
- 类时性热力图：每层每个 token 的 mq 分布

---

## 4. 实施阶段

### Phase 0：理论基础（Month 0-1）✅ 已完成

- [x] 写一份正式的数学描述：`Realizability_trimmed.tex` 的 Theorem 5（导出 Lorentzian 度规签名）就是数学基础
- [x] 模拟审稿小组（Noether/Einstein/Feynman/von Neumann）已 review 物理论文，结论：版本 C 逻辑完整无需修改
- [x] 阅读 Lorentz embeddings 文献、Krein space ML 文献——已嵌入 `paper_draft_v0.md` 的 §2 Related Work

**Gate 通过**：理论无矛盾，物理论文 finalize

### Phase 1：单支柱原型（Month 2-7）✅ 基本完成

- [x] 支柱 1（Split-Norm 表征）+ 支柱 3（SplitRMSNorm）：`lorentz_step1_splitnorm_v1-v3.py`
- [x] 玩具任务：从 MNIST 转向了**合成时序因果任务**（更直接验证因果先验）
- [x] 支柱 4（spacelike_residual gate）：v11 集成完成
- [x] 因果对比学习的简化形态：cross-entropy + m_q 自然分离（不需要显式对比 loss）
- [ ] 支柱 5（注意力）：未做（见上）

**Gate 通过**：v11 起 100+ epoch 稳定训练，类时比例稳定，训练无 NaN

### Phase 2：完整 backbone（Month 8-12）◐ 部分完成

- [ ] 整合五个支柱为完整 Lorentzian Transformer Block：未做（缺支柱 5）
- [x] 在 4 个对照任务上跑：
  - [x] 时序预测（AR(1)、Markov-2、Granger、do() intervention，v15c battery）
  - [ ] 文本分类：未做
  - [ ] 物理轨迹预测：未做
  - [ ] 视频帧预测：未做
- [x] **每个任务都有 Euclidean baseline 严格对照**（spacelike_residual=1.0 同结构对照）

**Gate 部分通过**：在合成时序上有清晰发现，但还没在标准 ML benchmark 上证明优势。**论文叙事策略调整**：从 "Lorentzian 在某 benchmark 赢 Euclidean" 转向 "Lorentzian 和 Euclidean 通过不同机制达到等价 OOD 性能；Lorentzian 给可解释 m_q，Euclidean 给大 magnitude"。后者更符合实测数据，也更诚实。

### Phase 3：scaling 与发表（Month 13-18）◐ 写论文中

- [ ] 在更大数据集和模型规模上验证：未做（可能不必要——见上策略调整）
- [x] 写论文：`paper_draft_v0.md` 完成（v0），三处 §6/§7/§8 已根据 v14c+v15c 数据修订
- [ ] 投稿：待定（见决策日志）
- [ ] 开源代码 + 预训练权重：未做

---

## 5. 风险与应对

### 风险 1：数值不稳定无解 ✅ 未实现

**预测概率**：高
**实际**：SplitNorm 自然把 |m_q| 控制在 4-8，从未需要除零保护。100+ epoch 稳定训练，0 NaN。

### 风险 2：理论上能 work，工程上没优势 ◐ 部分实现

**预测概率**：中
**实际**：Lorentzian 不"赢" Euclidean，但**两者达到等价 OOD 内容、通过不同机制**。这本身是非平凡发现：
- v14c calibration 测试反直觉显示 Lorentzian 不能"过渡到" Euclidean——是 distinct geometric regime
- markov2 暴露 Lorentzian 在 depth-of-dependence task 上的 informative limitation

**应对**（已采用）：放弃"Lorentzian 赢 Euclidean"叙事，改为"等价内容、不同机制"——更符合实测，也更有可解释性 angle 的论文价值

### 风险 3：被 scoop ✗ 未实现

**预测概率**：低-中
**实际**：到 2026-05 没有被 scoop。Krein space neural networks 方向仍冷门。Realizability 物理框架是 Y.Y.N. Li 独家。

### 风险 4：单人无法完成 ✗ 未实现

**预测概率**：很高
**实际**：Phase 0-1 单人完成，Phase 2 部分完成。原预计"博士论文级工程量"高估了——MLP backbone + 合成时序大幅降低工程负担。

**注**：如果走支柱 5（attention） + 真实数据集路线，单人难度会陡升。届时再评估招募。

### 风险 5：跑出来洛伦兹几何确实没用 ✗ 未实现

**预测概率**：未知
**实际**：18/18 cross-causal-structure direction-match，0/24 inversions——Lorentzian 几何先验确实在 m_q sign 上编码了"causally connected vs trivial"的区分。**不是没用，是用得 specific**：编码方式是 m_q 符号 + 中等 magnitude，不是大 magnitude。

### 新发现的风险（2026-04 后）

#### 风险 6：审稿人坚持要求 standard ML benchmark

**症状**：审稿人说"为什么不在 ImageNet/GLUE 上比？"
**应对**：诚实回答论文 scope 是 causal-structure-recognition + 几何先验对照；这不是 vision/text 替代品。如果被拒，转 FoP companion 路径（不需要 ML benchmark）

#### 风险 7：审稿人质疑 "为什么不在 markov2 上 100% direction-match"

**应对**：把 markov2 单独写成 informative limitation——bounded |m_q| 范围在 depth-of-dependence task 上的几何容量约束，而不是 inversion。0 inverted 数据保护这个 claim

---

## 6. 成功标准

### 最低成功（项目完成）✅ 已达到

- [x] 五个支柱：1, 3, 4 已实现；2 部分（标量 m_q 形式）；5 未做但叙事不依赖
- [x] 完整 backbone（MLP + Lorentzian + MqMLPHead）能稳定训练到收敛
- [x] 方法论论文 draft v0 完成

### 中度成功（值得继续）◐ 部分达到

- ✗ 在 benchmark 上系统性赢 Euclidean baseline (d > 0.5)：**未达到，且实测显示这不是正确比较框架**
- [x] m_q 可视化展示有意义的因果结构：18/18 cross-structure direction-match
- [ ] 论文被 top-tier 会议接受：未投

**叙事调整**：原标准是"赢 Euclidean"。实测显示等价 + 不同机制是更准确的 finding。如果走 FoP companion / TMLR 路径，"等价内容、不同机制"已经足够构成 contribution。

### 高度成功（颠覆性证据）✗ 未达到，可能不必要

- 多任务一致赢 Euclidean baseline：**实测路径已偏离这个目标**
- Scaling laws 显示 Lorentzian 优势：未做，且不一定要做（见上）
- 引发后续工作（其他实验室复现/扩展）：尚早

### 彻底失败（坦然承认）✗ 未实现

- 数值上无法稳定训练超过 100 epoch：**未发生**（v11 起稳定训练）
- 在所有任务上都显著输给 baseline：**未发生**（等价非输）

**实际结果落点**：在"最低成功"（已达）和"中度成功"（叙事调整后基本达到）之间。论文价值在于**证明 Lorentzian 是 distinct geometric regime + 提供可解释的因果结构识别**，而不是 SOTA replacement。

---

## 7. 资源需求

### 实际使用（截至 2026-05）

- Google Colab GPU（T4/A100，按需）
- 单 battery 约 25-35 分钟（24 trainings × 15 epochs × 8000 samples）
- 总开销：< $50 Colab Pro 订阅费
- 单人执行（理论 + 实验 + 论文写作）

**为什么实际比"最低配置"还低**：合成时序任务很轻；MLP backbone 参数量 ~67k；不需要 8 张 A100。

### 后续 scaling 情景

如果走 scale-up 路线（非 paper v0 必需）：

- 1 张 RTX 4090 或同级 GPU
- 12-18 个月时间
- AWS / Lambda 上偶尔租用 A100 做大实验（年预算 ~$10k）

如果走真实数据集 stress test 路线：

- 2-3 人团队（你 + 几何 DL 博士生 + 工程师）
- 8 张 A100 集群
- 18 个月时间
- 总预算估计 $200k-$500k（人力 + 算力）

### 不再考虑的"理想配置"

原始文档列了 $5M-$20M 的"小型实验室 + 64-128 张 H100"——基于"和 SOTA LLM 拼性能"的假设。当前数据已经显示**Lorentzian 不是和 Euclidean 拼性能的方向**，而是几何先验的可解释性方向。$5M 路线不必要。

---

## 8. 与 *Realizability* 物理论文的关系

**双轨独立推进**：

| 轨道 | 状态 | 时间线 |
|------|------|--------|
| 物理论文 *Realizability and the Origin of Causality* | 已 finalize，待投 *Foundations of Physics* | 2026 内发表 |
| AI 实现 LCA Backbone | 设计阶段 | 2026-2028 |

**物理论文不依赖 AI 实验成败**。AI 工作是物理框架的"延伸应用"，不是其证明。

**叙事策略**：
- 物理论文里：不提 AI，就讲 realizability 和因果性的关系
- AI 论文里：把物理框架作为"motivation"提一段，主要证据来自 AI 实验本身
- 两个圈子都不会因为对方的不熟悉而拒绝你

---

## 9. 立即可做的下一步（2026-05 之后）

### 当前优先（写论文）

1. **§4.3 新增 subsection "Generalization Across Causal Structures"**——把 v15c 的 18/18 indep_vs_dep 结果 + markov2 边界案例写进 main results
2. **§7.1 universality claim 升级**——从 "0 inverted across 24 OOD configs" 到 "0 inverted across 18 in-distribution × 3 causal structures + 24 OOD AR(1) = 42 evaluations"
3. **§7.4 limitation 拆分**——"single task domain (synthetic temporal)" + 新 limitation "depth-of-dependence 任务 Lorentzian 难以同时编码 indep-vs-dep 和 depth"
4. **Abstract 微调**——把"across all our experiments"改成具体数字
5. **决定投稿目标**：见决策日志 9.3

### 中期（Phase 2 收尾）

6. **Reverse calibration sanity check**：把 Euclidean 强行压到 |m_q| ≈ 5，看 sign-split 是否也塌缩。如果不塌缩，说明问题不对称——是 Lorentzian SplitNorm 特有的几何约束
7. **支柱 5（attention）**：如果走 NeurIPS 主会议路线，需要做；如果走 FoP companion / TMLR 路线，可以不做
8. **图表绘制**：m_q histograms by class、calibration trajectory、OOD scale curves、bilinear-quadratic concept figure
9. **填参考文献**：Bronstein, Ganea, Nickel-Kiela, Law-Stam, Xiong, Krein-space ML 占位符

### 长期（Phase 3）

10. **真实数据集 stress test**：image / text / 物理轨迹任意一个能 reproduce 18/18 direction-match，就有外延证据
11. **开源**：v15c 代码 + paper supplement
12. **后续工作 hooks**：do(y) intervention 任务、higher-order causality 任务、连续时间 SDE 数据

---

## 10. 决策日志

记录每次重大设计选择和理由，避免后期忘记为什么这么做。

| 日期 | 决策 | 理由 | 备选方案 |
|------|------|------|----------|
| 2026-05 | 项目启动 | 现有 F3/Riemannian 都不是真洛伦兹，需要从头设计 | 放弃 AI 方向，只发物理论文 |
| 2026-?? | 支柱 1 选 A（Split-Norm） | 显式切分 D_t/D_s 让号差结构在数据流中保留；不需要让模型从零学到这个先验 | B（sign-preserving norm）数值不稳；C（KL 正则）依赖 m_q 自发分化 |
| 2026-?? | 支柱 3 选 A（Time-Space Split RMSNorm） | 与支柱 1 一致：分别归一化 t 和 s 分量保持号差 | B（Indefinite Mahalanobis）m_q≈0 时除零；C（不归一化）训练不稳 |
| 2026-?? | 支柱 4 选 D 软形式（spacelike_residual gate） | 可同时给 Lorentzian (gate=0) 和 Euclidean (gate=1) baseline 做严格对照 | 硬投影（A）梯度跳变；adaptive scaling（B）不可微 |
| 2026-?? | Toy task 从 MNIST 转向合成时序 | 因果先验在 vision task 上不直接显现；因果结构识别任务（AR vs IID）能直接验证 m_q 分布 | 坚持 MNIST/CIFAR：和 baseline 比较优势会被 task 类型稀释 |
| 2026-04 | 论文叙事从"Lorentzian 赢 Euclidean"改成"等价 OOD 内容、不同机制" | v13/v14c/v15c 数据一致显示 mq_AUC 跨 backbone 等价；v14c calibration 还证明强行抬 |m_q| 反而塌缩 sign-split | 硬调指标找一个 Lorentzian 占优的子集——会被审稿人识破 |
| 2026-04 | 暂不做支柱 5（注意力） | MLP backbone 已经给出 18/18 universality 证据；attention 是 separate axis | 死磕 attention 才发论文：会大大延后投稿 |
| 2026-05-02 | v15c 4-task battery 完成；改为 18/18 cross-structure 框架 | 实测：indep_vs_dep 18/18 干净，markov2 4/6 同侧但 0 inverted | 假装 markov2 也是 indep_vs_dep（v15→v15b 之前犯过这个错） |
| TBD | 投稿目标 | 三选一：FoP companion paper（与 Realizability 配套，~80% 可接受）；TMLR（偏方法论，~50-60%）；NeurIPS workshop（~25% 主会接受） | （写完 v0.1 再决定） |
| TBD | 是否做 reverse calibration sanity | 如果做且 Euclidean 也塌缩，说明现象对称——削弱 paper 主 claim；如果做且不塌缩，加强 claim | 不做：审稿人可能要求 |

---

## 附录 A：术语表

- **不定度规 (indefinite metric)**：度规张量有正有负特征值的内积空间。Minkowski 度规 diag(-1, 1, 1, 1) 是典型例子。
- **类时 (timelike)**：满足 `<x, x>_M < 0` 的向量。物理上对应可被实物粒子穿越的间隔。
- **类空 (spacelike)**：满足 `<x, x>_M > 0` 的向量。物理上对应不可因果连接的间隔。
- **类光/零 (null/lightlike)**：满足 `<x, x>_M = 0` 的向量。光锥本身。
- **光锥 (light cone)**：以某点为顶点的所有零向量构成的锥面。划分时空为类时未来、类时过去、类空。
- **realizability**：Y.Y.N. Li 物理框架的核心概念。一个时空结构可实现 ⇔ 它满足某些自洽性条件，因果性是其衍生属性。

---

## 附录 B：参考文献种子列表

需要在 Phase 0 系统阅读的：

**物理基础**
- Y.Y.N. Li, *Realizability and the Origin of Causality*, in submission to *Foundations of Physics*

**几何深度学习**
- Bronstein et al., *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges* (2021)
- Nickel & Kiela, *Poincaré Embeddings for Learning Hierarchical Representations* (NeurIPS 2017)
- Nickel & Kiela, *Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry* (ICML 2018)

**Krein 空间与不定度规学习**
- Ong et al., *Learning with Non-positive Kernels* (ICML 2004)
- Loosli et al., *Learning SVM in Krein Spaces* (PAMI 2016)
- (这个方向的神经网络文献几乎为零，是 LCA 的机会)

**Lorentzian neural networks 已有工作**
- Law et al., *Lorentzian Distance Learning for Hyperbolic Representations* (ICML 2019)
- Chami et al., *Hyperbolic Graph Convolutional Neural Networks* (NeurIPS 2019)
- （这些都是 hyperbolic，仍禁止类空，需要批判性阅读）

---

**最后一句**（原版，2026-05 项目启动）：这是一个有可能完全失败的项目。但如果你的物理理论方向对，这条路上的任何中间结果都是有价值的——哪怕最后结论是"洛伦兹几何不能直接当 AI 表征空间用"，那也是一个对场域有贡献的负面结果。诚实做实验，结果说话。

**实证后跋**（2026-05-02）：项目没失败也没"颠覆"——落在中间。Lorentzian 几何先验**确实**编码了因果结构的方向（18/18 cross-structure direction-match, 0 inverted），但**不是**通过赢 Euclidean baseline 实现，而是通过 distinct geometric regime + 可解释 m_q signed magnitude。这是诚实数据告诉我们的事，论文据此叙事。物理论文层独立 finalize，与 ML 实验成败解耦——这条 "双轨独立" 的策略在 2026-05 启动时就写下了，现在被实证支持。
