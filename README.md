# 真正的洛伦兹 Backbone 设计工作计划

**项目代号**：Lorentzian Cognition Architecture (LCA)
**理论依托**：*Realizability and the Origin of Causality* (Y.Y.N. Li)
**起草日期**：2026-05
**状态**：设计阶段

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

### 支柱 1：不定度规表征向量

**目标**：构造一个 hidden state 空间，使 `inner(x, x) = -x_0² + Σx_i²` 在训练过程中可正可负可零，并且这个号差信息一路保留到输出层。

**候选方案 A：Split-Norm 表征**
- 将 `d` 维 hidden state 显式切分为 `(time_dim, space_dim) = (1, d-1)` 或 `(d_t, d_s)`
- 时间分量和空间分量分别有独立的尺度参数
- 不做联合归一化

**候选方案 B：Sign-Preserving Normalization**
- 计算 `mq = ||space||² - ||time||²`
- 归一化时使用 `sign(mq) * sqrt(|mq| + eps)`，把符号留在输出里
- 下游模块需要能处理可能为负的"范数"

**候选方案 C：Pseudo-Riemannian Manifold**
- 不强制约束，但加 KL-divergence 风格的正则项，鼓励 mq 分布跨越零点
- 让模型自己学会区分类时/类空

**验证**：
- 训练后 hidden state 的 mq 分布应该是双峰（类时一峰，类空一峰）或宽分布，不是单峰
- 类时比例应在 30%-70% 之间，不是 0% 或 100%

**工作量**：2-3 个月（设计 + 数值稳定性调试）

---

### 支柱 2：洛伦兹距离与相似度

**目标**：定义两个样本之间的"距离"，能够区分类时（因果可达）和类空（因果隔离）关系。

**候选方案 A：时空间隔**
```
s²(x, y) = -(x_0 - y_0)² + Σ(x_i - y_i)²
similarity(x, y) = -tanh(s² / τ)  # 类时时为正，类空时为负
```

**候选方案 B：洛伦兹内积**
```
<x, y>_M = -x_0 y_0 + Σ x_i y_i
similarity(x, y) = sign(<x,y>_M) * |<x,y>_M|^α  # 保留符号
```

**候选方案 C：因果带权距离**
- 类时间隔用真实时空间隔
- 类空间隔置为正无穷或大常数（"不可达"）

**验证**：
- 在已知因果结构的玩具数据上（比如时间序列因果链），相似度应该和真实因果可达性匹配
- 反事实样本（违反因果顺序的）应该获得正确的"类空"标签

**工作量**：1-2 个月（含玩具任务设计）

---

### 支柱 3：洛伦兹层归一化

**目标**：替换 LayerNorm，使归一化后保留号差信息。

**候选方案 A：Time-Space Split RMSNorm**
```python
def lorentz_rms_norm(x, gamma_t, gamma_s, beta):
    t, s = x[..., :d_t], x[..., d_t:]
    rms_t = sqrt((t**2).mean(-1, keepdim=True) + eps)
    rms_s = sqrt((s**2).mean(-1, keepdim=True) + eps)
    return cat([gamma_t * t / rms_t, gamma_s * s / rms_s], -1) + beta
```

**候选方案 B：Indefinite Mahalanobis Norm**
```python
def indefinite_norm(x):
    mq = -x[..., 0]**2 + (x[..., 1:]**2).sum(-1)
    sign = torch.sign(mq + eps)
    magnitude = sqrt(abs(mq) + eps)
    return x / (sign * magnitude + eps).unsqueeze(-1)
```
（注意：这会让类空向量被除以正数，类时向量被除以负数——意义需要数学论证）

**候选方案 C：不归一化**
- 取消 LayerNorm，改用更强的权重初始化 + gradient clipping
- 让 mq 自然演化

**验证**：
- 归一化前后，单个 token 的类时/类空标签应该保持一致（号差不被翻转）
- 训练稳定性：100 epoch 内不爆炸不消失

**工作量**：2 个月（数学正确性 + 数值稳定性都要做）

---

### 支柱 4：因果保持的残差连接

**目标**：解决 `x + f(x)` 在不定度规下可能破坏类时性的问题。

**问题数学描述**：
- 设 x 类时（mq(x) < 0），f(x) 类时
- 但 mq(x + f(x)) 不一定 < 0
- 例：x = (1, 0)（未来类时），f(x) = (-1, 2)（过去类时但空间分量大）
- x + f(x) = (0, 2)，mq = 4 > 0（变成类空！）

**候选方案 A：Causal Cone Projection**
```python
def causal_residual(x, fx):
    out = x + fx
    mq = compute_mq(out)
    # 如果 out 跑出了 x 所在的类时锥，投影回最近的类时点
    if mq > 0:
        out = project_to_causal_cone(out, x)
    return out
```

**候选方案 B：Adaptive Residual Scaling**
```python
def adaptive_residual(x, fx, max_alpha=1.0):
    # 自适应找最大的 alpha 使 x + alpha * fx 仍类时
    alpha = solve_max_alpha(x, fx, max_alpha)
    return x + alpha * fx
```

**候选方案 C：Highway-Style Gating**
- 完全放弃残差加法，改用 `gate * x + (1 - gate) * f(x)` 风格的混合
- gate 可以学习成保持类时性的形式

**候选方案 D：Soft Causal Loss**
- 残差不做硬投影，但加损失项 `λ * relu(mq(out))` 惩罚类空漂移
- 让模型在训练中自己学到稳定在类时区

**验证**：
- 训练 100 epoch 后，class token 的类时比例应稳定（不漂移到 0% 或 100%）
- 单层前后类时性的翻转率应 < 10%

**工作量**：3 个月（这是五个支柱里最难的，可能需要几次推倒重来）

---

### 支柱 5：洛伦兹注意力机制

**目标**：注意力的每一步都用洛伦兹度规，且 softmax 能正确处理号差 score。

**问题分析**：
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
（这种做法忽略号差，但保留洛伦兹尺度）

**候选方案 D：Two-Channel Attention**
- 类时 attention 和类空 attention 分两个通道独立计算
- 输出拼接

**验证**：
- 注意力权重的因果一致性：在时序数据上，t 时刻的 query 应主要关注 ≤ t 的 key
- attention pattern 可视化应显示明显的"光锥"结构

**工作量**：2-3 个月（含可视化工具开发）

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

### Phase 0：理论基础（Month 0-1）

**目标**：确认数学正确性，避免浪费工程力气。

- [ ] 写一份正式的数学描述：定义不定度规表征空间、洛伦兹归一化、因果残差的形式定义
- [ ] 与一两位几何深度学习/微分几何研究者交流，确认设计没有显然错误
- [ ] 阅读：Lorentz embeddings 文献、indefinite kernel methods、Krein space neural networks（这个方向几乎没人做，文献很少）

**Gate**：如果数学描述里发现根本性矛盾（比如某个支柱在原理上不可能），停止项目或重新设计。

### Phase 1：单支柱原型（Month 2-7）

每个支柱独立实现 + 在玩具任务上验证。

- [ ] 支柱 1（表征空间）+ 支柱 3（归一化）：先做这两个，因为它们决定了整个数据流的形态
- [ ] 玩具任务：MNIST / CIFAR 分类（不期待性能赢，只验证能稳定训练）
- [ ] 支柱 2（距离）+ 因果对比学习：在合成 DAG 时序数据上验证
- [ ] 支柱 4（残差）：和 1+3 整合，跑通 6 层网络
- [ ] 支柱 5（注意力）：和 1+3+4 整合，跑通 attention block

**Gate**：能否在 MNIST 上稳定训练 100 epoch 且类时比例不退化？如果不能，回到 Phase 0 重新设计支柱 4。

### Phase 2：完整 backbone（Month 8-12）

- [ ] 整合五个支柱为一个完整的 Lorentzian Transformer Block
- [ ] 在 4 个对照任务上跑：
  - 时序预测（验证因果结构有用的任务）
  - 文本分类（验证不会比 baseline 差太多）
  - 物理轨迹预测（你原本的 motivating task）
  - 视频帧预测（验证时空结构）
- [ ] **每个任务都要有 Euclidean baseline 做严格对照**，控制参数量和训练预算

**Gate**：在至少 1 个任务上系统性赢 Euclidean baseline（不是边际差异，是显著差异 d > 0.5）。如果一个都没赢，需要诚实评估理论可能错。

### Phase 3：scaling 与发表（Month 13-18）

- [ ] 在更大的数据集和模型规模上验证（如果有算力）
- [ ] 写论文，目标 NeurIPS / ICML / ICLR
- [ ] 开源代码 + 预训练权重

**Gate**：能否在某个 benchmark 上达到 SOTA 或接近 SOTA？如果只是"能跑但没优势"，可作为方法论论文发表，但不能宣称颠覆性。

---

## 5. 风险与应对

### 风险 1：数值不稳定无解

**概率**：高（这是这条路最大的工程风险）

**症状**：mq 接近零时归一化爆炸；类时/类空跨越时梯度跳变；训练 loss 周期性 NaN

**应对**：
- Phase 0 阶段就识别这个风险
- 准备 fallback：如果完全不定度规无法稳定训练，退而求其次只允许 mq < 0（即 hyperbolic embedding 的扩展），把"颠覆性"主张降级
- 与数值线性代数专家合作

### 风险 2：理论上能 work，工程上没优势

**概率**：中

**症状**：架构能稳定训练，但在所有 benchmark 上都不显著赢 Euclidean

**应对**：
- 这本身是有价值的负面结果——说明洛伦兹几何的优势（如果有）需要在更专门的任务上才能显现
- 转向：寻找洛伦兹几何**独有**的能力（比如可解释的因果可视化），即使性能持平也有研究价值
- 不要硬调指标——如果实测没用就承认

### 风险 3：被 scoop

**概率**：低-中（这个方向冷门，但任何时候可能有大组突然出手）

**应对**：
- Phase 1 完成时发布 arXiv preprint 占坑
- 即使被 scoop，物理论文层的 *Realizability* 框架是你独家的，AI 实现层的工作可以作为应用案例

### 风险 4：单人无法完成

**概率**：很高（如前所述，这是博士论文级工程量）

**应对**：
- 必须找合作者。理想画像：
  - 一位几何深度学习背景的博士生/postdoc（负责支柱 1-3）
  - 一位有 transformer 优化经验的工程师（负责支柱 4-5 和稳定性）
  - 你自己负责理论指导 + 因果对比学习设计
- 如果 12 个月内找不到合作者，建议把这个项目降级为"长期理论方向"，集中精力先发物理论文

### 风险 5：跑出来洛伦兹几何确实没用

**概率**：未知（这正是要验证的）

**应对**：
- 这是科学研究本来的样子。如果实验告诉你理论 prediction 错，承认，发表负面结果，调整理论
- 物理论文层的工作不依赖 AI 实验成败

---

## 6. 成功标准

### 最低成功（项目完成）

- 五个支柱各自有可工作的实现
- 完整 backbone 能在至少一个任务上稳定训练到收敛
- 写出方法论论文，记录设计选择和发现

### 中度成功（值得继续）

- 在至少 1 个 benchmark 上系统性赢 Euclidean baseline（d > 0.5）
- 类时性可视化能展示有意义的因果结构
- 论文被 top-tier 会议接受

### 高度成功（颠覆性证据）

- 在多个任务上一致赢 Euclidean baseline
- Scaling laws 显示洛伦兹优势随规模扩大
- 引发后续工作（其他实验室复现 / 扩展）

### 彻底失败（坦然承认）

- 数值上无法稳定训练超过 100 epoch
- 或者能训练但在所有任务上都显著输给 baseline
- → 论文写"洛伦兹几何不是表征学习的有效先验，原因如下"，作为负面结果发表

---

## 7. 资源需求

### 最低配置（单人勉强能做）

- 1 张 RTX 4090 或同级 GPU
- 你自己的时间（如果是兼职，预计 2-3 年）
- AWS / Lambda 上偶尔租用 A100 做大实验（年预算 ~$10k）

### 推荐配置（合作团队能做完）

- 2-3 人团队（你 + 几何 DL 博士生 + 工程师）
- 8 张 A100 集群（自有或云租）
- 18 个月时间
- 总预算估计 $200k-$500k（人力 + 算力）

### 理想配置（能 scale 到出 SOTA）

- 5-10 人小型实验室
- 64-128 张 H100
- 3-5 年时间
- 总预算 $5M-$20M
- → 现实地说，这个量级需要拉风投或加入大厂

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

## 9. 立即可做的下一步（本月）

1. **写支柱 1 + 3 的形式定义**（数学描述 + 伪代码），1-2 周
2. **联系 2-3 位几何深度学习方向的研究者**，请他们 review 数学定义。Twitter/X 上的 @MaxWelling、@taco_cohen、Bronstein 实验室都是潜在对象
3. **跑一次纯支柱 1 的实验**：把现有 F3 模型的 LayerNorm 替换为 Split-Norm，看类时比例分布是否改变。这是 1 周的工作，能给一个早期 sanity check
4. **决定合作者招募策略**：是发邮件给特定研究者，还是在 Reddit / Twitter 公开招募？

---

## 10. 决策日志

记录每次重大设计选择和理由，避免后期忘记为什么这么做。

| 日期 | 决策 | 理由 | 备选方案 |
|------|------|------|----------|
| 2026-05 | 项目启动 | 现有 F3/Riemannian 都不是真洛伦兹，需要从头设计 | 放弃 AI 方向，只发物理论文 |
| TBD | 支柱 3 选 A/B/C | | |
| TBD | 支柱 4 选 A/B/C/D | | |

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

**最后一句**：这是一个有可能完全失败的项目。但如果你的物理理论方向对，这条路上的任何中间结果都是有价值的——哪怕最后结论是"洛伦兹几何不能直接当 AI 表征空间用"，那也是一个对场域有贡献的负面结果。诚实做实验，结果说话。
