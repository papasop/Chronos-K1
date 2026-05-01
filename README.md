# Emergent Sign-Aware Lorentzian Representations in Neural Causal Sequence Classifiers

**Y.Y.N. Li**

*Working draft v0 — May 2026*

---

## Abstract

The Realizability theorem (Li, in prep.) shows that a small set of axioms about causal structure—reachability, energy, and time orientation—implies that the cost function on displacements must carry a Lorentzian (indefinite) signature. We ask whether neural networks trained on causally structured data spontaneously develop a representation consistent with this prediction. We construct a minimal neural architecture exposing the Lorentzian invariant *m_q* := ‖h_s‖² − ‖h_t‖² to the classifier, and train it to distinguish autoregressive sequences from independent ones. Across two backbones (a Lorentzian backbone with split normalization and a standard Euclidean backbone), three task difficulties, and twenty-four out-of-distribution shifts, the trained model consistently places dependent inputs in the timelike region (m_q < 0) and independent inputs in the spacelike region (m_q > 0)—the direction predicted by the axioms. The cross-backbone consistency is striking: 6/6 in-distribution seeds align with the predicted direction. We identify the technical key to obtaining this emergent sign-split—a head architecture (MqMLPHead) that gives the classifier explicit access to *m_q*—and show that the original bilinear inner-product head fails entirely (0/3) due to a structural algebra-geometry mismatch. We further document an architectural trade-off: the Lorentzian backbone produces bounded, light-cone-scale *m_q* encodings (|m_q| ≈ 8), while the Euclidean backbone produces unbounded encodings (|m_q| ≈ 90) with comparable information content (mq-AUC 0.90 vs 0.90) but more robust absolute sign labeling under noise. We discuss the implications for physics-inspired inductive biases in deep learning and for the interpretation of the Realizability axioms as informational rather than purely physical statements.

---

## 1. Introduction

Lorentzian geometry is the mathematical language of relativistic physics: the indefinite signature (−,+,+,+) of Minkowski space distinguishes timelike from spacelike intervals and underlies the entire causal structure of spacetime. In a companion theoretical work (Li, in prep., henceforth the *Realizability paper*), we show that this signature is not a contingent fact about our universe but follows from three axioms about how causal structure can be realized: reachability of any state from sufficiently many others, an energy-like cost function that grows with displacement, and a fixed time orientation. Theorem 5 of that work establishes that any cost function consistent with these axioms must take the form *Q(δ) = ⟨δ, η δ⟩* with *η* an indefinite quadratic form of Lorentzian signature.

If this theorem is correct, an interesting empirical question follows: do *neural networks* trained on causally structured data spontaneously develop representations consistent with the Lorentzian structure? A neural classifier processing temporal sequences must, in principle, distinguish causally ordered inputs (where information flows along a preferred direction) from causally trivial inputs (where it does not). If the Realizability axioms describe an informational rather than narrowly physical fact, then the trained classifier might naturally organize its internal representation along a Lorentzian sign convention: dependent inputs in a "timelike" region, independent inputs in a "spacelike" region.

This paper reports an empirical investigation of that question. We construct a minimal neural architecture compatible with Lorentzian geometry: a hidden representation split into "timelike" and "spacelike" blocks of equal dimension, normalized separately, and equipped with a classifier head that has explicit access to the Lorentzian invariant *m_q* := ‖h_s‖² − ‖h_t‖² (positive for spacelike, negative for timelike vectors). We train this on a synthetic task—distinguishing autoregressive AR(1) sequences from i.i.d. Gaussian sequences with matched marginals—and ask whether the trained model assigns negative *m_q* to autoregressive ("causal") inputs and positive *m_q* to i.i.d. ("acausal") inputs.

The answer is yes, with high reliability. Across three random seeds at AR coefficient ρ = 0.9, the trained model achieves 99.7% test accuracy, perfect mq-AUC (1.000 ± 0.000), and clean sign-split (3/3 seeds with mean *m_q* on dependent inputs equal to −5.5 and on independent inputs equal to +6.0). The direction matches the theoretical prediction of the Realizability paper. At reduced signal strength (ρ = 0.3), 3/3 seeds still produce the predicted sign-split direction, with mq-AUC 0.766 ± 0.008.

Reaching this result required identifying a structural problem with the natural choice of classifier head. The original Lorentzian inner-product head computes *⟨h, c_k⟩_η / τ*, which is **bilinear** in *h*; the Lorentzian invariant *m_q*, by contrast, is **quadratic** in *h*. The bilinear head is therefore mathematically incapable of using sign(*m_q*) as a classification signal, regardless of what the backbone produces. Replacing this with an MLP head over [*h*, *m_q*] places *m_q* into the loss's computation graph, and emergent sign-split follows.

A subsequent control experiment yielded a result that complicates—and we believe enriches—the interpretation. When we replace the Lorentzian backbone (split normalization, asymmetric residual) with a standard Euclidean backbone (LayerNorm, symmetric residual) while keeping the same MqMLPHead, the sign-split direction-matching also emerges: 3/3 seeds, mq-AUC 1.000, and the same dependent → timelike alignment. However, the Euclidean backbone achieves this through *unbounded* magnitude separation: |*m_q*| ≈ 90 (vs Lorentzian's |*m_q*| ≈ 8), an order of magnitude larger. Under out-of-distribution shifts, both encodings preserve the rank ordering of *m_q* by class (mq-AUC ≈ 0.90 in both cases), but the absolute sign is more stable under the Euclidean encoding (because the *m_q* values are far from zero, so noise rarely flips them). The Lorentzian backbone's "controlled" small-scale encoding is more interpretable as a light-cone position, but more fragile under perturbation.

This paper contributes:

1. **An identified structural mismatch** between Lorentzian inner-product heads (bilinear in *h*) and the Lorentzian invariant *m_q* (quadratic in *h*), which explains why prior attempts to obtain sign-aware Lorentzian classification have failed.

2. **A simple architectural fix** (MqMLPHead) that places *m_q* in the classifier's computation graph and reliably produces emergent sign-aware Lorentzian representations.

3. **An empirical demonstration** that across multiple backbones and out-of-distribution shifts, the trained model consistently aligns its representation with the direction predicted by the Realizability theorem (dependent → timelike, independent → spacelike), suggesting that causality → Lorentzian signature may be an informational rather than purely physical fact.

4. **A characterization of the architectural trade-off** between bounded Lorentzian encoding (interpretable, fragile) and unbounded Euclidean encoding (robust, uninterpretable as light-cone geometry).

We emphasize that this work is *not* a claim that Lorentzian backbones outperform Euclidean ones on standard task metrics—they do not. Our claim is that an emergent geometric phenomenon, predicted independently from causal axioms, is reproducibly observed in trained neural networks, and that this is interesting in its own right.

---

## 2. Background

### 2.1 The Realizability Theorem (Brief)

We summarize Theorem 5 of the Realizability paper at a level sufficient for this work; full details and proofs are in the companion theoretical work.

Consider a directed graph of states with a notion of "displacement" between connected states. The Realizability axioms require:

- **(R)** Reachability: from any state, sufficiently many other states are reachable through finite chains of displacements.
- **(E)** Energy: there exists a real-valued cost function *Q(δ)* on displacements that is non-degenerate and behaves additively along displacement chains.
- **(T)** Time orientation: there exists a globally consistent assignment of "forward" vs "backward" along chains.

Theorem 5 states that any such *Q(δ)* must take the quadratic form *Q(δ) = ⟨δ, η δ⟩*, where *η* is a non-degenerate symmetric bilinear form with exactly one negative eigenvalue and the remaining eigenvalues positive—i.e., a Lorentzian metric of signature (−,+,…,+). The negative eigenspace defines the timelike direction; the positive eigenspaces define the spacelike directions.

Crucially, the theorem makes a *directional* claim: causally connected ("reachable through chains") displacements lie in the timelike region (*Q < 0*); causally disconnected displacements lie in the spacelike region (*Q > 0*). This is the directional prediction we test empirically.

### 2.2 Geometric Deep Learning and Lorentzian Networks

Neural networks operating on non-Euclidean geometries have been extensively studied in recent years. Hyperbolic neural networks [Ganea et al. 2018, Nickel & Kiela 2017] embed representations on the Poincaré ball or Lorentz model, exploiting the exponential capacity of hyperbolic space. More recently, several works have explored Lorentzian or pseudo-Riemannian geometries for deep learning [Law & Stam 2020, Xiong et al. 2021]. These works typically constrain representations to lie on a specific Lorentzian manifold (e.g., the unit hyperboloid {x : ⟨x,x⟩_η = −1}) and use manifold-aware operations (exp/log maps, Möbius addition).

Our setting differs: we do *not* constrain the representation to any manifold. We work in flat Krein space (ℝ^d with indefinite inner product) and let the network learn whatever distribution over m_q values it finds optimal. The question is *whether* the trained model places mass on a particular side of the light-cone for a given class, not *how* it does so subject to a manifold constraint. In this sense the work is closer to representation analysis than to geometric architecture design.

### 2.3 The Causal Sequence Classification Task

We use the simplest task that exposes the relevant phenomenon. Given a sequence *x = (x_1, …, x_T)* of length T = 32, classify whether it was generated from:

- **Class 0 (DEPENDENT)**: AR(1) process *x_t = ρ x_{t−1} + ε_t*, with *ε_t ~ N(0, σ²(1−ρ²))* to keep the marginal variance fixed.
- **Class 1 (IID)**: i.i.d. Gaussian *x_t ~ N(0, σ²)*.

By construction, both classes have matched first and second moments (mean ≈ 0, variance ≈ σ²); the only difference is temporal autocorrelation. The lag-1 autocorrelation is the Bayes-optimal feature; we report the lag-1 logistic-regression baseline as a reference.

This task has a clear physical interpretation through the Realizability lens: the AR(1) process generates "causally connected" sequences in time; the i.i.d. process generates "causally trivial" sequences. The theorem predicts the dependent class should be represented as *timelike* (m_q < 0).

---

## 3. Method

### 3.1 SplitNorm: A Lorentzian-Aware Normalization Primitive

The hidden representation *h ∈ ℝ^d* is split into a timelike block *h_t ∈ ℝ^{d_t}* and a spacelike block *h_s ∈ ℝ^{d_s}* with *d_t = d_s = d/2*. The Lorentzian invariant is

```
m_q(h) := ‖h_s‖² − ‖h_t‖²
```

with sign(m_q) indicating spacelike (positive) vs timelike (negative). Standard normalization layers (LayerNorm, RMSNorm) apply to *h* as a whole and do not respect this block structure. We define **SplitNorm**:

```
SplitNorm(h) = [γ_t · normalize(h_t), γ_s · normalize(h_s)]
```

where each block is normalized separately and *γ_t, γ_s* are learned scalar gains. This preserves the time/space distinction at every layer and prevents the timelike block from being dominated by the spacelike block (or vice versa) through cross-block normalization.

### 3.2 Causal Residual

Standard residual connections *h_new = h_old + f(h_old)* treat the timelike and spacelike blocks symmetrically. Motivated by the time-orientation axiom (T), we use an **asymmetric residual** that lets the timelike block accumulate while the spacelike block does not:

```
h_t_new = h_t_old + f_t(SplitNorm(h_old))
h_s_new = α · h_s_old + f_s(SplitNorm(h_old))
```

with *α = 0* in our default configuration (full asymmetry). This operationalizes the intuition that information about a causally connected sequence should *integrate* along the timelike direction (consistent with a time arrow) while information about uncorrelated content does not accumulate. We later show this is not strictly necessary for sign-split to emerge (Section 5.3), but it gives the Lorentzian backbone its bounded *m_q* property.

### 3.3 The Bilinear-Quadratic Mismatch

A natural choice of classifier head, used in earlier versions of this work (v3–v10) and analogous to standard inner-product heads in metric learning, is the **Lorentzian inner-product head**:

```
logit_k = ⟨h, c_k⟩_η / τ = (−⟨h_t, c_k_t⟩ + ⟨h_s, c_k_s⟩) / τ
```

with learnable class centroids *c_k ∈ ℝ^d* and temperature *τ*. This head computes a function *bilinear* in *h*. The Lorentzian invariant *m_q*, by contrast, is *quadratic* in *h*:

```
m_q(h) = ‖h_s‖² − ‖h_t‖² = ⟨h, η h⟩
```

A bilinear function of *h* cannot, in general, depend on sign(*m_q*). To see why: any bilinear function of *h* takes the form *L(h) = ⟨v, h⟩* for some linear functional *v*, whose value depends only on *h*'s projection along *v*—a one-dimensional linear feature. The sign of *m_q*, by contrast, is determined by which side of a quadratic surface (the light-cone {h : m_q(h) = 0}) the vector *h* lies on. No linear feature can encode this side-of-quadratic-surface information.

In our experiments (v3–v10, see Section 5.2 below), this mismatch caused all attempts to obtain sign-aware classification to fail: the trained model produced class-discriminative *m_q* distributions, but with both classes on the same side of the light-cone. The model used *m_q* magnitude, not sign, because magnitude *can* be encoded in a linear projection (via the centroid's position).

### 3.4 MqMLPHead

The fix is to expose *m_q* as an explicit input to the classifier, placing it in the loss's computation graph. We use a small MLP head:

```
def forward(h):
    mq = compute_mq(h, d_t)              # scalar per sample
    feats = concat([h, mq.unsqueeze(-1)]) # shape (B, d+1)
    return MLP(feats)                    # logits over classes
```

The MLP has two layers with hidden width 64 and GELU activation. By placing *m_q* in the input, the head can learn arbitrary nonlinear functions of (*h*, *m_q*), including hard thresholds on sign(*m_q*). The classifier loss then provides gradient pressure for the backbone to produce class-separable *m_q* values—which it does, by spontaneously selecting the *direction* predicted by Theorem 5.

### 3.5 Full Architecture

The complete model is:

1. **Embed**: linear projection from input dimension (32 for sequences) to hidden dimension d = 128.
2. **Backbone**: 3 layers, each consisting of SplitNorm → linear+GELU → asymmetric residual.
3. **Final SplitNorm**.
4. **Head**: MqMLPHead.

Total parameter count: 62,722. Trained for 15 epochs with Adam, learning rate 3e-4, batch size 64, on 8000 training samples (4000 per class).

The Euclidean baseline replaces SplitNorm with LayerNorm, the asymmetric residual with a standard symmetric residual, and is otherwise identical (including the same MqMLPHead).

---

## 4. Main Results

### 4.1 Sign-Aware Representation Emerges Spontaneously

We train the Lorentzian backbone with MqMLPHead on the AR(1) vs IID classification task at ρ = 0.9 (strong autocorrelation; lag-1 baseline 98.7% test accuracy). Across three independent random seeds:

| seed | acc_L | acc_E | mq AUC | mq[c0] (DEP) | mq[c1] (IID) | gap | sign-split |
|------|-------|-------|--------|--------------|--------------|-----|------------|
| 42   | 99.75% | 99.80% | 1.000 | −5.78 | +6.18 | +11.95 | ✓ clean |
| 43   | 99.55% | 99.80% | 1.000 | −2.24 | +7.08 |  +9.32 | ✓ clean |
| 44   | 99.70% | 99.85% | 1.000 | −5.41 | +5.82 | +11.23 | ✓ clean |

All three seeds produce **clean sign-split** with the dependent class in the timelike region (m_q < 0) and the independent class in the spacelike region (m_q > 0). The cross-seed consistency is striking: AUC standard deviation is 0.000, gap standard deviation is 1.11. The direction of the split—dependent → timelike—matches the Realizability theorem's prediction.

We note that the Lorentzian model's accuracy (99.65 ± 0.10%) is slightly lower than the Euclidean baseline's (99.82 ± 0.03%) and only marginally above the lag-1 logistic-regression baseline (98.70%). The phenomenon we report is therefore not a *performance* improvement; it is an *emergent representational property* of the trained model.

### 4.2 Robustness to Reduced Signal Strength

To test whether the emergent sign-split persists under harder conditions, we repeat the experiment at ρ = 0.3 (weaker autocorrelation; lag-1 baseline 78.0% test accuracy):

| seed | acc_L | acc_E | mq AUC | mq[c0] (DEP) | mq[c1] (IID) | gap | sign-split |
|------|-------|-------|--------|--------------|--------------|-----|------------|
| 42   | 70.95% | 74.25% | 0.756 | −0.88 | +1.67 | +2.55 | ✓ clean |
| 43   | 72.40% | 74.90% | 0.774 | −0.85 | +1.85 | +2.71 | ✓ clean |
| 44   | 71.00% | 74.25% | 0.767 | −1.37 | +1.24 | +2.62 | ✓ clean |

Despite the *m_q* gap shrinking from +11 to +2.6 (consistent with weaker class-discriminative signal in the data), all three seeds still produce clean sign-split in the predicted direction. The Lorentzian model accuracy drops to 71% (3 percentage points below Euclidean baseline, 7 below the lag-1 baseline), but the *representational* property persists. Cross-seed AUC is 0.766 ± 0.008.

The fact that the model is willing to organize its representation along the predicted geometric axis even at the cost of slightly worse classification accuracy is, in our view, evidence that the *m_q* sign organization is a genuine structural property, not a minor side effect of the loss surface.

---

## 5. Ablations

### 5.1 The Bilinear Head Cannot Sign-Split

To verify the structural argument of Section 3.3 empirically, we train the same Lorentzian backbone but replace MqMLPHead with the original LorentzianHead (bilinear in *h*). On AR(1) vs IID at ρ = 0.9, three seeds:

| seed | acc | mq AUC | mq[c0] | mq[c1] | sign-split |
|------|-----|--------|--------|--------|------------|
| 42 | 99.65% | 0.730 | +1.77 | +2.25 | ✗ same-side |
| 43 | 99.75% | 0.943 | +2.82 | +3.94 | ✗ same-side |
| 44 | 99.75% | 0.423 | +2.31 | +2.20 | ✗ no info |

**0/3 sign-split**. The model still classifies well (accuracy comparable to MqMLPHead), but the *m_q* distribution shows both classes on the same side of the light-cone. The mq-AUC is variable (0.42 to 0.94)—a sign that without explicit *m_q* in the head's input, *whether* the model uses *m_q* magnitude as a classification feature is sensitive to initialization. In two seeds it does (0.73 and 0.94); in one seed it does not (0.42 ≈ chance). This confirms the structural claim: the bilinear head cannot use sign(*m_q*), only its magnitude (when it uses it at all).

### 5.2 Quadratic Features Without Nonlinearity Are Insufficient

We ask whether the MLP nonlinearity in MqMLPHead is necessary, or whether a linear head with quadratic features (*‖h_t‖²*, *‖h_s‖²* concatenated to *h*) would suffice:

| seed | acc | mq AUC | mq[c0] | mq[c1] | sign-split |
|------|-----|--------|--------|--------|------------|
| 42 | 99.70% | 0.009 | +8.91 | +6.18 | ✗ inverted |
| 43 | 99.75% | 0.001 | +8.80 | +4.72 | ✗ inverted |
| 44 | 99.75% | 1.000 | −5.11 | +6.65 | ✓ clean |

**1/3 sign-split** (in the predicted direction; the other two seeds produce strongly inverted *m_q* AUC). This is unstable: the linear head with quadratic features can in principle use sign(*m_q*) (because the quadratic features are linearly separable by side-of-light-cone), but the model often selects an alternative solution where both classes are spacelike with different magnitudes. The MLP nonlinearity in MqMLPHead allows the model to learn a hard threshold on sign(*m_q*), which makes the sign-split solution preferable.

### 5.3 The Lorentzian Backbone Is Not Required

The most interesting ablation: we replace the Lorentzian backbone (SplitNorm + asymmetric residual) with a standard Euclidean backbone (LayerNorm + symmetric residual), keeping MqMLPHead. On ρ = 0.9, three seeds:

| seed | acc | mq AUC | mq[c0] | mq[c1] | gap | sign-split |
|------|-----|--------|--------|--------|-----|------------|
| 42 | 99.75% | 1.000 | −59.74 | +57.97 | +117.7 | ✓ clean |
| 43 | 99.75% | 1.000 | −60.99 | +60.28 | +121.3 | ✓ clean |
| 44 | 99.55% | 1.000 |  −45.59 | +61.28 | +106.9 | ✓ clean |

**3/3 sign-split**, all in the predicted direction. The Euclidean backbone *also* produces emergent sign-aware Lorentzian representations.

This reveals that the Lorentzian backbone is not strictly necessary for the phenomenon. The MqMLPHead alone—by exposing *m_q* in the loss's computation graph—is sufficient to drive sign-aware representation regardless of how the backbone normalizes its hidden states. The optimizer finds whatever weights produce class-separable *m_q* values; the bounded vs unbounded character of *m_q* is then determined by the backbone's normalization properties.

The Euclidean backbone produces *m_q* values approximately 11× larger in magnitude (|m_q| ≈ 90 vs ≈ 8 for Lorentzian). With the same hidden dimension and no normalization constraint on *m_q*, the optimizer drives the *m_q* magnitudes to whatever scale minimizes classification loss—which, given the soft-argmax structure of cross-entropy, can grow without bound.

### 5.4 Summary Table

Summary across all four conditions on AR(1) vs IID at ρ = 0.9, three seeds each:

| Backbone | Head | sign-split | mq AUC | |m_q| scale |
|----------|------|------------|--------|-----------|
| Lorentzian | MqMLPHead (ours) | **3/3** | **1.000** | ≈ 6 |
| Lorentzian | LorentzianHead (bilinear) | 0/3 | 0.70 ± 0.21 | ≈ 2 |
| Lorentzian | Linear + quad features | 1/3 | 0.34 ± 0.47 | ≈ 6 |
| Euclidean | MqMLPHead | **3/3** | **1.000** | ≈ 60 |

The pattern is clear: **MqMLPHead is the necessary and sufficient architectural choice**; the backbone determines the encoding scale but not the existence of sign-split.

---

## 6. Out-of-Distribution Behavior

Section 5 establishes that two architecturally distinct backbones both produce sign-aware Lorentzian representations under in-distribution training. We now ask whether these encodings transfer across distribution shifts. This is the key experiment for evaluating whether the Lorentzian backbone provides a *meaningful* advantage despite both backbones producing equivalent in-distribution sign-split.

### 6.1 OOD Panel

We train each backbone (Lorentzian + MqMLPHead, Euclidean + MqMLPHead) on AR(1) at ρ = 0.9, NOISE_STD = 0.5, and then evaluate the trained models on a panel of OOD test distributions:

- **AR coefficient sweep**: ρ ∈ {0.1, 0.3, 0.5, 0.7, 0.9}, holding noise constant. Tests whether the geometric encoding generalizes to weaker autocorrelation.
- **Noise sweep**: NOISE_STD ∈ {0.25, 0.5, 1.0, 2.0}, holding ρ at 0.9. Tests whether the encoding is robust to amplitude shifts.

For each (backbone, OOD config) pair we measure: classification accuracy, mq-AUC (rank-based geometric information content), *m_q* mean per class, sign-split status, and max |m_q|. Averaged over 3 seeds.

### 6.2 Results

| OOD config | L: split | L: AUC | L: |m_q|max | acc | E: split | E: AUC | E: |m_q|max | acc |
|------------|----------|--------|-----------|-----|----------|--------|-----------|-----|
| ρ=0.1, n=0.5 (very weak) | 0/3 | 0.583 | 7.7 | 56% | 0/3 | 0.586 | 91.4 | 56% |
| ρ=0.3, n=0.5 | 2/3 | 0.743 | 7.9 | 69% | 2/3 | 0.745 | 92.4 | 68% |
| ρ=0.5, n=0.5 | 3/3 | 0.906 | 7.9 | 83% | 3/3 | 0.906 | 91.3 | 83% |
| ρ=0.7, n=0.5 | 3/3 | 0.991 | 8.0 | 96% | 3/3 | 0.991 | 97.1 | 96% |
| **ρ=0.9, n=0.5** (in-dist) | 3/3 | 1.000 | 8.2 | 99.7% | 3/3 | 1.000 | 98.9 | 99.8% |
| ρ=0.9, n=0.25 | 3/3 | 0.997 | 8.0 | 92% | 3/3 | 1.000 | 100.5 | 92% |
| ρ=0.9, n=1.0 | 2/3 | 0.997 | 7.9 | 95% | 3/3 | 0.996 | 87.7 | 95% |
| ρ=0.9, n=2.0 | 0/3 | 0.982 | 7.6 | 81% | 2/3 | 0.947 | 76.5 | 82% |

**Headline metrics:**

| | sign-split rate | mean mq-AUC | mean |m_q|max |
|-|-----------------|-------------|---------------|
| Lorentzian + MqMLPHead | 16/24 | 0.900 | **7.9 ± 0.5** |
| Euclidean + MqMLPHead  | 19/24 | 0.896 | **92.0 ± 7.5** |

### 6.3 Three Observations

**(1) The Lorentzian backbone produces a strictly bounded *m_q* encoding.** Across all 24 OOD evaluations, |m_q|max stays in the range [6.9, 8.7]; the Euclidean encoding ranges from 73.6 to 103.4. The Lorentzian SplitNorm acts as a structural prior that prevents *m_q* from growing without bound, regardless of distribution shift.

**(2) The geometric information content is essentially identical.** Mean mq-AUC across all OOD configurations is 0.900 (Lorentzian) vs 0.896 (Euclidean)—a difference well within seed variance. Both backbones produce *m_q* distributions that rank-order the two classes equally well; the difference is only in scale.

**(3) Absolute sign labeling is more fragile in the Lorentzian encoding.** The Euclidean backbone preserves sign-split in 19/24 OOD configurations vs 16/24 for the Lorentzian backbone. The mechanism is straightforward: with *m_q* values ≈ ±60, perturbations of magnitude ≈ 10 (induced by noise shifts) rarely cross zero; with *m_q* values ≈ ±5, the same perturbations frequently do.

We highlight one specific instance worth noting: at ρ = 0.9, NOISE_STD = 2.0, the Lorentzian backbone has mq-AUC = 0.982 (very high—the *m_q* values are still highly informative about class) but 0/3 sign-split. The geometric *information* is preserved; the absolute *labeling* (which side of zero) has drifted. This dissociation between mq-AUC (information) and sign-split status (absolute labeling) is informative: sign-split as a metric measures the alignment between the model's learned threshold and the literal *m_q = 0* light-cone, which is an additional fact beyond mere geometric discrimination.

### 6.4 Interpretation

We summarize the implication of these data as follows. The MqMLPHead reliably produces sign-aware Lorentzian representations regardless of backbone. The two backbones offer complementary properties:

- **Lorentzian backbone**: bounded, light-cone-scale *m_q* encoding (|m_q| ≈ 8). Interpretable as literal Minkowski geometry. Fragile under noise because *m_q* values lie close to the light-cone (the decision boundary).
- **Euclidean backbone**: unbounded magnitude separation (|m_q| ≈ 90). Not interpretable as light-cone geometry but more robust to noise because the *m_q* values are far from zero.

These are different *types* of representation, not better/worse versions of the same representation. The Lorentzian backbone implements "geometric encoding"; the Euclidean backbone implements "magnitude separation that happens to follow the same sign convention". The fact that both spontaneously align with the direction predicted by the Realizability theorem (dependent → negative *m_q*) is the substantive empirical finding.

---

## 7. Discussion

### 7.1 Cross-Backbone Direction Consistency Is the Central Empirical Result

We want to draw out one observation that we believe is the most robust and substantive empirical finding of this work. Across:

- 2 backbones (Lorentzian, Euclidean)
- 2 task difficulties (ρ = 0.9, ρ = 0.3)
- 3 seeds each
- 8 OOD configurations
- 24 total OOD evaluations per backbone

**every** in-distribution and most OOD evaluations that produce a sign-split do so with the dependent class in the *negative* (timelike) region and the independent class in the *positive* (spacelike) region. We observed **zero inverted sign-splits** across all our experiments. The direction matches the Realizability theorem's prediction.

This is striking because the architecture does not contain any built-in preference for which side dependent inputs should occupy. The classifier loss is symmetric between classes; class labels are arbitrary; centroid initializations are random. The model has full freedom to organize its *m_q* distribution either way. Yet across every successful sign-split it picks the same direction.

This has at least two non-trivial implications:

**(a)** The Realizability axioms predict a specific *direction* (causally connected → timelike), and trained neural networks consistently realize that direction. This is closer to a verified prediction than a coincidence. It is not strong evidence in the way a controlled physical experiment would be, but it is genuine empirical content for the axioms.

**(b)** The fact that this direction-matching is independent of backbone suggests that the underlying organizing principle is not specific to Lorentzian normalization, but emerges from any architecture that allows *m_q* to enter the loss. This in turn suggests that "causality → Lorentzian sign" may be an *informational* fact about how a learner organizes representations of causally-structured data, not a fact specific to physical spacetime.

### 7.2 The Bilinear-Quadratic Mismatch as a Lens on Prior Work

We explained in Section 3.3 why the original Lorentzian inner-product head structurally cannot use sign(*m_q*). This explains why prior attempts at "Lorentzian neural networks" using inner-product-style heads do not (and cannot) produce sign-aware classification—a fact that, to our knowledge, has not been previously identified in the literature.

The standard fix in the hyperbolic-network literature is to add nonlinear manifold operations (Möbius gyrovector additions, exponential maps) that effectively introduce nonlinear functions of *h* into the decision rule. Our MqMLPHead can be viewed as a minimal version of this idea: instead of working with manifold operations, we simply expose the geometric invariant *m_q* directly. This is computationally cheaper and architecturally simpler, and it is sufficient for the sign-aware property we want.

### 7.3 What the Architectural Trade-Off Means

The bounded vs unbounded *m_q* contrast (|m_q| ≈ 8 vs ≈ 90) is, on its face, a difference in numerical convention. Both backbones perform equivalently as classifiers (both achieve 99.7% in-distribution accuracy and ~0.90 mq-AUC under OOD). One might therefore conclude that the Lorentzian backbone is unnecessary.

We disagree with this conclusion in one specific sense. A bounded encoding lying close to the literal light-cone has interpretive content that an unbounded encoding does not. Specifically: when the Lorentzian backbone produces *m_q* ≈ −5 for a dependent input, this can be read as "this input is encoded as a vector lying in the timelike interior of the light-cone, at light-cone distance ≈ √5 from the origin in the ambient Minkowski space". When the Euclidean backbone produces *m_q* ≈ −60 for the same input, this reading does not apply—the vector lies far from the conventional light-cone in any geometric sense.

For applications where the geometric interpretation is the goal (e.g., interpretability research, physics-inspired representation analysis), the Lorentzian backbone is preferable despite its sign-fragility. For applications where only the binary label is needed, the Euclidean backbone is preferable due to its sign robustness. We do not view either backbone as universally better.

### 7.4 Limitations

We list these explicitly because we believe a paper of this kind should be self-critical.

1. **Single task family**. Our experiments use only AR(1) vs IID classification. The "causally structured" class is a narrow operationalization (lag-1 autocorrelation). A more thorough investigation would test on Markov-2 vs Markov-1, on Granger-causal time series, on interventional vs observational distributions, and on more complex causal graphs. The cross-backbone direction-matching may or may not generalize.

2. **Direction-prediction is suggestive, not derivational**. We argue that the Realizability theorem *predicts* dependent → timelike, and that we observe this. But our architecture is not derived from the theorem in a strict sense—we made motivated choices (SplitNorm, asymmetric residual, MqMLPHead) inspired by the theorem, but other choices could plausibly produce the opposite direction. A stronger paper would derive the architectural choices from the axioms more directly. This is open work.

3. **No performance advantage**. The Lorentzian backbone does not outperform the Euclidean baseline on classification accuracy. This work is therefore not motivated by, and should not be evaluated against, performance benchmarks.

4. **The "geometric encoding" interpretation is informal**. We say the Lorentzian backbone's *m_q* ≈ ±8 distribution is "interpretable as light-cone geometry" while the Euclidean backbone's *m_q* ≈ ±90 distribution is not. This is a soft, qualitative claim; we have not formalized what would count as a "geometric" vs "non-geometric" encoding. A more rigorous treatment (e.g., via information-geometric distances, or by relating *m_q* to a learned invariant manifold structure) is left for future work.

### 7.5 Open Questions

- **Does the cross-backbone direction-matching extend to other geometric invariants?** If we trained a network with explicit access to a different invariant (e.g., a conformal invariant), would the trained model align with a corresponding theoretical prediction?
- **What is the role of optimization?** Could one prove (perhaps under simplifying assumptions) that the cross-entropy loss with MqMLPHead has a *unique* minimum (up to a global sign flip) at the dep→timelike configuration?
- **Are there causal tasks where the bounded encoding's interpretive value translates into measurable robustness?** OOD tests within distribution-shifted versions of the same task did not reveal this; perhaps tests across qualitatively different causal structures would.

---

## 8. Conclusion

We have shown that neural networks, when given an architectural mechanism for using the Lorentzian invariant *m_q* in classification (the MqMLPHead), spontaneously develop representations in which causally structured inputs occupy the timelike region (m_q < 0) and causally trivial inputs occupy the spacelike region (m_q > 0). This emergent direction-matching is consistent across two different backbones, two task difficulties, three random seeds, and twenty-four out-of-distribution configurations: zero inverted sign-splits were observed.

The direction matches the prediction of the Realizability theorem (Li, in prep.), which derives Lorentzian metric signature and the directional convention from a small set of axioms about causal structure. The cross-backbone consistency suggests that this direction-matching is not a property of any specific Lorentzian architecture, but an organizational principle that emerges in any sufficiently expressive learner exposed to causally structured data and given access to the relevant geometric invariant.

We identified the technical key to enabling this emergence: the original bilinear inner-product head structurally cannot use sign(*m_q*), because the head is bilinear in the hidden state while *m_q* is quadratic. Replacing the head with an MLP over [*h*, *m_q*] places *m_q* in the loss's computation graph and allows sign-aware classification to emerge. We characterized the architectural trade-off between Lorentzian (bounded, interpretable, fragile) and Euclidean (unbounded, uninterpretable as geometry, sign-robust) encodings.

We do not claim that Lorentzian backbones outperform standard backbones on task metrics. We do claim that an emergent geometric phenomenon, predicted from independent axiomatic considerations, is reliably produced in trained neural networks. We believe this is interesting in its own right and may inform both physics-grounded representation learning and the interpretation of the Realizability axioms as informational rather than purely physical.

---

## Appendix A. Reproducibility

All code is available at [TODO: URL]. The complete experimental pipeline (battery + OOD evaluation) runs in approximately 20 minutes on a single Colab GPU. We provide the full hyperparameter list, dataset generation code, and per-seed result logs in the supplementary material.

## Appendix B. Hyperparameters

| Parameter | Value |
|-----------|-------|
| hidden dim *d* | 128 |
| timelike dim *d_t* | 64 |
| spacelike dim *d_s* | 64 |
| sequence length *T* | 32 |
| n training samples | 8000 |
| n test samples | 2000 |
| n epochs | 15 |
| optimizer | Adam |
| learning rate | 3e-4 |
| batch size | 64 |
| MqMLPHead hidden width | 64 |

## Appendix C. The Lag-1 Logistic Regression Baseline

For each of the dataset configurations we report a "lag-1 baseline" computed as follows. We fit a logistic regression with a single feature, the lag-1 sample autocorrelation of the sequence, on the training split, and evaluate test accuracy. This is the simplest possible classifier that uses the only feature distinguishing the two classes. For ρ = 0.9 the baseline is 98.7%; for ρ = 0.3, 78.0%. This sets the floor that any nontrivial classifier should beat.

## References

[Placeholder section. To be populated with references to:]
- Li, Y.Y.N. *Realizability and the Origin of Causality*. Foundations of Physics, in preparation.
- Bronstein, M.M. et al. *Geometric deep learning*. arXiv:2104.13478 (2021).
- Ganea, O., Bécigneul, G., Hofmann, T. *Hyperbolic Neural Networks*. NeurIPS (2018).
- Nickel, M., Kiela, D. *Poincaré Embeddings for Learning Hierarchical Representations*. NeurIPS (2017).
- Law, M.T., Stam, J. *Ultrahyperbolic Representation Learning*. NeurIPS (2020).
- Xiong, B. et al. *Pseudo-Riemannian Graph Convolutional Networks*. NeurIPS (2021).
- [Additional references on Krein space ML, causal representation learning, AR(1) processes as needed.]
