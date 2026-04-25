# Seed (Exploratory) — Flip Rate as Trajectory Divergence

**Working title**: Flip Rate as Trajectory Divergence in Autoregressive Models
**Subtitle**: *Two Regimes of Context-Perturbed Instability — An Exploratory Study*

> **STATUS: Exploratory (post-hoc analysis of pre-existing evaluation data).**
>
> This document describes *hypothesis-generating evidence*, not validated claims. All thresholds, metric definitions (including `k` in PDP), pooling decisions, and comparison protocols were selected post-hoc, after observing the data. Strong claims require pre-registered, out-of-sample validation — see `contract_validation.md` (Phase 2 protocol).
>
> Language convention: *we observe*, *we find*, *these observations suggest*. Never *we prove*, never *we show definitively*.
>
> v1 (2026-04-12). Pivot P1 "Two Regimes" validé par RD après discussion adversariale sur rigueur statistique.

---

## Hypothesis (one sentence, exploratory)

> *We hypothesize that divergence under context perturbation is not well captured by a single scalar metric, and may decompose into at least two method-dependent dimensions — **timing** (Persistent Divergence Point, PDP) and **density** (flip rate). The observations reported here are consistent with this hypothesis but do not establish it.*

---

## 1. Framework

### 1.1 Autoregressive generation as a constrained dynamical system

Let `x_t` be the token at step `t`, `z_t ∈ ℝ^d` the latent state, `f` the internal dynamics, `g` the projection to logits:

```
z_t = f(z_{t-1}, x_{t-1}) + Δ_adapt + Δ_constraint
p(x_t | x_{<t}) = softmax(g(z_t))
```

We posit an implicit **coherence manifold** `M ⊂ ℝ^d` such that `z ∈ M` corresponds to syntactically, semantically, and contextually coherent states. Generation approximates a trajectory `{z_t^*}` on `M`. **We do not construct `M`**; we postulate it as an interpretive lens.

### 1.2 Divergence as departure from M

Define `e_t = d(z_t, M)`. Under context perturbation `x_{<t} → x̃_{<t}`, the trajectory shifts `z_t → z̃_t`, producing a token-level observable: the **flip** `flip_t = 1[argmax g(z_t) ≠ argmax g(z̃_t)]`.

### 1.3 Two regimes (core contribution)

**Instantaneous divergence** — perturbation of `z_0` causes immediate token-level disagreement. Regime-defining under perturbations that affect the initial state (e.g., uniform KV-cache eviction).

**Persistent divergence** — alignment is preserved initially; divergence accumulates and becomes self-sustaining later. Regime-defining under perturbations that preserve recent context (e.g., recent-window KV-cache retention).

---

## 2. Metrics — Two candidate observables

| Observable | Symbol | What it measures | Formula |
|---|---|---|---|
| **Density** | `flip_rate` | fraction of tokens that flip | `(1/T) Σ flip_t` |
| **Timing** | `PDP_k` | first persistent divergence | `min {t : flip_t = ... = flip_{t+k-1} = 1}` |

We operationalize PDP with `k=2` throughout this exploratory analysis. This choice was made *post-hoc* as the minimal persistence criterion. Pre-registered robustness sweep across `k ∈ {2, 3, 5}` on held-out data is listed in `contract_validation.md`.

---

## 3. Observations (post-hoc, exploratory — not pre-registered claims)

All thresholds, pooling strategies, and metric definitions were chosen *after* inspecting the data. What follows is descriptive evidence, not hypothesis tests in the strict statistical sense.

### O1 — Density scales with perturbation intensity

**Observation**: on pooled data, `flip_rate` correlates positively with compression severity.

- **Measure**: Spearman ρ(1 − retention, flip_rate) on pooled conversations.
- **Value observed**: **ρ = 0.681** on mistral-7b synthetic, N=180.
- **Caveat**: pooled across two compression methods; not stratified.

### O2 — Timing distribution differs between two compression methods

**Observation**: on pooled retentions, PDP distributions for `uniform` vs `recent` compression differ.

- **Measure**: two-sample Kolmogorov–Smirnov test on PDP distributions (pooled across r∈{0.2, 0.5, 0.8}).
- **Value observed**: **D = 0.408, asymptotic p ≈ 10⁻⁶** (N_u = 90, N_r = 88).
- **Summary**: `uniform` mean PDP = 2.26, median = 0; `recent` mean PDP = 10.45, median = 5.
- **Caveats**:
  1. Pooled for statistical power — stratified analysis by retention left for pre-registered validation.
  2. Separation is dominated by `recent_r0.2` (mean PDP = 18.17). Without this condition, separation would weaken.

### O3 — Linear correlation between timing and density is weak

**Observation**: `flip_rate` and `PDP` show low linear correlation on this dataset.

- **Measure**: Pearson and Spearman correlation between `flip_rate` and `PDP` on pooled data.
- **Value observed**: Pearson **r = −0.083**, Spearman **ρ = 0.085** (N=178).
- **Within-method**: uniform r=−0.25, recent r=−0.18.
- **Interpretation (honest)**: this suggests that flip_rate and PDP *may* capture distinct aspects of divergence. **Weak linear correlation does not demonstrate orthogonality or independence** — it is compatible with non-linear dependence, shared latent cause, or absence of signal. A proper test of complementarity (e.g., classification-gain from adding PDP to flip_rate) is listed in `contract_validation.md`.

---

## 4. Empirical signature — Figure 1 (mental)

*Mistral-7B-Instruct, synthetic conversations, T=128, 30 convs × 6 conditions = 180 runs.*

| Method × Retention | PDP mean | PDP median | flip_rate mean | Regime |
|---|---:|---:|---:|---|
| uniform_r0.2 | 0.80 | 0 | 0.177 | instantaneous + dense |
| uniform_r0.5 | 1.27 | 0 | 0.128 | instantaneous + dense |
| uniform_r0.8 | 4.70 | 3 | 0.091 | instantaneous (attenuated) |
| recent_r0.2 | **18.17** | 5 | 0.183 | **persistent** + dense |
| recent_r0.5 | 8.00 | 6 | 0.202 | persistent + dense |
| recent_r0.8 | 4.82 | 2 | 0.089 | transitional |

**Reading**: rotating retention at fixed method moves density (flip_rate), but *not* timing (PDP) in a predictable way. Switching method at fixed retention moves *timing* dramatically while density stays comparable.

**Punchline**: *These regimes are method-dependent, not only perturbation-dependent.*

---

## 5. Positioning (exploratory contribution)

We do **not** propose a new geometric view of generation, nor do we claim definitive validation. We offer:

> *preliminary observations, from a post-hoc analysis of one model's evaluation data, that suggest context-perturbation divergence may be better described by two method-dependent dimensions — timing (PDP) and density (flip rate) — than by a single scalar. This reframing motivates a pre-registered cross-model validation, which we specify but do not conduct in this paper.*

Keyword: **operational candidate**. Flip rate and PDP are two measurable observables; whether they constitute the "right" pair of axes for divergence remains open.

**Known vulnerabilities, stated upfront (not defenses)**:
- Observations come from one model (mistral-7B) on one dataset family (synthetic conversations).
- The O2 separation is driven heavily by `recent_r0.2`; stripping that condition would weaken the effect.
- O3 is a *weak linear correlation*, not a demonstration of independence.
- Thresholds ("ρ > 0.5", "|r| < 0.3") were chosen after seeing the data; they function as description, not as pre-specified tests.

---

## 6. Scope

**IN (this paper = exploratory / position)**
- Theoretical reframing of AR generation as a constrained dynamical system with a postulated coherence manifold `M` (interpretive lens, not construction).
- Introduction of PDP as a timing observable complementary to flip_rate.
- Post-hoc empirical observations (O1, O2, O3) on LFCM V2 data, with explicit acknowledgement of their exploratory status.
- Specification (not execution) of a pre-registered validation protocol — see `contract_validation.md`.
- Open-source companion notebook reproducing O1/O2/O3 from `lfcm_mistral-7b_synthetic_v2_detailed.json`.

**OUT (to prevent dilution and overclaim)**
- Any claim of *proven* orthogonality or independence.
- Construction of `M`.
- TNT (Topological Navigation Theory) → future paper.
- Resonance Geometry → future paper.
- Active intervention / control (already in RBD/HHA papers).
- Cross-model validation (deferred to Phase 2 paper).

---

## 7. Limitations (front-loaded, stated explicitly in abstract and intro)

1. **Post-hoc, not pre-registered.** All metric definitions, thresholds, pooling choices, and comparisons were selected after observing the data. Nothing here functions as a strict hypothesis test.
2. **Single model, single dataset family.** Observations come from mistral-7B-Instruct on synthetic conversations — the only combination for which per-step flip logging (`steps[]`) is archived in LFCM V2 results.
3. **`k=2` chosen by minimal-persistence heuristic.** Robustness across `k ∈ {2,3,5}` not reported here; included in `contract_validation.md`.
4. **O2 separation is driven by `recent_r0.2`.** Removing this condition would substantially weaken the two-sample KS effect. Reported for transparency.
5. **O3 is weak linear correlation, not demonstrated independence.** A classification-gain test would be the appropriate falsifiable complementarity check; it is specified in Phase 2, not conducted here.
6. **`M` postulated, not constructed.** Interpretive framework, not a theorem.
7. **Perturbation = KV-cache eviction at prefill only.** Other perturbation families not tested.

---

## 8. Paper structure (6–8 pages)

1. **Introduction** — problem: stability under perturbation; limits of scalar metrics; contribution: two-regime decomposition.
2. **Framework** — AR generation as constrained dynamical system; manifold `M` (postulated); divergence as departure.
3. **Metrics** — flip_rate (density) + PDP (timing); definitions + `k` choice.
4. **Empirical Findings** — C1, C2, C3 on LFCM V2 data; regime table; orthogonality figure.
5. **Discussion** — method-dependence, implications for compression design, limitations.
6. **Conclusion** — divergence is not scalar; two orthogonal, operationally recoverable dimensions.

---

## 9. Timeline & priority

- **Priority**: *after* TCC (J-3 critical). This paper: 1–2 weeks after TCC starts.
- **No compute needed** for main claims — all analysis post-hoc on existing JSONs.
- Optional: 6h 3090 re-run with `steps[]` logging on qwen-7b/qwen-14b to add cross-model validation → upgrade from position paper to full paper. **Not in scope for v1.**

---

## 10. Next steps

1. **RD review (reject-proof)** of this exploratory seed.
2. `/literature-reviewer` — 10–15 papers (Sullivan 2024, Elhage *superposition*, JEPA, Tenney BERT probing, KV-cache: H2O, SnapKV, StreamingLLM).
3. **Write `contract_validation.md`** (Phase 2 protocol, pre-registered, see parallel file).
4. **Cross-LLM review** (Claude + ChatGPT + Gemini) on *both* the exploratory seed and the validation contract.
5. **Draft exploratory paper section-by-section** after TCC is underway.
6. **Phase 2 execution** (re-run + classification-gain test) after exploratory paper is submitted.

---

## Appendix — Raw numbers (ready to paste)

### C1 sanity
- N = 180 conversations × 6 (method × retention) conditions
- ρ(1−retention, flip_rate) = **0.681** (Spearman, all pooled)

### C2 KS test
- PDP uniform: n=90, mean=2.26, median=0, std=9.84
- PDP recent:  n=88, mean=10.45, median=5, std=13.51
- KS D = **0.408**, p ≈ **10⁻⁶**

### C3 orthogonality
- N=178 (2 *None* = no persistent divergence within T=128)
- Pearson r(flip_rate, PDP) = **−0.083**
- Spearman ρ(flip_rate, PDP) = **0.085**
- Within-method:
  - uniform: Pearson r = −0.252, Spearman ρ = −0.212, n=90
  - recent: Pearson r = −0.178, Spearman ρ = 0.034, n=88

### PDP pooled distribution (k=2)
- Mean 6.31, Median 2, Std 11.97, Range [0, 75]
- P10/P25/P50/P75/P90/P95 = 0/0/2/5/17/37

---

## The one-sentence differentiator (for abstract — exploratory version)

> *In a post-hoc analysis of one model's existing evaluation data, we observe a pattern consistent with the hypothesis that context-perturbation-induced divergence has at least two method-dependent dimensions — timing and density — and we specify a pre-registered cross-model protocol for falsifying or validating this hypothesis in future work.*
