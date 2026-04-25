# Pre-registered Falsification of Timing-Based Divergence Decomposition in KV-Cache Compression

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19757867.svg)](https://doi.org/10.5281/zenodo.19757867)

**Status**: Pre-registered Two Regimes hypothesis falsified (H1 KILL, H3 KILL, H2 WEAK).
**Framework**: RQZ. Related work: [LFCM V2](https://doi.org/10.5281/zenodo.19510356).
**Zenodo DOI**: [10.5281/zenodo.19757867](https://doi.org/10.5281/zenodo.19757867).
**Date**: 18 April 2026 (Phase 2 frozen) — 25 April 2026 (Zenodo deposit).

---

## Summary

This submission documents a **pre-registered falsification** on the *"Two Regimes"*
hypothesis, which proposed that KV-cache compression in LLMs induces a
divergence that decomposes into two (allegedly independent) axes: **density**
(flip rate Φ) and **timing** (Position of Double-Flip, PDP_2).

The hypothesis, born from a post-hoc observation in LFCM V2 where the metric
`ffp_mean` saturated in [1.0, 1.85], was formalized in `contract_validation.md`
**v0.4** (frozen at git tag `phase2_contract_v0.4_frozen` in the parent repo).

Under the pre-registered decision rule, Phase 2 data collected on 3 models
(mistral-7B-Instruct, qwen-7B-Instruct, qwen-14B-Instruct) × 2 datasets
(synthetic, ShareGPT) × 3 methods (uniform, recent, h2o_approx) × 4 retentions
({0.2, 0.5, 0.8, 0.9}) × N=50 conversations, T=128 tokens, yields:

| Hypothesis | Global verdict | Basis |
|---|---|---|
| **H1** (density ∝ severity monotonic) | **KILL** | 4/9 (method × model) cells have Spearman ρ < 0.3 |
| **H2** (timing discriminates uniform vs recent) | **WEAK** | Only mistral passes the strict all-retentions rule |
| **H3** (timing adds predictive value beyond density) | **KILL** | All per-model Δ macro-F1 < 0.05; **on qwen-7B Δ is negative (PDP actively degrades)** |
| **Outcome gate** | PROJECT KILLED (H1 KILL) | Contract v0.4 step 10 |

### Positive take-away (what was actually shown)

The bi-axial decomposition `divergence = (density, timing)` does **not** survive
falsification. What survives is mono-dimensional:

- **Φ (density) holds** robustly under `uniform` compression on all three models
  (ρ ∈ [0.35, 0.65]) and partially holds under `h2o_approx`. It is the only
  axis with empirical traction.
- **PDP (timing) does not hold** as an independent or complementary axis. It
  is not just "uninformative beyond density" — under leakage-free grouped CV
  it is *unstable* (only mistral passes H2 strictly) and *occasionally
  harmful* (qwen-7B: Δ macro-F1 = −0.018, CI95 strictly negative).

Interpreted strictly within the pre-registered scope, this archive is a
**falsification of the bi-axial claim**, not just an absence of result. A
follow-up direction sketched at the end (section *Hypothesis V2*) reframes
the residual signal toward *context structure* rather than *timing*.

**No paper is built on the original Two Regimes claim.** This archive exists
as a faithful record of the pre-registration, the data, the analysis, and
the verdict — to avoid the file-drawer effect and save other researchers
from retracing the same path.

---

## Key numbers

### H1: Spearman ρ(1 − retention, flip_rate), n = 400 per cell

| Model | `uniform` | `recent` | `h2o_approx` |
|---|---|---|---|
| mistral-7B | **+0.578** | +0.194 ⚠ | +0.304 |
| qwen-14B   | **+0.652** | +0.197 ⚠ | +0.385  |
| qwen-7B    | +0.349     | +0.073 ⚠ (p_bonf = 1.00) | +0.197 ⚠ |

Cells marked ⚠ satisfy the pre-registered KILL criterion (ρ < 0.3 on any
in-scope cell). The monotonic density–severity relation holds robustly for
`uniform`, weakly for `h2o_approx`, **not at all for `recent`**.

### H2: KS(uniform, recent) on PDP_2 per retention

| Model | r=0.2 | r=0.5 | r=0.8 | r=0.9 | All pass? |
|---|---|---|---|---|---|
| mistral-7B | p_bonf = 4e-5 | 0 | 1e-3 | 2e-3 | **yes** |
| qwen-14B | 1e-5 | 0 | 0.019 | 0.021 | no (0.8, 0.9) |
| qwen-7B | 1.000 | 0.011 | 0 | 0 | no (0.2) |

Only 1 model of 3 passes the strict all-retentions rule (contract requires
≥ 2 of 3). Leave-one-retention-out does **not** flip any per-model verdict
back to PASS, so H2 is weak but not fragile.

### H3: macro-F1 gain from adding PDP (grouped CV by conv_id, 30 repeats)

| Model | F1(flip_rate) | F1(flip_rate + PDP) | Δ | CI95 |
|---|---|---|---|---|
| mistral-7B | 0.388 | 0.413 | **+0.026** | [0.019, 0.032] |
| qwen-14B   | 0.412 | 0.434 | **+0.022** | [0.015, 0.031] |
| qwen-7B    | 0.389 | 0.371 | **−0.018** | [−0.028, −0.011] |

All three Δ are below the PASS threshold (0.10) and below the KILL threshold
(0.05 on every model). The reading is stronger than "no incremental gain":

- On **mistral-7B and qwen-14B**, the residual gain (~+0.022 to +0.026) is
  positive but well below the pre-registered threshold and within the noise
  band one would expect from any moderately correlated extra feature.
- On **qwen-7B**, Δ is **strictly negative** (CI95 = [−0.028, −0.011]).
  Adding PDP to flip_rate does not just leave classification flat — it
  actively degrades it. PDP is therefore not merely redundant; in this
  setting it behaves as a *noisy or anti-informative* feature.

Conclusion at the H3 level: under leakage-free grouped CV across 3 models,
PDP is **unstable and occasionally harmful**, not a complementary axis to Φ.

---

## Unexpected finding: `recent` is non-monotonic in retention (not pre-registered, reported as-is)

This is logged outside the pre-registered scope and is **not** a claim of
this archive. It is plausibly the most actionable observation in the data
set and is reported in full so future work can build on it cleanly.

**Phenomenon (cross-model, ShareGPT only).** The `recent` compression method
(StreamingLLM-style: keep first-k sinks + last-k tail) exhibits a
**non-monotonic "middle-gap"**: Φ rises sharply from r = 0.2 to r = 0.5
(e.g., qwen-14B sgpt_0000: Φ = 0.211 → 0.828) and only partially recovers
at higher retentions. The pattern is absent on short synthetic contexts
(~54 tokens) and pronounced on real ShareGPT conversations (≥ 512 tokens
after truncation). The peak around r = 0.5 is the worst point on the curve,
not the mid-point of a monotonic ramp.

This challenges the practical assumption that increasing retention should
monotonically reduce divergence under a fixed eviction rule.

**Interpretation (speculative, not tested here).** A natural reading is that
information in a conversation is **non-uniform along position and not
purely a function of recency**:

- **Sinks (very early tokens)** carry framing/instructions and are kept by
  design — their loss is rare.
- **Mid-conversation tokens** often carry the operative content and
  constraints (the user's actual problem, intermediate clarifications, the
  reference data). They are the first to be evicted as `last-k` advances.
- **Recent tokens** carry local context for the next token but are not, on
  their own, sufficient.

Under `recent`, raising retention from r = 0.2 to r = 0.5 expands the kept
*last* window *over the informative middle* while leaving the first-k sinks
unchanged; coverage of the critical middle therefore *worsens before it
improves*. At higher retentions, the kept window finally catches up and
includes the middle again, hence the partial recovery.

If true, the relevant axis governing divergence is not *timing* (PDP) but
**context structure** — the joint effect of position and role within the
conversation. This is the seed of *Hypothesis V2* below.

---

## Limitations (pre-registered run)

- **Loss of continuous divergence signal — KL_mean = NaN on every
  qwen-7B-sharegpt detail row.** Cause: underflow during softmax casting back
  to fp32 under the combination (4-bit model weights × eager attention × long
  context × qwen's broader vocabulary). H1, H2, H3 are formally computed from
  `flip_rate` and `PDP_2`, both integer-valued and unaffected by the NaN.
  However KL is **not** a secondary cosmetic metric: in the LFCM V2 paper
  KL is the early-warning continuous signal that captures *near-miss*
  divergence (next-token argmax preserved but distribution drifting). With
  KL unavailable on qwen-7B-sharegpt, the analysis on that cell is
  effectively binary (flip / no-flip), losing access to gradient and
  near-miss information that could inform H3 in particular. Conclusions
  on qwen-7B should therefore be read as conservative: PDP could only
  contribute on the binary view, where it does not.
- **Context truncation is asymmetric across models.** mistral-7B on ShareGPT
  uses `max_context_tokens = 1024` (4-bit + eager); qwen-14B on ShareGPT
  uses `max_context_tokens = 512` (to fit 4-bit + eager + 48 layers × 40
  heads in 24 GB VRAM); qwen-7B on ShareGPT uses 1024. Truncation is
  left-to-right on the first `N` tokens of the tokenized conversation.
  **Consequence: the "middle" of a ShareGPT conversation is not at the same
  absolute or relative position across models.** Cross-model H1/H2/H3
  comparisons on ShareGPT are therefore **not strictly distribution-matched**;
  the per-model verdicts hold within each model but apparent inter-model
  ordering should not be over-interpreted. This is a methodology constraint
  imposed by 3090 VRAM (24 GB) and is documented up-front in the contract,
  not a finding artefact discovered after the fact.
- **N = 50 conversations per cell**, satisfying the contract minimum
  (≥ 30). A follow-up run at N ≥ 100 would tighten the leave-one-out
  robustness checks.
- **Hardware/configuration dependency.** All Phase 2 runs were single
  RTX 3090 (24 GB), 4-bit weight quantization (bitsandbytes nf4),
  eager attention, deterministic seeds. SDPA / FlashAttention2 paths and
  full-precision weights were not exercised in this archive. Findings on
  `recent` (especially the middle-gap) should be re-checked under non-eager
  attention before being treated as model-intrinsic rather than
  configuration-dependent.

---

## Reproducibility

- **Contract** (frozen pre-data): `contract/contract_validation.md` (v0.4).
  Git SHA of freeze: tag `phase2_contract_v0.4_frozen` on branch `main`.
- **Phase 0bis pre-check** artefacts: `contract/phase_0bis_report.md`,
  `contract/phase_0bis_indices.json`, `contract/seed_exploratory.md`.
- **Benchmark code**: `code/benchmark.py`, `metrics.py`.
  Compression methods (uniform, recent, h2o_approx) defined in
  `benchmark.py:199-266`. Runtime requirements: CUDA ≥ 12.8, torch ≥ 2.10,
  transformers ≥ 4.57, scipy, scikit-learn. Full pin in `requirements.txt`.
- **Analysis**: `code/analyze_phase2.py` computes H1/H2/H3
  global verdicts and the outcome gate exactly as specified in the contract.
- **Raw results**: `results/lfcm_*_phase2_*.json` — six per-run JSONs with
  per-step token IDs and flip flags. Total ~87 MB.
- **Environment dumps**: `results/env_{mistral-7b,qwen-7b,qwen-14b}.json`
  (torch / transformers versions, 4-bit flag, seeds).
- **Final analysis**: `results/phase2_final_analysis.json` +
  `phase2_final_analysis.log`.

To replicate:

```bash
pip install -r code/requirements.txt
# Per-model runs (example, mistral-7b synthetic):
python code/benchmark.py \
  --model mistral-7b \
  --methods uniform,recent,h2o_approx \
  --retention 0.2,0.5,0.8,0.9 \
  --conversations 50 --T 128 --dataset synthetic \
  --dump-env --include-steps \
  --tag phase2_mistral_synth
# Then aggregate:
python code/analyze_phase2.py \
  --tag-prefix phase2 \
  --out phase2_final_analysis.json
```

---

## Conclusion

Two Regimes, as pre-registered, does not generalize across the intended
model × method × dataset grid. `uniform` is the only compression method
whose Φ scales monotonically with severity across all three models; `recent`
fails this basic property on all three. PDP as an auxiliary feature is, at
best, redundant with flip_rate (mistral, qwen-14B) and, at worst, mildly
harmful (qwen-7B) under leakage-free grouped CV.

The pre-registration was honored. The project is closed on these claims.
A potentially interesting orthogonal observation (`recent` middle-gap) is
logged in the *Unexpected finding* section as a seed for future work and is
**not** claimed here.

---

## Hypothesis V2 (sketch — not tested in this archive)

The falsified V1 framing was:

> **V1 (FAILED)**: divergence = density (Φ) + timing (PDP), independent and complementary axes.

A more honest framing of what the data actually shows is:

> **V2 (proposed, not yet pre-registered)**: divergence = density (Φ) + **context structure**, where context structure refers to the joint effect of token *position* and *role* within the conversation, not absolute timing.

Three minimal candidate features for context structure that future work
could pre-register and test:

1. **Relative position bucket** of the evicted span (early / mid / late
   thirds of the kept context, normalized by conversation length).
2. **Role tag** of the evicted span (system / instruction prefix vs.
   user content vs. assistant tail), exploitable from chat-template offsets.
3. **Local information density** in the evicted span (e.g., per-token
   entropy of the model's *own* next-token distribution at insertion time,
   or simpler: token-frequency-inverse on the conversation tokens).

The motivation is strictly empirical and bounded: under `recent` on real
ShareGPT conversations, retention r = 0.5 is consistently the worst
configuration, which only makes sense if information is non-uniform along
position. Whether (1)–(3) recover the lost predictive power is an open
question; this archive does not answer it.

---

## Citation

If this record helps your work (e.g., to avoid re-running the same
experiment, or to cite a pre-registered null in KV-cache compression
literature), please cite the Zenodo DOI assigned to this deposit.

Related: *Logit-Faithful Context Memory* (LFCM V2), Zenodo DOI
[10.5281/zenodo.19510356](https://doi.org/10.5281/zenodo.19510356),
which introduced the measurement infrastructure reused here.
