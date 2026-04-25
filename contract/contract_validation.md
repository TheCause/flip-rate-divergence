# Contract — Validation (Phase 2, pre-registered)

**Project**: Flip Rate as Trajectory Divergence — *Two Regimes* hypothesis
**Phase**: 2 (validation — written *before* new data is collected), preceded by Phase 0bis (structural sanity-check).
**Status**: DRAFT v0.4 (2026-04-14), ready for freeze. v0.3 hardened with 9 improvements from first /iterate round; v0.4 adds 3 further items from second /iterate round (deterministic sample selection, tightened N_flips threshold with synthetic bias calibration scan, tied-embedding clarification). To be frozen via git commit before any Phase 0bis compute.

---

## Purpose

This contract specifies the protocol for testing the *Two Regimes* hypothesis, which arose from an exploratory post-hoc analysis of mistral-7B-Instruct LFCM V2 data (see `seed_exploratory.md`). All decisions below are made **before** observing new data. Any deviation must be documented explicitly as a protocol amendment with date and justification.

A Phase 0bis is inserted as a cheap prerequisite: a competing hypothesis — that the compression-induced divergence is a *rigid shift* rather than a two-regime phenomenon — would, if true, invalidate the PDP/density framing. Phase 0bis tests this before Phase 2 commits compute to the (PDP, density) axes.

---

## Phase 0bis — Rigid-shift pre-check (gate for Phase 2)

### Motivation

Exploratory observation from the user: the compression-induced divergence *may* take the form `trajectory_compressed − trajectory_baseline ≈ constant`, i.e. a rigid translation in representation space rather than chaotic drift. If confirmed on LFCM V2 data, this would imply that `flip_rate` saturation observed in V2 is the *amplitude* of a rigid shift, not a genuine divergence measure — and that the (PDP, density) decomposition is incomplete without a third *rigidity* axis.

This pre-check runs on a small mistral-7B-Instruct re-run (~45 min total compute including the empirical null baseline) and gates whether Phase 2 analyzes the collected data on two axes or three.

### H0a and H0b — Pre-registered null and alternative

- **H0a (null, default expectation)**: compression-induced residuals are *not* dominated by a rigid shift. Variance of the residual across positions is non-negligible relative to baseline representation variance, and statistically indistinguishable from an empirical null (baseline-vs-baseline with different sampling seeds).
- **H0b (alternative)**: compression-induced residuals are dominated by a constant-shift component, statistically distinguishable from the empirical null. The (PDP, density) decomposition is structurally incomplete.

### Primary metric: residual variance on hidden states, conditioned on flips

For each conversation, each method, each retention level, and each position `t ∈ [1, T_eff]` (see EOS handling below):

- Capture hidden states pre-LM-head: `h_baseline(t), h_compressed(t) ∈ ℝ^{d_model}` (d_model = 4096 for Mistral-7B).
- Residual: `r_h(t) = h_compressed(t) − h_baseline(t) ∈ ℝ^{d_model}`.
- **Primary metric**: `σ²_h^flip = mean_{t: flip=True} Var_d(r_h(t)) / mean_{t: flip=True} Var_d(h_baseline(t))`.
- **Conditioning on flips**: positions where `argmax(logits_c(t)) == argmax(logits_0(t))` are excluded from the sum. At those positions both trajectories happen to coincide at the discrete output, and including them mechanically pulls σ² toward 0 regardless of the geometric structure of the divergence.
- **Exclusion threshold**: conversations with fewer than **20 flipped positions** (after EOS truncation) AND `flip_rate < 0.15` are excluded from the primary analysis and counted as NA. NA rate is reported per condition. If NA rate exceeds 50% on any in-scope condition, Phase 0bis is extended to N=50 via documented amendment.

### Secondary metrics

- `σ²_emb^flip` (same definition as primary, substituting embeddings `E[argmax(logits(t))]` for hidden states). Reported for comparison. Discrepancy between σ²_h and σ²_emb is itself a finding to report.
- `σ²_logits^flip` (same definition, substituting top-32 softmax logits). Reported for ablation.
- `flip_rate` per conversation. Reported alongside σ²_h so "few flips but rigid" is distinguishable from "many flips but rigid".

### Empirical null baseline

To calibrate σ²_h^flip against a no-compression null:

- Re-run the exact same prompts on mistral-7B-Instruct with no compression, `temperature=0.7, top_p=1.0, do_sample=True`, with a **different sampling seed** (seed=43 vs baseline seed=42).
- Apply the same metric computation: `σ²_h^flip(null) = mean_{t: flip_null=True} Var_d(h_baseline_seed43(t) − h_baseline_seed42(t)) / mean_{t: flip_null=True} Var_d(h_baseline_seed42(t))`.
- This quantifies the natural divergence induced by stochastic sampling alone, in the absence of KV-cache compression.
- Compute: `ratio = σ²_h^flip(compression) / σ²_h^flip(null)` per (method, retention).

### Decision rule (pre-registered, combines thresholds + formal test)

For each (method, retention) condition:

- **Point estimate**: median of `σ²_h^flip` and of `ratio` across conversations, with **bootstrap 95% CI** (1000 resamples over conversations).
- **Formal test**: one-sided **permutation test** on `σ²_h^flip(compression)` vs `σ²_h^flip(null)` at the conversation level (H0: distributions identical; Halt: compression < null). 10000 permutations, p-value corrected across conditions via **Benjamini-Hochberg FDR** (q=0.05), consistent with RQZ statistical conventions.
- **Condition-level decision**:
  - **Rigid-dominant**: CI_upper(ratio) ≤ 0.3 **AND** p_BH < 0.05.
  - **Rigid-rejected**: CI_lower(ratio) ≥ 0.7 **OR** p_BH > 0.10.
  - **Inconclusive**: intermediate.
- **Aggregate H0b / H0a gate**: "rigid-dominant" must hold on ≥ 2 of 3 in-scope methods, at ≥ 1 retention level each, to declare H0b SUPPORTED. Otherwise H0a NOT REJECTED (if no condition is inconclusive) or INCONCLUSIVE (if any in-scope condition is inconclusive).

The 0.3 / 0.7 ratio cutoffs are derived from the synthetic calibration step below. They are frozen at this value in v0.3 and are NOT to be tuned after observing compression data.

### Synthetic calibration (validates the metric itself before trusting its verdict)

Before interpreting compression results, validate σ²_h^flip on two synthetic controls using baseline logits only:

- **Positive control (rigid-by-construction)**: add a fixed bias vector `b ∼ N(0, σ_bias² · I_{d_model})` to all baseline hidden states, recompute argmax via the original LM head + b projection. The induced divergence IS a rigid shift by construction. Expected `σ²_h^flip ≤ 0.10`.
  - **Bias calibration scan (pre-registered)**: before the formal positive control, run a mini-scan on 5 baseline conversations with `σ_bias ∈ {0.1, 0.3, 0.5, 1.0}`. Select the `σ_bias` that produces `flip_rate ∈ [0.2, 0.4]` on the scan (natural regime, avoids both saturation at high σ and near-zero flips at low σ). The selected value is frozen and recorded in `phase_0bis_report.md` before the full positive control runs.
  - If the positive control fails its target (σ²_h^flip > 0.10) after calibration, the metric is broken and Phase 0bis is suspended pending metric revision.
- **Negative control (chaotic-by-construction)**: for positions where an artificial flip is injected, replace `argmax(logits_c(t))` by a uniformly random token. Expected `σ²_h^flip ≥ 0.7`. If not, the metric does not discriminate and Phase 0bis is suspended.
- Both controls are run on the same N=20 conversations before the compression analysis. Results are reported in `phase_0bis_report.md`.

### EOS and sequence-length handling

- **Truncation rule**: for each (baseline, compressed) pair, analysis stops at `T_eff = min(EOS_baseline, EOS_compressed, T=128)`. Positions beyond T_eff are excluded for both trajectories.
- **Minimum usable length**: conversations with `T_eff < 20` are excluded. Per-condition counts of excluded conversations are reported.
- **EOS flips excluded**: positions where the flip is the EOS token itself are excluded from σ²_h^flip computation (EOS is structurally distinct from semantic tokens).

### Environment specification (frozen)

All mini-benchmark runs (compression, baseline null, synthetic controls) MUST use:

- **Model**: `mistralai/Mistral-7B-Instruct-v0.3`. Revision SHA pinned at run time and recorded in `env.json`.
- **Embedding matrix `E`**: Mistral-7B-Instruct-v0.3 uses **tied embeddings** (input embedding = LM head weight). `σ²_emb^flip` uses this single matrix. If the protocol is later extended to qwen models (which are not tied), the choice of input vs output embedding must be made explicit in an amendment.
- **Tokenizer**: same model's tokenizer. SHA recorded.
- **Dtype**: `bfloat16` (per RQZ policy — never fp16 on RTX 3090).
- **Decoding**: `temperature=0.7, top_p=1.0, do_sample=True`. Non-deterministic sampling is required for the empirical null to be non-trivial.
- **Seeds**:
  - Baseline (reference trajectories for compression comparison): seed 42.
  - Compressed runs: each method × retention uses seed 42 (so divergence is attributable to compression, not sampling).
  - Empirical null: seed 43 with no compression.
- **Generation length T**: 128 tokens max.
- **Hardware**: RTX 3090, single GPU. Route per RQZ policy (no cloud for a 45-min run).
- **Library versions**: `torch`, `transformers`, `cuda` — dumped automatically to `env.json` at run start.
- **Reproducibility artifact**: one-command launcher `scripts/run_phase_0bis.sh` committed with this contract.

### Pre-registered conversation sample (frozen)

- The N=20 conversations analyzed in Phase 0bis are selected from the synthetic LFCM V2 dataset via:
  `numpy.random.default_rng(seed=42).choice(N_total, size=20, replace=False)`
  where `N_total` is the count of conversations in the synthetic dataset at the dataset SHA pinned in `env.json`.
- The 20 resolved indices are written to `phase_0bis_indices.json` in this directory at freeze time and committed with this contract. Any replication must use this exact list.
- Rationale: "N=20 synthetic conversations" is ambiguous. Pre-registering the indices makes Phase 0bis bit-for-bit reproducible.

### Protocol (execution order)

1. **Code modifications** (benchmark.py, metrics.py):
   - Add `--save-hidden` CLI flag (captures pre-LM-head hidden states, default off).
   - Add `--save-logits-topk` CLI flag (captures top-32 logits + top-32 indices, default off).
   - Extend `StepResult` dataclass to optionally carry hidden states and top-k logits.
   - Extend `compute_flip()` to populate these when flags are active.
   - Extend `to_dict(include_steps=True)` to serialize them.
   - Dump `env.json` at run start (library versions, model SHA, seeds).
2. **Synthetic calibration** (first, no compression needed): run the positive and negative controls on a small N=20 sample of baseline outputs. Verify σ²_h^flip hits the expected ranges. If not, STOP and revise.
3. **Empirical null run** on RTX 3090: mistral-7B-Instruct, no compression, seed 43, N=20 conversations. Compute σ²_h^flip(null) distribution. Runtime: ~10 min.
4. **Compression mini-benchmark** on RTX 3090: mistral-7B-Instruct, N=20, T=128, methods ∈ {`uniform`, `recent`, `h2o_approx` if available}, retention ∈ {`0.5`, `0.8`}. Runtime: ~20 min.
5. **Offline analysis script** (`analyze_rigidity.py`, ~80-120 lines — grown from the initial ~30 due to bootstrap, permutation, and conditioning logic):
   - Read mini-benchmark JSON.
   - Apply EOS truncation and flip-conditioning filters.
   - Compute `σ²_h^flip`, `σ²_emb^flip`, `σ²_logits^flip`, `flip_rate` per conversation.
   - Bootstrap 1000 resamples for CI per condition.
   - Permutation test (10000 permutations) vs null, BH-FDR corrected.
   - Output `phase_0bis_report.md` with: synthetic controls results, per-condition table (medians, CI, p_BH, decision), and aggregate H0b/H0a verdict.
6. **Decision**: apply the pre-registered rule. Freeze `phase_0bis_report.md` with a git commit.

### Gating for Phase 2 (reformulated: enrichment, not blocker)

- **H0b SUPPORTED** → Phase 2 proceeds *with rigidity added as a third axis* in the analysis framework. `seed_exploratory.md` is amended to "Three Regimes" (PDP, density, rigidity). Phase 2 collects the same data as planned, but analyzes it on three axes instead of two. No compute is lost.
  - **Sub-gate**: if a method shows `ratio ≤ 0.05` (rigidity dominates totally), Two Regimes is declared KILLED for that method specifically. The paper reports the asymmetry.
- **H0a NOT REJECTED** → Phase 2 proceeds exactly as specified below. The rigid-shift hypothesis is recorded as a tested-and-rejected alternative in supplementary material.
- **INCONCLUSIVE** → explicit protocol amendment required: either (a) expand N to 50 and re-run Phase 0bis, or (b) add theoretical justification for a threshold revision. No silent pivot. Decision documented in the Deviations log.

### Non-goals for Phase 0bis

- No tuning of ratio thresholds (0.3 / 0.7) after observing mini-benchmark data.
- No tuning of synthetic control targets (0.10 / 0.7) after observing their results.
- No extension to qwen models at this stage (mistral-7B-only for the pre-check).
- No downstream metric involvement (BLEU, QA). The rigidity question is purely geometric.
- No post-hoc metric search if all three (σ²_h, σ²_emb, σ²_logits) fail to discriminate — that would be a new exploratory phase requiring a new seed.

---

## H1, H2, H3 — Pre-registered hypotheses

### H1 — Density scales with perturbation severity (cross-model)

> For each compression method `m`, `flip_rate` has a positive monotonic association with `1 − retention`.

- **Metric**: Spearman ρ(1 − retention, flip_rate) **per method, per model**.
- **PASS criterion**: ρ ≥ 0.5 with p < 0.01 after Bonferroni correction across (methods × models).
- **Kill criterion**: ρ < 0.3 on any (method, model) pair that was pre-declared as in-scope.

### H2 — Timing discriminates methods at each retention level (stratified)

> PDP distributions differ between `uniform` and `recent` compression **at each retention level separately**, not only when pooled.

- **Metric**: two-sample KS test on PDP for (uniform_rX, recent_rX) for each X ∈ {0.2, 0.5, 0.8, 0.9}.
- **PASS criterion**: p < 0.01 after Bonferroni across retention levels, on at least 2 of 3 in-scope models (mistral-7B, qwen-7B, qwen-14B).
- **Kill criterion**: separation is driven by a single (method, retention) condition; removing that condition brings p > 0.05 in a leave-one-condition-out analysis.
- **Rationale**: the exploratory data showed the pooled effect may be carried by `recent_r0.2`. This test separates genuine two-regime structure from a single-condition outlier.

### H3 — Timing and density carry complementary information (classification gain)

> Adding PDP as a feature to a classifier predicting compression method improves predictive performance beyond what flip_rate alone achieves.

- **Setup**: multiclass classifier (logistic regression, 5-fold CV) predicting `method ∈ {uniform, recent, h2o_approx}` from features. Balanced classes per fold.
- **Baseline A**: features = (flip_rate).
- **Full B**: features = (flip_rate, PDP).
- **Metric**: macro-F1 on held-out folds, averaged across folds.
- **PASS criterion**: F1(B) − F1(A) ≥ 0.10 on at least 2 of 3 in-scope models.
- **Kill criterion**: gain < 0.05 on every model → PDP adds no predictive value beyond flip_rate.
- **Rationale**: "orthogonality" is not "|r| small". It is "PDP carries predictive information that flip_rate alone does not."

---

## Fixed definitions (no post-hoc tuning)

- **PDP operationalization**: `PDP = PDP_2 = min { t : flip_t = flip_{t+1} = 1 }`. If no such t exists within the first 128 tokens, `PDP = None` and the conversation is excluded from PDP-based analyses (reported as NA count).
- **Robustness check**: all analyses replicated with `k ∈ {3, 5}`. Results for `k=3` and `k=5` are reported as sensitivity analyses, not primary.
- **Perturbation axis**: `severity = 1 − retention`.
- **Pooling policy**: no pooling across retentions for H2 or H3 primary analyses. Pooled supplementary analyses allowed, explicitly labelled.
- **Datasets**: synthetic (in-scope) + sharegpt (in-scope if runtime allows).
- **Models in-scope**: mistral-7B-Instruct, qwen-7B-Instruct, qwen-14B-Instruct.
- **Methods in-scope**: uniform, recent. `h2o_approx` included if present in re-run.
- **Retention levels**: r ∈ {0.2, 0.5, 0.8, 0.9} where available.
- **T (generation length)**: 128 tokens.
- **N per condition**: ≥ 30 conversations (target 50).

---

## Required data (compute plan)

- **Existing**: `lfcm_mistral-7b_synthetic_v2_detailed.json` (180 conversations, has `steps[]`).
- **Needed** (Phase 2 re-run on RTX 3090):
  - qwen-7b synthetic with `steps[]` logging.
  - qwen-14b synthetic with `steps[]` logging.
  - mistral-7b synthetic replicate with `steps[]` (optional, for internal replication).
- **Runtime estimate**: 6–12 h on RTX 3090 depending on 4-bit quantization.
- **Constraint**: 3090 routing per RQZ policy. No M4 compute. Deploy via `ssh ia@192.168.1.163 + /data/venvs/rqz/bin/python3`.

---

## Analysis protocol (linear, no optional branches)

1. **Freeze this contract.** Commit to git with SHA. No changes after data arrives.
2. **Run re-runs** on 3090 with per-step flip logging. Verify integrity (n_conversations, T, method/retention grid).
3. **Compute `flip_rate` and `PDP_2`** for every conversation. Report `None` rates.
4. **H1 tests** across (model × method). Report Spearman ρ, 95% CI, Bonferroni-corrected p.
5. **H2 tests** stratified by retention. KS D, asymptotic p, Bonferroni across retention.
6. **H3 tests** via logistic regression with 5-fold CV. Report F1(A), F1(B), Δ, 95% CI via bootstrap (1000 resamples).
7. **Leave-one-condition-out robustness** for H2: for each condition, recompute KS without it; report if any single condition drives the effect.
8. **Sensitivity to k**: repeat H2 and H3 for k=3, k=5.
9. **Decision matrix**: declare PASS/KILL for H1, H2, H3 per model.
10. **Outcome gating**:
    - H1 + H2 + H3 all PASS on ≥ 2 of 3 models → *strong validation*, proceed to full paper.
    - H1 + H2 PASS, H3 KILL → *partial validation*, paper becomes "method-dependent timing of divergence, without complementarity claim".
    - H1 PASS only → the hypothesis is downgraded to "flip_rate is useful; timing story does not generalize".
    - H1 KILL → kill the project.

---

## Deviations log (to be filled during execution)

*None yet. Contract not yet frozen.*

---

## Non-goals (to prevent scope creep during execution)

- No tuning of `k` to maximize signal.
- No search over alternative metrics if H1–H3 fail (that would be a new exploratory phase, a new seed).
- No dataset expansion beyond synthetic + sharegpt in this contract.
- No theoretical proofs added during validation — theory stays at the level of the interpretive framework.

---

## Approvals

- **Regis (PI)**: pending.
- **Cross-LLM adversarial review (Claude + ChatGPT + Gemini)**: pending.
- **Freeze date**: TBD (must occur *before* Phase 2 data collection begins).
