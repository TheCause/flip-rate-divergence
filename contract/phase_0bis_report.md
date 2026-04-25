# Phase 0bis Report — Rigid-shift pre-check

**Source**: `unknown.json`
**Model**: mistral-7b
**Conversations**: 20
**T (max generation length)**: 128

## Verdict

**H0a_NOT_REJECTED**

## Synthetic controls (metric sanity)

- **Negative control** (random `hidden_comp` at flips): σ²_h = 1.253371238708496 — target ≥ 0.7 — **PASS**
- **Positive control** (constant bias σ=0.5): σ²_h = 8.038850889211663e-16 — target ≤ 0.10 — **PASS**

## Per-condition results

| Method | Retention | N used | σ²_h^flip (median) | CI 95% | σ²_h(null) | ratio | p_BH | Decision |
|--------|-----------|--------|--------------------|--------|------------|-------|------|----------|
| uniform | 0.5 | 3/20 | 0.3285 | [0.2492, 0.3900] | 0.3386 | 0.970 | 0.7567 | **rigid_rejected** |
| uniform | 0.8 | 0/20 | NA | [NA, NA] | NA | NA | NA | **no_data** |
| recent | 0.5 | 19/20 | 0.4257 | [0.3697, 0.4785] | 0.3287 | 1.295 | 1.0000 | **rigid_rejected** |
| recent | 0.8 | 1/20 | NA | [NA, NA] | NA | NA | NA | **no_data** |
| h2o_approx | 0.5 | 3/20 | 0.3285 | [0.2492, 0.3900] | 0.3386 | 0.970 | 0.7567 | **rigid_rejected** |
| h2o_approx | 0.8 | 0/20 | NA | [NA, NA] | NA | NA | NA | **no_data** |

## Decision rule (pre-registered, contract v0.4)

Per condition:
- **rigid_dominant** : CI_upper(σ²_h) ≤ 0.30 × σ²_null_median AND p_BH < 0.05
- **rigid_rejected** : CI_lower(σ²_h) ≥ 0.70 × σ²_null_median OR p_BH > 0.10
- **inconclusive** : otherwise

Aggregate gate: **H0b SUPPORTED** if rigid_dominant on ≥ 2 of 3 in-scope methods.

## Limitations of v1.0

- Empirical null is **synthetic** (gaussian noise on hidden states, σ_null_factor=0.5), not a real baseline-vs-baseline run with stochastic sampling. v1.1 will add the empirical null via a `--no-compression-mode` benchmark flag.
- Embedding-space metric (σ²_emb^flip) **deferred** to v1.1 (requires loading the model's embedding matrix post-hoc).
- Hidden states captured at the last layer pre-LM-head only.

## Gating for Phase 2

- **H0b SUPPORTED** → Phase 2 proceeds *with rigidity added as a third axis* (Three Regimes). `seed_exploratory.md` to be amended.
- **H0a NOT REJECTED** → Phase 2 proceeds as specified in contract v0.4.
- **INCONCLUSIVE** → explicit protocol amendment required.