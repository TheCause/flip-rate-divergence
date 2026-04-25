"""Phase 2 pre-registered analyses for FlipRateDivergence contract v0.4.

Contract interpretation (disambiguations frozen at script-writing time):

  H1 PASS: rho >= 0.5 AND p_bonf < 0.01  for every expected (method, model).
           Bonferroni correction across the full (methods x models) grid.
  H1 KILL: rho < 0.3 on any in-scope (method, model).

  H2 PASS (STRICT READING of "p < 0.01 after Bonferroni across retention"):
           for at least 2 of 3 in-scope models, ALL retentions must survive
           Bonferroni at p < 0.01. The contract text is ambiguous; we freeze
           the strict reading here (stronger than "at least one retention").
  H2 KILL: leave-one-retention-out re-application of the H2 PASS rule flips
           the per-model decision from PASS to FAIL. An effect that cannot
           survive removal of a single retention condition is fragile.

  H3 PASS: macro-F1(B) - macro-F1(A) >= 0.10 on >= 2 of 3 in-scope models.
  H3 KILL: delta < 0.05 on EVERY in-scope model (all three must be evaluated).

Completeness guard:
  Each global decision returns {"verdict": ..., "missing": [...]} when any
  expected model (or cell for H1) has no data. We never return PASS/KILL on
  an incomplete grid.

PDP operationalization (frozen):
  PDP_2 = min{t : flip_t = flip_{t+1} = 1} within first 128 tokens.
  Conversations with no such t are excluded from PDP-based analyses.

Outputs: JSON report + human-readable stdout trace.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score

IN_SCOPE_MODELS = ("mistral-7b", "qwen-7b", "qwen-14b")
IN_SCOPE_METHODS = ("uniform", "recent", "h2o_approx")
EXPECTED_H1_CELLS = {(m, meth) for m in IN_SCOPE_MODELS for meth in IN_SCOPE_METHODS}


def compute_pdp_2(steps, T_max=128):
    flips = [s["flip"] for s in steps[:T_max]]
    for t in range(len(flips) - 1):
        if flips[t] and flips[t + 1]:
            return t
    return None


def load_json(path):
    with open(path) as f:
        return json.load(f)


def summarise_file(path):
    d = load_json(path)
    model = d["model"]
    dataset = d["dataset"]
    rows = []
    for r in d["details"]:
        pdp = compute_pdp_2(r.get("steps", [])) if "steps" in r else None
        rows.append({
            "model": model,
            "dataset": dataset,
            "method": r["method"],
            "retention": r["retention"],
            "conv_id": r["conversation_id"],
            "flip_rate": r["flip_rate"],
            "pdp": pdp,
        })
    return rows


def bonferroni_clamped(p_values):
    """Bonferroni-adjusted p-values, clamped to [0, 1]. None preserved."""
    valid = [p for p in p_values if p is not None]
    m = len(valid)
    return [(min(p * m, 1.0) if p is not None else None) for p in p_values]


# -------------------------- H1 --------------------------

def h1_spearman_cell(rows, model, method):
    severity, phi = [], []
    for r in rows:
        if r["model"] == model and r["method"] == method:
            severity.append(1.0 - r["retention"])
            phi.append(r["flip_rate"])
    if len(severity) < 10:
        return None
    rho, p_raw = stats.spearmanr(severity, phi)
    return {"n": len(severity), "rho": float(rho), "p_raw": float(p_raw)}


def decide_h1_global(cells):
    observed = {(c["model"], c["method"]) for c in cells}
    missing = sorted(EXPECTED_H1_CELLS - observed)
    if missing:
        return {"verdict": "INCOMPLETE", "missing": [list(x) for x in missing]}
    if any(c["rho"] < 0.3 for c in cells):
        return {"verdict": "KILL"}
    all_pass = all(
        c["rho"] >= 0.5 and c["p_bonf"] is not None and c["p_bonf"] < 0.01
        for c in cells
    )
    return {"verdict": "PASS" if all_pass else "WEAK"}


# -------------------------- H2 --------------------------

def h2_ks_cell(rows, model, retention):
    u = [r["pdp"] for r in rows
         if r["model"] == model and r["method"] == "uniform"
         and r["retention"] == retention and r["pdp"] is not None]
    rec = [r["pdp"] for r in rows
           if r["model"] == model and r["method"] == "recent"
           and r["retention"] == retention and r["pdp"] is not None]
    if len(u) < 5 or len(rec) < 5:
        return {"n_u": len(u), "n_rec": len(rec), "D": None, "p_raw": None}
    D, p_raw = stats.ks_2samp(u, rec, alternative="two-sided", method="auto")
    return {"n_u": len(u), "n_rec": len(rec), "D": float(D), "p_raw": float(p_raw)}


def h2_check_rule_on_retentions(rows, model, retentions_subset):
    """Apply H2 PASS rule (strict) on a given subset of retentions for one model.
    Returns True if every retention in the subset has p_bonf < 0.01."""
    cells = []
    for ret in retentions_subset:
        cell = h2_ks_cell(rows, model, ret)
        if cell["p_raw"] is not None:
            cells.append(cell)
    if len(cells) < len(retentions_subset):
        # Missing test results for at least one retention in the subset.
        return None
    p_bonfs = bonferroni_clamped([c["p_raw"] for c in cells])
    return all(pb is not None and pb < 0.01 for pb in p_bonfs)


def h2_leave_one_out_rule(rows, model, retentions):
    """Drop each retention in turn and re-apply the H2 PASS rule on the rest.
    Returns {dropped_retention: passes_after_drop_or_None}.

    We mark None (uninterpretable) when fewer than 2 retentions remain, because
    H2 was designed as a cross-retention pattern and a single-retention decision
    no longer tests the same thing. Documented disambiguation.
    """
    out = {}
    for dropped in retentions:
        remaining = [r for r in retentions if r != dropped]
        if len(remaining) < 2:
            out[dropped] = None
            continue
        out[dropped] = h2_check_rule_on_retentions(rows, model, remaining)
    return out


def decide_h2_global(per_model_passes, per_model_loo_fragile):
    missing = sorted(
        m for m in IN_SCOPE_MODELS
        if m not in per_model_passes or m not in per_model_loo_fragile
    )
    if missing:
        return {"verdict": "INCOMPLETE", "missing": missing}
    if any(per_model_loo_fragile[m] for m in IN_SCOPE_MODELS):
        return {"verdict": "KILL"}
    n_pass = sum(1 for m in IN_SCOPE_MODELS if per_model_passes[m])
    return {"verdict": "PASS" if n_pass >= 2 else "WEAK"}


# -------------------------- H3 --------------------------

def h3_grouped_cv(rows, model, n_splits=5, n_repeats=30, seed=42):
    """Logistic regression predicting method from features, grouped by conv_id.
    Baseline A: [flip_rate]. Full B: [flip_rate, pdp/128].
    Repeated StratifiedGroupKFold; errors caught and counted.
    """
    X_A, X_B, y, groups = [], [], [], []
    for r in rows:
        if r["model"] != model or r["method"] not in IN_SCOPE_METHODS:
            continue
        if r["pdp"] is None:
            continue
        X_A.append([r["flip_rate"]])
        X_B.append([r["flip_rate"], r["pdp"] / 128.0])
        y.append(r["method"])
        groups.append(r["conv_id"])
    n = len(y)
    n_groups = len(set(groups))
    if n < 30 or n_groups < n_splits:
        return {"n": n, "n_groups": n_groups, "valid_repeats": 0,
                "errors": 0,
                "f1_A_mean": None, "f1_B_mean": None,
                "delta_mean": None, "delta_ci_95": None, "deltas": []}
    X_A = np.array(X_A)
    X_B = np.array(X_B)
    y_arr = np.array(y)
    grp = np.array(groups)

    all_deltas, all_f1A, all_f1B = [], [], []
    valid_repeats = 0
    errors = 0
    rng = np.random.default_rng(seed)
    for _ in range(n_repeats):
        try:
            skf = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            fold_delta, fold_A, fold_B = [], [], []
            for tr, te in skf.split(X_A, y_arr, groups=grp):
                clf_A = LogisticRegression(
                    max_iter=1000, solver="lbfgs", random_state=0,
                ).fit(X_A[tr], y_arr[tr])
                clf_B = LogisticRegression(
                    max_iter=1000, solver="lbfgs", random_state=0,
                ).fit(X_B[tr], y_arr[tr])
                a = f1_score(y_arr[te], clf_A.predict(X_A[te]), average="macro")
                b = f1_score(y_arr[te], clf_B.predict(X_B[te]), average="macro")
                fold_A.append(a)
                fold_B.append(b)
                fold_delta.append(b - a)
            all_f1A.append(float(np.mean(fold_A)))
            all_f1B.append(float(np.mean(fold_B)))
            all_deltas.append(float(np.mean(fold_delta)))
            valid_repeats += 1
        except ValueError:
            errors += 1
            continue

    if valid_repeats < 5:
        return {"n": n, "n_groups": n_groups,
                "valid_repeats": valid_repeats, "errors": errors,
                "f1_A_mean": None, "f1_B_mean": None,
                "delta_mean": None, "delta_ci_95": None, "deltas": all_deltas}

    ci_lo = float(np.percentile(all_deltas, 2.5))
    ci_hi = float(np.percentile(all_deltas, 97.5))
    return {
        "n": n, "n_groups": n_groups,
        "valid_repeats": valid_repeats, "errors": errors,
        "f1_A_mean": float(np.mean(all_f1A)),
        "f1_B_mean": float(np.mean(all_f1B)),
        "delta_mean": float(np.mean(all_deltas)),
        "delta_ci_95": (ci_lo, ci_hi),
        "deltas": all_deltas,
    }


def decide_h3_global(per_model_delta):
    missing = sorted(
        m for m in IN_SCOPE_MODELS
        if m not in per_model_delta or per_model_delta[m] is None
    )
    if missing:
        return {"verdict": "INCOMPLETE", "missing": missing}
    deltas = [per_model_delta[m] for m in IN_SCOPE_MODELS]
    if all(d < 0.05 for d in deltas):
        return {"verdict": "KILL"}
    n_pass = sum(1 for d in deltas if d >= 0.10)
    return {"verdict": "PASS" if n_pass >= 2 else "WEAK"}


# -------------------------- Main --------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="/data/rqz_workspace/results")
    parser.add_argument("--tag-prefix", default="phase2")
    parser.add_argument("--out", default="/data/rqz_workspace/flip_rate_divergence/phase2_analysis.json")
    args = parser.parse_args()

    files = sorted(Path(args.results_dir).glob(f"lfcm_*_{args.tag_prefix}*.json"))
    if not files:
        raise SystemExit(f"No result files matching {args.tag_prefix} in {args.results_dir}")
    print(f"Loaded {len(files)} result files:")
    for f in files:
        print(f"  {f.name}")

    all_rows = []
    for f in files:
        all_rows.extend(summarise_file(f))
    print(f"Total rows: {len(all_rows)}")

    models = sorted({r["model"] for r in all_rows if r["model"] in IN_SCOPE_MODELS})
    retentions = sorted({r["retention"] for r in all_rows})
    report = {"in_scope_models": list(IN_SCOPE_MODELS),
              "in_scope_methods": list(IN_SCOPE_METHODS),
              "observed_models": models, "retentions": retentions,
              "n_files": len(files), "n_rows": len(all_rows),
              "h2_interpretation": "strict (all retentions must pass Bonferroni)"}

    # ---- H1 ----
    print("\n=== H1: Spearman rho(1-retention, flip_rate) per (model, method) ===")
    h1_cells = []
    for model in models:
        for method in IN_SCOPE_METHODS:
            res = h1_spearman_cell(all_rows, model, method)
            if res is not None:
                h1_cells.append({"model": model, "method": method, **res})
    p_bonfs = bonferroni_clamped([c["p_raw"] for c in h1_cells])
    for c, pb in zip(h1_cells, p_bonfs):
        c["p_bonf"] = pb
    for c in h1_cells:
        print(f"  {c['model']:<12}{c['method']:<12}n={c['n']:3d}  "
              f"rho={c['rho']:+.3f}  p_raw={c['p_raw']:.4f}  p_bonf={c['p_bonf']:.4f}")
    h1_decision = decide_h1_global(h1_cells)
    print(f"  --> H1 GLOBAL: {h1_decision}")
    report["H1"] = {"cells": h1_cells, "decision": h1_decision}

    # ---- H2 ----
    print("\n=== H2: KS(uniform, recent) PDP per retention, per model ===")
    h2_cells = []
    per_model_passes = {}
    per_model_loo_fragile = {}
    for model in models:
        model_cells = []
        for ret in retentions:
            cell = h2_ks_cell(all_rows, model, ret)
            if cell["p_raw"] is None:
                continue
            model_cells.append({"model": model, "retention": ret, **cell})
        p_bonfs_model = bonferroni_clamped([c["p_raw"] for c in model_cells])
        for c, pb in zip(model_cells, p_bonfs_model):
            c["p_bonf"] = pb
        for c in model_cells:
            print(f"  {c['model']:<12}r={c['retention']:.1f}  n_u={c['n_u']:3d}  "
                  f"n_rec={c['n_rec']:3d}  D={c['D']:.3f}  "
                  f"p_raw={c['p_raw']:.4f}  p_bonf={c['p_bonf']:.4f}")

        passes = (
            len(model_cells) == len(retentions)
            and all(c["p_bonf"] is not None and c["p_bonf"] < 0.01 for c in model_cells)
        )
        per_model_passes[model] = passes

        loo = h2_leave_one_out_rule(all_rows, model, retentions)
        fragile = passes and any(p is False for p in loo.values())
        per_model_loo_fragile[model] = fragile

        print(f"  {model}: passes_all_retentions={passes}  loo_fragile={fragile}")
        for ret, p in loo.items():
            print(f"    drop r={ret:.1f} -> passes_after_drop={p}")
        h2_cells.extend(model_cells)

    h2_decision = decide_h2_global(per_model_passes, per_model_loo_fragile)
    print(f"  --> H2 GLOBAL: {h2_decision}")
    report["H2"] = {"cells": h2_cells,
                    "per_model_passes": per_model_passes,
                    "per_model_loo_fragile": per_model_loo_fragile,
                    "decision": h2_decision}

    # ---- H3 ----
    print("\n=== H3: macro-F1 gain adding PDP (grouped CV by conv_id, 30 repeats) ===")
    h3_cells = []
    per_model_delta = {}
    for model in models:
        res = h3_grouped_cv(all_rows, model, n_splits=5, n_repeats=30, seed=42)
        if res["f1_A_mean"] is None:
            print(f"  {model}: insufficient data "
                  f"(n={res['n']}, groups={res['n_groups']}, "
                  f"valid_repeats={res['valid_repeats']}, errors={res['errors']})")
            per_model_delta[model] = None
            continue
        per_model_delta[model] = res["delta_mean"]
        print(f"  {model:<12}n={res['n']:4d} groups={res['n_groups']:3d}  "
              f"F1_A={res['f1_A_mean']:.3f}  F1_B={res['f1_B_mean']:.3f}  "
              f"delta={res['delta_mean']:+.3f}  CI95={res['delta_ci_95']}  "
              f"(valid={res['valid_repeats']}/30, errors={res['errors']})")
        h3_cells.append({"model": model, **res})
    h3_decision = decide_h3_global(per_model_delta)
    print(f"  --> H3 GLOBAL: {h3_decision}")
    report["H3"] = {"cells": h3_cells, "per_model_delta": per_model_delta,
                    "decision": h3_decision}

    # ---- Outcome gating (contract v0.4 step 10) ----
    print("\n=== OUTCOME GATING (contract v0.4) ===")
    v1 = h1_decision["verdict"]
    v2 = h2_decision["verdict"]
    v3 = h3_decision["verdict"]
    if "INCOMPLETE" in (v1, v2, v3):
        gate = f"INCOMPLETE (H1={v1}, H2={v2}, H3={v3}) -> finish data collection first"
    elif (v1, v2, v3) == ("PASS", "PASS", "PASS"):
        gate = "STRONG VALIDATION -> proceed to full paper"
    elif v1 == "PASS" and v2 == "PASS" and v3 == "KILL":
        gate = "PARTIAL VALIDATION -> method-dependent timing, no complementarity claim"
    elif v1 == "PASS":
        gate = "H1-only -> downgrade to 'flip_rate useful; timing story does not generalize'"
    elif v1 == "KILL":
        gate = "PROJECT KILLED (H1 KILL)"
    else:
        gate = f"INTERMEDIATE (H1={v1}, H2={v2}, H3={v3}) -> discussion needed"
    print(f"  {gate}")
    report["outcome_gate"] = gate

    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport saved to {args.out}")


if __name__ == "__main__":
    main()
