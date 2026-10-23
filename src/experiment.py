"""Temporal firm-specific technology-entry prediction from patent records.

Input CSV columns: firm_id, year, cpc_code (one row per firm-patent-technology relation).
--demo generates synthetic records for a pipeline smoke test only; never report those
metrics as evidence about real firms. All features for prediction year t use records
dated <= t; target is first entry into a candidate CPC subclass in t+1.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, balanced_accuracy_score, brier_score_loss
from sklearn.model_selection import train_test_split

SEED = 41
BASE = ["firm_portfolio_size", "firm_patent_count", "candidate_prevalence", "candidate_recent_growth"]
GRAPH = ["common_neighbor_count", "peer_candidate_prevalence", "candidate_graph_degree"]
REQUIRED = {"firm_id", "year", "cpc_code"}


def make_demo(seed=SEED, n_firms=100, years=range(2010, 2024), n_tech=18):
    """Synthetic patent-like records for smoke testing only."""
    rng = np.random.default_rng(seed)
    techs = [f"{chr(65+i//10)}{i%10:02d}" for i in range(n_tech)]
    rows = []
    firm_base = [set(rng.choice(techs, size=int(rng.integers(2, 6)), replace=False)) for _ in range(n_firms)]
    for fi, base in enumerate(firm_base):
        portfolio = set(base)
        for year in years:
            # Slow portfolio evolution creates both repeat activity and new-class entry.
            if rng.random() < .28:
                candidates = [t for t in techs if t not in portfolio]
                if candidates:
                    portfolio.add(str(rng.choice(candidates)))
            for tech in portfolio:
                if rng.random() < .55:
                    rows.append((f"F{fi:04d}", year, tech))
    return pd.DataFrame(rows, columns=["firm_id", "year", "cpc_code"])


def load_records(path=None, demo=False):
    df = make_demo() if demo else pd.read_csv(path, dtype={"firm_id": str, "cpc_code": str})
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {sorted(missing)}")
    df = df[["firm_id", "year", "cpc_code"]].dropna().copy()
    df["firm_id"] = df["firm_id"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)
    # CPC subclass is the first four alphanumeric characters (e.g., A01B).
    df["cpc_code"] = df["cpc_code"].astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True).str[:4]
    df = df[(df["firm_id"] != "") & (df["cpc_code"].str.len() == 4)]
    df = df.drop_duplicates(["firm_id", "year", "cpc_code"]).sort_values(["year", "firm_id", "cpc_code"])
    if df.empty:
        raise ValueError("No valid firm-year-CPC records remain after normalization.")
    return df.reset_index(drop=True)


def make_candidate_panel(df, max_technologies=30, min_firms=3):
    """Create firm-candidate-year rows with a fixed candidate set from a burn-in window."""
    years = sorted(df.year.unique().tolist())
    if len(years) < 9:
        raise ValueError("At least 9 distinct years are required for burn-in, embargo, and future testing.")
    burn_years = years[:3]
    burn = df[df.year.isin(burn_years)]
    tech_counts = burn.groupby("cpc_code").firm_id.nunique().sort_values(ascending=False)
    tech_universe = tech_counts.head(max_technologies).index.tolist()
    if len(tech_universe) < 5:
        raise ValueError("Fewer than five CPC subclasses occur in the initial three-year burn-in period.")
    df = df[df.cpc_code.isin(tech_universe)].copy()
    eligible_prediction_years = years[3:-1]
    if len(eligible_prediction_years) < 5:
        raise ValueError("Not enough post-burn-in prediction years.")
    # Last two prediction years are test; one full prediction year is embargoed.
    train_years = eligible_prediction_years[:-3]
    embargo_year = eligible_prediction_years[-3]
    test_years = eligible_prediction_years[-2:]
    if not train_years:
        raise ValueError("Training period is empty after temporal split.")

    by_year_firm = {int(y): g.groupby("firm_id").cpc_code.apply(set).to_dict()
                    for y, g in df.groupby("year")}
    firms = sorted(df.firm_id.unique().tolist())
    rows = []
    for pred_year in eligible_prediction_years:
        history = df[df.year <= pred_year]
        hist_portfolios = history.groupby("firm_id").cpc_code.apply(set).to_dict()
        if not hist_portfolios:
            continue
        next_tech = by_year_firm.get(int(pred_year + 1), {})
        # Build technology co-occurrence graph using only the historical graph G_t.
        pair_counts = {}
        tech_neighbors = {c: set() for c in tech_universe}
        for portfolio in hist_portfolios.values():
            portfolio = sorted(portfolio)
            for i, a in enumerate(portfolio):
                for b in portfolio[i+1:]:
                    pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1
                    pair_counts[(b, a)] = pair_counts.get((b, a), 0) + 1
                    tech_neighbors.setdefault(a, set()).add(b)
                    tech_neighbors.setdefault(b, set()).add(a)
        # Activity/popularity and recent growth are calculated at the prediction cutoff.
        prev_counts = history.groupby("cpc_code").firm_id.nunique().to_dict()
        recent = df[(df.year <= pred_year) & (df.year >= pred_year - 1)].groupby("cpc_code").firm_id.nunique().to_dict()
        prior = df[(df.year <= pred_year - 2) & (df.year >= pred_year - 3)].groupby("cpc_code").firm_id.nunique().to_dict()
        total_firms = max(1, len(hist_portfolios))
        for firm, portfolio in hist_portfolios.items():
            if len(portfolio) == 0:
                continue
            patent_count = int(history.loc[history.firm_id == firm].shape[0])
            # Neighbour firms share >=1 observed class with this firm as of t.
            peer_firms = {other for other, op in hist_portfolios.items()
                          if other != firm and portfolio.intersection(op)}
            peer_den = max(1, len(peer_firms))
            for candidate in tech_universe:
                if candidate in portfolio:
                    continue
                new_next = candidate in next_tech.get(firm, set())
                neighbors = tech_neighbors.get(candidate, set())
                common = sum(pair_counts.get((candidate, tech), 0) for tech in portfolio if tech != candidate)
                peer_prev = sum(candidate in hist_portfolios[p] for p in peer_firms) / peer_den if peer_firms else 0.0
                recent_count = recent.get(candidate, 0)
                prior_count = prior.get(candidate, 0)
                growth = (recent_count - prior_count) / max(1, prior_count)
                rows.append({
                    "firm_id": firm, "prediction_year": int(pred_year), "candidate": candidate,
                    "target": int(new_next),
                    "firm_portfolio_size": len(portfolio),
                    "firm_patent_count": patent_count,
                    "candidate_prevalence": prev_counts.get(candidate, 0) / total_firms,
                    "candidate_recent_growth": float(np.clip(growth, -5, 5)),
                    "common_neighbor_count": float(common),
                    "peer_candidate_prevalence": float(peer_prev),
                    "candidate_graph_degree": len(neighbors),
                })
    panel = pd.DataFrame(rows)
    if panel.empty:
        raise ValueError("Candidate panel is empty; inspect the CPC field and date coverage.")
    return panel, train_years, [embargo_year], test_years, tech_universe, burn_years


def build_model(kind, features):
    prep = ColumnTransformer([("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ]), features)])
    if kind == "logistic":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    else:
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
            max_features="sqrt", class_weight="balanced_subsample",
            random_state=SEED, n_jobs=-1)
    return Pipeline([("prep", prep), ("model", clf)])


def metrics(y, p, threshold=.5):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        raise ValueError("Evaluation split must contain both positive and negative labels.")
    return {
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p >= threshold)),
        "brier_score": float(brier_score_loss(y, p)),
    }


def paired_bootstrap(y, p_new, p_base, n_boot=1000, seed=SEED):
    """Paired row bootstrap; cluster bootstrap by firm is recommended for final paper."""
    rng = np.random.default_rng(seed)
    y, p_new, p_base = np.asarray(y, int), np.asarray(p_new), np.asarray(p_base)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(average_precision_score(y[idx], p_new[idx]) -
                    average_precision_score(y[idx], p_base[idx]))
    if not vals:
        return {"delta_ap": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    return {"delta_ap": float(np.mean(vals)),
            "ci_low": float(np.quantile(vals, .025)),
            "ci_high": float(np.quantile(vals, .975))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", type=Path, help="normalized CSV: firm_id,year,cpc_code")
    src.add_argument("--demo", action="store_true", help="synthetic smoke test only")
    ap.add_argument("--outdir", type=Path, default=Path("outputs"))
    ap.add_argument("--max-technologies", type=int, default=30)
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()
    df = load_records(args.data, args.demo)
    panel, train_years, embargo_years, test_years, tech_universe, burn_years = make_candidate_panel(
        df, max_technologies=args.max_technologies)
    train = panel[panel.prediction_year.isin(train_years)].copy()
    test = panel[panel.prediction_year.isin(test_years)].copy()
    if train.target.nunique() < 2 or test.target.nunique() < 2:
        raise ValueError("Train and test must each contain both classes; check sample span and candidate count.")
    args.outdir.mkdir(parents=True, exist_ok=True)
    specs = {
        "M1_logistic": ("logistic", BASE),
        "M2_random_forest": ("forest", BASE),
        "M3_graph_features": ("forest", BASE + GRAPH),
    }
    results, probs = [], {}
    # Threshold is set on training data only using a separate stratified validation split.
    tr_fit, tr_val = train_test_split(train, test_size=.2, random_state=SEED, stratify=train.target)
    threshold_by_model = {}
    for name, (kind, features) in specs.items():
        val_model = build_model(kind, features).fit(tr_fit[features], tr_fit.target)
        val_p = val_model.predict_proba(tr_val[features])[:, 1]
        thresholds = np.linspace(.1, .9, 81)
        threshold = max(thresholds, key=lambda th: balanced_accuracy_score(tr_val.target, val_p >= th))
        threshold_by_model[name] = float(threshold)
        model = build_model(kind, features).fit(train[features], train.target)
        p = model.predict_proba(test[features])[:, 1]
        probs[name] = p
        results.append({"model": name, "n_train_pairs": len(train), "n_test_pairs": len(test),
            "n_test_firms": test.firm_id.nunique(), "candidate_classes": len(tech_universe),
            "test_positive_prevalence": float(test.target.mean()),
            "train_prediction_years": ",".join(map(str, train_years)),
            "test_prediction_years": ",".join(map(str, test_years)),
            "threshold_from_train_validation": threshold_by_model[name],
            **metrics(test.target, p, threshold_by_model[name])})
    res = pd.DataFrame(results)
    res.to_csv(args.outdir / "metrics.csv", index=False)
    delta = paired_bootstrap(test.target, probs["M3_graph_features"],
                             probs["M2_random_forest"], n_boot=args.bootstrap)
    pd.DataFrame([{"comparison": "M3 minus M2", "primary_metric": "average_precision",
                   **delta, "bootstrap_replicates_requested": args.bootstrap,
                   "note": "Row bootstrap is provisional; final inference should cluster by firm."}]).to_csv(
                       args.outdir / "paired_bootstrap.csv", index=False)
    # Two-panel visual is intentionally avoided: one clean performance figure.
    ax = res.set_index("model")[["average_precision", "roc_auc", "balanced_accuracy"]].plot(
        kind="bar", figsize=(9.5, 5.4), ylim=(0, 1), rot=8)
    ax.set_ylabel("Score (higher is better)")
    ax.set_title("Synthetic smoke test — not empirical evidence" if args.demo else "Future-period technology-entry prediction")
    ax.grid(axis="y", alpha=.25)
    ax.legend(loc="lower right", frameon=False)
    plt.tight_layout()
    plt.savefig(args.outdir / "figure2_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    summary = [
        f"mode={'synthetic demo only' if args.demo else 'real input CSV'}",
        f"source_rows={len(df)}", f"candidate_panel_rows={len(panel)}",
        f"candidate_classes={len(tech_universe)}", f"burn_in_years={burn_years}",
        f"train_prediction_years={train_years}", f"embargo_prediction_years={embargo_years}",
        f"test_prediction_years={test_years}", f"train_pairs={len(train)}",
        f"test_pairs={len(test)}", f"test_firms={test.firm_id.nunique()}",
        f"test_prevalence={test.target.mean():.6f}",
        "warning=synthetic demo results are not evidence" if args.demo else
        "warning=verify data provenance, firm linkage, timing, and cluster-level uncertainty",
    ]
    (args.outdir / "run_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(res.to_string(index=False))
    print("\nPaired bootstrap (M3 minus M2 average precision):", delta)
    print(f"Outputs written to {args.outdir.resolve()}")
    if args.demo:
        print("WARNING: synthetic data are a smoke test only; do not cite demo metrics in the paper.")


if __name__ == "__main__":
    main()
