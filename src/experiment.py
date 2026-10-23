"""Temporal firm-specific technology-entry prediction from patent records.

Input CSV columns: firm_id, year, cpc_code (one row per firm-patent-technology relation).
--demo generates synthetic records for a pipeline smoke test only; never report those
metrics as evidence about real firms. All features for prediction year t use records
dated <= t; target is first entry into a candidate CPC subclass in t+1.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, balanced_accuracy_score, brier_score_loss

SEED = 41
BASE = ["firm_portfolio_size", "firm_activity_count", "candidate_prevalence", "candidate_recent_growth"]
GRAPH = ["common_neighbor_count", "peer_candidate_prevalence", "candidate_graph_degree"]
REQUIRED = {"firm_id", "year", "cpc_code"}


def make_demo(seed=SEED, n_firms=100, years=range(2010, 2024), n_tech=18):
    """Synthetic patent-like records for smoke testing only."""
    rng = np.random.default_rng(seed)
    techs = [f"{chr(65+i//10)}{i:03d}" for i in range(n_tech)]
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
    tech_counts = tech_counts[tech_counts >= min_firms]
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
        print(f"Building candidate panel for prediction year {pred_year}", flush=True)
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
        tech_to_firms = {}
        for other, op in hist_portfolios.items():
            for tech in op:
                tech_to_firms.setdefault(tech, set()).add(other)
        patent_counts = history.groupby("firm_id").size().to_dict()
        for firm, portfolio in hist_portfolios.items():
            if len(portfolio) == 0:
                continue
            activity_count = int(patent_counts.get(firm, 0))
            # Inverted index avoids an O(number_of_firms^2) peer scan.
            peer_firms = set().union(*(tech_to_firms.get(tech, set()) for tech in portfolio)) - {firm}
            peer_den = max(1, len(peer_firms))
            for candidate in tech_universe:
                if candidate in portfolio:
                    continue
                new_next = candidate in next_tech.get(firm, set())
                neighbors = tech_neighbors.get(candidate, set())
                common = sum(pair_counts.get((candidate, tech), 0) for tech in portfolio if tech != candidate)
                cand_firms = tech_to_firms.get(candidate, set())
                peer_prev = (len(peer_firms & cand_firms) / peer_den) if peer_firms else 0.0
                recent_count = recent.get(candidate, 0)
                prior_count = prior.get(candidate, 0)
                growth = (recent_count - prior_count) / max(1, prior_count)
                rows.append({
                    "firm_id": firm, "prediction_year": int(pred_year), "candidate": candidate,
                    "target": int(new_next),
                    "firm_portfolio_size": len(portfolio),
                    "firm_activity_count": activity_count,
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


def paired_bootstrap(y, p_new, p_base, groups, n_boot=1000, seed=SEED):
    """Paired cluster bootstrap: resample firms, retaining all their held-out pairs."""
    rng = np.random.default_rng(seed)
    y, p_new, p_base, groups = np.asarray(y, int), np.asarray(p_new), np.asarray(p_base), np.asarray(groups)
    unique_groups = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in unique_groups}
    vals = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([members[g] for g in sampled])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(average_precision_score(y[idx], p_new[idx]) -
                    average_precision_score(y[idx], p_base[idx]))
    if not vals:
        return {"delta_ap": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    return {"delta_ap": float(np.mean(vals)),
            "ci_low": float(np.quantile(vals, .025)),
            "ci_high": float(np.quantile(vals, .975))}


def write_protocol_figures(figdir: Path):
    """Publication figures for the architecture and temporal protocol. Not results."""
    figdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    def box(ax, x, y, w, h, title, lines, facecolor):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                               facecolor=facecolor, edgecolor="#315d91", linewidth=1.4)
        ax.add_patch(patch)
        ax.text(x + 0.04, y + h - 0.10, title, fontsize=11, fontweight="bold", color="#132b3e", va="top")
        for i, line in enumerate(lines):
            ax.text(x + 0.04, y + h - 0.28 - 0.16 * i, line, fontsize=9.2, color="#31495e", va="top")

    fig, ax = plt.subplots(figsize=(11.2, 3.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(0.2, 5.7, "Knowledge-graph feature pipeline for firm-specific technology-entry prediction",
            fontsize=13, fontweight="bold", color="#132b3e")
    ax.text(0.2, 5.32, "Every feature at prediction year t uses only records dated <= t; the target is first CPC-subclass entry in t+1.",
            fontsize=9.2, color="#41566a")
    box(ax, 0.25, 2.55, 2.7, 2.4, "1  Time-stamped inputs",
        ["firm_id, year, cpc_code", "USPTO PatentsView", "Normalized CPC subclass"], "#eaf2fb")
    box(ax, 3.25, 2.55, 2.7, 2.4, "2  Graph G_t",
        ["Firms and CPC classes", "Edges observed by t", "Co-occurrence among classes"], "#e8f6f1")
    box(ax, 6.25, 2.55, 2.7, 2.4, "3  Inspectable features",
        ["Portfolio size, activity", "Class prevalence, growth", "CN, peer rate, degree"], "#fff3df")
    box(ax, 9.25, 2.55, 2.5, 2.4, "4  Controlled models",
        ["M1 logistic regression", "M2 random forest", "M3 graph-feature RF"], "#f2ecfb")
    for x0, x1 in ((2.95, 3.25), (5.95, 6.25), (8.95, 9.25)):
        ax.add_patch(FancyArrowPatch((x0, 3.75), (x1, 3.75), arrowstyle="-|>", mutation_scale=10,
                                     color="#49627a", linewidth=1.5))
    eval_box = FancyBboxPatch((0.25, 0.25), 11.5, 2.0, boxstyle="round,pad=0.02,rounding_size=0.08",
                              facecolor="#ffffff", edgecolor="#9db0c0", linewidth=1.3)
    ax.add_patch(eval_box)
    ax.text(0.45, 1.95, "5  Shared evaluation", fontsize=11, fontweight="bold", color="#132b3e")
    for i, (title, note) in enumerate([
        ("Discrimination", "Average precision + prevalence"),
        ("Calibration", "Brier score"),
        ("Incremental value", "Firm-clustered bootstrap"),
        ("Executed split", "One embargoed holdout"),
    ]):
        x = 0.5 + i * 2.85
        inner = FancyBboxPatch((x, 0.45), 2.6, 1.15, boxstyle="round,pad=0.02,rounding_size=0.06",
                               facecolor="#f5f8fb", edgecolor="#c3d0dc", linewidth=1.0)
        ax.add_patch(inner)
        ax.text(x + 1.3, 1.2, title, ha="center", fontsize=10, fontweight="bold", color="#132b3e")
        ax.text(x + 1.3, 0.85, note, ha="center", fontsize=8.6, color="#526a7e")
    fig.tight_layout()
    fig.savefig(figdir / "figure1_architecture.pdf", bbox_inches="tight")
    fig.savefig(figdir / "figure1_architecture.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.2, 3.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(0.2, 5.7, "Leakage-aware temporal evaluation protocol",
            fontsize=13, fontweight="bold", color="#132b3e")
    ax.text(0.2, 5.32, "Predict from information available at time t; score outcomes only after the cutoff and embargo.",
            fontsize=9.2, color="#41566a")
    ax.annotate("", xy=(11.6, 4.55), xytext=(0.4, 4.55),
                arrowprops=dict(arrowstyle="-|>", color="#49627a", lw=1.8))
    periods = [
        (0.4, 3.2, "#dceafe", "#315d91", "Burn-in (3 years)", "Freeze candidate CPC universe"),
        (3.7, 3.6, "#cfe4d8", "#28734d", "Training years", "Labels end before embargo"),
        (7.4, 1.7, "#fff0d4", "#a56a12", "Embargo", "One unused year"),
        (9.2, 2.4, "#d9f4e5", "#28734d", "Future test", "Last two prediction years"),
    ]
    for x, w, face, edge, title, note in periods:
        patch = FancyBboxPatch((x, 3.85), w, 1.15, boxstyle="round,pad=0.02,rounding_size=0.06",
                               facecolor=face, edgecolor=edge, linewidth=1.3)
        ax.add_patch(patch)
        ax.text(x + w / 2, 4.6, title, ha="center", fontsize=10, fontweight="bold", color="#132b3e")
        ax.text(x + w / 2, 4.2, note, ha="center", fontsize=8.4, color="#31495e")
    steps = [
        (0.4, "A. Fit", ["Imputation and scaling", "Model parameters", "Historical graph rates"]),
        (4.2, "B. Compare", ["M1 logistic baseline", "M2 tabular random forest", "M3 graph-feature RF"]),
        (8.0, "C. Report", ["Average precision, ROC-AUC", "Balanced accuracy, Brier", "Firm-clustered intervals"]),
    ]
    for x, title, lines in steps:
        patch = FancyBboxPatch((x, 0.45), 3.5, 2.7, boxstyle="round,pad=0.02,rounding_size=0.08",
                               facecolor="#ffffff", edgecolor="#9db0c0", linewidth=1.3)
        ax.add_patch(patch)
        ax.text(x + 0.18, 2.8, title, fontsize=11, fontweight="bold", color="#132b3e")
        for i, line in enumerate(lines):
            ax.text(x + 0.18, 2.3 - 0.45 * i, line, fontsize=9.4, color="#31495e")
    ax.text(0.4, 0.15, "This diagram defines the protocol; it is not a report of measured model performance.",
            fontsize=8.5, color="#526a7e")
    fig.tight_layout()
    fig.savefig(figdir / "figure2_evaluation.pdf", bbox_inches="tight")
    fig.savefig(figdir / "figure2_evaluation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", type=Path, help="normalized CSV: firm_id,year,cpc_code")
    src.add_argument("--demo", action="store_true", help="synthetic smoke test only")
    src.add_argument("--figures", action="store_true", help="write protocol figures only; no model scores")
    ap.add_argument("--outdir", type=Path, default=Path("outputs"))
    ap.add_argument("--figdir", type=Path, default=Path("figures"))
    ap.add_argument("--max-technologies", type=int, default=30)
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()
    if args.figures:
        write_protocol_figures(args.figdir)
        print(f"Protocol figures written to {args.figdir.resolve()}")
        return
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
    # Threshold is selected on the latest training prediction year (temporal validation),
    # never by random row splitting or by inspecting the final test period.
    validation_year = max(train_years)
    tr_fit = train[train.prediction_year < validation_year]
    tr_val = train[train.prediction_year == validation_year]
    if tr_fit.empty or tr_val.empty or tr_fit.target.nunique() < 2 or tr_val.target.nunique() < 2:
        raise ValueError("Temporal threshold-validation split lacks both classes; extend the time span or revise the pre-specified split.")
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
                             probs["M2_random_forest"], test.firm_id,
                             n_boot=args.bootstrap)
    pd.DataFrame([{"comparison": "M3 minus M2", "primary_metric": "average_precision",
                   **delta, "bootstrap_replicates_requested": args.bootstrap,
                   "note": "Paired cluster bootstrap resamples firms and retains all held-out pairs."}]).to_csv(
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
