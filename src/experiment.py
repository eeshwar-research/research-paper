"""Temporal experiment for knowledge-graph-enhanced innovation prediction.

--demo is synthetic and intended only for pipeline smoke-testing.
Never report demo metrics as empirical findings. Graph features are derived
from firm-technology relations and lagged peer labels; verify timing and data
provenance carefully before using real data.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss
)

SEED = 41
BASE_NUMERIC = ["digital_intensity", "data_analytics", "patent_count",
                "rd_intensity", "market_growth"]
CATEGORICAL = ["industry"]
GRAPH_FEATURES = ["technology_diversity", "technology_peer_prior",
                  "industry_peer_prior"]
REQUIRED = set(["firm_id", "year", *BASE_NUMERIC, *CATEGORICAL,
                "technology_codes", "innovation_next_year"])


def make_demo(seed: int = SEED, n_firms: int = 420,
              years=range(2017, 2025)) -> pd.DataFrame:
    """Generate synthetic firm-year data solely to smoke-test the pipeline."""
    rng = np.random.default_rng(seed)
    industries = ["software", "manufacturing", "health", "retail", "finance"]
    tech_pool = [f"T{i:02d}" for i in range(18)]
    firm_profiles = []
    for i in range(n_firms):
        industry = industries[i % len(industries)]
        ntech = int(rng.integers(1, 5))
        techs = rng.choice(tech_pool, size=ntech, replace=False).tolist()
        firm_profiles.append((industry, techs, rng.normal(0, .5)))
    rows = []
    for i, (industry, techs, firm_effect) in enumerate(firm_profiles):
        for year in years:
            digital = float(rng.beta(2, 2))
            analytics = float(np.clip(.55 * digital + rng.normal(.22, .18), 0, 1))
            patents = int(rng.poisson(.6 + digital))
            rd = float(np.clip(rng.gamma(2, .08), 0, 1))
            market = float(rng.normal(.025, .055))
            tech_signal = np.mean([int(t[1:]) % 5 for t in techs]) / 5
            logit = -2.3 + .8 * digital + .6 * analytics + .45 * rd + .16 * patents + .35 * tech_signal + firm_effect
            p = 1 / (1 + np.exp(-logit))
            y = int(rng.random() < p)
            rows.append([f"F{i:04d}", year, digital, analytics, patents, rd,
                         market, industry, ";".join(techs), y])
    return pd.DataFrame(rows, columns=[
        "firm_id", "year", *BASE_NUMERIC, *CATEGORICAL,
        "technology_codes", "innovation_next_year"
    ])


def parse_tech(value) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    return sorted({x.strip() for x in str(value).split(";") if x.strip()})


def temporal_split(df: pd.DataFrame):
    """Hold out the latest years with a one-year embargo for next-year labels."""
    years = sorted(pd.to_numeric(df["year"], errors="coerce").dropna().unique())
    if len(years) < 5:
        raise ValueError("At least five distinct years are required for a temporal split.")
    test_start = years[-2]
    train_end = years[-4]
    train = df[df["year"] <= train_end].copy()
    test = df[df["year"] >= test_start].copy()
    if train.empty or test.empty:
        raise ValueError("Temporal split produced an empty train/test partition.")
    if train["innovation_next_year"].nunique() != 2 or test["innovation_next_year"].nunique() != 2:
        raise ValueError("Both train and test must contain both classes; revise dataset or time window.")
    return train, test, f"train year <= {train_end}; test year >= {test_start}; embargo year(s) between"


def historical_graph_features(query_rows: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Build relational features using only labels available at least one year earlier.

    A source row at year t has an outcome in t+1; for a query row at year y,
    only source rows with year <= y-2 are eligible. This avoids using a peer's
    next-year label before it would have been observable.
    """
    q = query_rows.copy()
    q["technology_diversity"] = q["technology_codes"].map(lambda x: len(parse_tech(x))).astype(float)
    tech_lists = history["technology_codes"].map(parse_tech)
    expanded = history[["year", "innovation_next_year", "industry"]].copy()
    expanded["techs"] = tech_lists
    # Neutral prior avoids leaking labels from future rows when no historical peers exist.
    fallback_prior = 0.5
    tech_rates, industry_rates = [], []
    for _, row in q.iterrows():
        eligible = history[pd.to_numeric(history["year"], errors="coerce") <= float(row["year"]) - 2]
        if eligible.empty:
            tech_rates.append(fallback_prior)
            industry_rates.append(fallback_prior)
            continue
        label_mean = float(eligible["innovation_next_year"].mean())
        row_techs = set(parse_tech(row["technology_codes"]))
        eligible_techs = eligible["technology_codes"].map(parse_tech)
        matched = eligible.loc[eligible_techs.map(lambda ts: bool(row_techs.intersection(ts)))]
        tech_rates.append(float(matched["innovation_next_year"].mean()) if len(matched) else label_mean)
        ind = eligible[eligible["industry"].astype(str) == str(row["industry"])]
        industry_rates.append(float(ind["innovation_next_year"].mean()) if len(ind) else label_mean)
    q["technology_peer_prior"] = tech_rates
    q["industry_peer_prior"] = industry_rates
    return q


def build_model(kind: str, features: list[str]) -> Pipeline:
    numeric = [f for f in features if f not in CATEGORICAL]
    categorical = [f for f in features if f in CATEGORICAL]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical))
    prep = ColumnTransformer(transformers)
    if kind == "logistic":
        clf = LogisticRegression(max_iter=2500, class_weight="balanced", random_state=SEED)
    else:
        clf = RandomForestClassifier(
            n_estimators=350, min_samples_leaf=5, max_features="sqrt",
            class_weight="balanced_subsample", random_state=SEED, n_jobs=-1
        )
    return Pipeline([("prep", prep), ("model", clf)])


def score(y, p, threshold=.5):
    y = np.asarray(y, dtype=int)
    pred = (np.asarray(p) >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "brier_score": float(brier_score_loss(y, p))
    }


def bootstrap_delta(y, p_new, p_base, metric, n_boot=1500, seed=SEED):
    """Paired row bootstrap CI for M3 minus M2 on the same test cases."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    p_new, p_base = np.asarray(p_new), np.asarray(p_base)
    values = []
    n = len(y)
    fn = {"roc_auc": roc_auc_score, "average_precision": average_precision_score,
          "brier_score": brier_score_loss}[metric]
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        if metric == "brier_score":
            delta = fn(y[idx], p_base[idx]) - fn(y[idx], p_new[idx])
        else:
            delta = fn(y[idx], p_new[idx]) - fn(y[idx], p_base[idx])
        values.append(delta)
    if not values:
        return (np.nan, np.nan, np.nan)
    arr = np.asarray(values)
    return (float(np.mean(arr)), float(np.quantile(arr, .025)), float(np.quantile(arr, .975)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use synthetic smoke-test data only")
    ap.add_argument("--data", type=Path, help="CSV path with required columns")
    ap.add_argument("--outdir", type=Path, default=Path("outputs"))
    ap.add_argument("--bootstrap", type=int, default=1500)
    args = ap.parse_args()
    if args.demo == (args.data is not None):
        ap.error("Choose exactly one of --demo or --data path/to/file.csv")
    df = make_demo() if args.demo else pd.read_csv(args.data)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="raise")
    df["innovation_next_year"] = pd.to_numeric(df["innovation_next_year"], errors="raise").astype(int)
    if not set(df["innovation_next_year"].unique()).issubset({0, 1}):
        raise ValueError("innovation_next_year must be binary 0/1.")
    if df.duplicated(["firm_id", "year"]).any():
        raise ValueError("Duplicate firm_id/year rows detected; resolve before evaluation.")
    df = df.sort_values(["year", "firm_id"]).reset_index(drop=True)
    train, test, split_desc = temporal_split(df)

    # Build temporal relational features independently for train and test rows.
    # Each row only receives peer labels observable before its own prediction year.
    train_feat = historical_graph_features(train, train)
    test_feat = historical_graph_features(test, train)
    base_features = BASE_NUMERIC + CATEGORICAL
    graph_features = base_features + GRAPH_FEATURES
    y_train = train_feat["innovation_next_year"].astype(int)
    y_test = test_feat["innovation_next_year"].astype(int)
    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError("Temporal train/test must each contain both classes.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    specs = {
        "M1_logistic_baseline": ("logistic", base_features),
        "M2_random_forest": ("forest", base_features),
        "M3_knowledge_graph_features": ("forest", graph_features)
    }
    predictions, rows = {}, []
    for name, (kind, features) in specs.items():
        model = build_model(kind, features)
        model.fit(train_feat[features], y_train)
        prob = model.predict_proba(test_feat[features])[:, 1]
        predictions[name] = prob
        metrics = score(y_test, prob)
        rows.append({
            "model": name, "split": split_desc, "n_train": len(train_feat),
            "n_test": len(test_feat), "test_prevalence": float(y_test.mean()),
            **metrics
        })
    results = pd.DataFrame(rows)
    results.to_csv(args.outdir / "metrics.csv", index=False)

    delta_rows = []
    base = predictions["M2_random_forest"]
    enhanced = predictions["M3_knowledge_graph_features"]
    for metric in ["roc_auc", "average_precision", "brier_score"]:
        mean, lo, hi = bootstrap_delta(y_test, enhanced, base, metric, args.bootstrap)
        delta_rows.append({"comparison": "M3 minus M2 (positive favours M3)",
                           "metric": metric, "delta_mean": mean,
                           "ci_2.5%": lo, "ci_97.5%": hi,
                           "bootstrap_replicates_requested": args.bootstrap})
    pd.DataFrame(delta_rows).to_csv(args.outdir / "paired_bootstrap.csv", index=False)

    # Figure 2 is generated from the actual selected dataset. Demo charts are labelled.
    metric_cols = ["roc_auc", "average_precision", "balanced_accuracy"]
    ax = results.set_index("model")[metric_cols].plot(kind="bar", figsize=(9.5, 5.2), ylim=(0, 1), rot=12)
    ax.set_ylabel("Score (higher is better)")
    ax.set_title("Synthetic smoke-test — not empirical findings" if args.demo
                 else "Out-of-time model comparison")
    ax.grid(axis="y", alpha=.25)
    ax.legend(loc="lower right", frameon=False)
    plt.tight_layout()
    plt.savefig(args.outdir / "figure2_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Save an auditable split summary, including the gap.
    (args.outdir / "run_summary.txt").write_text(
        f"seed={SEED}\nmode={'synthetic demo' if args.demo else 'user data'}\n"
        f"split={split_desc}\ntrain_n={len(train_feat)}\ntest_n={len(test_feat)}\n"
        f"test_prevalence={y_test.mean():.6f}\n"
        f"warning={'Synthetic metrics are smoke-test only.' if args.demo else 'Verify provenance, outcome timing, and feature availability.'}\n",
        encoding="utf-8"
    )
    print(results.to_string(index=False))
    print("\nPaired bootstrap M3 vs M2:")
    print(pd.DataFrame(delta_rows).to_string(index=False))
    print(f"\nOutputs written to {args.outdir.resolve()}")
    if args.demo:
        print("WARNING: synthetic demo metrics are not evidence about real firms.")


if __name__ == "__main__":
    main()
