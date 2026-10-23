"""Reproducible prototype for knowledge-enhanced innovation prediction.

Demo mode generates synthetic data only for pipeline smoke-testing. Do not
report synthetic-demo metrics as scientific findings. The knowledge features
in this starter are proxies, not outputs from a real knowledge graph.
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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix
)
from sklearn.model_selection import train_test_split

REQUIRED = {
    "firm_id", "year", "digital_intensity", "data_analytics",
    "patent_count", "rd_intensity", "market_growth", "innovation_next_year"
}
BASE_FEATURES = ["digital_intensity", "data_analytics", "patent_count",
                 "rd_intensity", "market_growth"]
KNOWLEDGE_FEATURES = ["technology_diversity", "industry_signal"]


def make_demo(seed: int = 17, n_firms: int = 500,
              years=range(2018, 2025)) -> pd.DataFrame:
    """Create synthetic panel data for smoke-testing only."""
    rng = np.random.default_rng(seed)
    rows = []
    firm_effect = rng.normal(0, 0.8, n_firms)
    for i in range(n_firms):
        industry_signal = rng.normal()
        for year in years:
            digital = np.clip(rng.beta(2, 2), 0, 1)
            analytics = np.clip(0.55 * digital + rng.normal(0.25, 0.2), 0, 1)
            patents = rng.poisson(0.8 + digital)
            rd = np.clip(rng.gamma(2, 0.08), 0, 1)
            market = rng.normal(0.02, 0.06)
            diversity = np.clip(0.25 + 0.5 * digital + rng.normal(0, 0.2), 0, 1)
            logit = (-2.2 + digital + 0.8 * analytics + 0.5 * rd
                     + 0.25 * patents + 0.3 * diversity
                     + 0.2 * industry_signal + firm_effect[i] * 0.25)
            p = 1 / (1 + np.exp(-logit))
            y = int(rng.random() < p)
            rows.append([f"F{i:04d}", year, digital, analytics, patents, rd,
                         market, diversity, industry_signal, y])
    return pd.DataFrame(rows, columns=[
        "firm_id", "year", "digital_intensity", "data_analytics",
        "patent_count", "rd_intensity", "market_growth",
        "technology_diversity", "industry_signal", "innovation_next_year"
    ])


def add_knowledge_proxy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create explicitly labelled proxies until genuine graph data are available."""
    out = df.copy()
    if "technology_diversity" not in out:
        out["technology_diversity"] = (
            out["digital_intensity"].fillna(0)
            * (1 + np.log1p(out["patent_count"].fillna(0)))
        )
    if "industry_signal" not in out:
        out["industry_signal"] = out.groupby("year")["market_growth"].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-8)
        )
    return out


def build_model(kind: str, features: list[str]) -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    prep = ColumnTransformer([("numeric", numeric, features)], remainder="drop")
    if kind == "logistic":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                 random_state=17)
    else:
        clf = RandomForestClassifier(
            n_estimators=400, min_samples_leaf=4, class_weight="balanced",
            random_state=17, n_jobs=-1
        )
    return Pipeline([("prep", prep), ("model", clf)])


def temporal_split(df: pd.DataFrame):
    years = sorted(df["year"].dropna().unique())
    if len(years) >= 3:
        cutoff = years[max(0, int(len(years) * 0.75) - 1)]
        train = df[df.year <= cutoff]
        test = df[df.year > cutoff]
        if (not train.empty and not test.empty
                and test["innovation_next_year"].nunique() == 2):
            return train, test, f"temporal holdout (train <= {cutoff})"
    train, test = train_test_split(
        df, test_size=0.25, random_state=17,
        stratify=df["innovation_next_year"]
    )
    return train, test, "random stratified fallback (use cautiously; panel leakage risk)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true",
                    help="use synthetic smoke-test data")
    ap.add_argument("--data", type=Path,
                    help="CSV containing required real-data columns")
    ap.add_argument("--outdir", type=Path, default=Path("outputs"))
    args = ap.parse_args()
    if not args.demo and args.data is None:
        ap.error("Provide --demo or --data path/to/data.csv")
    df = make_demo() if args.demo else pd.read_csv(args.data)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = add_knowledge_proxy_features(df).sort_values("year")
    if df["innovation_next_year"].nunique() != 2:
        raise ValueError("Target must contain both binary classes (0 and 1).")
    train, test, split_name = temporal_split(df)
    if train["innovation_next_year"].nunique() < 2:
        raise ValueError("Training split contains only one target class; revise split/data.")
    args.outdir.mkdir(parents=True, exist_ok=True)

    experiments = {
        "M1_logistic": ("logistic", BASE_FEATURES),
        "M2_random_forest": ("forest", BASE_FEATURES),
        "M3_knowledge_enhanced": ("forest", BASE_FEATURES + KNOWLEDGE_FEATURES),
    }
    records = []
    for name, (kind, features) in experiments.items():
        model = build_model(kind, features)
        model.fit(train[features], train["innovation_next_year"])
        prob = model.predict_proba(test[features])[:, 1]
        pred = (prob >= 0.5).astype(int)
        y = test["innovation_next_year"].astype(int)
        records.append({
            "model": name, "split": split_name, "n_train": len(train),
            "n_test": len(test), "prevalence_test": float(y.mean()),
            "roc_auc": float(roc_auc_score(y, prob)) if y.nunique() == 2 else np.nan,
            "average_precision": float(average_precision_score(y, prob)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "brier_score": float(brier_score_loss(y, prob)),
            "tn_fp_fn_tp": confusion_matrix(y, pred, labels=[0, 1]).ravel().tolist()
        })
    results = pd.DataFrame(records)
    results.to_csv(args.outdir / "metrics.csv", index=False)
    print(results.to_string(index=False))
    print("\nWARNING: synthetic-demo results are for code smoke-testing only."
          if args.demo else
          "\nVerify data provenance, leakage controls, and target validity before interpreting results.")

    # Generate a compact architecture figure.
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.axis("off")
    boxes = [
        (0.03, 0.38, 0.19, 0.32, "Time-stamped\nfirm + market data"),
        (0.28, 0.38, 0.19, 0.32, "Domain knowledge\nentities + relations"),
        (0.53, 0.38, 0.19, 0.32, "Leakage-safe\nfeature construction"),
        (0.78, 0.38, 0.19, 0.32, "Baseline vs.\nknowledge-enhanced AI")
    ]
    for x, y, w, h, label in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, linewidth=1.5))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)
    for x in [0.22, 0.47, 0.72]:
        ax.annotate("", xy=(x + 0.055, 0.54), xytext=(x, 0.54),
                    arrowprops=dict(arrowstyle="->", lw=1.4))
    ax.text(0.5, 0.12, "Output: out-of-time prediction + calibration + decision metrics",
            ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.outdir / "figure1_architecture.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    metric = results.set_index("model")[["roc_auc", "average_precision",
                                         "balanced_accuracy"]]
    ax = metric.plot(kind="bar", figsize=(9, 4.5), ylim=(0, 1), rot=12)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison — synthetic demo only" if args.demo
                 else "Model comparison on supplied dataset")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(args.outdir / "figure2_model_comparison.png", dpi=220, bbox_inches="tight")
    plt.close()
    print(f"\nOutputs written to {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
