# Temporal Knowledge-Graph AI for Firm-Specific Technology Opportunity Prediction

Short communication manuscript and reproducible experiment for *Knowledge-Based Systems*.

## Research question
Do time-valid firm–technology graph features improve out-of-time prediction of a firm's entry into a previously unadopted CPC subclass, beyond conventional firm and class activity features?

## Result in brief
On a PatentsView organization-assignee extract (2010–2023; 10,878 firms), the graph-feature random forest improved average precision versus the same-learner forest on 2021–2022 pairs (ΔAP 0.017; firm-cluster 95% interval 0.015–0.019). Logistic regression is a competitive ranking baseline. Absolute AP stays low against a 1.11% positive rate.

The manuscript has **two figures and two tables**:
- Figure 1 — PatentsView pipeline and chronological protocol
- Figure 2 — measured 2021–2022 ranking performance
- Table 1 — evaluation sample
- Table 2 — holdout metrics with the M3 − M2 interval

## Structure
- `paper.md` — manuscript
- `src/prepare_patentsview.py` — download, join, and normalize PatentsView
- `src/experiment.py` — temporal experiment and Figure 2 generator
- `data/processed/firm_year_cpc.csv.gz` — normalized extract
- `data/LITERATURE_SEARCH.md` — documented reference expansion
- `figures/figure1_architecture.svg`, `figures/figure2_results.svg`

## Data
Source: USPTO PatentsView granted-patent metadata through 31 December 2024, [10.5281/zenodo.15058362](https://doi.org/10.5281/zenodo.15058362), downloaded 23 October 2026.

```bash
python src/prepare_patentsview.py --raw-dir data/raw --out data/processed/firm_year_cpc.csv
python src/experiment.py --data data/processed/firm_year_cpc.csv.gz --outdir outputs
```

`--demo` is a synthetic smoke test. Do not cite demo metrics.
