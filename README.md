# Knowledge-Graph-Enhanced AI for Entrepreneurial Opportunity Prediction

Compact, reproducible prototype for a proposed *Knowledge-Based Systems* short communication.

## Research integrity
This is a research scaffold, not a completed empirical study. Demo data are synthetic and must not be presented as findings. Before submission, use a documented real dataset, implement genuine graph-derived features, verify the literature, and report results transparently. No publication or Q1 outcome is guaranteed.

## Research question
Do structured, pre-outcome knowledge features improve out-of-time prediction of a clearly defined entrepreneurial innovation outcome beyond conventional tabular machine-learning models?

## Minimal structure
- `paper.md` — working manuscript with two tables and two figures.
- `src/experiment.py` — single-script pipeline for baselines, metrics, and plots.
- `requirements.txt` — minimal dependencies.
- `figures/` — conceptual SVG figures; they illustrate the proposed design, not empirical findings.

## Run
```bash
python -m pip install -r requirements.txt
python src/experiment.py --demo --outdir outputs
```

For real data, provide a CSV with one row per firm-year and columns:
`firm_id, year, digital_intensity, data_analytics, patent_count, rd_intensity, market_growth, innovation_next_year`.

The target must be defensibly defined and sourced. The current knowledge features are transparent proxies, not a true knowledge graph. Replace them with graph-derived features from documented firm–technology–industry relations before making claims about knowledge-graph effects.

## Evaluation cautions
- Prevent temporal leakage; predictors must predate the outcome window.
- Prefer time-based evaluation for longitudinal firm data.
- Report prevalence, sample sizes, missingness, calibration, uncertainty, and subgroup performance.
- Compare strong baselines and ablations; do not assume the knowledge-enhanced model will win.
