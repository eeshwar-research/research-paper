# Knowledge-Graph-Enhanced AI for Entrepreneurial Innovation Prediction

A compact research prototype for a proposed short communication targeting *Knowledge-Based Systems*.

## Research design
This project follows a rigorous short-communication structure: explicit problem formulation, transparent method, baseline comparison, component ablation, uncertainty-aware reporting, reproducibility, and limitations. An accepted paper supplied by the author is used only as a structural and experimental-design reference; its subject matter, wording, results, and claims are not reused.

## Research question
Do temporally available knowledge-graph features add predictive value for a firm's subsequent innovation outcome beyond conventional firm-level predictors?

## Minimal project layout
- `paper.md` — manuscript scaffold with exactly two main tables and two main figures.
- `src/experiment.py` — one-file reproducible pipeline, temporal holdout, graph-feature construction, baseline/ablation models, bootstrap intervals, and plots.
- `requirements.txt` — dependencies.
- `figures/` — two publication-oriented conceptual SVGs.

## Install and run the smoke test
```bash
python -m pip install -r requirements.txt
python src/experiment.py --demo --outdir outputs
```

**Important:** demo data are synthetic. Their scores only test the pipeline and must not be reported as empirical findings.

## Real-data schema
Supply a CSV with one row per firm-year and these columns:
`firm_id, year, digital_intensity, data_analytics, patent_count, rd_intensity, market_growth, industry, technology_codes, innovation_next_year`.

- `technology_codes`: semicolon-separated technology identifiers linked to the firm in that year.
- `innovation_next_year`: binary outcome with a documented, defensible definition.
- Each predictor and relation must have a documented source and timestamp.
- Use a chronological holdout with an explicit gap so labels from the training outcome window cannot overlap the test window.

The pipeline constructs a transparent firm–technology bipartite relation from `technology_codes`, plus temporally lagged technology-peer and industry-peer innovation signals. These are graph-derived relational features, but they are not a substitute for a richer ontology unless the research question and data warrant one.

## Before journal submission
1. Complete a verified literature review and articulate the precise novelty gap.
2. Document dataset provenance, inclusion criteria, missingness, outcome validity, and class prevalence.
3. Confirm temporal cutoffs and feature availability; inspect the data for leakage.
4. Run repeated seeds or additional time splits, calibration and subgroup analyses, and an ablation that isolates relational features.
5. Replace all planned or demo-only statements with measured results, uncertainty, and limitations.
6. Verify references and journal requirements. Publication is never guaranteed.
