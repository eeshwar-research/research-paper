# Temporal Knowledge-Graph AI for Firm-Specific Technology Opportunity Prediction

Compact research repository for a proposed *Knowledge-Based Systems* short communication.

## Research question
Do time-valid firm–technology graph features improve out-of-time prediction of a firm's entry into a previously unadopted CPC technology subclass, beyond conventional firm and technology activity features?

## Minimal structure
- `paper.md` — full working manuscript, research gap, method, exactly two main tables and two main figures.
- `src/experiment.py` — one-file temporal experiment with baselines, graph features, chronological validation, paired firm-cluster bootstrap, and output generation.
- `requirements.txt` — minimal dependencies.
- `figures/` — two publication-oriented vector diagrams.

## Data
The proposed source is the public USPTO PatentsView / Open Data Portal. Download the appropriate patent, assignee/organization, and CPC classification tables from the official [PatentsView page](https://www.uspto.gov/ip-policy/economic-research/patentsview), record the exact release and download date, disambiguate firms, and normalize the result to CSV with these columns:

```csv
firm_id,year,cpc_code
firm_001,2015,A01B
firm_001,2016,A01B
firm_001,2017,G06F
```

One row represents a unique firm-year-CPC subclass relation. CPC subclass codes are normalized to their first four alphanumeric characters. This is an input contract; the repository does not include a downloaded real dataset. Patent-based technology entry is a proxy, not a comprehensive measure of innovation or entrepreneurship.

## Run
Install dependencies:
```bash
python -m pip install -r requirements.txt
```

Smoke test:
```bash
python src/experiment.py --demo --outdir outputs
```

Real-data run:
```bash
python src/experiment.py --data path/to/firm_year_cpc.csv --outdir outputs
```

The pipeline creates `metrics.csv`, `paired_bootstrap.csv`, `run_summary.txt`, and a 300-dpi model-comparison chart. The demo mode is synthetic and only checks software execution; never report demo metrics as empirical findings.

## Experimental safeguards
- Candidate CPC classes are fixed using the first three burn-in years, not the future test window.
- Features at prediction year (t) use only data through (t); the target is entry in (t+1).
- The last two eligible prediction years form the test period, with an intervening embargo year.
- Thresholds are selected on a later training-period validation year, not on test labels.
- M3 is compared with M2 on identical held-out candidate pairs; the paired bootstrap resamples firms.
- Inspect data timing, firm disambiguation, duplicate relations, CPC code revisions, class prevalence, and year coverage before interpretation.

## Publication status
This is a manuscript and experiment scaffold, not a completed empirical paper. Real-data extraction, a full literature search, execution, robustness checks, and evidence-based results are still required. Do not claim model superiority or guaranteed acceptance before those steps are complete.
