# Temporal Knowledge-Graph AI for Firm-Specific Technology Opportunity Prediction

Compact research repository for a proposed *Knowledge-Based Systems* short communication.

## Research question
Do time-valid firm–technology graph features improve out-of-time prediction of a firm's entry into a previously unadopted CPC technology subclass, beyond conventional firm and technology activity features?

## Result in brief
On a PatentsView organization-assignee extract (grant years 2010–2023; 10,878 firms), the graph-feature random forest improved average precision versus the same-learner forest on 2021–2022 pairs (ΔAP 0.017; firm-cluster 95% interval 0.015–0.019). Logistic regression was a competitive ranking baseline. Absolute AP stayed low against a 1.11% positive rate. See `paper.md` and `outputs/`.

## Structure
- `paper.md` — manuscript with measured results, three tables, and three figures
- `src/prepare_patentsview.py` — download, join, and normalize PatentsView tables
- `src/experiment.py` — temporal experiment, baselines, graph features, firm-cluster bootstrap
- `data/processed/firm_year_cpc.csv.gz` — normalized `firm_id,year,cpc_code` extract
- `data/processed/firm_year_cpc.provenance.json` — source DOI, filters, hashes, counts
- `data/LITERATURE_SEARCH.md` — documented reference expansion
- `outputs/` — metrics, bootstrap, run summary, comparison chart
- `figures/` — architecture, evaluation protocol, and observed performance plot

## Data
Source: USPTO PatentsView granted-patent metadata through 31 December 2024, Zenodo DOI [10.5281/zenodo.15058362](https://doi.org/10.5281/zenodo.15058362), downloaded 23 October 2026. Tables: `g_patent`, `g_assignee_disambiguated`, `g_cpc_current`, joined on `patent_id`.

```csv
firm_id,year,cpc_code
0019b8b6-cab8-4f5c-aff0-8ea114ceec37,2010,D02J
```

One row is a unique firm-year-CPC subclass. Codes are four alphanumeric characters. `Y10*` indexing subclasses are excluded. Firms need eight years, eight subclasses, twenty relations, and at least one G or H subclass. Patent-based entry is a proxy, not a full measure of innovation.

Rebuild the extract (downloads ~1 GB of zip files):

```bash
python src/prepare_patentsview.py --raw-dir data/raw --out data/processed/firm_year_cpc.csv
```

## Run
```bash
python -m pip install -r requirements.txt
python src/experiment.py --demo --outdir outputs
python src/experiment.py --data data/processed/firm_year_cpc.csv.gz --outdir outputs
```

`--demo` is a synthetic smoke test. Do not cite demo metrics.

## Experimental safeguards
- Candidate CPC classes come from the first three burn-in years, not the test window.
- Features at year (t) use only records through (t); the target is entry in (t+1).
- Test years are the last two eligible prediction years, with an embargo year in between.
- Thresholds are chosen on a training-period validation year.
- M3 is compared with M2 on identical held-out pairs; the bootstrap resamples firms.

## Publication status
The extract, experiment, and results section are completed for the pre-specified split. Remaining work includes a second chronological cutoff, sector slices, and a Scopus/Web of Science search if the journal requires one.
