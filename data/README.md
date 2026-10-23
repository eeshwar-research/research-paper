# PatentsView extract

This folder stores the normalized firm–year–CPC table used by `src/experiment.py`. Raw USPTO files are not committed.

## Source
- USPTO PatentsView granted-patent metadata, public Zenodo archive of the 31 December 2024 dump.
- Record: https://zenodo.org/records/15058362
- DOI: https://doi.org/10.5281/zenodo.15058362
- License: Creative Commons Attribution 4.0.
- The USPTO Open Data Portal now hosts later PatentsView files (coverage through 31 December 2025) behind authenticated access. This extract uses the last fully public dump.

## Tables and join
| Table | Role |
|---|---|
| `g_patent` | Utility granted patents; `year` is the grant year from `patent_date`; withdrawn patents dropped |
| `g_assignee_disambiguated` | `firm_id` = PatentsView `assignee_id`; organizations only (non-empty organization name) |
| `g_cpc_current` | `cpc_code` = 4-character CPC subclass |

Join key: `patent_id`. Joint assignees create one firm–technology row per organization.

## Inclusion
- Grant years 2010–2023.
- Firms with at least 8 distinct years, 8 distinct subclasses, and 20 firm–year–CPC rows.
- Firms with at least one G or H subclass (electrical / computing activity, matching the digital-innovation scope).
- Former-USPC indexing subclasses `Y10*` excluded.

## Reproduce
```bash
python src/prepare_patentsview.py --raw-dir data/raw --out data/processed/firm_year_cpc.csv
python src/experiment.py --data data/processed/firm_year_cpc.csv.gz --outdir outputs
```

Exact file hashes, download timestamps, and row counts are in `data/processed/firm_year_cpc.provenance.json`.
