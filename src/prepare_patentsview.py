"""Download, join, and normalize USPTO PatentsView granted-patent tables.

Output CSV columns: firm_id, year, cpc_code (one unique firm-year-CPC subclass row).

The current USPTO Open Data Portal bulk files require an authenticated ODP account.
This script therefore uses the last fully public PatentsView granted-patent metadata
dump archived on Zenodo (coverage through 31 December 2024).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import duckdb

ZENODO_RECORD = "https://zenodo.org/records/15058362"
ZENODO_DOI = "10.5281/zenodo.15058362"
ZENODO_FILES = {
    "g_patent.tsv.zip": {
        "url": "https://zenodo.org/records/15058362/files/g_patent.tsv.zip?download=1",
        "md5": "f74fbde4b2adbf980b8e4ed5394f16d2",
        "bytes": 223192856,
    },
    "g_assignee_disambiguated.tsv.zip": {
        "url": "https://zenodo.org/records/15058362/files/g_assignee_disambiguated.tsv.zip?download=1",
        "md5": "7fa0f23627570ce599d81d6fbb97fb98",
        "bytes": 346950350,
    },
    "g_cpc_current.tsv.zip": {
        "url": "https://zenodo.org/records/15058362/files/g_cpc_current.tsv.zip?download=1",
        "md5": "8211b681c392b2810b5a848e1fcfa308",
        "bytes": 485195636,
    },
}
JOIN_KEY = "patent_id"


def md5sum(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url} -> {dest}", flush=True)
    with urlopen(url, timeout=120) as resp, tmp.open("wb") as out:
        while True:
            block = resp.read(1024 * 1024)
            if not block:
                break
            out.write(block)
    tmp.replace(dest)


def ensure_raw_tables(raw_dir: Path, skip_checksum: bool = False) -> dict:
    log = {}
    for name, meta in ZENODO_FILES.items():
        zpath = raw_dir / name
        tsv_name = name.replace(".zip", "")
        tsv_path = raw_dir / tsv_name
        if not zpath.exists() or zpath.stat().st_size != meta["bytes"]:
            download(meta["url"], zpath)
        digest = None if skip_checksum else md5sum(zpath)
        if digest and digest != meta["md5"]:
            raise ValueError(f"MD5 mismatch for {name}: {digest} != {meta['md5']}")
        if not tsv_path.exists():
            print(f"Extracting {zpath}", flush=True)
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(raw_dir)
        log[name] = {
            "zip_path": str(zpath),
            "zip_bytes": zpath.stat().st_size,
            "md5": digest or "skipped",
            "tsv_path": str(tsv_path),
            "tsv_bytes": tsv_path.stat().st_size,
        }
    return log


def normalize(raw_dir: Path, out_csv: Path, year_start: int, year_end: int,
              min_years: int, min_subclasses: int, min_relations: int) -> dict:
    con = duckdb.connect()
    patent_tsv = str(raw_dir / "g_patent.tsv")
    assignee_tsv = str(raw_dir / "g_assignee_disambiguated.tsv")
    cpc_tsv = str(raw_dir / "g_cpc_current.tsv")
    con.execute(f"""
        CREATE OR REPLACE TABLE patents AS
        SELECT
            patent_id,
            CAST(substr(patent_date, 1, 4) AS INTEGER) AS year
        FROM read_csv(
            '{patent_tsv}',
            delim='\t', quote='"', escape='"', header=true,
            columns={{
                'patent_id':'VARCHAR','patent_type':'VARCHAR','patent_date':'VARCHAR',
                'patent_title':'VARCHAR','wipo_kind':'VARCHAR','num_claims':'VARCHAR',
                'withdrawn':'VARCHAR','filename':'VARCHAR'
            }},
            max_line_size=50000000, ignore_errors=true, parallel=false
        )
        WHERE patent_type = 'utility'
          AND (withdrawn IS NULL OR withdrawn IN ('0', '0.0'))
          AND CAST(substr(patent_date, 1, 4) AS INTEGER) BETWEEN {year_start} AND {year_end}
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE assignees AS
        SELECT DISTINCT patent_id, assignee_id AS firm_id
        FROM read_csv(
            '{assignee_tsv}',
            delim='\t', quote='"', escape='"', header=true,
            columns={{
                'patent_id':'VARCHAR','assignee_sequence':'VARCHAR','assignee_id':'VARCHAR',
                'disambig_assignee_individual_name_first':'VARCHAR',
                'disambig_assignee_individual_name_last':'VARCHAR',
                'disambig_assignee_organization':'VARCHAR',
                'assignee_type':'VARCHAR','location_id':'VARCHAR'
            }},
            max_line_size=50000000, ignore_errors=true, parallel=false
        )
        WHERE assignee_id IS NOT NULL AND length(trim(assignee_id)) > 0
          AND disambig_assignee_organization IS NOT NULL
          AND length(trim(disambig_assignee_organization)) > 0
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE cpc AS
        SELECT DISTINCT
            patent_id,
            upper(substr(regexp_replace(cpc_subclass, '[^A-Za-z0-9]', '', 'g'), 1, 4)) AS cpc_code
        FROM read_csv(
            '{cpc_tsv}',
            delim='\t', quote='"', escape='"', header=true,
            columns={{
                'patent_id':'VARCHAR','cpc_sequence':'VARCHAR','cpc_section':'VARCHAR',
                'cpc_class':'VARCHAR','cpc_subclass':'VARCHAR','cpc_group':'VARCHAR',
                'cpc_type':'VARCHAR'
            }},
            max_line_size=50000000, ignore_errors=true, parallel=false
        )
        WHERE cpc_subclass IS NOT NULL
          AND length(upper(substr(regexp_replace(cpc_subclass, '[^A-Za-z0-9]', '', 'g'), 1, 4))) = 4
          AND upper(substr(regexp_replace(cpc_subclass, '[^A-Za-z0-9]', '', 'g'), 1, 3)) != 'Y10'
    """)
    con.execute("""
        CREATE OR REPLACE TABLE relations AS
        SELECT DISTINCT a.firm_id, p.year, c.cpc_code
        FROM patents p
        JOIN assignees a USING (patent_id)
        JOIN cpc c USING (patent_id)
    """)
    pre = con.execute("""
        SELECT
            count(*) AS n_rows,
            count(DISTINCT firm_id) AS n_firms,
            count(DISTINCT year) AS n_years,
            count(DISTINCT cpc_code) AS n_subclasses,
            min(year) AS year_min,
            max(year) AS year_max
        FROM relations
    """).fetchone()
    con.execute(f"""
        CREATE OR REPLACE TABLE eligible_firms AS
        SELECT firm_id
        FROM relations
        GROUP BY firm_id
        HAVING count(DISTINCT year) >= {min_years}
           AND count(DISTINCT cpc_code) >= {min_subclasses}
           AND count(*) >= {min_relations}
           AND count(DISTINCT CASE WHEN substr(cpc_code, 1, 1) IN ('G', 'H') THEN 1 END) > 0
    """)
    con.execute("""
        CREATE OR REPLACE TABLE final AS
        SELECT r.firm_id, r.year, r.cpc_code
        FROM relations r
        JOIN eligible_firms e USING (firm_id)
        ORDER BY year, firm_id, cpc_code
    """)
    post = con.execute("""
        SELECT
            count(*) AS n_rows,
            count(DISTINCT firm_id) AS n_firms,
            count(DISTINCT year) AS n_years,
            count(DISTINCT cpc_code) AS n_subclasses,
            min(year) AS year_min,
            max(year) AS year_max
        FROM final
    """).fetchone()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY final TO '{out_csv}' (HEADER, DELIMITER ',')")
    patents_n = con.execute("SELECT count(*) FROM patents").fetchone()[0]
    assignee_patents = con.execute("SELECT count(DISTINCT patent_id) FROM assignees").fetchone()[0]
    org_assignees = con.execute("SELECT count(DISTINCT firm_id) FROM assignees").fetchone()[0]
    cpc_n = con.execute("SELECT count(*) FROM cpc").fetchone()[0]
    return {
        "utility_patents_in_window": int(patents_n),
        "organization_assignee_ids": int(org_assignees),
        "patents_with_organization_assignee": int(assignee_patents),
        "patent_cpc_subclass_pairs": int(cpc_n),
        "before_firm_filters": {
            "rows": int(pre[0]), "firms": int(pre[1]), "years": int(pre[2]),
            "subclasses": int(pre[3]), "year_min": int(pre[4]), "year_max": int(pre[5]),
        },
        "after_firm_filters": {
            "rows": int(post[0]), "firms": int(post[1]), "years": int(post[2]),
            "subclasses": int(post[3]), "year_min": int(post[4]), "year_max": int(post[5]),
        },
        "output_csv": str(out_csv),
        "output_bytes": out_csv.stat().st_size,
        "output_md5": md5sum(out_csv),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/firm_year_cpc.csv"))
    ap.add_argument("--year-start", type=int, default=2010)
    ap.add_argument("--year-end", type=int, default=2023)
    ap.add_argument("--min-years", type=int, default=8)
    ap.add_argument("--min-subclasses", type=int, default=8)
    ap.add_argument("--min-relations", type=int, default=20)
    ap.add_argument("--skip-checksum", action="store_true")
    args = ap.parse_args()
    download_utc = datetime.now(timezone.utc).isoformat()
    raw_log = ensure_raw_tables(args.raw_dir, skip_checksum=args.skip_checksum)
    stats = normalize(
        args.raw_dir, args.out, args.year_start, args.year_end,
        args.min_years, args.min_subclasses, args.min_relations,
    )
    provenance = {
        "source_name": "USPTO PatentsView granted-patent metadata, Zenodo archive",
        "source_record": ZENODO_RECORD,
        "source_doi": ZENODO_DOI,
        "coverage_note": "Granted patents and related tables through 31 December 2024.",
        "download_started_utc": download_utc,
        "download_finished_utc": datetime.now(timezone.utc).isoformat(),
        "join_key": JOIN_KEY,
        "tables": {
            "g_patent": "utility patents; year = grant-year from patent_date; withdrawn patents excluded",
            "g_assignee_disambiguated": "firm_id = PatentsView assignee_id; organizations only (non-empty organization name); joint assignees create multiple firm-technology rows",
            "g_cpc_current": "cpc_code = 4-character CPC subclass from cpc_subclass, current classification; Y10 indexing codes associated with former USPC are excluded",
        },
        "inclusion": {
            "patent_type": "utility",
            "year_field": "grant year",
            "year_start": args.year_start,
            "year_end": args.year_end,
            "assignee_rule": "disambiguated organization assignees",
            "min_distinct_years_per_firm": args.min_years,
            "min_distinct_subclasses_per_firm": args.min_subclasses,
            "min_firm_year_cpc_rows_per_firm": args.min_relations,
            "digital_activity_filter": "firm has at least one G or H CPC subclass in the window",
            "excluded_cpc": "Y10* former-USPC indexing subclasses",
        },
        "files": raw_log,
        "stats": stats,
    }
    prov_path = args.out.with_suffix(".provenance.json")
    prov_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance["stats"], indent=2))
    print(f"Wrote {args.out}")
    print(f"Wrote {prov_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
