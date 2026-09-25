# MpoxMirror

A digital early-warning system for mpox in Nigeria.

MpoxMirror combines surveillance, climate, rodent-reservoir, online-text and spatial
data into one pipeline that estimates state-level mpox outbreak risk four weeks ahead
and explains every alert with the expert rules that fired.

**Live dashboard:** https://egwusamuel.github.io/MpoxMirror/dashboard/

## What's inside

| Folder | Purpose |
|--------|---------|
| [`p1_warehouse/`](p1_warehouse/) | Data pipelines and storage |
| [`p2_portal/`](p2_portal/) | Data-entry portal with admin review |
| [`p3_scanner/`](p3_scanner/) | Multilingual online-text scanner |
| [`p4_early_warning/`](p4_early_warning/) | Outbreak detection and backtests |
| [`p5_dashboard/`](p5_dashboard/) | Surveillance dashboard |
| [`p6_api/`](p6_api/) | API, alerts, and cross-border feeds |

## Run locally

```bash
git clone https://github.com/EgwuSamuel/MpoxMirror.git
cd MpoxMirror
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # add your credentials
python test_connection.py
```

Built on PostgreSQL/PostGIS and Python. See [`requirements.txt`](requirements.txt)
and [`.env.example`](.env.example) for setup.

## Reproducing the paper

Every number, table and results figure in the MpoxMirror manuscript can be
regenerated **without database access** from the frozen snapshot in
[`p4_early_warning/frozen/`](p4_early_warning/frozen/) (derived, aggregate data only;
SHA-256 checksums in `MANIFEST.json`). The online-text stream is pinned to posts
scraped on or before 2026-09-21 08:05:52 UTC, because the scanner keeps back-filling
older posts.

```bash
pip install -r requirements-lock.txt          # exact versions used for the paper
export MPOX_DATA=frozen                       # Windows PowerShell: $env:MPOX_DATA="frozen"
python p4_early_warning/multiseed_analysis.py --seeds 10   # Tables 1-3 (10 seeds)
python p4_early_warning/nb_regression.py --save            # Table 4 (IRRs)
python p4_early_warning/reservoir_robustness.py            # Section 4.9
python p4_early_warning/clinical_utility.py                # Section 4.10, Fig. 3
python p4_early_warning/prospective_validation.py          # Section 4.11, Fig. 4
python p4_early_warning/generate_results_figures.py        # Figs. 2-4 -> results/
```

| Manuscript item | Script | Result file |
|---|---|---|
| Tables 1–3, Fig. 2 | `multiseed_analysis.py` | `models/multiseed_results.json` |
| Table 4 | `nb_regression.py` | `models/nb_regression_v1.json` |
| Section 4.9 | `reservoir_robustness.py` | `models/reservoir_robustness.json` |
| Section 4.10, Fig. 3 | `clinical_utility.py` | `models/clinical_utility_results.json` |
| Section 4.11, Fig. 4 | `prospective_validation.py` | `models/prospective_validation_results.json` |

With `DATABASE_URL` set (and `MPOX_DATA` unset) the same scripts read the live
warehouse instead; `p4_early_warning/export_frozen_snapshot.py` refreshes the snapshot.
