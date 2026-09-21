# MpoXMirror

**A digital early-warning and genomic-surveillance system for mpox in Nigeria.**

MpoXMirror (formerly `smartmpox-nigeria`) integrates open epidemiological, genomic, and
environmental data into a single pipeline that detects mpox signals earlier than
traditional case-based reporting. The system is built around a seven-phase methodology,
from data warehousing through a public API and alerting layer.

> **Repository note:** this project was renamed from `smartmpox-nigeria` to `MpoXMirror`.
> GitHub redirects the old URL, but please update any bookmarks, CI configs, or clones to
> `https://github.com/EgwuSamuel/MpoXMirror`.

## Architecture

The codebase is organized by phase:

| Phase | Directory | Purpose |
|-------|-----------|---------|
| P1 | [`p1_warehouse/`](p1_warehouse/) | Data warehouse: ETL pipelines, schema, feature store, data-quality scorecards |
| P2 | [`p2_portal/`](p2_portal/) | Data-entry portal and admin review workflow |
| P3 | [`p3_scanner/`](p3_scanner/) | Lesion/image scanner models, evaluation, and export tooling |
| P4 | [`p4_early_warning/`](p4_early_warning/) | Early-warning engine: CUSUM detector, backtests, baselines, ablations, clade analysis |
| P5 | [`p5_dashboard/`](p5_dashboard/) | Surveillance dashboard |
| P6 | [`p6_api/`](p6_api/) | Public API, alert engine, and cross-border consumers (Benin, Cameroon, Niger) |

Supporting directories: [`shared/`](shared/), [`docs/`](docs/), [`admin/`](admin/),
[`dashboard/`](dashboard/), [`portal/`](portal/).

## Getting started

```bash
git clone https://github.com/EgwuSamuel/MpoXMirror.git
cd MpoXMirror
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                 # then fill in credentials
python test_connection.py                            # verify database connectivity
```

The stack uses PostgreSQL/PostGIS with SQLAlchemy and Alembic migrations, Apache Airflow
for scheduled ETL, and the standard scientific-Python stack (pandas, NumPy). See
[`requirements.txt`](requirements.txt) for the full dependency list and
[`.env.example`](.env.example) for required configuration.

## License

See repository settings for license details.
