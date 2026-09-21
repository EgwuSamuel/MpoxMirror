# MpoxMirror

A digital early-warning system for mpox in Nigeria.

MpoxMirror combines open health, genomic, and environmental data into one pipeline
that flags mpox outbreaks earlier than traditional case reporting.

Formerly `smartmpox-nigeria`.

## What's inside

| Folder | Purpose |
|--------|---------|
| [`p1_warehouse/`](p1_warehouse/) | Data pipelines and storage |
| [`p2_portal/`](p2_portal/) | Data-entry portal with admin review |
| [`p3_scanner/`](p3_scanner/) | Lesion image models |
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
