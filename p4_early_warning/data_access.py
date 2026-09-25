"""
Shared data access for the P4 analyses — live warehouse OR frozen snapshot.

Every analysis in the manuscript reads its inputs through this module, so the
paper can be reproduced either against the Supabase warehouse or, without any
credentials, from the frozen CSV snapshot committed in p4_early_warning/frozen/.

Mode selection (env var MPOX_DATA):
  * "frozen" — read p4_early_warning/frozen/*.csv (no database needed)
  * "db"     — read the live warehouse via DATABASE_URL
  * unset    — "db" if DATABASE_URL is set, otherwise "frozen"

The P3 scanner keeps running and back-fills posts with old publication dates, so
the digital stream is pinned to SOCIAL_SNAPSHOT_CUTOFF (posts scraped on or
before it). Without the pin, the training-fitted surge threshold drifts between
runs and the digital-stream results cannot be reproduced.

Create/refresh the snapshot:  python p4_early_warning/export_frozen_snapshot.py
"""
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN_DIR = os.path.join(HERE, "frozen")

# Scrape-time pin for social_media_signals (the snapshot used by the 10-seed
# analysis reported in the manuscript).
SOCIAL_SNAPSHOT_CUTOFF = "2026-09-21 08:05:52+00"

FEATURE_COLUMNS = [
    "state_id", "state_code", "epi_year", "epi_week", "week_start_date",
    "cases_t1", "cases_t2", "cases_t4",
    "cases_rolling4w_mean", "cases_rolling8w_mean", "cases_log1p",
    "rainfall_t2_mm", "rainfall_t4_mm", "temp_mean_t1_c",
    "reservoir_risk_index", "is_border_state", "neighbour_cases_t1",
    "target_outbreak_4w", "target_cases_4w",
]


def mode() -> str:
    m = os.getenv("MPOX_DATA", "").strip().lower()
    if m in ("frozen", "db"):
        return m
    return "db" if os.getenv("DATABASE_URL") else "frozen"


def _conn():
    import psycopg2
    return psycopg2.connect(os.getenv("DATABASE_URL"))


def _query(sql, params=None) -> pd.DataFrame:
    conn = _conn(); cur = conn.cursor()
    cur.execute(sql, params)
    cols = [d[0] for d in cur.description]
    df = pd.DataFrame(cur.fetchall(), columns=cols)
    cur.close(); conn.close()
    return df


def _frozen(name) -> pd.DataFrame:
    # keep_default_na=False: the state code for Nasarawa is literally "NA".
    return pd.read_csv(os.path.join(FROZEN_DIR, name), keep_default_na=False,
                       na_values=[""])


# ─────────────────────────── feature store ───────────────────────────────────
def _all_feature_rows() -> pd.DataFrame:
    """Every features_weekly row (complete or not), with state_code and is_complete."""
    if mode() == "frozen":
        return _frozen("features_weekly.csv")
    return _query("""
        SELECT f.state_id, r.state_code, f.epi_year, f.epi_week, f.week_start_date,
               f.cases_t1, f.cases_t2, f.cases_t4,
               f.cases_rolling4w_mean, f.cases_rolling8w_mean, f.cases_log1p,
               f.rainfall_t2_mm, f.rainfall_t4_mm, f.temp_mean_t1_c,
               f.reservoir_risk_index,
               f.is_border_state::INT AS is_border_state,
               f.neighbour_cases_t1,
               f.target_outbreak_4w::INT AS target_outbreak_4w, f.target_cases_4w,
               f.is_complete::INT AS is_complete
        FROM features_weekly f
        JOIN ref_states r ON r.state_id = f.state_id
        ORDER BY f.epi_year, f.epi_week, f.state_id
    """)


def load_feature_rows() -> pd.DataFrame:
    """All complete state-weeks (labelled and unlabelled), with state_code."""
    df = _all_feature_rows()
    df = df[pd.to_numeric(df["is_complete"]).astype(int) == 1][FEATURE_COLUMNS].copy()
    num = [c for c in FEATURE_COLUMNS if c not in ("state_code", "week_start_date")]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("state_id", "epi_year", "epi_week", "is_border_state"):
        df[c] = df[c].astype(int)
    df["week_start_date"] = pd.to_datetime(df["week_start_date"]).dt.date
    return df.sort_values(["epi_year", "epi_week", "state_id"]).reset_index(drop=True)


def load_labelled_features() -> pd.DataFrame:
    df = load_feature_rows()
    return df[df["target_outbreak_4w"].notna()].reset_index(drop=True)


# ─────────────────────────── surveillance ────────────────────────────────────
def load_surveillance() -> pd.DataFrame:
    """surveillance_weekly: one row per ingested state-week report."""
    if mode() == "frozen":
        df = _frozen("surveillance_weekly.csv")
    else:
        df = _query("""
            SELECT state_id, epi_year, epi_week, week_start_date,
                   COALESCE(total_cases, 0) AS total_cases,
                   COALESCE(confirmed, 0)   AS confirmed
            FROM surveillance_weekly
            ORDER BY epi_year, epi_week, state_id
        """)
    for c in ("state_id", "epi_year", "epi_week", "total_cases", "confirmed"):
        df[c] = pd.to_numeric(df[c]).astype(int)
    df["week_start_date"] = pd.to_datetime(df["week_start_date"]).dt.date
    return df


def load_case_series() -> pd.DataFrame:
    """Full state-week grid with reported cases (implicit zeros filled)."""
    # The full grid (incomplete rows included), as the ARIMA baseline always used.
    grid = _all_feature_rows()[["state_id", "epi_year", "epi_week"]].astype(int)
    sw = load_surveillance()[["state_id", "epi_year", "epi_week", "total_cases"]]
    df = grid.merge(sw, on=["state_id", "epi_year", "epi_week"], how="left")
    df["cases"] = df["total_cases"].fillna(0).astype(int)
    return (df[["state_id", "epi_year", "epi_week", "cases"]]
            .sort_values(["state_id", "epi_year", "epi_week"]).reset_index(drop=True))


def first_confirmations(year: int):
    """(per_state {state_id: {week, date}}, national first date) for a year."""
    sw = load_surveillance()
    s = sw[(sw["epi_year"] == year) & (sw["confirmed"] > 0)]
    per_state = {int(sid): {"week": int(g["epi_week"].min()),
                            "date": g["week_start_date"].min()}
                 for sid, g in s.groupby("state_id")}
    national = s["week_start_date"].min() if len(s) else None
    return per_state, national


def surveillance_weeks(year: int):
    """Epi-weeks of a year for which any situation report was ingested."""
    sw = load_surveillance()
    return sorted(sw.loc[sw["epi_year"] == year, "epi_week"].unique().tolist())


# ─────────────────────────── digital stream ──────────────────────────────────
def load_social_weekly() -> pd.DataFrame:
    """
    Weekly count of mpox-relevant posts (ISO year/week of publication), pinned to
    SOCIAL_SNAPSHOT_CUTOFF. n_nigeria = posts whose title/text mention Nigeria or
    carry a state tag (descriptive only; not used by the model).
    """
    if mode() == "frozen":
        df = _frozen("social_weekly_counts.csv")
    else:
        df = _query("""
            SELECT EXTRACT(ISOYEAR FROM published_at)::int AS iso_year,
                   EXTRACT(WEEK    FROM published_at)::int AS iso_week,
                   COUNT(*) AS n,
                   COUNT(*) FILTER (WHERE state_id IS NOT NULL
                                       OR title ILIKE '%%nigeria%%'
                                       OR content_snippet ILIKE '%%nigeria%%'
                                       OR full_text ILIKE '%%nigeria%%') AS n_nigeria
            FROM social_media_signals
            WHERE is_mpox_relevant = TRUE AND published_at IS NOT NULL
              AND scraped_at <= %s
            GROUP BY 1, 2 ORDER BY 1, 2
        """, (SOCIAL_SNAPSHOT_CUTOFF,))
    return df.astype(int)


def load_social_languages() -> pd.DataFrame:
    """Posts per detected language (all vs mpox-relevant), pinned to the cutoff."""
    if mode() == "frozen":
        return _frozen("social_languages.csv")
    return _query("""
        SELECT detected_language, COUNT(*) AS n_all,
               COUNT(*) FILTER (WHERE is_mpox_relevant) AS n_relevant
        FROM social_media_signals WHERE scraped_at <= %s
        GROUP BY 1 ORDER BY 3 DESC
    """, (SOCIAL_SNAPSHOT_CUTOFF,))


# ─────────────────────────── GBIF sampling effort ────────────────────────────
EFFORT_FILE = os.path.join(FROZEN_DIR, "gbif_rodentia_effort_by_state.json")


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["week_sin"]       = np.sin(2 * np.pi * df["epi_week"] / 52)
    df["week_cos"]       = np.cos(2 * np.pi * df["epi_week"] / 52)
    df["cases_velocity"] = df["cases_t1"] - df["cases_t2"]
    df["cases_accel"]    = (df["cases_t1"] - df["cases_t2"]) - (df["cases_t2"] - df["cases_t4"]) / 2
    return df
