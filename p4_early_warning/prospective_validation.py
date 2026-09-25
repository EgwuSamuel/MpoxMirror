"""
P4 Task 10 — Simulated Real-Time (Forward-Chaining) Prospective Validation
==========================================================================
Freezes the ENTIRE system on data <= 2023, then replays 2024
week-by-week, using only information available at each decision
point, and measures how far ahead of NCDC laboratory confirmation the system
alerts.

This is a *simulated* prospective test (retrospective replay under strict temporal
freezing), not a live real-time deployment — reported honestly as such. It is the
standard, defensible way to estimate prospective lead time before real forward data
accrue.

Two alerting channels are evaluated separately and honestly:
  A. Digital surveillance (national)  — first threshold crossing as a function of
     the alert threshold, with the number of posts in that week that mention
     Nigeria (the digital stream is national and not Nigeria-specific).
  B. Case-based ML + expert layer (state) — does a model trained on 2017-2023 pre-empt
     the 2024 reports at all? (tests the temporal-shift limitation directly).

Reference date: first confirmed case in the ingested 2024 NCDC reports (surveillance_weekly).

Run:  python p4_early_warning/prospective_validation.py
Out:  p4_early_warning/models/prospective_validation_results.json
"""
import os, sys, json, math
from datetime import date, datetime, timezone
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_access as da
from expert_system import (
    FEATURE_COLS, load_features, add_derived, build_engine,
    load_national_digital_signal, base_tier,
)

FREEZE_YEAR = 2023          # system frozen on <= this year
TEST_YEAR   = 2024          # replay this year forward


def iso_monday(year, week):
    try:
        return date.fromisocalendar(int(year), int(week), 1)
    except Exception:
        return None


# ───────── ground truth: first NCDC confirmation in the ingested reports ─────────
def ncdc_confirmations():
    return da.first_confirmations(TEST_YEAR)


# ───────── channel A: digital lead-time vs threshold ─────────
def digital_weekly_counts():
    """{iso_week: (n_relevant_posts, n_mentioning_nigeria)} for TEST_YEAR."""
    wk = da.load_social_weekly()
    wk = wk[wk["iso_year"] == TEST_YEAR]
    return {int(r.iso_week): (int(r.n), int(r.n_nigeria)) for r in wk.itertuples()}


def digital_lead_analysis(national_ncdc_date):
    # frozen surge threshold from <=2023 posts (same as the deployed expert engine)
    _, frozen_p75 = load_national_digital_signal(FREEZE_YEAR)
    counts = digital_weekly_counts()
    weeks = sorted(counts)
    results = []
    for thr in sorted({1.0, 2.0, float(frozen_p75), 8.0}):
        first_wk = next((w for w in weeks if counts[w][0] >= thr), None)
        alert_date = iso_monday(TEST_YEAR, first_wk) if first_wk else None
        lead = (national_ncdc_date - alert_date).days if alert_date else None
        results.append({
            "threshold_posts_per_week": thr,
            "is_frozen_operating_point": thr == float(frozen_p75),
            "first_alert_week": first_wk,
            "first_alert_date": alert_date.isoformat() if alert_date else None,
            "posts_in_alert_week": counts[first_wk][0] if first_wk else None,
            "posts_mentioning_nigeria_in_alert_week": counts[first_wk][1] if first_wk else None,
            "lead_days_vs_first_ingested_confirmation": lead,
        })
    return {"frozen_surge_threshold": frozen_p75,
            "social_snapshot_cutoff": da.SOCIAL_SNAPSHOT_CUTOFF,
            "by_threshold": results,
            "weekly_counts_2024": {w: c[0] for w, c in counts.items()},
            "weekly_nigeria_mentions_2024": {w: c[1] for w, c in counts.items()}}


# ───────── channel B: case-based ML + expert forward replay ─────────
def load_test_year_features():
    """All 2024 state-weeks (features only; target not required for alerting)."""
    df = da.load_feature_rows()
    return add_derived(df[df["epi_year"] == TEST_YEAR].sort_values(["epi_week", "state_id"])
                       .reset_index(drop=True))


def _first_alert_per_state(test, probs, threshold_fn):
    first_alert = {}
    for i, (_, row) in enumerate(test.iterrows()):
        if threshold_fn(float(probs[i])):
            sid, wk = int(row["state_id"]), int(row["epi_week"])
            if sid not in first_alert or wk < first_alert[sid]:
                first_alert[sid] = wk
    return first_alert


def _lead_stats(per_state_ncdc, first_alert):
    leads = []
    for sid, info in per_state_ncdc.items():
        conf_wk = info["week"]
        alert_wk = first_alert.get(sid)
        leads.append({"state_id": sid, "ncdc_confirm_week": conf_wk,
                      "ml_alert_week": alert_wk,
                      "lead_weeks": (conf_wk - alert_wk) if alert_wk is not None else None})
    return leads


def model_forward_replay(per_state_ncdc):
    labelled = load_features()
    frozen = labelled[labelled["epi_year"] <= FREEZE_YEAR]
    model, engine, meta = build_engine(frozen, FREEZE_YEAR)
    youden = meta["youden"]

    test = load_test_year_features()
    X = test[FEATURE_COLS].fillna(0).values.astype(np.float32)
    probs = model.predict_proba(X)[:, 1]

    national_week = min(v["week"] for v in per_state_ncdc.values())  # week 34

    def alarm_context(threshold_fn, label):
        """How non-specific is a 'pre-emption' claim under this threshold?"""
        pre = test.reset_index(drop=True)
        mask = np.array([threshold_fn(float(p)) for p in probs])
        pre_window = pre["epi_week"].values < national_week
        alerted_states = set(pre["state_id"].values[mask & pre_window].astype(int))
        first_alert = _first_alert_per_state(test, probs, threshold_fn)
        leads = _lead_stats(per_state_ncdc, first_alert)
        before = [l for l in leads if l["lead_weeks"] is not None and l["lead_weeks"] > 0]
        n_alerted_sw = int((mask & pre_window).sum())
        n_total_sw   = int(pre_window.sum())
        return {
            "threshold": label,
            "distinct_states_alerted_pre_confirmation": len(alerted_states),
            "of_total_states": int(pre["state_id"].nunique()),
            "pre_confirmation_alert_rate": round(n_alerted_sw / n_total_sw, 4) if n_total_sw else None,
            "n_confirmed_states_alerted_before": len(before),
            "per_state": sorted(leads, key=lambda x: x["ncdc_confirm_week"]),
        }

    deployed = alarm_context(lambda p: base_tier(p) in ("red", "critical"),
                             "deployed tiers (red>=0.20)")
    calibrated = alarm_context(lambda p: p >= youden,
                               f"calibrated Youden (>={round(youden,3)})")

    return {"engine_thresholds": meta,
            "n_confirmed_states": len(per_state_ncdc),
            "national_confirmation_week": national_week,
            "deployed_threshold": deployed,
            "calibrated_threshold": calibrated,
            "note": ("A high 'alerted before confirmation' count under the lenient deployed "
                     "tiers is a NON-SPECIFIC standing prior, not detection: the same "
                     "threshold alarms most states all year. The calibrated Youden threshold "
                     "shows the genuine (near-zero) prospective skill of a 2017-2023-trained "
                     "case model in 2024.")}


def main():
    print("=== Simulated Real-Time Prospective Validation (freeze<=2023, replay 2024) ===")
    per_state_ncdc, national_ncdc = ncdc_confirmations()
    print(f"NCDC national first confirmation 2024: {national_ncdc} "
          f"({len(per_state_ncdc)} states confirmed during 2024)")

    print("\n── Channel A: digital surveillance lead time vs alert threshold ──")
    dig = digital_lead_analysis(national_ncdc)
    print(f"  Frozen surge threshold (<=2023 p75) = {dig['frozen_surge_threshold']} posts/week")
    for r in dig["by_threshold"]:
        star = "  <-- frozen operating point" if r["is_frozen_operating_point"] else ""
        print(f"    thr>={r['threshold_posts_per_week']:>4} posts: first crossing "
              f"{r['first_alert_date']} (wk {r['first_alert_week']}, "
              f"{r['posts_mentioning_nigeria_in_alert_week']}/{r['posts_in_alert_week']} posts mention Nigeria) -> "
              f"nominal lead {r['lead_days_vs_first_ingested_confirmation']} days{star}")

    print("\n── Channel B: case-based ML + expert forward replay ──")
    mdl = model_forward_replay(per_state_ncdc)
    print(f"  States confirmed in 2024: {mdl['n_confirmed_states']} | "
          f"national confirmation week {mdl['national_confirmation_week']}")
    for key in ("deployed_threshold", "calibrated_threshold"):
        d = mdl[key]
        print(f"  [{d['threshold']}]")
        print(f"     distinct states alerted pre-confirmation: "
              f"{d['distinct_states_alerted_pre_confirmation']}/{d['of_total_states']} "
              f"(pre-confirm alert rate {d['pre_confirmation_alert_rate']})")
        print(f"     confirmed states 'alerted before': {d['n_confirmed_states_alerted_before']}")
    print("  => the lenient count is a non-specific standing prior; the calibrated")
    print("     threshold shows the true (near-zero) case-based prospective skill.")

    frozen_row = next(r for r in dig["by_threshold"] if r["is_frozen_operating_point"])
    report_weeks = da.surveillance_weeks(TEST_YEAR)
    print("\n=== PROSPECTIVE SUMMARY ===")
    print(f"  Frozen operating point (>= {frozen_row['threshold_posts_per_week']} posts/week): first "
          f"crossing wk {frozen_row['first_alert_week']} with "
          f"{frozen_row['posts_mentioning_nigeria_in_alert_week']}/{frozen_row['posts_in_alert_week']} "
          f"posts mentioning Nigeria")
    print(f"  {TEST_YEAR} situation reports ingested for epi-weeks {report_weeks[0]}-{report_weeks[-1]} only"
          f" -> 'first confirmation' is the first ingested report, not the true first case.")

    out = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "method": ("Simulated real-time (forward-chaining) replay: system frozen on "
                   "<=2023, replayed weekly through 2024 with no look-ahead. Not a live "
                   "real-time deployment."),
        "ncdc_first_ingested_confirmation": national_ncdc.isoformat(),
        "surveillance_weeks_ingested_2024": report_weeks,
        "channel_a_digital": dig,
        "channel_b_model_expert": mdl,
        "summary": {
            "frozen_threshold_first_crossing_week": frozen_row["first_alert_week"],
            "frozen_threshold_nominal_lead_days": frozen_row["lead_days_vs_first_ingested_confirmation"],
            "frozen_threshold_crossing_posts_mentioning_nigeria":
                frozen_row["posts_mentioning_nigeria_in_alert_week"],
            "interpretation": (
                ("The frozen-threshold crossing is driven by posts that do not mention "
                 "Nigeria. " if frozen_row["posts_mentioning_nigeria_in_alert_week"] == 0 else "")
                + "The digital stream is a national (continental-news) chatter count, so a "
                "Nigeria-specific lead time cannot be claimed from it. The reference date is "
                f"the first ingested {TEST_YEAR} situation report (reports exist only for "
                f"epi-weeks {report_weeks[0]}-{report_weeks[-1]}), not the true first "
                "confirmation. The case-based ML shows little genuine prospective skill."),
        },
    }
    os.makedirs("p4_early_warning/models", exist_ok=True)
    path = "p4_early_warning/models/prospective_validation_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
