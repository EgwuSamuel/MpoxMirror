"""
P4 — Reservoir suitability: sampling-effort robustness (reviewer #4)
==================================================================
The reservoir_risk_index is an occurrence-DENSITY proxy: log-normalised counts of
Cricetomys gambianus + Funisciurus spp. GBIF records per state. A fair criticism
is that GBIF occurrence counts partly track COLLECTOR EFFORT (better-surveyed /
more accessible states accrue more records of everything), which could also
correlate with case detection -- i.e. the IRR=4.49 might be a sampling artefact.

This script runs a "target-group background" bias check: it fetches the broader
GBIF sampling effort for the whole order RODENTIA in Nigeria (same collection
methods, so a proxy for effort independent of the focal reservoir taxa), then
refits the interpretable NB-GLM WITH log(effort) as an extra covariate. If the
reservoir IRR remains >1 and significant after controlling for effort, the signal
is not purely a sampling artefact. We also report rank correlations.

Run:  python p4_early_warning/reservoir_robustness.py
Out:  p4_early_warning/models/reservoir_robustness.json
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "p1_warehouse", "etl"))
import data_access as da
from negbin import design, fit_negbin

NB_PREDICTORS = [
    "cases_t1", "cases_velocity", "rainfall_t2_mm", "temp_mean_t1_c",
    "reservoir_risk_index", "is_border_state", "neighbour_cases_t1",
    "week_sin", "week_cos",
]
TARGET_COUNT = "target_cases_4w"


def lookup_order_key(name="Rodentia") -> int:
    from gbif_rodent_etl import GBIF_BASE, HEADERS
    r = requests.get(f"{GBIF_BASE}/species/match",
                     params={"name": name, "rank": "ORDER"},
                     headers=HEADERS, timeout=30)
    r.raise_for_status()
    d = r.json()
    key = d.get("orderKey") or d.get("usageKey")
    print(f"  {name} -> GBIF key {key}")
    return key


# Frozen GBIF effort counts (committed); fetched from GBIF only if absent.
_EFFORT_CACHE = da.EFFORT_FILE


def state_effort_from_gbif() -> dict:
    """Per-state count of all Rodentia occurrence records in Nigeria (collector effort proxy)."""
    if os.path.exists(_EFFORT_CACHE):
        with open(_EFFORT_CACHE) as f:
            counts = json.load(f)
        print(f"  [cache] Rodentia effort loaded for {len(counts)} states "
              f"(total {sum(counts.values())})")
        return counts
    from gbif_rodent_etl import REQUEST_DELAY, assign_state, fetch_occurrences
    key = lookup_order_key("Rodentia")
    time.sleep(REQUEST_DELAY)
    recs = fetch_occurrences(key, "taxonKey", country="NG")
    print(f"  Rodentia NGA records fetched: {len(recs)}")
    counts = {}
    for rec in recs:
        code = assign_state(rec, {})
        if code:
            counts[code] = counts.get(code, 0) + 1
    print(f"  States with effort: {len(counts)} | total assigned: {sum(counts.values())}")
    os.makedirs(os.path.dirname(_EFFORT_CACHE), exist_ok=True)
    with open(_EFFORT_CACHE, "w") as f:
        json.dump(counts, f)
    return counts


def load_features_with_state_code() -> pd.DataFrame:
    df = da.add_derived(da.load_labelled_features())
    return df[df["epi_year"] <= 2022].reset_index(drop=True)


def fit_nb(df, predictors):
    y = df[TARGET_COUNT].fillna(0).astype(np.float64).values
    return fit_negbin(y, design(df, predictors))


def irr_row(res, name):
    if name not in res.params.index:
        return None
    ci = res.conf_int()
    return {"irr": float(np.exp(res.params[name])),
            "lo95": float(np.exp(ci.loc[name, 0])),
            "hi95": float(np.exp(ci.loc[name, 1])),
            "pvalue": float(res.pvalues[name])}


def main():
    print("=== Reservoir suitability: sampling-effort robustness ===")
    effort = state_effort_from_gbif()

    df = load_features_with_state_code()

    df["rodentia_effort"] = df["state_code"].map(effort).fillna(0).astype(float)
    df["log_effort"] = np.log1p(df["rodentia_effort"])

    # --- per-state summary for correlations (one row per state) ---
    per_state = (df.groupby("state_code")
                 .agg(reservoir=("reservoir_risk_index", "first"),
                      effort=("rodentia_effort", "first"),
                      total_cases=("cases_t1", "sum"))
                 .reset_index())
    rho_eff, p_eff = spearmanr(per_state["reservoir"], per_state["effort"])
    rho_cas, p_cas = spearmanr(per_state["reservoir"], per_state["total_cases"])

    print(f"\n  Spearman rho(reservoir, rodentia_effort) = {rho_eff:.3f} (p={p_eff:.3g})")
    print(f"  Spearman rho(reservoir, total_cases)     = {rho_cas:.3f} (p={p_cas:.3g})")

    # --- NB-GLM before / after controlling for effort ---
    res_base = fit_nb(df, NB_PREDICTORS)
    res_ctrl = fit_nb(df, NB_PREDICTORS + ["log_effort"])

    base_irr = irr_row(res_base, "reservoir_risk_index")
    ctrl_irr = irr_row(res_ctrl, "reservoir_risk_index")
    eff_irr = irr_row(res_ctrl, "log_effort")
    alphas = {"baseline": float(res_base.params["alpha"]),
              "effort_controlled": float(res_ctrl.params["alpha"])}

    print("\n  Reservoir IRR (baseline model)          : "
          f"{base_irr['irr']:.3f} ({base_irr['lo95']:.3f}-{base_irr['hi95']:.3f}), p={base_irr['pvalue']:.2g}")
    print("  Reservoir IRR (+ log sampling effort)   : "
          f"{ctrl_irr['irr']:.3f} ({ctrl_irr['lo95']:.3f}-{ctrl_irr['hi95']:.3f}), p={ctrl_irr['pvalue']:.2g}")
    print("  Sampling-effort IRR (in controlled model): "
          f"{eff_irr['irr']:.3f} ({eff_irr['lo95']:.3f}-{eff_irr['hi95']:.3f}), p={eff_irr['pvalue']:.2g}")

    survives = ctrl_irr["irr"] > 1.0 and ctrl_irr["pvalue"] < 0.05
    print(f"\n  Reservoir signal survives effort control: {'YES' if survives else 'NO'}")

    out = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "method": ("Target-group background bias check: order Rodentia GBIF records "
                   "per state as a collector-effort proxy; NB-GLM refit with log(effort)."),
        "rodentia_effort_by_state": effort,
        "n_states_with_effort": len(effort),
        "total_rodentia_records": int(sum(effort.values())),
        "spearman_reservoir_vs_effort": {"rho": round(float(rho_eff), 4), "p": float(p_eff)},
        "spearman_reservoir_vs_totalcases": {"rho": round(float(rho_cas), 4), "p": float(p_cas)},
        "nb_alpha_mle": alphas,
        "reservoir_irr_baseline": base_irr,
        "reservoir_irr_effort_controlled": ctrl_irr,
        "sampling_effort_irr": eff_irr,
        "reservoir_survives_effort_control": bool(survives),
    }
    os.makedirs(os.path.join(os.path.dirname(__file__), "models"), exist_ok=True)
    path = os.path.join(os.path.dirname(__file__), "models", "reservoir_robustness.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
