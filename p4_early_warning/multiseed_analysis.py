"""
P4 — Multi-seed uncertainty analysis
====================================
Reviewer request (#5): single-run numbers from one XGBoost seed are fragile given
only ~40 positive test weeks. This driver re-fits every XGBoost-based configuration
under N random seeds and reports mean +/- SD, so the paper's headline metrics carry
honest uncertainty bands.

Deterministic systems (ARIMA, NB-GLM) do not depend on the seed and are computed
once; their SD is reported as 0 and noted as deterministic.

Covers:
  * Table 1  — three-system comparison (SmartMpox AUC/FAR/PPV across seeds; WF + 2024)
  * Table 2  — knowledge-based recovery on the held-out 2024 test (XGB vs +expert),
               plus the same expert layer with the digital stream removed (R3 off)
  * Table 3  — data-stream ablation (mean WF AUC per configuration; same XGBoost
               configuration as the main system)
  * Fig. 2   — per-fold walk-forward AUCs (mean over seeds)

Data: live warehouse or frozen snapshot (data_access.py; MPOX_DATA=frozen).

Run:  PYTHONIOENCODING=utf-8 python p4_early_warning/multiseed_analysis.py [--seeds 10]
Out:  p4_early_warning/models/multiseed_results.json
"""
import os, sys, json, argparse, warnings
import numpy as np
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

from expert_system import (
    load_features, evaluate_split, FEATURE_COLS, TARGET,
)
import baseline_comparison as bc
import ablation as ab


def msd(vals):
    """mean, sd (population-style ddof=1 when >1 sample) as rounded floats."""
    v = [x for x in vals if x is not None]
    if not v:
        return {"mean": None, "sd": None, "n": 0, "values": []}
    arr = np.asarray(v, dtype=float)
    sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return {"mean": round(float(arr.mean()), 4), "sd": round(sd, 4),
            "n": len(arr), "values": [round(float(x), 4) for x in arr]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    seeds = list(range(42, 42 + args.seeds))
    print(f"=== Multi-seed uncertainty analysis (seeds {seeds[0]}..{seeds[-1]}) ===")

    df = load_features()
    print(f"Loaded {len(df)} state-weeks ({df['epi_year'].min()}-{df['epi_year'].max()})")

    # ---------- deterministic baselines (computed once) --------------------------
    print("\n[1/3] Deterministic baselines (ARIMA, NB-GLM) — seed-independent ...")
    series_df = bc.load_case_series()
    arima_wf, nb_wf = {k: [] for k in ("auc", "false_alarm_rate", "ppv", "sensitivity", "sens_at_far10")}, \
                      {k: [] for k in ("auc", "false_alarm_rate", "ppv", "sensitivity", "sens_at_far10")}
    for train_max, test_year in bc.FOLDS:
        te = df[df["epi_year"] == test_year]
        tr = df[df["epi_year"] <= train_max]
        if len(te) == 0 or te[TARGET].sum() == 0:
            continue
        arima_test = bc.arima_scores_for_fold(series_df, train_max, test_year)
        arima_train = bc.arima_scores_for_fold(series_df, train_max - 1, train_max)
        tr_year = df[df["epi_year"] == train_max]
        thr = bc.youden_threshold(tr_year[TARGET].values.astype(int),
                                  bc._scores_for_rows(tr_year, arima_train))
        ra = bc.eval_arima(te, arima_test, thr)
        rn = bc.eval_nb_glm(tr, te)
        for k in arima_wf:
            arima_wf[k].append(ra.get(k)); nb_wf[k].append(rn.get(k))
    # 2024 clade fold (deterministic parts)
    tr23 = df[df["epi_year"] <= 2023]; te24 = df[df["epi_year"] == 2024]
    arima_test24 = bc.arima_scores_for_fold(series_df, 2023, 2024)
    arima_train24 = bc.arima_scores_for_fold(series_df, 2022, 2023)
    tr_year23 = df[df["epi_year"] == 2023]
    thr24 = bc.youden_threshold(tr_year23[TARGET].values.astype(int),
                                bc._scores_for_rows(tr_year23, arima_train24))
    arima_2024 = bc.eval_arima(te24, arima_test24, thr24)
    nb_2024 = bc.eval_nb_glm(tr23, te24)

    def det_summary(wf_lists, single2024):
        return {
            "wf": {k: round(float(np.mean([x for x in wf_lists[k] if x is not None])), 4)
                   for k in wf_lists},
            "clade_2024": {k: single2024.get(k) for k in
                           ("auc", "false_alarm_rate", "ppv", "sensitivity", "sens_at_far10")},
            "fold_auc": [None if x is None else round(float(x), 4) for x in wf_lists["auc"]],
            "seed_dependent": False,
        }

    arima_summary = det_summary(arima_wf, arima_2024)
    nb_summary = det_summary(nb_wf, nb_2024)
    print(f"    ARIMA  WF AUC={arima_summary['wf']['auc']}  2024 AUC={arima_2024['auc']}")
    print(f"    NB-GLM WF AUC={nb_summary['wf']['auc']}  2024 AUC={nb_2024['auc']}")

    # ---------- per-seed XGBoost / SmartMpox ------------------------------------
    print("\n[2/3] SmartMpox (XGBoost + expert layer) across seeds ...")
    sm_wf = {k: [] for k in ("auc", "false_alarm_rate", "ppv", "sensitivity", "sens_at_far10")}
    sm_2024 = {k: [] for k in ("auc", "false_alarm_rate", "ppv", "sensitivity", "sens_at_far10")}
    # Table 2 recovery on 2024 clade test
    rec = {"xgb_sens": [], "xgb_far": [], "xgb_ppv": [],
           "exp_sens": [], "exp_far": [], "exp_ppv": [],
           "recovered": [], "added_fa": [],
           "nodig_sens": [], "nodig_far": [], "nodig_ppv": [], "nodig_recovered": []}
    sm_fold_auc = []
    # Table 3 ablation
    abl = {name: [] for name in ab.ABLATION_GROUPS}

    for si, seed in enumerate(seeds, 1):
        # --- 3-system SmartMpox row: WF folds + 2024 ---
        fold_auc, fold_far, fold_ppv, fold_sens, fold_s10 = [], [], [], [], []
        for train_max, test_year in bc.FOLDS:
            te = df[df["epi_year"] == test_year]; tr = df[df["epi_year"] <= train_max]
            if len(te) == 0 or te[TARGET].sum() == 0:
                continue
            r = bc.eval_smartmpox(tr, te, train_max, seed=seed)
            fold_auc.append(r["auc"]); fold_far.append(r["false_alarm_rate"])
            fold_ppv.append(r["ppv"]); fold_sens.append(r["sensitivity"])
            fold_s10.append(r["sens_at_far10"])
        sm_fold_auc.append(fold_auc)
        sm_wf["auc"].append(np.mean([a for a in fold_auc if a is not None]))
        sm_wf["false_alarm_rate"].append(np.mean(fold_far))
        sm_wf["ppv"].append(np.mean(fold_ppv))
        sm_wf["sensitivity"].append(np.mean(fold_sens))
        sm_wf["sens_at_far10"].append(np.mean([s for s in fold_s10 if s is not None]))

        r24 = bc.eval_smartmpox(tr23, te24, 2023, seed=seed)
        for k in sm_2024:
            sm_2024[k].append(r24.get(k))

        # --- Table 2 recovery (2024 clade) ---
        ev = evaluate_split(tr23, te24, 2023, "clade", seed=seed)
        rec["xgb_sens"].append(ev["xgboost_alone"]["sensitivity"])
        rec["xgb_far"].append(ev["xgboost_alone"]["false_alarm_rate"])
        rec["xgb_ppv"].append(ev["xgboost_alone"]["ppv"])
        rec["exp_sens"].append(ev["expert_system"]["sensitivity"])
        rec["exp_far"].append(ev["expert_system"]["false_alarm_rate"])
        rec["exp_ppv"].append(ev["expert_system"]["ppv"])
        rec["recovered"].append(ev["recovered_outbreaks"])
        rec["added_fa"].append(ev["added_false_alarms"])
        nd = evaluate_split(tr23, te24, 2023, "clade-nodigital", seed=seed,
                            use_digital=False, verbose=False)
        rec["nodig_sens"].append(nd["expert_system"]["sensitivity"])
        rec["nodig_far"].append(nd["expert_system"]["false_alarm_rate"])
        rec["nodig_ppv"].append(nd["expert_system"]["ppv"])
        rec["nodig_recovered"].append(nd["recovered_outbreaks"])

        # --- Table 3 ablation (WF mean AUC per configuration) ---
        for group_name, remove_cols in ab.ABLATION_GROUPS.items():
            active = [c for c in ab.FEATURE_COLS if c not in remove_cols]
            aucs = []
            for fold in ab.WALK_FORWARD_FOLDS:
                a = ab.run_fold(df, fold["train_max"], fold["test_year"], active, seed=seed)
                if a is not None:
                    aucs.append(a)
            abl[group_name].append(float(np.mean(aucs)) if aucs else None)

        print(f"    seed {seed} ({si}/{len(seeds)}): "
              f"SM WF AUC={sm_wf['auc'][-1]:.3f}  2024 AUC={r24['auc']:.3f}  "
              f"expert sens(2024)={ev['expert_system']['sensitivity']:.3f}")

    # ---------- assemble ---------------------------------------------------------
    print("\n[3/3] Summarising ...")
    smartmpox_summary = {
        "wf": {k: msd(sm_wf[k]) for k in sm_wf},
        "fold_auc_mean_over_seeds": [round(float(np.mean(col)), 4)
                                     for col in zip(*sm_fold_auc)],
        "clade_2024": {k: msd(sm_2024[k]) for k in sm_2024},
        "seed_dependent": True,
    }
    recovery_summary = {
        "xgboost_alone": {"sensitivity": msd(rec["xgb_sens"]),
                          "false_alarm_rate": msd(rec["xgb_far"]),
                          "ppv": msd(rec["xgb_ppv"])},
        "expert_system": {"sensitivity": msd(rec["exp_sens"]),
                          "false_alarm_rate": msd(rec["exp_far"]),
                          "ppv": msd(rec["exp_ppv"])},
        "recovered_outbreaks": msd(rec["recovered"]),
        "added_false_alarms": msd(rec["added_fa"]),
        "expert_system_no_digital": {"sensitivity": msd(rec["nodig_sens"]),
                                     "false_alarm_rate": msd(rec["nodig_far"]),
                                     "ppv": msd(rec["nodig_ppv"]),
                                     "recovered_outbreaks": msd(rec["nodig_recovered"])},
    }
    ablation_summary = {name: msd(vals) for name, vals in abl.items()}

    out = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "n_seeds": len(seeds), "seeds": seeds,
        "note": ("XGBoost/MpoxMirror metrics are mean +/- SD across seeds; ARIMA and "
                 "NB-GLM are deterministic (seed-independent) and reported as single values. "
                 "Walk-forward means are over the folds with positives (test 2021-2023)."),
        "data_mode": __import__("data_access").mode(),
        "social_snapshot_cutoff": __import__("data_access").SOCIAL_SNAPSHOT_CUTOFF,
        "table1_three_system": {
            "arima": arima_summary, "nb_glm": nb_summary,
            "smartmpox": smartmpox_summary,
        },
        "table2_recovery_2024_clade": recovery_summary,
        "table3_ablation_wf": ablation_summary,
    }
    os.makedirs(os.path.join(os.path.dirname(__file__), "models"), exist_ok=True)
    path = os.path.join(os.path.dirname(__file__), "models", "multiseed_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    # console summary
    print("\n" + "=" * 78)
    print(f"  MULTI-SEED SUMMARY  (n={len(seeds)} seeds)")
    print("=" * 78)
    s = smartmpox_summary
    print(f"  SmartMpox  WF AUC   = {s['wf']['auc']['mean']} +/- {s['wf']['auc']['sd']}")
    print(f"  SmartMpox  2024 AUC = {s['clade_2024']['auc']['mean']} +/- {s['clade_2024']['auc']['sd']}")
    print(f"  SmartMpox  2024 FAR = {s['clade_2024']['false_alarm_rate']['mean']} +/- {s['clade_2024']['false_alarm_rate']['sd']}")
    print(f"  SmartMpox  2024 PPV = {s['clade_2024']['ppv']['mean']} +/- {s['clade_2024']['ppv']['sd']}")
    r = recovery_summary
    print(f"  Expert  sens(2024)  = {r['expert_system']['sensitivity']['mean']} +/- {r['expert_system']['sensitivity']['sd']}"
          f"  (XGB alone {r['xgboost_alone']['sensitivity']['mean']} +/- {r['xgboost_alone']['sensitivity']['sd']})")
    print(f"  Recovered outbreaks = {r['recovered_outbreaks']['mean']} +/- {r['recovered_outbreaks']['sd']} of 40")
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
