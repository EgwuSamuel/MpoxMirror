"""
P4 Clinical Utility — Decision Curve Analysis on the held-out 2024 year
======================================================================
Net benefit of acting on MpoxMirror's alerts versus alerting on every state
(treat-all) or on none (treat-none), evaluated on the SAME held-out test used
everywhere else in the paper:

  * test set  : all 2024 state-weeks (n = 1,924)
  * outcome   : target_outbreak_4w (the four-week-ahead label the system predicts)
  * model     : system frozen on data <= 2023 (no 2024 information)
  * seeds     : mean over the same 10 XGBoost seeds as multiseed_analysis.py

Two curves are reported:
  1. XGBoost probability thresholded at each decision threshold p_t. The
     probabilities are trained with class weighting and are NOT calibrated, so
     p_t here is a score cut-off rather than a calibrated risk.
  2. The MpoxMirror alert rule itself (XGBoost + corroboration gate). It is a
     fixed binary decision, so its net benefit at p_t is
         NB(p_t) = TP/N - FP/N * p_t / (1 - p_t),
     which does not depend on probability calibration (Vickers & Elkin, 2006).

Run:    python p4_early_warning/clinical_utility.py
Output: p4_early_warning/models/clinical_utility_results.json
"""
import os, sys, json, warnings
import numpy as np
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
from expert_system import FEATURE_COLS, TARGET, load_features, build_engine

SEEDS = list(range(42, 52))
THRESHOLDS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
              0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def net_benefit(y, alert, pt):
    y = np.asarray(y).astype(int); a = np.asarray(alert).astype(bool)
    n = len(y)
    tp = int((a & (y == 1)).sum()); fp = int((a & (y == 0)).sum())
    return tp / n - fp / n * pt / (1 - pt)


def main():
    print("=== Decision-curve analysis (held-out 2024, 4-week target) ===")
    df = load_features()
    tr, te = df[df["epi_year"] <= 2023], df[df["epi_year"] == 2024]
    y = te[TARGET].values.astype(int)
    prev = float(y.mean())
    X = te[FEATURE_COLS].fillna(0).values.astype(np.float32)
    print(f"Test 2024: n={len(y)}  positives={y.sum()}  prevalence={prev:.4f}")

    nb_prob = {t: [] for t in THRESHOLDS}
    nb_rule = {t: [] for t in THRESHOLDS}
    rule_tp, rule_fp = [], []
    for seed in SEEDS:
        model, engine, _ = build_engine(tr, 2023, seed=seed)
        probs = model.predict_proba(X)[:, 1]
        alert = engine.infer_frame(te, probs)["expert_alert"].values.astype(bool)
        rule_tp.append(int((alert & (y == 1)).sum())); rule_fp.append(int((alert & (y == 0)).sum()))
        for t in THRESHOLDS:
            nb_prob[t].append(net_benefit(y, probs >= t, t))
            nb_rule[t].append(net_benefit(y, alert, t))

    curve = []
    print(f"\n  {'p_t':>5} {'XGB prob':>10} {'MpoxMirror':>11} {'treat-all':>10}")
    for t in THRESHOLDS:
        all_nb = prev - (1 - prev) * t / (1 - t)
        row = {
            "threshold": t,
            "model_prob_nb": round(float(np.mean(nb_prob[t])), 5),
            "model_prob_nb_sd": round(float(np.std(nb_prob[t], ddof=1)), 5),
            "mpoxmirror_rule_nb": round(float(np.mean(nb_rule[t])), 5),
            "mpoxmirror_rule_nb_sd": round(float(np.std(nb_rule[t], ddof=1)), 5),
            "treat_all_nb": round(all_nb, 5),
            "treat_none_nb": 0.0,
        }
        row["rule_preferred"] = bool(row["mpoxmirror_rule_nb"] > max(all_nb, 0.0))
        row["prob_preferred"] = bool(row["model_prob_nb"] > max(all_nb, 0.0))
        curve.append(row)
        print(f"  {t:>5.2f} {row['model_prob_nb']:>+10.5f} {row['mpoxmirror_rule_nb']:>+11.5f} {all_nb:>+10.5f}")

    tp, fp = float(np.mean(rule_tp)), float(np.mean(rule_fp))
    breakeven = tp / (tp + fp)       # NB(rule) > 0  <=>  p_t < TP / (TP + FP)
    out = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_set": "held-out 2024 state-weeks; outcome target_outbreak_4w; model frozen on <=2023",
        "n": int(len(y)), "positives": int(y.sum()), "prevalence": round(prev, 4),
        "seeds": SEEDS,
        "mpoxmirror_rule_mean_tp": tp, "mpoxmirror_rule_mean_fp": fp,
        "mpoxmirror_rule_positive_nb_below_threshold": round(breakeven, 4),
        "treat_all_positive_nb_below_threshold": round(prev, 4),
        "dca_curve": curve,
        "note": ("XGBoost probabilities are class-weighted and uncalibrated; the "
                 "MpoxMirror-rule curve is calibration-free."),
    }
    print(f"\nMpoxMirror rule: mean TP={tp:.1f} FP={fp:.1f} -> positive net benefit for p_t < {breakeven:.3f}"
          f" (treat-all only for p_t < {prev:.3f})")
    os.makedirs(os.path.join(os.path.dirname(__file__), "models"), exist_ok=True)
    path = os.path.join(os.path.dirname(__file__), "models", "clinical_utility_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
