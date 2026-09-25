"""
P4 Ablation Study — Feature Group Contribution Analysis
Walk-forward CV with each feature group removed to prove marginal value.

Uses exactly the XGBoost configuration of the main system (expert_system.make_xgb),
so the "Full model" row equals the system's walk-forward AUC. The digital stream
is not an XGBoost feature (it enters only through rule R3), so its contribution is
measured at system level in multiseed_analysis.py ("no digital" expert layer).

Run: python p4_early_warning/ablation.py
Output: p4_early_warning/models/ablation_results.json
"""
import os, sys, json
import numpy as np
from datetime import datetime, timezone
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expert_system import FEATURE_COLS, TARGET, load_features, make_xgb

ABLATION_GROUPS = {
    "Full model (all streams)": [],   # remove nothing
    "No climate":               ["rainfall_t2_mm", "rainfall_t4_mm", "temp_mean_t1_c"],
    "No reservoir/ecology":     ["reservoir_risk_index"],
    "No spatial":               ["is_border_state", "neighbour_cases_t1"],
    "Cases only (baseline)":    ["rainfall_t2_mm", "rainfall_t4_mm", "temp_mean_t1_c",
                                 "reservoir_risk_index", "is_border_state", "neighbour_cases_t1"],
}

WALK_FORWARD_FOLDS = [
    {"train_max": 2019, "test_year": 2020},
    {"train_max": 2020, "test_year": 2021},
    {"train_max": 2021, "test_year": 2022},
    {"train_max": 2022, "test_year": 2023},
]


def run_fold(df, train_max, test_year, feature_cols, seed: int = 42):
    train = df[df["epi_year"] <= train_max]
    test  = df[df["epi_year"] == test_year]
    if len(test) == 0 or test[TARGET].sum() == 0:
        return None

    X_tr = train[feature_cols].fillna(0).values.astype(np.float32)
    y_tr = train[TARGET].values.astype(int)
    X_te = test[feature_cols].fillna(0).values.astype(np.float32)
    y_te = test[TARGET].values.astype(int)

    model = make_xgb(y_tr, seed)
    model.fit(X_tr, y_tr, verbose=False)
    probs = model.predict_proba(X_te)[:, 1]
    return roc_auc_score(y_te, probs)


def main():
    print("=== Ablation Study — Feature Group Contribution ===\n")
    df = load_features()
    print(f"Loaded {len(df)} rows ({df['epi_year'].min()}–{df['epi_year'].max()})\n")

    results = {}
    for group_name, remove_cols in ABLATION_GROUPS.items():
        active_cols = [c for c in FEATURE_COLS if c not in remove_cols]
        fold_aucs   = []

        for fold in WALK_FORWARD_FOLDS:
            auc = run_fold(df, fold["train_max"], fold["test_year"], active_cols)
            if auc is not None:
                fold_aucs.append(auc)

        mean_auc = round(float(np.mean(fold_aucs)), 4) if fold_aucs else None
        delta    = None
        results[group_name] = {
            "features_used":   len(active_cols),
            "features_removed": remove_cols,
            "fold_aucs":        [round(a, 4) for a in fold_aucs],
            "mean_auc":         mean_auc,
        }
        print(f"  {group_name}")
        print(f"    Features: {len(active_cols)}  |  Mean AUC: {mean_auc}")
        print(f"    Fold AUCs: {[round(a,4) for a in fold_aucs]}\n")

    # Compute delta vs full model
    full_auc = results["Full model (all streams)"]["mean_auc"]
    for name, r in results.items():
        if r["mean_auc"] is not None and name != "Full model (all streams)":
            r["delta_vs_full"] = round(r["mean_auc"] - full_auc, 4)
        else:
            r["delta_vs_full"] = 0.0

    # Summary table
    print("\n" + "="*65)
    print(f"  {'Configuration':<30} {'Features':>8} {'AUC':>8} {'ΔAUC':>8}")
    print("  " + "-"*61)
    for name, r in results.items():
        delta_str = f"{r['delta_vs_full']:+.4f}" if r["delta_vs_full"] != 0.0 else "  (ref)"
        print(f"  {name:<30} {r['features_used']:>8} {r['mean_auc']:>8.4f} {delta_str:>8}")
    print("="*65)

    out = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "full_model_auc": full_auc,
        "folds": [f"{f['train_max']}→{f['test_year']}" for f in WALK_FORWARD_FOLDS],
        "results": results,
    }
    os.makedirs("p4_early_warning/models", exist_ok=True)
    with open("p4_early_warning/models/ablation_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved → p4_early_warning/models/ablation_results.json")


if __name__ == "__main__":
    main()
