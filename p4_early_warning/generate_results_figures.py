"""
Generate the three results figures of the MpoxMirror manuscript, strictly from
this repository's result JSONs (no values are typed in by hand).

Inputs  (p4_early_warning/models/):
  - multiseed_results.json               -> Fig. 2 (10-seed AUC + operating points)
  - clinical_utility_results.json        -> Fig. 3 (decision-curve analysis, 2024)
  - prospective_validation_results.json  -> Fig. 4 (2024 digital chatter timeline)

Outputs (results/):
  - fig02_auc_operating.png
  - fig03_decision_curve.png
  - fig04_lead_time.png

Run:
    python p4_early_warning/generate_results_figures.py
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "savefig.dpi": 200,
})

BLUE = "#3B6EA5"    # walk-forward
ORANGE = "#D2833B"  # held-out 2024
GREEN = "#2E7D32"   # MpoxMirror / model
RED = "#B23B3B"     # NCDC / treat-all
GREY = "#7A7A7A"


def load(name):
    with open(os.path.join(MODELS, name), encoding="utf-8") as f:
        return json.load(f)


def _clean(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _save(fig, name):
    fig.tight_layout()
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Wrote", p)


# ---------------------------------------------------------------- Fig. 2
def fig_auc_operating():
    d = load("multiseed_results.json")["table1_three_system"]
    systems = ["arima", "nb_glm", "smartmpox"]
    labels = ["ARIMA\n(cases only)", "NB-GLM\n(4 streams)", "MpoxMirror\n(5 streams)"]

    def val(s, part, k):
        v = d[s][part][k]
        return v["mean"] if isinstance(v, dict) else v

    folds = {"arima": d["arima"]["fold_auc"], "nb_glm": d["nb_glm"]["fold_auc"],
             "smartmpox": d["smartmpox"]["fold_auc_mean_over_seeds"]}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.4, 4.1))
    x = np.arange(len(systems))
    w = 0.38
    wf_auc = [val(s, "wf", "auc") for s in systems]
    cc_auc = [val(s, "clade_2024", "auc") for s in systems]
    axA.bar(x - w / 2, wf_auc, w, label="Walk-forward (mean of 2021-2023 folds)",
            color=BLUE, edgecolor="#22384d", linewidth=0.6, zorder=2)
    axA.bar(x + w / 2, cc_auc, w, label="2024 (held out)",
            color=ORANGE, edgecolor="#7c4d1c", linewidth=0.6, zorder=2)
    sd_wf = d["smartmpox"]["wf"]["auc"]["sd"]
    sd_24 = d["smartmpox"]["clade_2024"]["auc"]["sd"]
    axA.errorbar([x[2] - w / 2, x[2] + w / 2], [wf_auc[2], cc_auc[2]], yerr=[sd_wf, sd_24],
                 fmt="none", ecolor="#222222", capsize=3, lw=0.9, zorder=5)
    for i, s in enumerate(systems):
        pts = [p for p in folds[s] if p is not None]
        axA.scatter([x[i] - w / 2] * len(pts), pts, s=22, color="#12263a",
                    zorder=4, alpha=0.85, marker="o", linewidths=0)
    axA.axhline(0.5, color=GREY, ls=":", lw=1.0, zorder=1)
    axA.text(len(systems) - 0.5, 0.505, "chance", color=GREY, fontsize=8, va="bottom", ha="right")
    axA.set_xticks(x)
    axA.set_xticklabels(labels, fontsize=9.5)
    axA.set_ylabel("AUC")
    axA.set_ylim(0.40, 0.95)
    axA.set_title("(a) Walk-forward vs held-out 2024 discrimination", fontsize=11)
    axA.legend(fontsize=8.2, loc="upper left", frameon=False)
    _clean(axA)

    marks = {"arima": ("ARIMA", "s", ORANGE), "nb_glm": ("NB-GLM", "^", BLUE),
             "smartmpox": ("MpoxMirror", "o", GREEN)}
    axB.plot([0, 0.7], [0, 0.7], ls="--", color=GREY, lw=1.0, zorder=1)
    for s in systems:
        name, mk, col = marks[s]
        far = val(s, "clade_2024", "false_alarm_rate")
        sens = val(s, "clade_2024", "sensitivity")
        if s == "smartmpox":
            axB.errorbar([far], [sens],
                         xerr=[d[s]["clade_2024"]["false_alarm_rate"]["sd"]],
                         yerr=[d[s]["clade_2024"]["sensitivity"]["sd"]],
                         fmt="none", ecolor="#222222", capsize=3, lw=0.9, zorder=2)
        axB.scatter([far], [sens], s=95, marker=mk, color=col, edgecolor="#222222",
                    linewidth=0.7, zorder=3)
        ha = "right" if s == "nb_glm" else "left"
        dx = -0.006 if s == "nb_glm" else 0.006
        axB.annotate(name, (far, sens), xytext=(far + dx, sens + 0.035), fontsize=9.2,
                     ha=ha, color="#222222")
    axB.set_xlabel("False-alarm rate")
    axB.set_ylabel("Sensitivity")
    axB.set_xlim(-0.01, 0.24)
    axB.set_ylim(0.0, 0.72)
    axB.set_title("(b) Held-out 2024 operating points", fontsize=11)
    _clean(axB)
    _save(fig, "fig02_auc_operating.png")


# ---------------------------------------------------------------- Fig. 3
def fig_decision_curve():
    d = load("clinical_utility_results.json")
    prev = d["prevalence"]
    curve = d["dca_curve"]
    thr = [r["threshold"] for r in curve]
    rule = [r["mpoxmirror_rule_nb"] for r in curve]
    prob = [r["model_prob_nb"] for r in curve]
    allnb = [r["treat_all_nb"] for r in curve]
    be = d["mpoxmirror_rule_positive_nb_below_threshold"]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(thr, rule, color=GREEN, lw=2.0, marker="o", ms=4,
            label="MpoxMirror alert rule", zorder=4)
    ax.plot(thr, prob, color=BLUE, lw=1.3, marker=".", ms=4,
            label="XGBoost score (uncalibrated)", zorder=3)
    ax.plot(thr, [0.0] * len(thr), color=GREY, lw=1.3, ls=":", label="Treat none", zorder=2)
    ax.plot(thr, allnb, color=RED, lw=1.3, ls="--", label="Treat all", zorder=2)
    ax.axvline(prev, color="#555555", lw=1.0, ls="-.", zorder=1)
    ax.text(prev + 0.002, -0.0145, f"prevalence {prev:.3f}", fontsize=8.3, color="#555555",
            rotation=90, va="bottom")
    ax.axvline(be, color=GREEN, lw=0.8, ls=":", zorder=1)
    ax.text(be + 0.003, 0.009, f"rule NB > 0\nfor p$_t$ < {be:.3f}", fontsize=8.2, color=GREEN)
    ax.set_xlabel("Decision threshold probability p$_t$")
    ax.set_ylabel("Net benefit")
    ax.set_xlim(0, 0.31)
    ax.set_ylim(-0.015, 0.016)
    ax.set_title(f"Decision-curve analysis, held-out 2024 (n = {d['n']:,}, 4-week target)",
                 fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right", frameon=False)
    _clean(ax)
    _save(fig, "fig03_decision_curve.png")


# ---------------------------------------------------------------- Fig. 4
def fig_lead_time():
    d = load("prospective_validation_results.json")
    ca = d["channel_a_digital"]
    wk = {int(k): v for k, v in ca["weekly_counts_2024"].items()}
    ng = {int(k): v for k, v in ca["weekly_nigeria_mentions_2024"].items()}
    thr = ca["frozen_surge_threshold"]
    frozen = next(r for r in ca["by_threshold"] if r["is_frozen_operating_point"])
    cross_wk = frozen["first_alert_week"]
    rep_wks = d["surveillance_weeks_ingested_2024"]
    ncdc_wk = d["channel_b_model_expert"]["national_confirmation_week"]

    weeks = list(range(1, 53))
    tot = [wk.get(w, 0) for w in weeks]
    nig = [ng.get(w, 0) for w in weeks]
    other = [t - n for t, n in zip(tot, nig)]
    ymax = max(tot) * 1.15

    fig, ax = plt.subplots(figsize=(8.8, 4.1))
    ax.axvspan(min(rep_wks) - 0.5, max(rep_wks) + 0.5, color="#EFE7DA", zorder=0,
               label=f"weeks with ingested NCDC reports ({min(rep_wks)}-{max(rep_wks)})")
    ax.bar(weeks, nig, width=0.82, color=GREEN, edgecolor="#1f4f22", linewidth=0.3,
           zorder=2, label="posts mentioning Nigeria")
    ax.bar(weeks, other, bottom=nig, width=0.82, color=BLUE, edgecolor="#22384d",
           linewidth=0.3, zorder=2, label="other mpox posts")
    ax.axhline(thr, color=GREY, ls="--", lw=1.1, zorder=3)
    ax.text(0.5, thr + ymax * 0.012,
            f"frozen surge threshold ({thr:g} posts/week, fitted on <=2023)",
            fontsize=8.2, color=GREY, va="bottom")
    ax.annotate(f"first crossing: wk {cross_wk}\n"
                f"({frozen['posts_mentioning_nigeria_in_alert_week']} of "
                f"{frozen['posts_in_alert_week']} posts mention Nigeria)",
                (cross_wk, wk.get(cross_wk, 0)), xytext=(max(cross_wk - 17, 1), ymax * 0.35),
                fontsize=8.2, color="#333333",
                arrowprops=dict(arrowstyle="->", color="#777777", lw=0.9))
    ax.axvline(ncdc_wk, color=RED, lw=1.4, zorder=4)
    ax.text(ncdc_wk + 0.6, ymax * 0.97,
            f"first confirmed case in\ningested reports (wk {ncdc_wk})",
            fontsize=8.2, color=RED, ha="left", va="top")
    ax.set_xlabel("ISO / epidemiological week, 2024")
    ax.set_ylabel("Mpox-relevant posts per week")
    ax.set_xlim(0, 53)
    ax.set_ylim(0, ymax)
    ax.set_title("National digital chatter in 2024 (simulated real-time replay)", fontsize=11.5)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    _clean(ax)
    _save(fig, "fig04_lead_time.png")


if __name__ == "__main__":
    fig_auc_operating()
    fig_decision_curve()
    fig_lead_time()
    print("Done.")
