"""Build manuscript Table 3 and Figure 3 from their traceable model outputs."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from analysis_helpers import METRICS, load_dual_trials, display_metric  # noqa: E402

HERE = Path(__file__).resolve().parent
TABLES, FIGURES = HERE / "tables", HERE / "figures"
COLORS = {"Low": "#0072B2", "High": "#D55E00"}


def observed_summaries(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome, (column, scale, _) in METRICS.items():
        for (condition, trial), group in data.groupby(["condition", "trial_index"], sort=True):
            values = group[column].to_numpy(float)
            if scale == "log":
                logs = np.log(values)
                estimate = np.exp(logs.mean())
                half = 1.96 * logs.std(ddof=1) / np.sqrt(len(logs))
                low, high = np.exp(logs.mean() - half), np.exp(logs.mean() + half)
            else:
                estimate = values.mean()
                half = 1.96 * values.std(ddof=1) / np.sqrt(len(values))
                low, high = estimate - half, estimate + half
            rows.append({"outcome": outcome, "condition": condition, "trial_index": int(trial),
                         "estimate": estimate, "CI_low": low, "CI_high": high})
    return pd.DataFrame(rows)


def manuscript_table(dynamic: pd.DataFrame, contrasts: pd.DataFrame) -> pd.DataFrame:
    """All five rows use the identical Low/High model and BH family of five."""
    rows = []
    for outcome in METRICS:
        model = dynamic[(dynamic.metric == outcome) & (dynamic.term == "Block x progress")].iloc[0]
        endpoint = contrasts[(contrasts.metric == outcome) & contrasts.trial_index.isin([1, 10])]
        values = dict(zip(endpoint.trial_index, endpoint.reported_effect))
        rows.append({"outcome": outcome, "trajectory_change": model.reported_effect,
                     "CI_low": model.CI_low, "CI_high": model.CI_high,
                     "high_minus_low_trial_1": values[1],
                     "high_minus_low_trial_10": values[10],
                     "q": model.q_BH_5_outcomes,
                     "source_model": "Low/High trial-level mixed model"})
    return pd.DataFrame(rows)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    dynamic = pd.read_csv(TABLES / "low_high_trajectory_models.csv", encoding="utf-8-sig")
    dynamic_predictions = pd.read_csv(TABLES / "low_high_trial_predictions.csv", encoding="utf-8-sig")
    contrasts = pd.read_csv(TABLES / "low_high_trial_contrasts.csv", encoding="utf-8-sig")
    table = manuscript_table(dynamic, contrasts)
    table.to_csv(TABLES / "Table_03_trajectory_change.csv", index=False, encoding="utf-8-sig")
    observed = observed_summaries(load_dual_trials())

    panels = [("DDK Rate", "DDK Rate (syll/s)"), ("DDK Duration", "DDK Duration (ms)"),
              ("Pause Duration", "Pause Duration (ms)"), ("Pause Regularity", "Pause Regularity (ms)")]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.55), constrained_layout=True)
    fig.suptitle("Repeated-trial speech-timing trajectories", fontsize=12, fontweight="bold")
    for panel, (outcome, ylabel) in zip("ABCD", panels):
        ax = axes.flat["ABCD".index(panel)]
        q = float(table.loc[table.outcome == outcome, "q"].iloc[0])
        for condition in ["Low", "High"]:
            pred_source = dynamic_predictions
            outcome_col = "metric"
            pred = pred_source[(pred_source[outcome_col] == outcome) & (pred_source.condition == condition)]
            obs = observed[(observed.outcome == outcome) & (observed.condition == condition)]
            ax.plot(pred.trial_index, pred.predicted_value, color=COLORS[condition], linewidth=2.0, label=condition)
            ax.errorbar(obs.trial_index, obs.estimate,
                        yerr=[obs.estimate - obs.CI_low, obs.CI_high - obs.estimate], fmt="o",
                        color=COLORS[condition], markersize=3.2, capsize=1.6, alpha=.62)
        q_text = "< .001" if q < .001 else f"= {q:.3f}".replace("0.", ".")
        ax.set_title(f"{panel}  {display_metric(outcome)}", loc="left", fontsize=9.5, fontweight="bold")
        ax.text(.02, .96, f"High-Low trajectory divergence: q {q_text}", transform=ax.transAxes,
                va="top", fontsize=8, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .88})
        ax.set(xlabel="Trial within block", ylabel=display_metric(ylabel), xticks=range(1, 11))
        ax.spines[["top", "right"]].set_visible(False)
    axes.flat[0].legend(title="Block condition", frameon=False, loc="lower left", fontsize=7, title_fontsize=7)
    fig.savefig(FIGURES / "Figure_03.png", dpi=600, facecolor="white", bbox_inches="tight")
    fig.savefig(FIGURES / "Figure_03.pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
