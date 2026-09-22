"""Primary Low/High trial-level speech-timing dynamics for five acoustic outcomes."""
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
from analysis_helpers import METRICS, DiagonalLMM, bh_adjust, dynamic_design, load_dual_trials, reported_effect, response, save_json_qc  # noqa: E402


OUT = Path(__file__).resolve().parent
TABLES, FIGURES = OUT / "tables", OUT / "figures"
COLORS = {"Low": "#E9A93D", "High": "#4B89B5"}


def fit_primary(data: pd.DataFrame):
    x, z, names = dynamic_design(data, include_age=False)
    models, rows = {}, []
    for metric, (_, scale, unit) in METRICS.items():
        model = DiagonalLMM(response(data, metric), x, z, data.subject_id, names, ["Intercept", "Block slope", "Progress slope"])
        result = model.fit()
        models[metric] = result
        for term, estimate, se, p in zip(names, result["beta"], result["se"], result["p"]):
            effect, low, high = reported_effect(float(estimate), float(se), scale)
            rows.append({
                "metric": metric, "scale": scale, "term": term, "estimate_model_scale": estimate,
                "SE": se, "z": estimate / se, "p": p, "reported_effect": effect,
                "CI_low": low, "CI_high": high, "unit": unit,
                "random_intercept_SD": result["random_sd"][0],
                "random_block_SD": result["random_sd"][1],
                "random_progress_SD": result["random_sd"][2],
                "residual_SD": result["residual_sd"], "converged": bool(result["optimizer"].success),
            })
    table = pd.DataFrame(rows)
    mask = table.term.eq("Block x progress")
    table.loc[mask, "q_BH_5_outcomes"] = bh_adjust(table.loc[mask, "p"].to_numpy())
    return models, table


def model_predictions(models: dict, data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, (_, scale, _) in METRICS.items():
        result = models[metric]
        for condition, block in [("Low", -0.5), ("High", .5)]:
            for trial in range(1, 11):
                progress = (trial - 1) / 9
                vector = np.array([1.0, block, progress, block * progress])
                estimate = float(vector @ result["beta"])
                se = float(np.sqrt(vector @ result["beta_cov"] @ vector))
                transform = np.exp if scale == "log" else lambda x: x
                rows.append({
                    "metric": metric, "condition": condition, "trial_index": trial,
                    "predicted_value": float(transform(estimate)),
                    "CI_low": float(transform(estimate - 1.96 * se)),
                    "CI_high": float(transform(estimate + 1.96 * se)),
                })
    return pd.DataFrame(rows)


def high_low_contrasts(models: dict) -> pd.DataFrame:
    rows = []
    for metric, (_, scale, unit) in METRICS.items():
        result = models[metric]
        for trial in range(1, 11):
            progress = (trial - 1) / 9
            vector = np.array([0.0, 1.0, 0.0, progress])
            estimate = float(vector @ result["beta"])
            se = float(np.sqrt(vector @ result["beta_cov"] @ vector))
            effect, low, high = reported_effect(estimate, se, scale)
            rows.append({"metric": metric, "trial_index": trial, "estimate_model_scale": estimate, "SE": se, "p": 2 * __import__("scipy").stats.norm.sf(abs(estimate / se)), "reported_effect": effect, "CI_low": low, "CI_high": high, "unit": unit})
    table = pd.DataFrame(rows)
    for trial in range(1, 11):
        index = table.index[table.trial_index.eq(trial)]
        table.loc[index, "q_BH_5_outcomes"] = bh_adjust(table.loc[index, "p"].to_numpy())
    return table


def figure(data: pd.DataFrame, prediction: pd.DataFrame, table: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.0), constrained_layout=True)
    for axis, metric in zip(axes.flat, METRICS):
        column, _, unit = METRICS[metric]
        raw_unit = "syllables/s" if metric == "DDK Rate" else "ms"
        observed = data.groupby(["condition", "trial_index"], as_index=False)[column].mean()
        for condition in ["Low", "High"]:
            obs = observed[observed.condition.eq(condition)]
            pred = prediction[(prediction.metric.eq(metric)) & (prediction.condition.eq(condition))]
            axis.scatter(obs.trial_index, obs[column], s=16, color=COLORS[condition], alpha=.42)
            axis.plot(pred.trial_index, pred.predicted_value, color=COLORS[condition], linewidth=2, label=condition)
            axis.fill_between(pred.trial_index, pred.CI_low, pred.CI_high, color=COLORS[condition], alpha=.14)
        row = table[(table.metric.eq(metric)) & (table.term.eq("Block x progress"))].iloc[0]
        q = row.q_BH_5_outcomes
        q_label = "<.001" if q < .001 else f"={q:.3f}"
        axis.text(.03, .96, f"Block x progress: q {q_label}", transform=axis.transAxes, va="top", fontsize=8)
        axis.set(title=metric, xlim=(1, 10), xlabel="Within-block trial", ylabel=raw_unit)
        axis.spines[["top", "right"]].set_visible(False)
    axes.flat[-1].axis("off")
    axes.flat[0].legend(frameon=False, loc="lower left")
    fig.suptitle("Trial-level speech-timing trajectories", fontsize=13)
    for suffix, kwargs in [("png", {"dpi": 300}), ("pdf", {})]:
        fig.savefig(FIGURES / f"Figure_03_trial_level_dynamics.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = load_dual_trials()
    models, table = fit_primary(data)
    prediction = model_predictions(models, data)
    contrasts = high_low_contrasts(models)
    table.to_csv(TABLES / "low_high_trajectory_models.csv", index=False, encoding="utf-8-sig")
    prediction.to_csv(TABLES / "low_high_trial_predictions.csv", index=False, encoding="utf-8-sig")
    contrasts.to_csv(TABLES / "low_high_trial_contrasts.csv", index=False, encoding="utf-8-sig")
    save_json_qc(data, OUT / "analysis_qc.json")
    print(f"Trial-level dynamics written to {OUT}")


if __name__ == "__main__":
    main()
