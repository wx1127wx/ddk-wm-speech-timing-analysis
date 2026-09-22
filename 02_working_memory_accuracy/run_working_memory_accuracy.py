"""Working-memory accuracy manipulation check using existing dual-task trials."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from analysis_helpers import LogisticRandomInterceptSlope, load_dual_trials, save_json_qc  # noqa: E402


OUT = Path(__file__).resolve().parent
TABLES, FIGURES = OUT / "tables", OUT / "figures"


def effect_rows(result: dict) -> pd.DataFrame:
    names = ["Intercept", "High-Low at block midpoint", "Progress", "Block x progress"]
    rows = []
    for name, estimate, se, p in zip(names, result["beta"], result["se"], result["p"]):
        lo, hi = estimate - 1.96 * se, estimate + 1.96 * se
        rows.append({
            "term": name, "log_odds": estimate, "SE": se, "z": estimate / se, "p": p,
            "odds_ratio": float(np.exp(estimate)), "OR_CI_low": float(np.exp(lo)),
            "OR_CI_high": float(np.exp(hi)),
        })
    return pd.DataFrame(rows)


def predictions(result: dict) -> pd.DataFrame:
    rows = []
    for condition, block in [("Low", -0.5), ("High", 0.5)]:
        for trial in range(1, 11):
            progress = (trial - 1) / 9 - .5
            x = np.array([1.0, block, progress, block * progress])
            eta = float(x @ result["beta"])
            se = float(np.sqrt(x @ result["cov"][:4, :4] @ x))
            rows.append({
                "condition": condition, "trial_index": trial,
                "predicted_accuracy": float(1 / (1 + np.exp(-eta))),
                "CI_low": float(1 / (1 + np.exp(-(eta - 1.96 * se)))),
                "CI_high": float(1 / (1 + np.exp(-(eta + 1.96 * se)))),
            })
    return pd.DataFrame(rows)


def figure(subject_means: pd.DataFrame, prediction: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)
    colors = {"Low": "#E9A93D", "High": "#4B89B5"}
    pivot = subject_means.pivot(index="subject_id", columns="condition", values="wm_accuracy")
    rng = np.random.default_rng(20260908)
    for index, condition in enumerate(["Low", "High"]):
        x = rng.normal(index, 0.045, len(pivot))
        axes[0].scatter(x, pivot[condition], s=13, color=colors[condition], alpha=.35, linewidth=0)
        axes[0].errorbar(index, pivot[condition].mean(), yerr=1.96 * pivot[condition].sem(), fmt="o", color="#26343B", capsize=3)
    axes[0].set(xlim=(-.4, 1.4), ylim=(-.02, 1.03), xticks=[0, 1], xticklabels=["Low", "High"], ylabel="WM accuracy")
    axes[0].set_title("Participant condition means", loc="left", fontweight="bold")
    for condition in ["Low", "High"]:
        work = prediction[prediction.condition.eq(condition)]
        axes[1].plot(work.trial_index, work.predicted_accuracy, color=colors[condition], label=condition, linewidth=2)
        axes[1].fill_between(work.trial_index, work.CI_low, work.CI_high, color=colors[condition], alpha=.18)
    axes[1].set(xlim=(1, 10), ylim=(0, 1), xlabel="Within-block trial", ylabel="Predicted WM accuracy")
    axes[1].legend(frameon=False)
    axes[1].set_title("Block trajectory", loc="left", fontweight="bold")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    for suffix, kwargs in [("png", {"dpi": 300}), ("pdf", {})]:
        fig.savefig(FIGURES / f"Figure_01_WM_manipulation.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    data = load_dual_trials()
    data["progress_centered"] = data.progress - .5
    x = np.column_stack([np.ones(len(data)), data.block, data.progress_centered, data.block * data.progress_centered])
    model = LogisticRandomInterceptSlope(data.wm_correct, x, data.block, data.subject_id, gh_points=11)
    result = model.fit()
    primary = effect_rows(result)
    primary.to_csv(TABLES / "working_memory_GLMM.csv", index=False, encoding="utf-8-sig")
    descriptive = data.groupby("condition", as_index=False)["wm_correct"].agg(correct="sum", responses="count", accuracy="mean")
    descriptive.to_csv(TABLES / "working_memory_descriptives.csv", index=False, encoding="utf-8-sig")
    save_json_qc(data, OUT / "analysis_qc.json")
    print(f"WM results written to {OUT}")


if __name__ == "__main__":
    main()
