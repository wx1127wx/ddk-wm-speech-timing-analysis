"""Build manuscript Figure 4 from the participant-mean models in Table 4."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.formula.api import mixedlm

HERE = Path(__file__).resolve().parent
MEANS = HERE / "data_processed" / "speech_condition_means.csv"
TABLE4 = HERE / "tables" / "Table_04_age_moderation.csv"
FIGURES = HERE / "figures"
COLORS = {"Single": "#737373", "Low": "#0072B2", "High": "#D55E00"}
SPECS = [("Pause Duration", "pause_dur"), ("Pause Regularity", "pause_reg")]


def fixed_prediction(fit, condition: str, age_z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    duality = {"Single": -2 / 3, "Low": 1 / 3, "High": 1 / 3}[condition]
    load = {"Single": 0.0, "Low": -0.5, "High": 0.5}[condition]
    names = ["Intercept", "duality", "load", "age_z", "duality:age_z", "load:age_z"]
    beta = fit.params[names].to_numpy(float)
    covariance = fit.cov_params().loc[names, names].to_numpy(float)
    design = np.column_stack([np.ones(len(age_z)), np.full(len(age_z), duality), np.full(len(age_z), load),
                              age_z, duality * age_z, load * age_z])
    eta = design @ beta
    se = np.sqrt(np.einsum("ij,jk,ik->i", design, covariance, design))
    return np.exp(eta), np.exp(eta - 1.96 * se), np.exp(eta + 1.96 * se)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    means = pd.read_csv(MEANS, dtype={"analysis_id": str})
    table4 = pd.read_csv(TABLE4, encoding="utf-8-sig")
    participant_age = means[["analysis_id", "age_years"]].drop_duplicates("analysis_id").age_years
    age_mean, age_sd = participant_age.mean(), participant_age.std(ddof=1)
    age = np.linspace(means.age_years.min(), means.age_years.max(), 130)
    age_z = (age - age_mean) / age_sd
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.2), constrained_layout=True)
    fig.suptitle("Age associations with pause timing across ordered task conditions", fontsize=12, fontweight="bold")
    for panel, ax, (outcome, column) in zip("AB", axes, SPECS):
        work = means.copy()
        work["outcome"] = np.log(work[column])
        model = mixedlm("outcome ~ duality + load + age_z + duality:age_z + load:age_z",
                        work, groups=work.analysis_id, re_formula="~duality+load")
        try:
            fit = model.fit(reml=True, method="lbfgs", maxiter=2000, disp=False)
        except Exception:
            fit = model.fit(reml=True, method="powell", maxiter=2000, disp=False)
        for condition in ["Single", "Low", "High"]:
            observed = work[work.condition == condition]
            prediction, low, high = fixed_prediction(fit, condition, age_z)
            ax.scatter(observed.age_years, observed[column], s=8, alpha=.10, color=COLORS[condition], edgecolors="none")
            ax.fill_between(age, low, high, color=COLORS[condition], alpha=.09, linewidth=0)
            ax.plot(age, prediction, color=COLORS[condition], linewidth=2, label=condition)
        row = table4[table4.outcome == outcome].iloc[0]
        dual_q = "<.001" if row.single_to_dual_q < .001 else f"={row.single_to_dual_q:.3f}".replace("0.", ".")
        load_q = "<.001" if row.high_minus_low_q < .001 else f"={row.high_minus_low_q:.3f}".replace("0.", ".")
        ax.text(.025, .965, f"Single-Dual x age: {row.single_to_dual_x_age:+.1f}%, q {dual_q}\n"
                f"High-Low x age: {row.high_minus_low_x_age:+.1f}%, q {load_q}",
                transform=ax.transAxes, va="top", fontsize=7.6,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .88})
        ax.set_title(f"{panel}  {outcome}", loc="left", fontsize=9.3, fontweight="bold")
        ax.set(xlabel="Age (years)", ylabel=f"{outcome} (ms)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(title="Task condition", frameon=False, fontsize=7.5, title_fontsize=7.5)
    fig.savefig(FIGURES / "Figure_04.png", dpi=600, facecolor="white", bbox_inches="tight")
    fig.savefig(FIGURES / "Figure_04.pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
