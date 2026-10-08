"""Build Figure 4 with the same age scaling for model fitting and prediction.

The condition-mean models retain the 465-row sample-SD convention. The
155-participant convention used in pause-trajectory and CMMS models is unchanged.
"""
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
    age_mean = means.age_years.mean()
    age_sd = means.age_years.std(ddof=1)
    if not np.allclose(means.age_z, (means.age_years - age_mean) / age_sd,
                       rtol=1e-12, atol=1e-12):
        raise ValueError("Prediction age scaling does not match the fitted age_z column.")
    age = np.linspace(means.age_years.min(), means.age_years.max(), 130)
    age_z = (age - age_mean) / age_sd
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 1.0})
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.492))
    fig.subplots_adjust(left=.065, right=.985, bottom=.18, top=.82, wspace=.25)
    fig.suptitle("Age associations with pause timing from participant-mean models",
                 fontsize=15, fontweight="bold", y=.97)
    for panel, ax, (outcome, column) in zip("ab", axes, SPECS):
        work = means.copy()
        work["outcome"] = np.log(work[column])
        model = mixedlm("outcome ~ duality + load + age_z + duality:age_z + load:age_z",
                        work, groups=work.analysis_id, re_formula="~duality+load")
        try:
            fit = model.fit(reml=True, method="lbfgs", maxiter=2000, disp=False)
        except Exception:
            fit = model.fit(reml=True, method="powell", maxiter=2000, disp=False)
        if not fit.converged:
            raise RuntimeError(f"Figure 4 model did not converge: {outcome}")
        for condition in ["Single", "Low", "High"]:
            observed = work[work.condition == condition]
            prediction, low, high = fixed_prediction(fit, condition, age_z)
            ax.scatter(observed.age_years, observed[column], s=13, alpha=.18,
                       color=COLORS[condition], edgecolors="none", rasterized=True)
            ax.fill_between(age, low, high, color=COLORS[condition], alpha=.13, linewidth=0)
            ax.plot(age, prediction, color=COLORS[condition], linewidth=2.1, label=condition)
        row = table4[table4.outcome == outcome].iloc[0]
        dual_q = "< .001" if row.single_to_dual_q < .001 else "= " + f"{row.single_to_dual_q:.3f}".lstrip("0")
        load_q = "< .001" if row.high_minus_low_q < .001 else "= " + f"{row.high_minus_low_q:.3f}".lstrip("0")
        ax.text(.025, .965, f"Dual–Single × age: {row.single_to_dual_x_age:+.1f}%, q {dual_q}\n"
                f"High–Low × age: {row.high_minus_low_x_age:+.1f}%, q {load_q}\n"
                f"Age effects per {age_sd:.2f} years",
                transform=ax.transAxes, va="top", fontsize=8.6,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .94, "pad": 3})
        ax.set_title(f"{panel}  {outcome}", loc="left", fontsize=12.5, fontweight="bold", pad=10)
        ax.set_xlabel("Age (years)", fontsize=11)
        ax.set_ylabel(f"{outcome} (ms)", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#D9DEE3", linewidth=.8)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .025),
               ncol=3, frameon=False, fontsize=10)
    fig.savefig(FIGURES / "Figure_04.png", dpi=600, facecolor="white")
    fig.savefig(FIGURES / "Figure_04.pdf", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
