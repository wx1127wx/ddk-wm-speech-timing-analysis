"""Sensitivity analyses reported in the manuscript Results section only.

The primary Low/High trajectory model is repeated (1) after retaining only
working-memory-correct trials and (2) after omitting Trials 1 and 10.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from analysis_helpers import (  # noqa: E402
    METRICS, DiagonalLMM, bh_adjust, dynamic_design, load_dual_trials,
    reported_effect, response,
)

OUT = Path(__file__).resolve().parent
TABLES = OUT / "tables"


def fit_subset(data: pd.DataFrame, label: str) -> list[dict]:
    x, z, names = dynamic_design(data, include_age=False)
    rows = []
    for metric, (_, scale, unit) in METRICS.items():
        result = DiagonalLMM(
            response(data, metric), x, z, data.subject_id, names,
            ["Intercept", "Block slope", "Progress slope"],
        ).fit()
        index = names.index("Block x progress")
        estimate, se = float(result["beta"][index]), float(result["se"][index])
        effect, low, high = reported_effect(estimate, se, scale)
        rows.append({
            "analysis": label, "outcome": metric, "n_trials": len(data),
            "n_participants": data.subject_id.nunique(), "trajectory_change": effect,
            "CI_low": low, "CI_high": high,
            "p": float(2 * stats.norm.sf(abs(estimate / se))), "unit": unit,
            "converged": bool(result["optimizer"].success),
        })
    q_values = bh_adjust([row["p"] for row in rows])
    for row, q_value in zip(rows, q_values):
        row["q_BH_5_outcomes"] = q_value
    return rows


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    data = load_dual_trials()
    rows = []
    rows.extend(fit_subset(data[data.wm_correct.eq(1)].copy(), "WM-correct trials only"))
    rows.extend(fit_subset(data[data.trial_index.between(2, 9)].copy(), "Trials 2-9 only"))
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "reported_trial_trajectory_sensitivity.csv", index=False, encoding="utf-8-sig")
    print(table[["analysis", "outcome", "trajectory_change", "CI_low", "CI_high", "q_BH_5_outcomes"]].to_string(index=False))


if __name__ == "__main__":
    main()
