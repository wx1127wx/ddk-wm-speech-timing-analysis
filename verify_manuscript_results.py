"""Compare generated aggregate results with values printed in the submission manuscript."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
EXPECTED = ROOT / "expected_results" / "tables"
checks: list[dict] = []


def check(label: str, actual: float, expected: float, tolerance: float = 0.0015) -> None:
    difference = abs(float(actual) - float(expected))
    checks.append({"check": label, "actual": float(actual), "expected": float(expected),
                   "absolute_difference": difference, "tolerance": tolerance,
                   "pass": bool(difference <= tolerance)})


def main() -> None:
    t2e = pd.read_csv(EXPECTED / "Table_02.csv").set_index("outcome")
    t2a = pd.read_csv(ROOT / "01_sample_and_condition_means" / "tables" / "Table_02_condition_means_and_contrasts.csv").set_index("outcome")
    mapping2 = {"Single": "Single", "Low": "Low", "High": "High",
                "dual_effect": "dual_minus_single_effect", "dual_CI_low": "dual_minus_single_CI_low",
                "dual_CI_high": "dual_minus_single_CI_high", "dual_q": "dual_minus_single_q",
                "high_low_effect": "high_minus_low_effect", "high_low_CI_low": "high_minus_low_CI_low",
                "high_low_CI_high": "high_minus_low_CI_high", "high_low_q": "high_minus_low_q"}
    for outcome in t2e.index:
        for expected_col, actual_col in mapping2.items():
            check(f"Table 2 | {outcome} | {expected_col}", t2a.loc[outcome, actual_col], t2e.loc[outcome, expected_col])

    t3e = pd.read_csv(EXPECTED / "Table_03.csv").set_index("outcome")
    t3a = pd.read_csv(ROOT / "03_trial_trajectories" / "tables" / "Table_03_trajectory_change.csv").set_index("outcome")
    for outcome in t3e.index:
        for col in ["trajectory_change", "CI_low", "CI_high", "high_minus_low_trial_1", "high_minus_low_trial_10", "q"]:
            if pd.notna(t3e.loc[outcome, col]):
                check(f"Table 3 | {outcome} | {col}", t3a.loc[outcome, col], t3e.loc[outcome, col], 0.0055)

    t4e = pd.read_csv(EXPECTED / "Table_04.csv").set_index("outcome")
    t4a = pd.read_csv(ROOT / "01_sample_and_condition_means" / "tables" / "Table_04_age_moderation.csv").set_index("outcome")
    mapping4 = {"single_to_dual_x_age": "single_to_dual_x_age", "CI_low": "single_to_dual_CI_low",
                "CI_high": "single_to_dual_CI_high", "q": "single_to_dual_q",
                "high_minus_low_x_age": "high_minus_low_x_age", "high_minus_low_q": "high_minus_low_q"}
    for outcome in t4e.index:
        for expected_col, actual_col in mapping4.items():
            check(f"Table 4 | {outcome} | {expected_col}", t4a.loc[outcome, actual_col], t4e.loc[outcome, expected_col], 0.0055)

    t5e = pd.read_csv(EXPECTED / "Table_05.csv").set_index("outcome")
    raw5 = pd.read_csv(ROOT / "05_cmms_moderation" / "tables" / "Table_03_condition_mean_moderation.csv")
    for outcome in t5e.index:
        dual = raw5[(raw5.metric == outcome) & (raw5.term == "Duality x scale")].iloc[0]
        load = raw5[(raw5.metric == outcome) & (raw5.term == "High-Low x scale")].iloc[0]
        values = {"single_to_dual_x_CMMS": dual.reported_effect, "CI_low": dual.CI_low,
                  "CI_high": dual.CI_high, "q": dual.q_BH_5_outcomes,
                  "high_minus_low_x_CMMS": load.reported_effect, "high_CI_low": load.CI_low,
                  "high_CI_high": load.CI_high, "high_q": load.q_BH_5_outcomes}
        for col, value in values.items():
            check(f"Table 5 | {outcome} | {col}", value, t5e.loc[outcome, col], 0.0055)

    t6e = pd.read_csv(EXPECTED / "Table_06.csv").set_index("outcome")
    raw6 = pd.read_csv(ROOT / "06_manual_validation" / "results" / "agreement_summary.csv")
    feature_map = {"DDK rate (syll/s)": "DDK Rate", "DDK regularity (ms)": "DDK Regularity",
                   "DDK duration (ms)": "DDK Duration", "pause regularity (ms)": "Pause Regularity",
                   "pause duration (ms)": "Pause Duration"}
    raw6["outcome"] = raw6.feature.map(feature_map)
    raw6 = raw6.set_index("outcome")
    mapping6 = {"pearson_r": "pearson_r", "pearson_CI_low": "pearson_r_ci_low_cluster_bootstrap",
                "pearson_CI_high": "pearson_r_ci_high_cluster_bootstrap", "CCC": "ccc",
                "CCC_CI_low": "ccc_ci_low_cluster_bootstrap", "CCC_CI_high": "ccc_ci_high_cluster_bootstrap",
                "ICC_A1": "icc_a1_absolute_agreement", "bias": "bias_automatic_minus_manual",
                "bias_CI_low": "bias_automatic_minus_manual_ci_low_cluster_bootstrap",
                "bias_CI_high": "bias_automatic_minus_manual_ci_high_cluster_bootstrap"}
    for outcome in t6e.index:
        for expected_col, actual_col in mapping6.items():
            # Bootstrap limits use 2,000 resamples in the release; the manuscript
            # limits came from an earlier Monte-Carlo realization and are allowed
            # a small simulation tolerance. Point estimates use rounding tolerance.
            tolerance = 0.6 if expected_col.startswith("bias_CI") else (0.05 if "CI" in expected_col else 0.0055)
            check(f"Table 6 | {outcome} | {expected_col}", raw6.loc[outcome, actual_col], t6e.loc[outcome, expected_col], tolerance)

    wm = pd.read_csv(ROOT / "02_working_memory_accuracy" / "tables" / "working_memory_descriptives.csv").set_index("condition")
    check("WM Low correct", wm.loc["Low", "correct"], 1349, 0)
    check("WM Low responses", wm.loc["Low", "responses"], 1550, 0)
    check("WM High correct", wm.loc["High", "correct"], 1198, 0)
    check("WM High responses", wm.loc["High", "responses"], 1550, 0)

    frame = pd.DataFrame(checks)
    frame.to_csv(ROOT / "verification_report.csv", index=False, encoding="utf-8-sig")
    summary = {"checks": len(frame), "passed": int(frame["pass"].sum()),
               "failed": int((~frame["pass"]).sum()), "all_passed": bool(frame["pass"].all())}
    (ROOT / "verification_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["all_passed"]:
        print(frame.loc[~frame["pass"]].to_string(index=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
