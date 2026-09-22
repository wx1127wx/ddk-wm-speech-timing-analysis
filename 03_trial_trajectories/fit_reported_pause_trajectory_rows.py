"""Reproduce the Pause rows actually reported in manuscript Table 3.

Important: these two rows came from an age-adjusted three-condition model
(Single/Low/High), whereas the manuscript Methods describes the Stage-2 model
as Low/High only.  The release keeps this script so the published numbers are
traceable; README.md flags the specification mismatch for author resolution.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from dynamic_model_utils import DiagonalLMM, bh_adjust  # noqa: E402

DATA = ROOT / "data_raw"
OUT = Path(__file__).resolve().parent / "tables"
PAUSE = {
    "pause regularity (ms)": "Pause Regularity",
    "pause duration (ms)": "Pause Duration",
}


def pct(value: float) -> float:
    return 100.0 * math.expm1(value)


def load_data() -> pd.DataFrame:
    ddk = pd.read_csv(DATA / "ddk_trials.csv", dtype={"subject_id": str})
    dual = pd.read_csv(DATA / "ddkwm_trials.csv", dtype={"subject_id": str})
    demo = pd.read_csv(DATA / "participants.csv", dtype={"subject_id": str})[["subject_id", "age"]]
    demo = demo.drop_duplicates("subject_id")
    demo["age_z"] = (demo.age - demo.age.mean()) / demo.age.std(ddof=1)
    ddk = ddk.merge(demo, on="subject_id", validate="many_to_one")
    dual = dual.merge(demo, on="subject_id", validate="many_to_one")
    ddk["condition"] = "Single"
    ddk["progress"] = (ddk.trial_index - 1) / 2 - 0.5
    dual["condition"] = dual.ddkwm_condition.map({1: "Low", 2: "High"})
    dual["progress"] = (dual.trial_index - 1) / 9 - 0.5
    combined = pd.concat([ddk, dual], ignore_index=True, sort=False)
    combined["duality"] = combined.condition.map({"Single": -2 / 3, "Low": 1 / 3, "High": 1 / 3})
    combined["load"] = combined.condition.map({"Single": 0.0, "Low": -0.5, "High": 0.5})
    return combined


def contrast(result: dict, vector: np.ndarray) -> tuple[float, float, float, float]:
    estimate = float(vector @ result["beta"])
    se = float(np.sqrt(vector @ result["beta_cov"] @ vector))
    return estimate, se, estimate - 1.96 * se, estimate + 1.96 * se


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    names = ["Intercept", "Dual vs single", "High vs low", "Age (+1 SD)",
             "Duality x age", "Load x age", "Progress", "Duality x progress", "Load x progress"]
    x = np.column_stack([
        np.ones(len(data)), data.duality, data.load, data.age_z,
        data.duality * data.age_z, data.load * data.age_z, data.progress,
        data.duality * data.progress, data.load * data.progress,
    ])
    z = np.column_stack([np.ones(len(data)), data.duality, data.load])
    rows, predictions = [], []
    for column, outcome in PAUSE.items():
        result = DiagonalLMM(np.log(data[column]), x, z, data.subject_id, names,
                             ["Intercept", "Duality slope", "Load slope"]).fit()
        for index, term in enumerate(names):
            estimate, se = float(result["beta"][index]), float(result["se"][index])
            rows.append({"outcome": outcome, "term": term, "estimate_log_scale": estimate,
                         "SE": se, "p": float(result["p"][index]), "percent_change": pct(estimate),
                         "CI_low_percent": pct(estimate - 1.96 * se),
                         "CI_high_percent": pct(estimate + 1.96 * se),
                         "converged": bool(result["optimizer"].success)})
        for condition, duality, load in [("Low", 1 / 3, -0.5), ("High", 1 / 3, 0.5)]:
            for trial in range(1, 11):
                progress = (trial - 1) / 9 - 0.5
                vector = np.array([1, duality, load, 0, 0, 0, progress, duality * progress, load * progress])
                estimate, _, low, high = contrast(result, vector)
                predictions.append({"outcome": outcome, "condition": condition, "trial_index": trial,
                                    "predicted_value": math.exp(estimate), "CI_low": math.exp(low), "CI_high": math.exp(high)})
    table = pd.DataFrame(rows)
    table["q_BH_within_term"] = np.nan
    for term in names[1:]:
        idx = table.index[table.term == term]
        table.loc[idx, "q_BH_within_term"] = bh_adjust(table.loc[idx, "p"])
    table["family"] = table.term.map({
        "Dual vs single": "Mean condition effects", "High vs low": "Mean condition effects",
        "Age (+1 SD)": "Age effects", "Duality x age": "Age moderation", "Load x age": "Age moderation",
        "Progress": "Dynamic effects", "Duality x progress": "Dynamic effects", "Load x progress": "Dynamic effects",
    })
    table["q_BH_within_family"] = np.nan
    for family in table.family.dropna().unique():
        idx = table.index[table.family == family]
        table.loc[idx, "q_BH_within_family"] = bh_adjust(table.loc[idx, "p"])
    table.to_csv(OUT / "reported_pause_models.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(predictions).to_csv(OUT / "reported_pause_predictions.csv", index=False, encoding="utf-8-sig")
    print(table[table.term.eq("Load x progress")][["outcome", "percent_change", "CI_low_percent", "CI_high_percent", "q_BH_within_family"]].to_string(index=False))


if __name__ == "__main__":
    main()
