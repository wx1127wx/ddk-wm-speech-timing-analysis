"""Shared data loading and mixed-model helpers for revised DDK-WM analyses.

Uses existing trial tables only. No eye-tracking or neuroimaging data are read.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
LOCAL_MODEL_CODE = ROOT
if str(LOCAL_MODEL_CODE) not in sys.path:
    sys.path.insert(0, str(LOCAL_MODEL_CODE))

from dynamic_model_utils import DiagonalLMM, LogisticRandomInterceptSlope, bh_adjust  # noqa: E402


METRICS = {
    "DDK Rate": ("DDK rate (syll/s)", "raw", "syllables/s"),
    "DDK Regularity": ("DDK regularity (ms)", "log", "% change"),
    "DDK Duration": ("DDK duration (ms)", "log", "% change"),
    "Pause Regularity": ("pause regularity (ms)", "log", "% change"),
    "Pause Duration": ("pause duration (ms)", "log", "% change"),
}


def load_dual_trials() -> pd.DataFrame:
    """Load current 3,100-row Low/High table without altering observations."""
    path = ROOT / "data_raw" / "ddkwm_trials.csv"
    data = pd.read_csv(path, dtype={"subject_id": str})
    required = {"subject_id", "ddkwm_condition", "trial_index", "wm_correct", *[x[0] for x in METRICS.values()]}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing trial-table columns: {sorted(missing)}")
    data = data.copy()
    data["condition"] = data["ddkwm_condition"].map({1: "Low", 2: "High"})
    if data["condition"].isna().any():
        raise ValueError("ddkwm_condition must contain only 1 (Low) and 2 (High).")
    data["block"] = data["condition"].map({"Low": -0.5, "High": 0.5}).astype(float)
    data["progress"] = (pd.to_numeric(data["trial_index"], errors="raise") - 1.0) / 9.0
    data["wm_correct"] = pd.to_numeric(data["wm_correct"], errors="raise").astype(float)
    if not set(data["wm_correct"].unique()).issubset({0.0, 1.0}):
        raise ValueError("wm_correct must be binary.")
    demo = pd.read_csv(ROOT / "data_raw" / "participants.csv", dtype={"subject_id": str})
    data = data.merge(demo[["subject_id", "age"]], on="subject_id", how="left", validate="many_to_one")
    if data["age"].isna().any():
        raise ValueError("Age is missing after merge.")
    age = data.drop_duplicates("subject_id")["age"].astype(float)
    data["age_z"] = (data["age"].astype(float) - age.mean()) / age.std(ddof=1)
    data = data.sort_values(["subject_id", "condition", "trial_index"]).reset_index(drop=True)
    counts = data.groupby(["subject_id", "condition"], observed=True).size().unstack(fill_value=0)
    if len(data) != 3100 or data["subject_id"].nunique() != 155 or not (counts == 10).all().all():
        raise ValueError("Expected 155 participants with 10 Low and 10 High trials each.")
    return data


def response(data: pd.DataFrame, metric: str) -> np.ndarray:
    column, scale, _ = METRICS[metric]
    values = pd.to_numeric(data[column], errors="raise").to_numpy(float)
    if (values <= 0).any() or not np.isfinite(values).all():
        raise ValueError(f"{metric} has non-positive or non-finite values.")
    return np.log(values) if scale == "log" else values


def dynamic_design(data: pd.DataFrame, include_age: bool = False):
    block = data["block"].to_numpy(float)
    progress = data["progress"].to_numpy(float)
    if include_age:
        age = data["age_z"].to_numpy(float)
        names = [
            "Intercept", "High-Low at Trial 1", "Progress", "Age (+1 SD)",
            "Block x progress", "Block x age", "Progress x age", "Block x progress x age",
        ]
        x = np.column_stack([
            np.ones(len(data)), block, progress, age, block * progress,
            block * age, progress * age, block * progress * age,
        ])
    else:
        names = ["Intercept", "High-Low at Trial 1", "Progress", "Block x progress"]
        x = np.column_stack([np.ones(len(data)), block, progress, block * progress])
    z = np.column_stack([np.ones(len(data)), block, progress])
    return x, z, names


def reported_effect(estimate: float, se: float, scale: str):
    low, high = estimate - 1.96 * se, estimate + 1.96 * se
    if scale == "log":
        return 100 * np.expm1(estimate), 100 * np.expm1(low), 100 * np.expm1(high)
    return estimate, low, high


def save_json_qc(data: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "participants": int(data["subject_id"].nunique()),
        "trials": int(len(data)),
        "low_trials": int((data["condition"] == "Low").sum()),
        "high_trials": int((data["condition"] == "High").sum()),
        "age_mean": float(data.drop_duplicates("subject_id")["age"].mean()),
        "age_sd": float(data.drop_duplicates("subject_id")["age"].std(ddof=1)),
        "age_range": [float(data["age"].min()), float(data["age"].max())],
        "input_table": "data_raw/ddkwm_trials.csv",
    }
    import json
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
