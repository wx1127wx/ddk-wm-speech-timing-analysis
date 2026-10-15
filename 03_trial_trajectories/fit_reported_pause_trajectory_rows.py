"""Compatibility export; obsolete Single-inclusive fitting has been retired.

Runs the unified five-outcome Low/High analysis and exports the two intersyllabic-gap
rows with legacy filenames. Figure 3 reads unified outputs directly.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from fit_low_high_trajectory_models import main as fit_unified

def main():
    fit_unified()
    tables = Path(__file__).resolve().parent / "tables"
    data = pd.read_csv(tables / "low_high_trajectory_models.csv")
    data = data[data.metric.str.startswith("Pause")].copy()
    data = data.rename(columns={"metric":"outcome", "estimate_model_scale":"estimate_log_scale",
                                "reported_effect":"percent_change", "CI_low":"CI_low_percent",
                                "CI_high":"CI_high_percent", "q_BH_5_outcomes":"q_BH_within_family"})
    data["term"] = data.term.replace({"High-Low at Trial 1":"High vs low", "Block x progress":"Load x progress"})
    data["FDR_family"] = "five unified Low/High condition x progress tests"
    data["source_model"] = "Low/High only; progress 0 to 1"
    data.to_csv(tables / "reported_pause_models.csv", index=False, encoding="utf-8-sig")
    pred = pd.read_csv(tables / "low_high_trial_predictions.csv")
    pred = pred[pred.metric.str.startswith("Pause")].rename(columns={"metric":"outcome"})
    pred.to_csv(tables / "reported_pause_predictions.csv", index=False, encoding="utf-8-sig")
    print("Legacy filenames exported from unified Low/High models; no Single observations.")

if __name__ == "__main__":
    main()
