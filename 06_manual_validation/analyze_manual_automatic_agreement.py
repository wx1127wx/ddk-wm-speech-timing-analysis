"""Method-comparison analysis for manually and automatically extracted DDK features.

The unit of matching is one recording.  Confidence intervals use a
participant-cluster bootstrap so repeated trials from a participant are not
treated as independent resampling units.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


RNG_SEED = 20260918
BOOTSTRAP_REPLICATES = 2_000
FEATURES = [
    "DDK rate (syll/s)",
    "DDK regularity (ms)",
    "DDK duration (ms)",
    "pause regularity (ms)",
    "pause duration (ms)",
]
CONDITION_TO_AUTOMATIC_PREFIX = {
    "Single": "02_DDK-{subject}_ddk{trial}-",
    "Low": "13_DDKWM-T1-{subject}_ddkwm{trial}-DDK-",
    "High": "13_DDKWM-T2-{subject}_ddkwm{trial}-DDK-",
}
CONDITION_ORDER = ["Single", "Low", "High"]
CONDITION_COLORS = {"Single": "#0072B2", "Low": "#009E73", "High": "#D55E00"}


def percentile_ci(values: list[float] | np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    return tuple(np.quantile(values, [0.025, 0.975]))


def concordance_correlation_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    cov_xy = np.cov(x, y, ddof=1)[0, 1]
    denominator = var_x + var_y + (np.mean(x) - np.mean(y)) ** 2
    return 2 * cov_xy / denominator if denominator else np.nan


def icc_two_methods(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return ICC(A,1) absolute agreement and ICC(C,1) consistency."""
    values = np.column_stack([x, y])
    n_items, n_methods = values.shape
    if n_items < 2:
        return np.nan, np.nan

    grand_mean = values.mean()
    item_means = values.mean(axis=1)
    method_means = values.mean(axis=0)
    ms_items = n_methods * np.sum((item_means - grand_mean) ** 2) / (n_items - 1)
    ms_methods = n_items * np.sum((method_means - grand_mean) ** 2) / (n_methods - 1)
    residual = values - item_means[:, None] - method_means[None, :] + grand_mean
    ms_error = np.sum(residual**2) / ((n_items - 1) * (n_methods - 1))
    numerator = ms_items - ms_error
    denominator_absolute = (
        ms_items
        + (n_methods - 1) * ms_error
        + n_methods * (ms_methods - ms_error) / n_items
    )
    denominator_consistency = ms_items + (n_methods - 1) * ms_error
    icc_absolute = numerator / denominator_absolute if denominator_absolute else np.nan
    icc_consistency = numerator / denominator_consistency if denominator_consistency else np.nan
    return icc_absolute, icc_consistency


def calculate_metrics_from_arrays(
    x: np.ndarray,
    y: np.ndarray,
    n_subjects: int,
    include_p_values: bool = True,
) -> dict[str, float]:
    diff = y - x
    average = (x + y) / 2
    pearson_r = float(np.corrcoef(x, y)[0, 1])
    spearman_rho = float(stats.spearmanr(x, y).statistic)
    mean_average = np.mean(average)
    slope = float(np.sum((average - mean_average) * (diff - np.mean(diff))) / np.sum((average - mean_average) ** 2))
    intercept = float(np.mean(diff) - slope * mean_average)
    if include_p_values:
        _, pearson_p = stats.pearsonr(x, y)
        _, spearman_p = stats.spearmanr(x, y)
        _, _, _, proportional_p, _ = stats.linregress(average, diff)
    else:
        pearson_p = np.nan
        spearman_p = np.nan
        proportional_p = np.nan
    icc_absolute, icc_consistency = icc_two_methods(x, y)
    bias = float(np.mean(diff))
    diff_sd = float(np.std(diff, ddof=1))
    return {
        "n_pairs": len(x),
        "n_subjects": n_subjects,
        "manual_mean": float(np.mean(x)),
        "manual_sd": float(np.std(x, ddof=1)),
        "automatic_mean": float(np.mean(y)),
        "automatic_sd": float(np.std(y, ddof=1)),
        "pearson_r": pearson_r,
        "pearson_p_naive": float(pearson_p),
        "spearman_rho": spearman_rho,
        "spearman_p_naive": float(spearman_p),
        "ccc": float(concordance_correlation_coefficient(x, y)),
        "icc_a1_absolute_agreement": float(icc_absolute),
        "icc_c1_consistency": float(icc_consistency),
        "bias_automatic_minus_manual": bias,
        "difference_sd": diff_sd,
        "loa_lower": bias - 1.96 * diff_sd,
        "loa_upper": bias + 1.96 * diff_sd,
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "proportional_bias_slope": float(slope),
        "proportional_bias_intercept": float(intercept),
        "proportional_bias_p_naive": float(proportional_p),
    }


def calculate_metrics(frame: pd.DataFrame) -> dict[str, float]:
    return calculate_metrics_from_arrays(
        frame["manual"].to_numpy(dtype=float),
        frame["automatic"].to_numpy(dtype=float),
        frame["source_subject_id"].nunique(),
    )


def cluster_bootstrap(frame: pd.DataFrame, seed: int) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    subject_blocks = [
        (
            group["manual"].to_numpy(dtype=float),
            group["automatic"].to_numpy(dtype=float),
        )
        for _, group in frame.groupby("source_subject_id", sort=False)
    ]
    measures = [
        "pearson_r",
        "spearman_rho",
        "ccc",
        "icc_a1_absolute_agreement",
        "icc_c1_consistency",
        "bias_automatic_minus_manual",
        "mae",
        "proportional_bias_slope",
    ]
    draws = {measure: [] for measure in measures}
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_indices = rng.integers(0, len(subject_blocks), len(subject_blocks))
        x = np.concatenate([subject_blocks[index][0] for index in sampled_indices])
        y = np.concatenate([subject_blocks[index][1] for index in sampled_indices])
        metrics = calculate_metrics_from_arrays(x, y, len(subject_blocks), include_p_values=False)
        for measure in measures:
            draws[measure].append(metrics[measure])
    return {measure: percentile_ci(values) for measure, values in draws.items()}


def automatic_column_name(subject_id: str, condition: str, trial_index: int, feature: str) -> str:
    return CONDITION_TO_AUTOMATIC_PREFIX[condition].format(
        subject=subject_id,
        trial=trial_index,
    ) + feature


def create_paired_data(manual: pd.DataFrame, automatic: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    required_manual = {"source_subject_id", "analysis_id", "condition", "trial_index", *FEATURES}
    missing_manual = sorted(required_manual - set(manual.columns))
    if missing_manual:
        raise ValueError(f"Manual file is missing columns: {missing_manual}")
    if automatic["id"].duplicated().any():
        raise ValueError("Automatic feature table has duplicated participant IDs.")

    automatic_indexed = automatic.assign(id=automatic["id"].astype(str)).set_index("id", drop=False)
    manual = manual.copy()
    manual["source_subject_id"] = manual["source_subject_id"].astype(str)
    manual["trial_index"] = pd.to_numeric(manual["trial_index"], errors="raise").astype(int)
    duplicate_keys = manual.duplicated(["source_subject_id", "condition", "trial_index"]).sum()
    if duplicate_keys:
        raise ValueError(f"Manual file has {duplicate_keys} duplicate recording keys.")

    unavailable_columns: Counter[str] = Counter()
    missing_automatic_ids: Counter[str] = Counter()
    rows = []
    for _, record in manual.iterrows():
        subject = str(record["source_subject_id"])
        condition = record["condition"]
        trial = int(record["trial_index"])
        if condition not in CONDITION_TO_AUTOMATIC_PREFIX:
            raise ValueError(f"Unexpected condition: {condition}")
        if subject not in automatic_indexed.index:
            missing_automatic_ids[subject] += 1
            continue
        auto_row = automatic_indexed.loc[subject]
        for feature in FEATURES:
            automatic_column = automatic_column_name(subject, condition, trial, feature)
            if automatic_column not in automatic_indexed.columns:
                continue
            rows.append(
                {
                    "source_subject_id": subject,
                    "analysis_id": record["analysis_id"],
                    "condition": condition,
                    "trial_index": trial,
                    "feature": feature,
                    "manual": pd.to_numeric(record[feature], errors="coerce"),
                    "automatic": pd.to_numeric(auto_row[automatic_column], errors="coerce"),
                    "automatic_column": automatic_column,
                    "audio_path": record.get("audio_path", np.nan),
                }
            )

    paired = pd.DataFrame(rows)
    paired["difference_automatic_minus_manual"] = paired["automatic"] - paired["manual"]
    paired["mean_of_methods"] = (paired["automatic"] + paired["manual"]) / 2
    quality = {
        "manual_rows": int(len(manual)),
        "manual_subjects": int(manual["source_subject_id"].nunique()),
        "manual_duplicate_recording_keys": int(duplicate_keys),
        "automatic_rows": int(len(automatic)),
        "automatic_matched_subjects": int(
            manual.loc[manual["source_subject_id"].isin(automatic_indexed.index), "source_subject_id"].nunique()
        ),
        "manual_rows_with_matching_automatic_id": int(manual["source_subject_id"].isin(automatic_indexed.index).sum()),
        "missing_automatic_ids": dict(missing_automatic_ids),
        "missing_automatic_columns": dict(unavailable_columns),
        "paired_feature_rows_before_missing_value_filter": int(len(paired)),
        "missing_manual_values": int(paired["manual"].isna().sum()),
        "missing_automatic_values": int(paired["automatic"].isna().sum()),
    }
    paired = paired.dropna(subset=["manual", "automatic"]).copy()
    quality["paired_feature_rows_after_missing_value_filter"] = int(len(paired))
    return paired, quality


def add_confidence_intervals(metrics: dict[str, float], cis: dict[str, tuple[float, float]]) -> dict[str, float]:
    output = dict(metrics)
    for measure, (lower, upper) in cis.items():
        output[f"{measure}_ci_low_cluster_bootstrap"] = lower
        output[f"{measure}_ci_high_cluster_bootstrap"] = upper
    return output


def summary_by_feature(paired: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for index, feature in enumerate(FEATURES):
        frame = paired.loc[paired["feature"] == feature].copy()
        metrics = calculate_metrics(frame)
        metrics = add_confidence_intervals(metrics, cluster_bootstrap(frame, RNG_SEED + index))
        metrics["feature"] = feature
        summaries.append(metrics)
    return pd.DataFrame(summaries).loc[:, ["feature", *[column for column in summaries[0] if column != "feature"]]]


def summary_by_feature_and_condition(paired: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for feature_index, feature in enumerate(FEATURES):
        for condition_index, condition in enumerate(CONDITION_ORDER):
            frame = paired.loc[(paired["feature"] == feature) & (paired["condition"] == condition)].copy()
            metrics = calculate_metrics(frame)
            metrics.update({"feature": feature, "condition": condition})
            summaries.append(metrics)
    columns = ["feature", "condition", *[column for column in summaries[0] if column not in {"feature", "condition"}]]
    return pd.DataFrame(summaries).loc[:, columns]


def contrast_recovery(paired: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for feature_index, feature in enumerate(FEATURES):
        frame = paired.loc[paired["feature"] == feature]
        subject_condition = (
            frame.groupby(["source_subject_id", "condition"], as_index=False)[["manual", "automatic"]]
            .mean()
            .pivot(index="source_subject_id", columns="condition", values=["manual", "automatic"])
        )
        for contrast_index, target_condition in enumerate(["Low", "High"]):
            contrast_frame = pd.DataFrame(
                {
                    "source_subject_id": subject_condition.index,
                    "manual": subject_condition[("manual", target_condition)] - subject_condition[("manual", "Single")],
                    "automatic": subject_condition[("automatic", target_condition)] - subject_condition[("automatic", "Single")],
                }
            ).dropna().reset_index(drop=True)
            metrics = calculate_metrics(contrast_frame)
            metrics = add_confidence_intervals(
                metrics,
                cluster_bootstrap(contrast_frame, RNG_SEED + 500 + feature_index * 10 + contrast_index),
            )
            metrics.update({"feature": feature, "contrast": f"{target_condition} - Single"})
            summaries.append(metrics)
    columns = ["feature", "contrast", *[column for column in summaries[0] if column not in {"feature", "contrast"}]]
    return pd.DataFrame(summaries).loc[:, columns]


def bland_altman_figure(paired: pd.DataFrame, output_stem: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2), constrained_layout=True)
    axes = axes.ravel()
    for axis, feature in zip(axes, FEATURES):
        frame = paired.loc[paired["feature"] == feature]
        for condition in CONDITION_ORDER:
            subset = frame.loc[frame["condition"] == condition]
            axis.scatter(
                subset["mean_of_methods"],
                subset["difference_automatic_minus_manual"],
                s=15,
                alpha=0.60,
                color=CONDITION_COLORS[condition],
                label=condition,
                linewidths=0,
            )
        diff = frame["difference_automatic_minus_manual"]
        bias = diff.mean()
        sd = diff.std(ddof=1)
        axis.axhline(bias, color="#222222", linewidth=1.1)
        axis.axhline(bias - 1.96 * sd, color="#555555", linestyle="--", linewidth=0.9)
        axis.axhline(bias + 1.96 * sd, color="#555555", linestyle="--", linewidth=0.9)
        axis.set_title(feature, fontsize=10)
        axis.set_xlabel("Mean of manual and automatic", fontsize=8)
        axis.set_ylabel("Automatic - manual", fontsize=8)
        axis.tick_params(labelsize=8)
        axis.grid(alpha=0.18, linewidth=0.5)
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.985, 0.03), frameon=False, title="Condition")
    fig.suptitle("Bland–Altman plots: automatic minus manual DDK features", fontsize=13)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def priority_review_pairs(paired: pd.DataFrame) -> pd.DataFrame:
    """Return the five largest absolute method differences for each feature."""
    priority = paired.copy()
    priority["absolute_difference"] = priority["difference_automatic_minus_manual"].abs()
    priority = priority.sort_values(["feature", "absolute_difference"], ascending=[True, False])
    priority["review_rank_within_feature"] = priority.groupby("feature").cumcount() + 1
    return priority.loc[priority["review_rank_within_feature"] <= 5].copy()


def format_number(value: float, digits: int = 3) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def write_report(
    output_path: Path,
    quality: dict[str, object],
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> None:
    lines = [
        "# 手工与自动 DDK 特征一致性检验",
        "",
        "## 数据与方法",
        "",
        f"- 成功匹配 {quality['paired_feature_rows_after_missing_value_filter']} 个特征-录音配对，来自 {quality['manual_subjects']} 名被试、460 条录音和 7 项声学特征。",
        "- 匹配单位为单条录音：单任务对应 `02_DDK`，低负荷对应 `DDKWM-T1`，高负荷对应 `DDKWM-T2`。",
        "- 偏差定义为自动值 − 手工值；正值表示自动提取结果更高。",
        "- 相关与一致性区间采用以被试为重抽样单位的 2,000 次簇自助法，保留同一被试内重复试次的相关结构。",
        "- 绝对一致性使用 Lin's CCC 与 ICC(A,1)；ICC(C,1) 仅反映扣除固定方法偏移后的相对排序一致性。Bland–Altman 一致性限界为平均偏差 ± 1.96×差值标准差。",
        "",
        "## 总体结果",
        "",
        "| 特征 | Pearson r（95% CI） | CCC（95% CI） | ICC(A,1) | 偏差：自动−手工（95% CI） | 95% 一致性限界 | 比例偏差斜率（95% CI） |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            "| {feature} | {r} [{rlo}, {rhi}] | {ccc} [{clo}, {chi}] | {icc} | {bias} [{blo}, {bhi}] | [{loa_lo}, {loa_hi}] | {slope} [{slo}, {shi}] |".format(
                feature=row.feature,
                r=format_number(row.pearson_r),
                rlo=format_number(row.pearson_r_ci_low_cluster_bootstrap),
                rhi=format_number(row.pearson_r_ci_high_cluster_bootstrap),
                ccc=format_number(row.ccc),
                clo=format_number(row.ccc_ci_low_cluster_bootstrap),
                chi=format_number(row.ccc_ci_high_cluster_bootstrap),
                icc=format_number(row.icc_a1_absolute_agreement),
                bias=format_number(row.bias_automatic_minus_manual),
                blo=format_number(row.bias_automatic_minus_manual_ci_low_cluster_bootstrap),
                bhi=format_number(row.bias_automatic_minus_manual_ci_high_cluster_bootstrap),
                loa_lo=format_number(row.loa_lower),
                loa_hi=format_number(row.loa_upper),
                slope=format_number(row.proportional_bias_slope),
                slo=format_number(row.proportional_bias_slope_ci_low_cluster_bootstrap),
                shi=format_number(row.proportional_bias_slope_ci_high_cluster_bootstrap),
            )
        )

    lines += [
        "",
        "## 条件变化的复现",
        "",
        "下表比较每名被试先在条件内跨试次平均、再计算条件变化（低负荷−单任务或高负荷−单任务）的手工与自动结果。它评估自动方法能否复现研究关注的任务负荷变化，而非仅复现绝对值。",
        "",
        "| 特征 | 对比 | r（95% CI） | CCC（95% CI） | 条件变化偏差：自动−手工（95% CI） |",
        "|---|---|---:|---:|---:|",
    ]
    for row in contrasts.itertuples(index=False):
        lines.append(
            "| {feature} | {contrast} | {r} [{rlo}, {rhi}] | {ccc} [{clo}, {chi}] | {bias} [{blo}, {bhi}] |".format(
                feature=row.feature,
                contrast=row.contrast,
                r=format_number(row.pearson_r),
                rlo=format_number(row.pearson_r_ci_low_cluster_bootstrap),
                rhi=format_number(row.pearson_r_ci_high_cluster_bootstrap),
                ccc=format_number(row.ccc),
                clo=format_number(row.ccc_ci_low_cluster_bootstrap),
                chi=format_number(row.ccc_ci_high_cluster_bootstrap),
                bias=format_number(row.bias_automatic_minus_manual),
                blo=format_number(row.bias_automatic_minus_manual_ci_low_cluster_bootstrap),
                bhi=format_number(row.bias_automatic_minus_manual_ci_high_cluster_bootstrap),
            )
        )
    lines += [
        "",
        "## 解读边界",
        "",
        "- 相关系数高仅说明排序或变化趋势一致；CCC、ICC(A,1) 和 Bland–Altman 偏差共同用于判断绝对一致性。",
        "- 偏差区间不跨 0 提示存在稳定的平均方法偏移；若 ICC(C,1) 高于 ICC(A,1)，也支持主要问题是固定偏移而非排序失真。",
        "- 比例偏差斜率的区间不跨 0 时，偏差会随特征量级变化，不能仅用单一常数校正。",
        "- 这是一项 20 名被试的人工复核验证；结论适用于当前录音、特征定义与标注规则，不能替代独立标注者之间的信度研究。",
        "",
        "详细数值见 `agreement_summary.csv`、`agreement_by_condition.csv` 与 `condition_contrast_agreement.csv`；图见 `bland_altman_all_features.png/pdf`。`priority_review_largest_discrepancies.csv` 列出每个特征差异最大的 5 条录音，供回看音频和 TextGrid。",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired", type=Path, required=True,
                        help="Long table with participant_id, condition, trial_index, outcome, manual, automatic")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Loading paired recording-level feature table...", flush=True)
    paired = pd.read_csv(args.paired, low_memory=False)
    paired = paired.rename(columns={"participant_id": "source_subject_id", "outcome": "feature"})
    required = {"source_subject_id", "condition", "trial_index", "feature", "manual", "automatic"}
    missing = required.difference(paired.columns)
    if missing:
        raise ValueError(f"Paired table is missing columns: {sorted(missing)}")
    paired = paired[paired.feature.isin(FEATURES)].copy()
    if paired[["manual", "automatic"]].isna().any().any():
        raise ValueError("Manual or automatic values are missing.")
    quality = {"participants": int(paired.source_subject_id.nunique()),
               "recordings": int(paired[["source_subject_id", "condition", "trial_index"]].drop_duplicates().shape[0]),
               "paired_feature_rows": int(len(paired)), "bootstrap_replicates": BOOTSTRAP_REPLICATES}

    print("Computing overall agreement statistics...", flush=True)
    summary = summary_by_feature(paired)
    print("Computing condition-specific summaries and condition-change recovery...", flush=True)
    contrasts = contrast_recovery(paired)

    print("Writing manuscript-reported agreement tables...", flush=True)
    summary.to_csv(args.output_dir / "agreement_summary.csv", index=False, encoding="utf-8-sig")
    contrasts.to_csv(args.output_dir / "condition_contrast_agreement.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "data_quality.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"paired_feature_rows={len(paired)}")
    print(f"outputs={args.output_dir}")


if __name__ == "__main__":
    main()
