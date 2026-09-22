"""CMMS analyses reported in the final DDK manuscript.

CMMS is evaluated as a secondary/exploratory moderator of condition means and
Low/High repeated-trial trajectories. Missing scores are never imputed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
from dynamic_model_utils import DiagonalLMM, bh_adjust


SCALE_INFO = {
    "CMMS": {
        "folder": ".",
        "label": "CMMS total score",
        "direction": "higher scores indicate better global cognitive performance",
        "color": "#0072B2",
    },
}

METRICS = {
    "DDK Rate": ("rate", "raw", "syllables/s"),
    "DDK Regularity": ("reg", "log", "% change"),
    "DDK Duration": ("dur", "log", "% change"),
    "Pause Regularity": ("pause_reg", "log", "% change"),
    "Pause Duration": ("pause_dur", "log", "% change"),
}
RENAME = {
    "subject_id": "analysis_id",
    "DDK rate (syll/s)": "rate",
    "DDK regularity (ms)": "reg",
    "DDK duration (ms)": "dur",
    "pause regularity (ms)": "pause_reg",
    "pause duration (ms)": "pause_dur",
}
CONDITIONS = ["Single", "Low", "High"]
RNG_SEED = 20260917
BOOTSTRAP_N = 2000


def zscore(series: pd.Series) -> pd.Series:
    sd = series.std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError(f"Cannot standardize {series.name}: SD={sd}")
    return (series - series.mean()) / sd


def rank_zscore(series: pd.Series) -> pd.Series:
    ranks = stats.rankdata(series, method="average")
    q = (ranks - 0.5) / len(ranks)
    return pd.Series(stats.norm.ppf(q), index=series.index, name=series.name)


def effect_transform(est: float, lo: float, hi: float, scale: str) -> tuple[float, float, float]:
    if scale == "log":
        return tuple(float(100 * np.expm1(x)) for x in (est, lo, hi))
    return float(est), float(lo), float(hi)


def p_text(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    return "<.001" if value < .001 else f"{value:.3f}".lstrip("0")


def partial_spearman(data: pd.DataFrame, x: str, y: str, covars: list[str]) -> tuple[float, float, int]:
    cols = [x, y, *covars]
    work = data[cols].dropna().copy()
    ranked = work.rank(method="average")
    design = np.column_stack([np.ones(len(ranked)), ranked[covars].to_numpy(float)])
    rx = ranked[x].to_numpy(float) - design @ np.linalg.lstsq(design, ranked[x], rcond=None)[0]
    ry = ranked[y].to_numpy(float) - design @ np.linalg.lstsq(design, ranked[y], rcond=None)[0]
    r = float(stats.pearsonr(rx, ry).statistic)
    df = len(work) - len(covars) - 2
    t = r * math.sqrt(df / max(1e-15, 1 - r * r))
    p = float(2 * stats.t.sf(abs(t), df))
    return r, p, len(work)


def bootstrap_partial_ci(data: pd.DataFrame, x: str, y: str, covars: list[str], seed: int) -> tuple[float, float]:
    work = data[[x, y, *covars]].dropna().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(BOOTSTRAP_N):
        sample = work.iloc[rng.integers(0, len(work), len(work))]
        try:
            values.append(partial_spearman(sample, x, y, covars)[0])
        except Exception:
            continue
    return tuple(np.quantile(values, [0.025, 0.975]))


def load_data(root: Path, scale: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    demo_path = root / "data_raw" / "participants.csv"
    means_path = root / "01_sample_and_condition_means" / "data_processed" / "speech_condition_means.csv"
    trial_path = root / "data_raw" / "ddkwm_trials.csv"
    for path in (demo_path, means_path, trial_path):
        if not path.exists():
            raise FileNotFoundError(path)

    demo = pd.read_csv(demo_path, dtype={"subject_id": str}).rename(columns={"subject_id": "analysis_id", "age": "age_years", "edu": "education_years"})
    means = pd.read_csv(means_path, dtype={"analysis_id": str})
    trials = pd.read_csv(trial_path, dtype={"subject_id": str}).rename(columns=RENAME)
    trials["condition"] = trials["ddkwm_condition"].map({1: "Low", 2: "High"})

    demo[scale] = pd.to_numeric(demo[scale], errors="coerce")
    for col in ["age_years", "education_years"]:
        demo[col] = pd.to_numeric(demo[col], errors="coerce")
    ids = set(demo.loc[demo[scale].notna(), "analysis_id"])
    if not ids:
        raise ValueError(f"No nonmissing {scale} scores")

    selected = demo.loc[demo.analysis_id.isin(ids)].copy()
    selected["scale_z"] = zscore(selected[scale])
    selected["scale_rank_z"] = rank_zscore(selected[scale])
    selected["age_z"] = zscore(selected["age_years"])
    selected["education_z"] = zscore(selected["education_years"])

    means = means.drop(columns=[c for c in ["age_years", "age_z"] if c in means.columns]).merge(
        selected[["analysis_id", scale, "scale_z", "scale_rank_z", "age_years", "education_years", "age_z", "education_z"]],
        on="analysis_id", how="inner", validate="many_to_one"
    )
    means["duality"] = means["condition"].map({"Single": -2 / 3, "Low": 1 / 3, "High": 1 / 3}).astype(float)
    means["block"] = means["condition"].map({"Single": 0.0, "Low": -0.5, "High": 0.5}).astype(float)

    metric_cols = [v[0] for v in METRICS.values()]
    trials = trials[["analysis_id", "condition", "trial_index", *metric_cols]].merge(
        selected[["analysis_id", scale, "scale_z", "scale_rank_z", "age_years", "education_years", "age_z", "education_z"]],
        on="analysis_id", how="inner", validate="many_to_one"
    )
    trials["block"] = trials["condition"].map({"Low": -0.5, "High": 0.5}).astype(float)
    trials["progress"] = (trials["trial_index"] - 1) / 9 - 0.5

    expected_means = len(selected) * 3
    expected_trials = len(selected) * 20
    if len(means) != expected_means or len(trials) != expected_trials:
        raise ValueError(f"Incomplete repeated measures: means={len(means)}/{expected_means}, trials={len(trials)}/{expected_trials}")
    if means[metric_cols].isna().any().any() or trials[metric_cols].isna().any().any():
        raise ValueError("Speech outcomes contain missing values")
    if (means[metric_cols] <= 0).any().any() or (trials[metric_cols] <= 0).any().any():
        raise ValueError("Speech outcomes contain non-positive values")
    return demo, means, trials


def scale_description(demo: pd.DataFrame, scale: str) -> pd.DataFrame:
    x = demo[scale].dropna()
    total_n = len(demo)
    row = {
        "scale": scale, "total_n": total_n, "available_n": len(x), "missing_n": total_n - len(x),
        "missing_percent": 100 * (total_n - len(x)) / total_n,
        "mean": x.mean(), "SD": x.std(ddof=1), "median": x.median(), "Q1": x.quantile(.25), "Q3": x.quantile(.75),
        "minimum": x.min(), "maximum": x.max(), "skewness": stats.skew(x, bias=False), "kurtosis_excess": stats.kurtosis(x, bias=False),
        "n_at_minimum": int((x == x.min()).sum()), "n_at_maximum": int((x == x.max()).sum()),
        "floor_percent_observed_min": 100 * (x == x.min()).mean(), "ceiling_percent_observed_max": 100 * (x == x.max()).mean(),
        "unique_values": int(x.nunique()),
    }
    return pd.DataFrame([row])


def association_analysis(demo: pd.DataFrame, scale: str) -> pd.DataFrame:
    targets = ["age_years", "education_years", *[s for s in SCALE_INFO if s != scale]]
    rows = []
    for i, target in enumerate(targets):
        columns = list(dict.fromkeys([scale, target, "age_years", "education_years"]))
        work = demo[columns].dropna()
        r, p = stats.spearmanr(work[scale], work[target])
        row = {"scale": scale, "target": target, "analysis": "Spearman", "n": len(work), "rho": float(r), "p": float(p), "CI_low": np.nan, "CI_high": np.nan}
        rows.append(row)
        if target not in {"age_years", "education_years"}:
            pr, pp, n = partial_spearman(work, scale, target, ["age_years", "education_years"])
            lo, hi = bootstrap_partial_ci(work, scale, target, ["age_years", "education_years"], RNG_SEED + i)
            rows.append({"scale": scale, "target": target, "analysis": "Partial Spearman adjusted for age and education", "n": n, "rho": pr, "p": pp, "CI_low": lo, "CI_high": hi})
    out = pd.DataFrame(rows)
    out["q_BH_within_scale"] = bh_adjust(out["p"].to_numpy(float))
    return out


def fit_lmm(y: np.ndarray, X: np.ndarray, Z: np.ndarray, groups: pd.Series, fixed_names: list[str], random_names: list[str]) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = DiagonalLMM(y, X, Z, groups, fixed_names, random_names, reml=True).fit()
    if not bool(result["optimizer"].success):
        warnings.warn(f"Optimizer did not report success: {result['optimizer'].message}")
    return result


def mean_models(means: pd.DataFrame, score_column: str, analysis_label: str) -> pd.DataFrame:
    rows = []
    fixed_names = ["Intercept", "Dual vs Single", "High vs Low", "Scale (+1 SD)", "Age (+1 SD)", "Education (+1 SD)", "Duality x scale", "High-Low x scale"]
    for metric, (column, outcome_scale, unit) in METRICS.items():
        d = means.copy()
        y = np.log(d[column].to_numpy(float)) if outcome_scale == "log" else d[column].to_numpy(float)
        s = d[score_column].to_numpy(float); dual = d.duality.to_numpy(float); block = d.block.to_numpy(float)
        X = np.column_stack([np.ones(len(d)), dual, block, s, d.age_z, d.education_z, dual * s, block * s])
        Z = np.column_stack([np.ones(len(d)), dual, block])
        result = fit_lmm(y, X, Z, d.analysis_id, fixed_names, ["Intercept", "Duality", "Block"])
        for name, est, se, z, p in zip(fixed_names, result["beta"], result["se"], result["z"], result["p"]):
            lo, hi = est - 1.96 * se, est + 1.96 * se
            rep, rep_lo, rep_hi = effect_transform(est, lo, hi, outcome_scale)
            rows.append({"analysis": analysis_label, "metric": metric, "outcome_scale": outcome_scale, "term": name, "estimate_model_scale": est, "SE": se, "z": z, "p": p, "reported_effect": rep, "CI_low": rep_lo, "CI_high": rep_hi, "unit": unit, "n_participants": d.analysis_id.nunique(), "n_observations": len(d), "optimizer_success": bool(result["optimizer"].success)})
    out = pd.DataFrame(rows)
    out["q_BH_5_outcomes"] = np.nan
    for term in ["Duality x scale", "High-Low x scale"]:
        mask = out.term.eq(term)
        out.loc[mask, "q_BH_5_outcomes"] = bh_adjust(out.loc[mask, "p"].to_numpy(float))
    return out


def trial_models(trials: pd.DataFrame, score_column: str, analysis_label: str) -> pd.DataFrame:
    rows = []
    fixed_names = ["Intercept", "Block", "Progress", "Scale (+1 SD)", "Age (+1 SD)", "Education (+1 SD)", "Block x progress", "Block x scale", "Progress x scale", "Block x progress x scale"]
    for metric, (column, outcome_scale, unit) in METRICS.items():
        d = trials.copy()
        y = np.log(d[column].to_numpy(float)) if outcome_scale == "log" else d[column].to_numpy(float)
        b = d.block.to_numpy(float); pgr = d.progress.to_numpy(float); s = d[score_column].to_numpy(float)
        X = np.column_stack([np.ones(len(d)), b, pgr, s, d.age_z, d.education_z, b * pgr, b * s, pgr * s, b * pgr * s])
        Z = np.column_stack([np.ones(len(d)), b, pgr])
        result = fit_lmm(y, X, Z, d.analysis_id, fixed_names, ["Intercept", "Block", "Progress"])
        for name, est, se, z, p in zip(fixed_names, result["beta"], result["se"], result["z"], result["p"]):
            lo, hi = est - 1.96 * se, est + 1.96 * se
            rep, rep_lo, rep_hi = effect_transform(est, lo, hi, outcome_scale)
            rows.append({"analysis": analysis_label, "metric": metric, "outcome_scale": outcome_scale, "term": name, "estimate_model_scale": est, "SE": se, "z": z, "p": p, "reported_effect": rep, "CI_low": rep_lo, "CI_high": rep_hi, "unit": unit, "n_participants": d.analysis_id.nunique(), "n_observations": len(d), "optimizer_success": bool(result["optimizer"].success)})
    out = pd.DataFrame(rows)
    out["q_BH_5_outcomes"] = np.nan
    for term in ["Block x scale", "Progress x scale", "Block x progress x scale"]:
        mask = out.term.eq(term)
        out.loc[mask, "q_BH_5_outcomes"] = bh_adjust(out.loc[mask, "p"].to_numpy(float))
    return out


def participant_slope_analysis(trials: pd.DataFrame, scale: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    slope_rows = []
    for metric, (column, outcome_scale, _) in METRICS.items():
        d = trials.copy()
        d["outcome"] = np.log(d[column]) if outcome_scale == "log" else d[column]
        for participant, part in d.groupby("analysis_id"):
            slopes = {}
            for condition, group in part.groupby("condition"):
                slopes[condition] = float(stats.linregress(group.progress, group.outcome).slope)
            slope_rows.append({"analysis_id": participant, "metric": metric, "Low_slope": slopes["Low"], "High_slope": slopes["High"], "High_minus_Low_slope": slopes["High"] - slopes["Low"]})
    slopes = pd.DataFrame(slope_rows).merge(trials[["analysis_id", scale, "age_years", "education_years"]].drop_duplicates(), on="analysis_id", validate="many_to_one")
    rows = []
    for i, (metric, group) in enumerate(slopes.groupby("metric", sort=False)):
        r, p, n = partial_spearman(group, scale, "High_minus_Low_slope", ["age_years", "education_years"])
        lo, hi = bootstrap_partial_ci(group, scale, "High_minus_Low_slope", ["age_years", "education_years"], RNG_SEED + 100 + i)
        influential_id = group.loc[group.High_minus_Low_slope.abs().idxmax(), "analysis_id"]
        reduced = group[group.analysis_id.ne(influential_id)]
        r_red, p_red, _ = partial_spearman(reduced, scale, "High_minus_Low_slope", ["age_years", "education_years"])
        rows.append({"metric": metric, "n": n, "partial_rho": r, "CI_low_bootstrap": lo, "CI_high_bootstrap": hi, "p": p, "removed_for_influence_check": influential_id, "partial_rho_after_removal": r_red, "p_after_removal": p_red})
    out = pd.DataFrame(rows)
    out["q_BH_5_outcomes"] = bh_adjust(out.p.to_numpy(float))
    return slopes, out


def make_distribution_figure(demo: pd.DataFrame, scale: str, outpath: Path) -> None:
    cfg = SCALE_INFO[scale]; x = demo[scale].dropna().to_numpy(float)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), gridspec_kw={"width_ratios": [2.2, 1]})
    axes[0].hist(x, bins="auto", color=cfg["color"], alpha=.78, edgecolor="white", linewidth=.8)
    axes[0].axvline(np.mean(x), color="#263238", linewidth=1.5, label=f"Mean = {np.mean(x):.2f}")
    axes[0].axvline(np.median(x), color="#263238", linewidth=1.2, linestyle="--", label=f"Median = {np.median(x):.2f}")
    axes[0].set(xlabel=cfg["label"], ylabel="Participants", title="Score distribution")
    axes[0].legend(frameon=False, fontsize=8)
    rng = np.random.default_rng(RNG_SEED)
    axes[1].boxplot(x, vert=True, widths=.35, showfliers=False, patch_artist=True, boxprops={"facecolor": "white", "edgecolor": cfg["color"]}, medianprops={"color": "#263238", "linewidth": 1.4}, whiskerprops={"color": cfg["color"]}, capprops={"color": cfg["color"]})
    axes[1].scatter(rng.normal(1, .045, len(x)), x, s=14, alpha=.42, color=cfg["color"], linewidth=0)
    axes[1].set(xticks=[], ylabel=cfg["label"], title="Individual scores")
    for ax in axes:
        ax.grid(axis="y", color="#D9DEE2", linewidth=.6); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{scale}: observed measurement distribution (n = {len(x)})", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(outpath.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def forest_figure(table: pd.DataFrame, terms: list[str], labels: list[str], scale: str, title: str, outpath: Path) -> None:
    cfg = SCALE_INFO[scale]
    fig, axes = plt.subplots(1, len(terms), figsize=(5.3 * len(terms), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    metric_order = list(METRICS)
    for ax, term, label in zip(axes, terms, labels):
        d = table[table.term.eq(term)].set_index("metric").loc[metric_order].reset_index()
        y = np.arange(len(d))[::-1]
        ax.axvline(0, color="#66727A", linewidth=.9)
        ax.errorbar(d.reported_effect, y, xerr=[d.reported_effect - d.CI_low, d.CI_high - d.reported_effect], fmt="o", color=cfg["color"], ecolor=cfg["color"], capsize=3, markersize=5)
        for xi, yi, q in zip(d.reported_effect, y, d.q_BH_5_outcomes):
            ax.annotate(f"q={p_text(q)}", (xi, yi), xytext=(5, 5), textcoords="offset points", fontsize=7, color="#455A64")
        ax.set(yticks=y, yticklabels=metric_order, xlabel="Effect per 1-SD higher score\n(rate: syllables/s; others: % change)", title=label)
        ax.grid(axis="x", color="#D9DEE2", linewidth=.6); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(outpath.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_results(scale: str, outdir: Path, desc: pd.DataFrame, assoc: pd.DataFrame, mean_primary: pd.DataFrame, trial_primary: pd.DataFrame, mean_rank: pd.DataFrame, trial_rank: pd.DataFrame, slope_summary: pd.DataFrame) -> None:
    cfg = SCALE_INFO[scale]; d = desc.iloc[0]
    mean_key = mean_primary[mean_primary.term.isin(["Duality x scale", "High-Low x scale"])]
    trial_key = trial_primary[trial_primary.term.isin(["Block x scale", "Progress x scale", "Block x progress x scale"])]
    primary_key = pd.concat([mean_key, trial_key], ignore_index=True)
    rank_key = pd.concat([mean_rank, trial_rank], ignore_index=True)
    significant = primary_key[primary_key.q_BH_5_outcomes.lt(.05)].copy()
    robust = []
    for _, row in significant.iterrows():
        match = rank_key[(rank_key.metric.eq(row.metric)) & (rank_key.term.eq(row.term))]
        robust.append(bool(len(match) and match.iloc[0].q_BH_5_outcomes < .05))
    significant["rank_robust"] = robust
    lines = [
        f"# {scale} 独立分析结果", "",
        "## 定位与解释边界", "",
        f"本模块把 {cfg['label']} 作为次要、探索性个体差异变量；{cfg['direction']}。它用于检验认知水平是否与当前文章两阶段DDK效应的个体差异相关，而不是用于临床诊断。", "",
        "仅有量表总分，没有条目级数据、重复测量或外部金标准，因此本模块**不能**估计 Cronbach's α、McDonald's ω、重测信度或诊断效度。这里报告的是分布/测量特征、量表间关联以及与DDK效应的关联性证据。", "",
        "## 数据与缺失", "",
        f"共有 {int(d.total_n)} 名受试者，其中 {scale} 可用 {int(d.available_n)} 名，缺失 {int(d.missing_n)} 名（{d.missing_percent:.1f}%）。缺失分数未插补。" + (" DST与IQCODE只在同一63人子样本中可用，因此属于结构性子样本，不能把未测量者视为随机缺失。" if scale in {"DST", "IQCODE"} else ""), "",
        f"均值 {d['mean']:.2f}（SD {d.SD:.2f}），中位数 {d['median']:.2f}（IQR {d.Q1:.2f}–{d.Q3:.2f}），范围 {d.minimum:.2f}–{d.maximum:.2f}。观测最大值占 {d.ceiling_percent_observed_max:.1f}%。", "",
        "## 主要发现摘要", "",
    ]
    if significant.empty:
        lines += ["预设交互项在五个语音指标内进行BH-FDR校正后，没有达到 q < .05 的量表调节效应。该结果不支持当前数据中存在稳定的量表相关DDK效应差异。", ""]
    else:
        lines += [f"主分析共有 {len(significant)} 个量表调节项达到 q < .05，其中 {int(significant.rank_robust.sum())} 个在秩正态量表编码下仍达到 q < .05。", "", "|指标|交互项|方向性估计|FDR q|秩转换复现|", "|---|---|---:|---:|---|"]
        for _, r in significant.iterrows():
            lines.append(f"|{r.metric}|{r.term}|{r.reported_effect:.2f} {r.unit}|{p_text(r.q_BH_5_outcomes)}|{'是' if r.rank_robust else '否'}|")
        lines += ["", "这些结果表示统计关联，不构成认知机制、临床效度或纯负荷因果效应的证据；未能在秩转换中复现的信号尤其应谨慎处理。", ""]
    lines += [
        "## 关联效度/构念关联", "",
        "|变量|分析|n|ρ|95% CI|p|FDR q|", "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in assoc.iterrows():
        ci = "—" if pd.isna(r.CI_low) else f"[{r.CI_low:.2f}, {r.CI_high:.2f}]"
        lines.append(f"|{r.target}|{r.analysis}|{int(r.n)}|{r.rho:.2f}|{ci}|{p_text(r.p)}|{p_text(r.q_BH_within_scale)}|")
    lines += ["", "## 第一阶段：条件均值调节", "", "模型：`g(Y_mean) ~ duality + ordered_block + scale_z + age_z + education_z + duality×scale_z + ordered_block×scale_z + (1 + duality + ordered_block || participant)`。Duality表示平均双任务条件与Single的差异；ordered block表示High–Low固定次序对比。", "", "|指标|交互项|效应/1 SD|95% CI|p|FDR q|", "|---|---|---:|---:|---:|---:|"]
    for _, r in mean_key.iterrows():
        lines.append(f"|{r.metric}|{r.term}|{r.reported_effect:.2f}|[{r.CI_low:.2f}, {r.CI_high:.2f}]|{p_text(r.p)}|{p_text(r.q_BH_5_outcomes)}|")
    lines += ["", "## 第二阶段：Low/High重复试次轨迹调节", "", "模型：`g(Y_trial) ~ block × progress × scale_z + age_z + education_z + (1 + block + progress || participant)`。progress以区块中点为0；High固定在Low之后，因此Block相关项描述区块/次序相关差异，不能单独解释为负荷因果效应。", "", "|指标|交互项|效应/1 SD|95% CI|p|FDR q|", "|---|---|---:|---:|---:|---:|"]
    for _, r in trial_key.iterrows():
        lines.append(f"|{r.metric}|{r.term}|{r.reported_effect:.2f}|[{r.CI_low:.2f}, {r.CI_high:.2f}]|{p_text(r.p)}|{p_text(r.q_BH_5_outcomes)}|")
    lines += ["", "## 被试轨迹差异的敏感性分析", "", "对每名受试者分别估计Low和High区块内斜率，并计算High−Low斜率差；再计算控制年龄和教育后的偏Spearman相关。", "", "|指标|偏ρ|bootstrap 95% CI|p|FDR q|移除最大绝对斜率差后ρ|", "|---|---:|---:|---:|---:|---:|"]
    for _, r in slope_summary.iterrows():
        lines.append(f"|{r.metric}|{r.partial_rho:.2f}|[{r.CI_low_bootstrap:.2f}, {r.CI_high_bootstrap:.2f}]|{p_text(r.p)}|{p_text(r.q_BH_5_outcomes)}|{r.partial_rho_after_removal:.2f}|")
    lines += ["", "## 稳健性与报告原则", "", "主分析使用原始量表的z分数；表 `Table_05_rank_score_sensitivity.csv` 使用秩正态分数重复关键模型。若显著性仅出现在单一编码、单一指标或影响点移除前，应视为不稳定探索性信号。每个预设交互项分别在五个语音指标内进行BH-FDR校正。", "", "量表分析不改变01–06的主分析结论，也不应被表述为量表的正式心理测量学验证。"]
    (outdir / "results_CN.md").write_text("\n".join(lines), encoding="utf-8")


def run(scale: str, root: Path, output_root: Path) -> None:
    cfg = SCALE_INFO[scale]
    outdir = output_root / cfg["folder"]
    tables = outdir / "tables"; figures = outdir / "figures"; qcdir = outdir / "qc"
    for p in (tables, figures, qcdir): p.mkdir(parents=True, exist_ok=True)

    demo, means, trials = load_data(root, scale)
    desc = scale_description(demo, scale)
    assoc = association_analysis(demo.rename(columns={"age": "age_years", "edu": "education_years"}), scale)
    mean_primary = mean_models(means, "scale_z", "Primary z-score")
    trial_primary = trial_models(trials, "scale_z", "Primary z-score")
    mean_rank = mean_models(means, "scale_rank_z", "Rank-normal score sensitivity")
    trial_rank = trial_models(trials, "scale_rank_z", "Rank-normal score sensitivity")
    desc.to_csv(tables / "Table_01_scale_distribution.csv", index=False, encoding="utf-8-sig")
    assoc.to_csv(tables / "Table_02_association_validity.csv", index=False, encoding="utf-8-sig")
    mean_primary.to_csv(tables / "Table_03_condition_mean_moderation.csv", index=False, encoding="utf-8-sig")
    trial_primary.to_csv(tables / "Table_04_trial_level_moderation.csv", index=False, encoding="utf-8-sig")
    pd.concat([mean_rank, trial_rank], ignore_index=True).to_csv(tables / "Table_05_rank_score_sensitivity.csv", index=False, encoding="utf-8-sig")

    qc = {
        "scale": scale, "input_tables": {
            "demographics": "data_raw/participants.csv",
            "condition_means": "01_sample_and_condition_means/data_processed/speech_condition_means.csv",
            "dual_task_trials": "data_raw/ddkwm_trials.csv",
        },
        "available_n": int(means.analysis_id.nunique()), "condition_mean_rows": len(means), "trial_rows": len(trials),
        "missing_scores_not_imputed": True, "age_and_education_adjusted": True, "bootstrap_repetitions": BOOTSTRAP_N,
        "all_primary_models_optimizer_success": bool(mean_primary.optimizer_success.all() and trial_primary.optimizer_success.all()),
        "all_rank_models_optimizer_success": bool(mean_rank.optimizer_success.all() and trial_rank.optimizer_success.all()),
        "random_effect_structure": "diagonal (uncorrelated) random intercept + planned within-participant slopes",
        "fixed_order_warning": "High always followed Low; High-Low terms are ordered-block contrasts, not pure causal load effects.",
    }
    (qcdir / "analysis_qc.json").write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"scale": scale, "output": str(outdir), "n": qc["available_n"], "primary_models_ok": qc["all_primary_models_optimizer_success"], "rank_models_ok": qc["all_rank_models_optimizer_success"]}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=list(SCALE_INFO), required=True)
    parser.add_argument("--analysis-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    script_root = Path(__file__).resolve().parents[2]
    analysis_root = args.analysis_root.resolve() if args.analysis_root else script_root
    output_root = args.output_root.resolve() if args.output_root else analysis_root / "05_cmms_moderation"
    run(args.scale, analysis_root, output_root)


if __name__ == "__main__":
    main()
