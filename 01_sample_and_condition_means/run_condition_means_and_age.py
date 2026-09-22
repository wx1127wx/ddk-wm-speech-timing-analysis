"""Phase 1: traditional condition-mean analysis for DDK-WM.

Reads the organized data copy, aggregates trials within participant and
condition, and writes repeated-measures ANOVA, planned paired contrasts,
age-adjusted mixed models, diagnostics, figures, and a Chinese summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import mixedlm
from statsmodels.stats.anova import AnovaRM


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_raw"
OUT = Path(__file__).resolve().parent
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
PROCESSED = OUT / "data_processed"

CONDITIONS = ["Single", "Low", "High"]
SPEECH = {
    "DDK Rate": ("rate", "raw", "syllables/s"),
    "DDK Regularity": ("reg", "log", "% change"),
    "DDK Duration": ("dur", "log", "% change"),
    "Pause Regularity": ("pause_reg", "log", "% change"),
    "Pause Duration": ("pause_dur", "log", "% change"),
}


def bh_adjust(values: list[float]) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def transform_effect(estimate: float, lo: float, hi: float, scale: str) -> tuple[float, float, float]:
    if scale == "log":
        return tuple(100.0 * np.expm1(v) for v in (estimate, lo, hi))
    return estimate, lo, hi


def load_and_aggregate() -> tuple[pd.DataFrame, pd.DataFrame]:
    ddk = pd.read_csv(DATA / "ddk_trials.csv", dtype={"subject_id": str})
    dual = pd.read_csv(DATA / "ddkwm_trials.csv", dtype={"subject_id": str})
    demo = pd.read_csv(DATA / "participants.csv", dtype={"subject_id": str})
    rename = {
        "subject_id": "analysis_id",
        "DDK rate (syll/s)": "rate",
        "DDK regularity (ms)": "reg",
        "DDK duration (ms)": "dur",
        "pause regularity (ms)": "pause_reg",
        "pause duration (ms)": "pause_dur",
    }
    ddk = ddk.rename(columns=rename)
    dual = dual.rename(columns=rename)
    demo = demo.rename(columns={"subject_id": "analysis_id", "age": "age_years"})
    dual["condition"] = dual["ddkwm_condition"].map({1: "Low", 2: "High"})
    ddk["condition"] = "Single"
    columns = ["analysis_id", "condition", *[v[0] for v in SPEECH.values()]]
    trial = pd.concat([ddk[columns], dual[columns]], ignore_index=True)
    for col in [v[0] for v in SPEECH.values()]:
        trial[col] = pd.to_numeric(trial[col], errors="raise")
    if trial.isna().any().any() or (trial[[v[0] for v in SPEECH.values()]] <= 0).any().any():
        raise ValueError("Speech data contain missing or non-positive values.")
    means = trial.groupby(["analysis_id", "condition"], as_index=False).mean(numeric_only=True)
    means["condition"] = pd.Categorical(means["condition"], categories=CONDITIONS, ordered=True)
    means = means.merge(demo[["analysis_id", "age_years"]], on="analysis_id", validate="many_to_one")
    if means.shape[0] != 155 * 3 or means.analysis_id.nunique() != 155:
        raise ValueError(f"Expected 465 participant-condition means, found {means.shape}.")
    if not means.groupby("analysis_id", observed=True).size().eq(3).all():
        raise ValueError("Not every participant has all three conditions.")
    means["age_z"] = (means["age_years"] - means["age_years"].mean()) / means["age_years"].std(ddof=1)
    means["duality"] = means["condition"].map({"Single": -2 / 3, "Low": 1 / 3, "High": 1 / 3}).astype(float)
    means["load"] = means["condition"].map({"Single": 0.0, "Low": -0.5, "High": 0.5}).astype(float)
    means.to_csv(PROCESSED / "speech_condition_means.csv", index=False, encoding="utf-8-sig")
    trial.to_csv(PROCESSED / "speech_trial_data_used.csv", index=False, encoding="utf-8-sig")
    return means, pd.DataFrame()


def repeated_measures_and_contrasts(means: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    anova_rows = []
    contrast_rows = []
    for metric, (column, scale, unit) in SPEECH.items():
        work = means[["analysis_id", "condition", column]].copy()
        y = np.log(work[column]) if scale == "log" else work[column]
        work["outcome"] = y
        fit = AnovaRM(work, depvar="outcome", subject="analysis_id", within=["condition"]).fit()
        row = fit.anova_table.loc["condition"]
        df1, df2, f_value, p_value = float(row["Num DF"]), float(row["Den DF"]), float(row["F Value"]), float(row["Pr > F"])
        eta_p2 = (f_value * df1) / (f_value * df1 + df2)
        anova_rows.append({"metric": metric, "scale": scale, "F": f_value, "df_condition": df1, "df_error": df2, "p": p_value, "partial_eta_squared": eta_p2})
        wide = work.pivot(index="analysis_id", columns="condition", values="outcome").loc[:, CONDITIONS]
        planned = {
            "Dual vs Single": wide[["Low", "High"]].mean(axis=1) - wide["Single"],
            "High vs Low": wide["High"] - wide["Low"],
        }
        for contrast_name, difference in planned.items():
            t_value, p = stats.ttest_1samp(difference, 0.0)
            sd = difference.std(ddof=1)
            dz = difference.mean() / sd
            se = sd / np.sqrt(len(difference))
            lo, hi = difference.mean() - stats.t.ppf(0.975, len(difference) - 1) * se, difference.mean() + stats.t.ppf(0.975, len(difference) - 1) * se
            reported, reported_lo, reported_hi = transform_effect(float(difference.mean()), float(lo), float(hi), scale)
            contrast_rows.append({"metric": metric, "scale": scale, "contrast": contrast_name, "estimate_model_scale": difference.mean(), "CI_low_model_scale": lo, "CI_high_model_scale": hi, "reported_effect": reported, "CI_low": reported_lo, "CI_high": reported_hi, "unit": unit, "t": t_value, "df": len(difference) - 1, "p": p, "cohens_dz": dz})
    anova = pd.DataFrame(anova_rows)
    contrasts = pd.DataFrame(contrast_rows)
    contrasts["q_BH_within_contrast"] = np.nan
    for contrast_name in contrasts["contrast"].unique():
        mask = contrasts["contrast"].eq(contrast_name)
        contrasts.loc[mask, "q_BH_within_contrast"] = bh_adjust(contrasts.loc[mask, "p"].tolist())
    anova.to_csv(TABLES / "Table_01_repeated_measures_ANOVA.csv", index=False, encoding="utf-8-sig")
    contrasts.to_csv(TABLES / "Table_02_planned_paired_contrasts.csv", index=False, encoding="utf-8-sig")
    return anova, contrasts


def age_adjusted_models(means: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, (column, scale, unit) in SPEECH.items():
        work = means.copy()
        work["outcome"] = np.log(work[column]) if scale == "log" else work[column]
        model = mixedlm("outcome ~ duality + load + age_z + duality:age_z + load:age_z", work, groups=work["analysis_id"], re_formula="~duality+load")
        try:
            fit = model.fit(reml=True, method="lbfgs", maxiter=2000, disp=False)
        except Exception:
            fit = model.fit(reml=True, method="powell", maxiter=2000, disp=False)
        terms = ["Intercept", "duality", "load", "age_z", "duality:age_z", "load:age_z"]
        labels = ["Intercept", "Dual vs Single", "High vs Low", "Age (+1 SD)", "Duality x age", "Load x age"]
        for term, label in zip(terms, labels):
            estimate, se = float(fit.params[term]), float(fit.bse[term])
            lo, hi = estimate - 1.96 * se, estimate + 1.96 * se
            reported, reported_lo, reported_hi = transform_effect(estimate, lo, hi, scale)
            rows.append({"metric": metric, "scale": scale, "term": label, "estimate_model_scale": estimate, "SE": se, "z": estimate / se, "p": float(fit.pvalues[term]), "reported_effect": reported, "CI_low": reported_lo, "CI_high": reported_hi, "unit": unit, "converged": bool(getattr(fit, "converged", True)), "random_effects": "intercept + duality + load"})
    out = pd.DataFrame(rows)
    out["q_BH_within_term"] = np.nan
    for term in ["Dual vs Single", "High vs Low", "Duality x age", "Load x age"]:
        mask = out.term.eq(term)
        out.loc[mask, "q_BH_within_term"] = bh_adjust(out.loc[mask, "p"].tolist())
    out.to_csv(TABLES / "Table_03_age_adjusted_mean_LMM.csv", index=False, encoding="utf-8-sig")
    return out


def diagnostics(means: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, (column, scale, _) in SPEECH.items():
        work = means.copy()
        work["outcome"] = np.log(work[column]) if scale == "log" else work[column]
        condition_groups = [work.loc[work.condition == c, "outcome"] for c in CONDITIONS]
        lev = stats.levene(*condition_groups, center="median")
        # Marginal residual after removing participant and condition means.
        residual = work["outcome"] - work.groupby("analysis_id", observed=True)["outcome"].transform("mean") - work.groupby("condition", observed=True)["outcome"].transform("mean") + work["outcome"].mean()
        shapiro = stats.shapiro(residual)
        outlier_count = int((np.abs(stats.zscore(work["outcome"])) > 3).sum())
        rows.append({"metric": metric, "scale": scale, "n": len(work), "Levene_W": float(lev.statistic), "Levene_p": float(lev.pvalue), "residual_Shapiro_W": float(shapiro.statistic), "residual_Shapiro_p": float(shapiro.pvalue), "marginal_abs_z_gt_3_n": outlier_count})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "Table_04_assumption_diagnostics.csv", index=False, encoding="utf-8-sig")
    return out


def make_figures(means: pd.DataFrame) -> None:
    colors = {"Single": "#66727A", "Low": "#2B7A78", "High": "#C46D3C"}
    fig, axes = plt.subplots(1, len(SPEECH), figsize=(16, 3.8))
    for ax, (metric, (column, _, unit)) in zip(axes, SPEECH.items()):
        values = means.pivot(index="analysis_id", columns="condition", values=column).loc[:, CONDITIONS]
        for i, condition in enumerate(CONDITIONS):
            x = np.random.default_rng(20260907 + i).normal(i, 0.035, len(values))
            ax.scatter(x, values[condition], s=9, alpha=0.16, color=colors[condition], linewidth=0)
        m = values.mean(); ci = values.sem() * stats.t.ppf(0.975, len(values) - 1)
        ax.plot(range(3), m, color="#26343C", linewidth=1.7, zorder=3)
        ax.errorbar(range(3), m, yerr=ci, fmt="o", color="#26343C", markersize=5, capsize=3, zorder=4)
        ax.set_xticks(range(3), CONDITIONS, rotation=25)
        ax.set_title(metric, fontsize=9)
        ax.set_ylabel(unit)
        ax.grid(axis="y", color="#D9DEE2", linewidth=0.6); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("DDK-WM condition means", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure_01_condition_means.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "Figure_01_condition_means.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(means: pd.DataFrame, anova: pd.DataFrame, contrasts: pd.DataFrame, lmm: pd.DataFrame, diag: pd.DataFrame) -> None:
    lines = ["# 第一阶段：传统条件均值分析结果", "", f"分析样本为 {means.analysis_id.nunique()} 名受试者，每名受试者在 Single、Low、High 各有一个条件均值。Single 均值来自 3 次试次，Low 和 High 均值来自各 10 次试次。", "", "## 分析方法", "", "对五项语音指标分别进行三水平重复测量 ANOVA，并报告两个预先定义的被试内计划比较：Dual–Single = 平均(Low, High) − Single；High–Low = High − Low。随后拟合年龄校正的混合模型：`Y_mean ~ duality + load + age_z + duality:age_z + load:age_z + (1 + duality + load | participant)`。", "", "DDK Rate 使用原始尺度；其余指标使用自然对数，模型结果转换为百分比变化。重复测量分析和计划比较均以受试者为配对单位。", "", "## 条件均值", "", "|指标|Single M|Low M|High M|", "|---|---:|---:|---:|"]
    for metric, (column, _, _) in SPEECH.items():
        vals = means.groupby("condition", observed=True)[column].mean()
        lines.append(f"|{metric}|{vals['Single']:.3f}|{vals['Low']:.3f}|{vals['High']:.3f}|")
    lines += ["", "## 重复测量总体检验", "", "|指标|F(2,308)|p|偏η²|", "|---|---:|---:|---:|"]
    for _, r in anova.iterrows():
        p_text = "<.001" if r.p < .001 else f"{r.p:.3f}"
        lines.append(f"|{r.metric}|{r.F:.3f}|{p_text}|{r.partial_eta_squared:.3f}|")
    lines += ["", "## 计划比较", "", "|指标|比较|估计值|95% CI|p|FDR q|", "|---|---|---:|---:|---:|---:|"]
    for _, r in contrasts.iterrows():
        p_text = "<.001" if r.p < .001 else f"{r.p:.3f}"
        q_text = "<.001" if r.q_BH_within_contrast < .001 else f"{r.q_BH_within_contrast:.3f}"
        lines.append(f"|{r.metric}|{r.contrast}|{r.reported_effect:+.3f} {r.unit}|[{r.CI_low:+.3f}, {r.CI_high:+.3f}]|{p_text}|{q_text}|")
    lines += ["", "## 年龄校正模型", "", "年龄校正模型用于估计年龄是否改变一般双任务代价或 High–Low 区块差异。应重点查看 `Duality x age` 和 `Load x age`，不能用年龄主效应替代年龄调节效应。", "", "|指标|Duality × age|p|q|Load × age|p|q|", "|---|---:|---:|---:|---:|---:|---:|"]
    for metric in SPEECH:
        a = lmm[(lmm.metric == metric) & (lmm.term == "Duality x age")].iloc[0]
        b = lmm[(lmm.metric == metric) & (lmm.term == "Load x age")].iloc[0]
        lines.append(f"|{metric}|{a.reported_effect:+.3f}|{a.p:.3g}|{a.q_BH_within_term:.3g}|{b.reported_effect:+.3f}|{b.p:.3g}|{b.q_BH_within_term:.3g}|")
    lines += ["", "## 假设和解释边界", "", "诊断表报告了条件方差、边际残差正态性和极端值检查。若模型尺度残差明显偏态，应优先保留对数变换并报告百分比效应。重复测量结果描述总体条件差异，不能说明差异从哪个试次开始，也不能区分练习、疲劳、策略或区块顺序；这些问题留给第二阶段试次分析。", "", "High 固定在 Low 之后时，High–Low 结果同时混合负荷和区块顺序，本文不能将其解释为纯粹的因果负荷效应。", "", "## 文件", "", "- `data_processed/speech_condition_means.csv`：被试内语音条件均值。", "- `tables/Table_01_repeated_measures_ANOVA.csv`：重复测量总体检验。", "- `tables/Table_02_planned_paired_contrasts.csv`：两个计划比较。", "- `tables/Table_03_age_adjusted_mean_LMM.csv`：年龄校正均值混合模型。", "- `tables/Table_04_assumption_diagnostics.csv`：假设诊断。", "- `figures/Figure_01_condition_means.png`：条件均值图。"]
    (OUT / "phase1_results_CN.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    for directory in [OUT, TABLES, FIGURES, PROCESSED]:
        directory.mkdir(parents=True, exist_ok=True)
    means, _ = load_and_aggregate()
    anova, contrasts = repeated_measures_and_contrasts(means)
    lmm = age_adjusted_models(means)
    sample = pd.DataFrame([
        ["Single-task DDK", 155, 465, "3 trials per participant; condition means"],
        ["Concurrent DDK + working-memory task", 155, 3100, "10 Low and 10 High trials per participant"],
        ["Age", 155, 155, "Moderator of task-condition contrasts"],
        ["Working-memory responses", 155, 3100, "Description of concurrent task performance"],
        ["CMMS", 155, 155, "Exploratory moderator of task-condition contrasts"],
        ["Manual-annotation subset", 20, 460, "Validation of automated feature extraction"],
    ], columns=["component", "participants", "observations", "role_in_analysis"])
    sample.to_csv(TABLES / "Table_01_sample.csv", index=False, encoding="utf-8-sig")
    condition_rows = []
    for metric, (column, _, _) in SPEECH.items():
        row = {"outcome": metric}
        row.update(means.groupby("condition", observed=True)[column].mean().to_dict())
        for contrast_name, prefix in [("Dual vs Single", "dual_minus_single"), ("High vs Low", "high_minus_low")]:
            value = contrasts[(contrasts.metric == metric) & (contrasts.contrast == contrast_name)].iloc[0]
            row.update({f"{prefix}_effect": value.reported_effect, f"{prefix}_CI_low": value.CI_low,
                        f"{prefix}_CI_high": value.CI_high, f"{prefix}_q": value.q_BH_within_contrast})
        condition_rows.append(row)
    pd.DataFrame(condition_rows).to_csv(TABLES / "Table_02_condition_means_and_contrasts.csv", index=False, encoding="utf-8-sig")
    age_rows = []
    for metric in SPEECH:
        dual = lmm[(lmm.metric == metric) & (lmm.term == "Duality x age")].iloc[0]
        load = lmm[(lmm.metric == metric) & (lmm.term == "Load x age")].iloc[0]
        age_rows.append({"outcome": metric, "single_to_dual_x_age": dual.reported_effect,
                         "single_to_dual_CI_low": dual.CI_low, "single_to_dual_CI_high": dual.CI_high,
                         "single_to_dual_q": dual.q_BH_within_term, "high_minus_low_x_age": load.reported_effect,
                         "high_minus_low_CI_low": load.CI_low, "high_minus_low_CI_high": load.CI_high,
                         "high_minus_low_q": load.q_BH_within_term})
    pd.DataFrame(age_rows).to_csv(TABLES / "Table_04_age_moderation.csv", index=False, encoding="utf-8-sig")
    qc = {"n_participants": int(means.analysis_id.nunique()), "n_condition_means": int(len(means)), "conditions": CONDITIONS, "metrics": list(SPEECH), "all_lmm_converged": bool(lmm.converged.all())}
    (OUT / "phase1_qc.json").write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(qc, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
