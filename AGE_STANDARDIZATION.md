# Age standardization and Figure 4

Updated October 8, 2026, for manuscript version 16.

## Retained conventions

Age is continuous and standardized as `(age - mean_age) / age_SD`, with the sample standard deviation (`ddof=1`).

| Model family | Records used to compute the SD | Mean age (years) | SD (years) |
|---|---|---:|---:|
| Condition-mean age moderation and Figure 4 | 465 condition records, 155 participants represented three times each | 39.5419354839 | 18.9870323407 |
| Age-adjusted three-condition pause trajectories | 155 participant records, before trial merging | 39.5419354839 | 19.0280854316 |
| CMMS condition-mean and trial-level moderation | Participants with available CMMS scores, before repeated-record merging; all 155 had scores | 39.5419354839 | 19.0280854316 |

The primary Low/High trajectory and trial-selection sensitivity models do not include age. A loader creating an `age_z` column does not imply that every downstream model adjusts for age.

## Derivation

Let `Q` be the sum of squared age deviations over the 155 participants. The participant-level sample variance is `Q / 154`. Repeating each age three times preserves the mean but gives sample variance `3 * Q / 464`.

```text
SD_condition_records / SD_participants
    = sqrt(3 * 154 / 464)
    = 0.9978425002

19.0280854316 * 0.9978425002 = 18.9870323407 years
```

There are 155 independent participants; the 465 condition records are repeated observations. Retaining both model conventions is valid when their units are described accurately. Condition-mean age effects and Table 4 therefore correspond to approximately 18.99 years, while the participant-level age covariates use approximately 19.03 years. The descriptive sample age SD remains 19.03 years.

Relevant implementations are `01_sample_and_condition_means/run_condition_means_and_age.py`, `03_trial_trajectories/fit_reported_pause_trajectory_rows.py`, and `05_cmms_moderation/cognitive_scale_analysis.py`. The CMMS loader discards the previous condition-record `age_z` and merges its participant-level standardized covariates.

## Figure 4 correction

The original plotting script used the participant-level SD to construct the age grid while fitting the condition-mean models with the condition-record `age_z`. The updated script computes the grid with the fitting mean and SD and verifies consistency with the existing `age_z` column. Both prediction lines and confidence bands use that grid.

The contrast label is now `Dual–Single`, meaning the average of Low and High minus Single. `High–Low` means High minus Low. Figure annotations state that age effects are per approximately 18.99 years. The script and the version 16 manuscript figure use the same layout; the tested PNG is byte-identical to the version 16 embedded figure. The corresponding reference file in `expected_results/figures/Figure_04.png` is updated, while all numerical table targets are retained.

## Validation and remaining review

The version 16 local validation reproduced the five original condition-mean age models using the released fixed-effect formula, correlated participant random intercept/condition slopes, original `age_z`, REML, and the original optimizer sequence. It matched 90 coefficient/SE/p quantities and 40 age-interaction effect/interval/FDR quantities against the released result tables; the largest fixed-effect-check numerical difference was approximately `9.22e-14`.

On the 130-point grid from ages 22 to 86, over two pause outcomes and three conditions, correcting the prediction grid changed the center line by at most approximately 0.354 ms relative to the same fitted model evaluated with the old grid. Existing model estimates and inferential results are unchanged. Direct formula predictions and the plotting design-matrix predictions were also checked for agreement.

The four log-outcome models (DDK Regularity, DDK Duration, Pause Regularity, and Pause Duration) converged but emitted `The MLE may be on the boundary of the parameter space.` This warning is retained for author review of random-effect variance/covariance and boundary diagnostics; it is not treated as evidence of nonconvergence or as an issue already ruled out.

These checks are a targeted reproduction of existing condition-mean analyses and validation of the updated figure. They are not a complete rerun of the trial-level, CMMS, or manual-agreement pipelines. Participant inputs used for local validation are not added to this public repository. The 207-check verifier can separately check the released aggregate tables against the unchanged numerical targets.

## 中文说明

保留两种年龄标准化方式：条件均值模型及图4采用465条条件记录计算的SD（约18.99年）；暂停轨迹与CMMS模型在155人表上计算SD（约19.03年）。年龄均值相同，SD差异来自`ddof=1`的有限样本分母。

图4的预测网格、中心线和置信带现已与拟合尺度一致，对比名称及年龄单位已标明。统计模型、系数、区间、p及q值不改。四个对数指标的边界警告仍需作者检查。此次仅同步代码、汇总图及说明，未加入参与者级输入或原始录音。
