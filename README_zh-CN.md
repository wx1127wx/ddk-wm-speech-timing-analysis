# JSLHR 稿件分析代码

[English version](README.md)

**稿件题目：** *Speech Timing Level and Trajectory Across Three Ordered Diadochokinetic Task Conditions: Mean-Level Differences, Repeated-Trial Change, and Age Associations*

本仓库是与当前投稿稿件对应的最小代码发布版本，仅包含稿件报告的分析：条件均值与年龄调节、工作记忆正确率、Low/High 试次轨迹、两项已报告的轨迹敏感性分析、CMMS 调节（包括秩分数敏感性分析）以及手工与自动特征的一致性分析。早期探索性分析、未使用的图形和已被替代的工作流不包含在本版本中。

## 数据可用性与隐私

仓库不包含参与者级输入数据。这与稿件中的数据可用性声明一致：去标识化特征表可在合理请求并满足伦理和数据保护要求的前提下向通讯作者获取。运行流程前，请按照 `data_raw/README.md` 的说明，将四个受控访问文件放入 `data_raw/`。公开仓库中不得加入原始音频、直接身份标识、录音日期或行政性录音字段。

## 环境与运行

本版本已在 Windows、Python 3.11.15 和 `env_codex` Conda 环境中验证。精确的软件包版本见 `requirements.txt` 和 `environment.yml`。

```powershell
conda activate env_codex
python -m pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Python python
```

最后一条命令会运行所有稿件分析，并执行针对锁定投稿结果的 207 项数值检查。各阶段的汇总结果写入对应的编号目录；验证完成后，`verification_summary.json` 中的 `"all_passed"` 应为 `true`。

## 目录说明

- `01_sample_and_condition_means`：表 1、表 2、表 4，以及图 2、图 4。
- `02_working_memory_accuracy`：Low/High 正确率统计和稿件报告的二项混合模型。
- `03_trial_trajectories`：表 3 和图 3。
- `04_reported_sensitivity`：结果部分报告的仅正确反应分析和 Trials 2–9 敏感性分析。
- `05_cmms_moderation`：表 5、CMMS 描述统计与关联分析、试次级调节分析和秩分数敏感性分析。
- `06_manual_validation`：表 6 统计量和条件变化一致性分析。
- `expected_results`：用于不可变结果比较的四舍五入数值和投稿图形文件。

`MANUSCRIPT_OUTPUT_MAP.csv` 给出了稿件结论、分析脚本和输出文件之间的对应关系。图 1 是任务流程示意图，不由统计分析流水线生成。

## 可追溯性说明

1. 表 3 中 DDK Rate、DDK Regularity 和 DDK Duration 三行来自 Methods 中描述的 Low/High-only 试次级模型。投稿稿件中的两个 Pause 行来自经过年龄校正的 Single/Low/High 模型。`fit_reported_pause_trajectory_rows.py` 保留了实际使用的来源，以便复现投稿数值。这构成稿件方法描述与模型规格之间的不一致，投稿或公开存档前应进一步处理。
2. 本版本按照稿件所述实现了 2,000 次参与者聚类 bootstrap 重抽样。投稿表格保留了早期固定随机种子实现得到的区间；点估计一致，但部分 bootstrap 区间可能略有差异。验证脚本对这些区间使用明确的 Monte Carlo 容差，对其他结果使用严格的四舍五入容差。
3. 图 2 与投稿文件逐字节一致。图 3 和图 4 根据最终模型输出重新生成；由于栅格化和投稿导出过程不同，投稿 PNG 文件保留在 `expected_results/figures` 中供视觉比较。

## 可复现性规则

不要因为早期项目目录中存在某项分析就将其加入本发布版本。只有当前稿件报告的分析，或生成已报告表格/图形所必需的脚本和输出，才应加入本目录。
