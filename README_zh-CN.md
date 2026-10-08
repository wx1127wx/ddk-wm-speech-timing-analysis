# JSLHR 稿件分析代码

[English version](README.md)

**稿件题目：** *Sequential Diadochokinesis Performance During Visual-Memory Dual Tasking: Repeated-Trial Dynamics Across Adulthood*

本仓库是与当前投稿稿件对应的最小代码发布版本，仅包含稿件报告的分析：条件均值与年龄调节、工作记忆正确率、Low/High 试次轨迹、两项已报告的轨迹敏感性分析、CMMS 调节（包括秩分数敏感性分析）以及手工与自动特征的一致性分析。早期探索性分析、未使用的图形和已被替代的工作流不包含在本版本中。

## 数据可用性与隐私

代码、汇总结果和图已公开；仓库不包含参与者级输入数据。源数据文件及访问安排见 `data_raw/README.md`：去标识化特征表可在合理请求并满足伦理和数据保护要求的前提下向通讯作者获取。运行流程前，请按照该说明将四个受控访问文件放入 `data_raw/`。公开仓库中不得加入原始音频、直接身份标识、录音日期或行政性录音字段。

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
- `expected_results`：锁定的数值比较目标及当前稿件参考图。

`MANUSCRIPT_OUTPUT_MAP.csv` 给出了稿件结论、分析脚本和输出文件之间的对应关系。图 1 是任务流程示意图，不由统计分析流水线生成。

## 可追溯性说明

1. 表 3 中 DDK Rate、DDK Regularity 和 DDK Duration 三行来自 Low/High-only 试次级模型；两个 Pause 行来自年龄校正的 Single/Low/High 模型。修订后的 Methods 已明确说明这两种规格及不同的 FDR 家族。`fit_reported_pause_trajectory_rows.py` 保留实际的暂停结果来源，原模型和表格数值保持不变。
2. 本版本按照稿件所述实现了 2,000 次参与者聚类 bootstrap 重抽样。投稿表格保留了早期固定随机种子实现得到的区间；点估计一致，但部分 bootstrap 区间可能略有差异。验证脚本对这些区间使用明确的 Monte Carlo 容差，对其他结果使用严格的四舍五入容差。
3. 图 2 与稿件参考图逐字节一致。图 3 根据最终模型输出重新生成，其参考 PNG 保留在 `expected_results/figures` 中供视觉比较。图 4 及其参考 PNG 已于2026-10-08更新为第16版：拟合和预测网格使用同一年龄尺度，图内注明年龄单位及对比方向。在本次验证环境中，更新脚本生成的 PNG 与第16版 Word 内嵌图逐字节一致。

## 年龄标准化与图4修正

保留现有两种年龄计算口径。条件均值年龄调节模型在465条参与者×条件记录上使用样本标准差（`ddof=1`）：年龄均值39.5419年、SD 18.9870年，即模型一单位标准化年龄约对应18.99年。年龄校正的暂停轨迹及CMMS模型先在参与者表标准化、再合并重复记录：均值相同、SD 19.0281年，即约19.03年。两者独立参与者人数均为155人。

图4原预测网格使用155人尺度，而拟合沿用465条条件记录的 `age_z`；现已将中心线和置信带的预测尺度对齐为拟合尺度。表4及稿件条件均值年龄效应标签对应约18.99年。模型系数、置信区间、p值和FDR q值不变。推导与核验范围见 [AGE_STANDARDIZATION.md](AGE_STANDARDIZATION.md)。

第16版核对使用本地输入复现了五个原有条件均值年龄模型，系数、SE、p、区间及FDR结果与公开表一致。这是针对性验证，并非对所有研究分析的全流程重跑。四个对数结局模型虽然优化器收敛，但提示随机效应方差可能位于边界；保留该提示，供作者进一步检查。

## 可复现性规则

不要因为早期项目目录中存在某项分析就将其加入本发布版本。只有当前稿件报告的分析，或生成已报告表格/图形所必需的脚本和输出，才应加入本目录。
