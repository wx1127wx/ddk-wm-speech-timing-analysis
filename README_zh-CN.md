# JSLHR 稿件分析代码

[English version](README.md)

**稿件题目：** *Speech Timing During Sequential Diadochokinesis Under Dual-Task Conditions: Age-Related Differences and Trial-Level Changes*

本仓库对应正文报告的条件均值与年龄关联、记忆正确率、Low/High 动态轨迹、两项独立敏感性分析、CMMS 分析和人工—自动测量一致性。参与者级数据和本地补充实验不上传。

## 数据与运行

公开内容为代码、汇总表格和图像。四个受控输入的格式见 `data_raw/README.md`，须向通讯作者申请并直接放在 `data_raw/` 下。原始音频、逐人/逐试次数据、可识别文件名和音频路径不应公开；生成的 `data_processed` 目录也已排除。

使用 Windows、Python 3.11.15、Conda `env_codex`，依赖版本见 `requirements.txt`、`environment.yml`。有完整四个输入时：

```powershell
conda activate env_codex
powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Python python
```

无需参与者数据即可核对已发布的汇总结果：

```powershell
conda activate env_codex
python -B verify_manuscript_results.py
```

当前共 **244 项检查**，`verification_summary.json` 应显示 `all_passed: true`。表3同时核对统一模型来源、十个端点、五项BH及独立保存的当前数值目标。数值一致不代表所有统计设定已经验证。本次发布重建图2–4及表3、检查汇总结果，没有重跑完整研究流程。

## 文件对应关系

| 目录 | 内容 |
|---|---|
| 01_sample_and_condition_means | 表1、2、4及图2、4 |
| 02_working_memory_accuracy | Low/High正确率及二项混合模型 |
| 03_trial_trajectories | 五指标表3、四面板图3 |
| 04_reported_sensitivity | 仅正确试次、全部Trials2–9两项独立筛选 |
| 05_cmms_moderation | 表5、量表关联、动态调节、秩分数敏感性 |
| 06_manual_validation | 表6测量一致性、条件对比一致性 |
| expected_results/current | 当前表3独立锁定数值 |
| expected_results/tables及figures | 保留不变的历史投稿目标和参考图 |

精确映射见 `MANUSCRIPT_OUTPUT_MAP.csv`。图1为任务示意图，不由统计流水线生成；各分析目录的figures为当前图像。

## 最新同步内容

- 表3五指标统一为 `g(Y) ~ Condition * Progress`，只纳入Low/High；Low=-0.5、High=+0.5，Progress=(trial−1)/9；参与者随机截距、条件斜率、进展斜率采用对角协方差。Rate原尺度，其他四项取自然对数，五项交互统一BH校正。主动态模型不加入Single或年龄。
- 图3保持正文四面板；表3完整报告五指标，补齐DDK Regularity的Trial1、Trial10端点。旧pause入口只兼容导出统一模型，不再拟合另一套三条件模型。
- 图2–4显示名称统一为Intersyllabic-gap，原始Pause/pause字段保留。图4显示Dual–Single，含义为Low/High平均减Single，效应值不反号。
- 图4拟合和预测均使用约18.99年的年龄单位；CMMS参与者级年龄协变量约19.03年。保留远端已有标准化检查与收敛检查，详见 `AGE_STANDARDIZATION.md`。
- 正确试次与Trials2–9分别从完整Low/High数据筛选；人工标注仅补充测量一致性，不解释为对全样本轨迹结果的复现。表6保留2000次参与者整簇bootstrap，历史区间容差在核验脚本中明示。
- Single/Low/High录音均为5秒；3.8秒是视觉记忆刺激序列及间隔，加准备、注视各0.6秒，共5秒。录音长度不等于已确认的rate计算分母，自动提取原始实现及实际测量跨度仍待核实。

## 尚需复核

表4/5当前模型及数值保留。每人三个条件均值时，随机截距、两个随机条件斜率加残差的协方差分解存在参数冗余，不能只凭收敛判定解决。CMMS模型含年龄、教育主效应，但缺少对应条件交互；共同主效应不能充分控制条件差异层面的年龄/教育影响，较强的“独立于年龄、教育”解释需要联合调节或参与者条件对比重分析。本次没有声称已完成这些重分析。

固定Single→Low→High顺序仍混杂顺序、练习和疲劳。主动态交互不能用来证明年龄使轨迹分化扩大。本次修改范围及验证见 `CHANGELOG_20261015.md`。
