param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $Root

$RequiredInputs = @(
    ".\data_raw\ddk_trials.csv",
    ".\data_raw\ddkwm_trials.csv",
    ".\data_raw\participants.csv",
    ".\data_raw\manual_automatic_paired.csv"
)
foreach ($InputFile in $RequiredInputs) {
    if (-not (Test-Path -LiteralPath $InputFile)) {
        throw "Missing controlled-access input: $InputFile. See data_raw\README.md."
    }
}

function Invoke-Analysis([string[]]$Arguments) {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Analysis failed: $($Arguments -join ' ')"
    }
}

Invoke-Analysis @(".\01_sample_and_condition_means\run_condition_means_and_age.py")
Invoke-Analysis @(".\01_sample_and_condition_means\build_figure_02.py")
Invoke-Analysis @(".\01_sample_and_condition_means\build_figure_04.py")
Invoke-Analysis @(".\02_working_memory_accuracy\run_working_memory_accuracy.py")
Invoke-Analysis @(".\03_trial_trajectories\fit_low_high_trajectory_models.py")
Invoke-Analysis @(".\03_trial_trajectories\fit_reported_pause_trajectory_rows.py")
Invoke-Analysis @(".\03_trial_trajectories\build_figure_03.py")
Invoke-Analysis @(".\04_reported_sensitivity\run_reported_sensitivity.py")
Invoke-Analysis @(".\05_cmms_moderation\run_cmms_analysis.py")
Invoke-Analysis @(".\06_manual_validation\analyze_manual_automatic_agreement.py", "--paired", ".\data_raw\manual_automatic_paired.csv", "--output-dir", ".\06_manual_validation\results")
Invoke-Analysis @(".\verify_manuscript_results.py")

Write-Host "All manuscript analyses and verification checks completed."
