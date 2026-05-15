# Register two Windows Scheduled Tasks that run the ta-agent pipeline at
# 08:00 CT and 17:00 CT each weekday. No third-party tools needed - this
# uses Windows' built-in Task Scheduler.
#
# Usage (PowerShell, run as your normal user - NOT admin required):
#
#     cd C:\dev\ta-agent
#     PowerShell -ExecutionPolicy Bypass -File .\scripts\register_windows_tasks.ps1
#
# The script can be re-run safely; it overwrites any existing tasks
# with the same names.
#
# To inspect / disable / delete the tasks later, use Task Scheduler GUI
# (taskschd.msc) or:
#   Get-ScheduledTask -TaskName "ta-agent-*"
#   Disable-ScheduledTask -TaskName "ta-agent-pipeline-8am-ct"
#   Unregister-ScheduledTask -TaskName "ta-agent-pipeline-8am-ct" -Confirm:$false
#
# Logging:
#   Each fire's stdout/stderr lands at:
#       C:\dev\ta-agent\logs\scheduled_run_YYYY-MM-DD.log
#   (handled by scripts\run_pipeline.cmd which Task Scheduler invokes)

$ErrorActionPreference = "Stop"

$RepoRoot          = "C:\dev\ta-agent"
$PipelineWrapper   = "$RepoRoot\scripts\run_pipeline.cmd"
$MonthlyWrapper    = "$RepoRoot\scripts\run_monthly_retrain.cmd"
$QuarterlyWrapper  = "$RepoRoot\scripts\run_quarterly_retune.cmd"
$LogDir            = "$RepoRoot\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# Sanity checks
foreach ($w in @($PipelineWrapper, $MonthlyWrapper, $QuarterlyWrapper)) {
    if (-not (Test-Path $w)) {
        Write-Error "$w not found - wrong checkout?"
        exit 1
    }
}

# Direct .cmd invocation. The .cmd handles cd to repo root, log capture,
# and exit-code propagation. Avoids both the PowerShell quoting hell of
# v1 and the WorkingDirectory-not-honored bug seen with direct python.exe
# invocation when the user session isn't foreground.
function Build-Action {
    param([string]$Wrapper)
    return New-ScheduledTaskAction `
        -Execute "cmd.exe" `
        -Argument "/c `"$Wrapper`"" `
        -WorkingDirectory $RepoRoot
}

# Trigger: weekdays at the requested local time. Windows reads the system
# timezone - set the box to America/Chicago (CT) for the trigger times to
# match the user's intent. If you're on a different tz, edit the hours.
function Build-WeekdayTrigger {
    param([int]$Hour, [int]$Minute)
    return New-ScheduledTaskTrigger `
        -Weekly `
        -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
        -At ([datetime]::Today.AddHours($Hour).AddMinutes($Minute))
}

$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
    -MultipleInstances IgnoreNew

# Longer settings for quarterly Optuna re-tune -~3 hours per universe.
$LongSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 8) `
    -MultipleInstances IgnoreNew

# 8:35 AM CT - post-open refresh (5 min after the equity bell at 08:30 CT)
# We deliberately fire AFTER the open so today's OPEN bar is reliably
# available from yfinance -needed by the paper backtest for fills.
# daily_predict itself still uses yesterday's complete bar for features.
# Pipeline tail also runs the drift detector (cheap, with cooldown).
$TaskName8am = "ta-agent-pipeline-8am-ct"
Register-ScheduledTask `
    -TaskName $TaskName8am `
    -Action (Build-Action $PipelineWrapper) `
    -Trigger (Build-WeekdayTrigger -Hour 8 -Minute 35) `
    -Settings $Settings `
    -Description "ta-agent: refresh OHLCV + predict + paper backtest + drift_check at 08:35 CT" `
    -Force | Out-Null
Write-Host "registered: $TaskName8am (weekdays 08:35 local)"

# 5 PM CT - post-close refresh
$TaskName5pm = "ta-agent-pipeline-5pm-ct"
Register-ScheduledTask `
    -TaskName $TaskName5pm `
    -Action (Build-Action $PipelineWrapper) `
    -Trigger (Build-WeekdayTrigger -Hour 17 -Minute 0) `
    -Settings $Settings `
    -Description "ta-agent: refresh OHLCV + predict + paper backtest + drift_check at 17:00 CT" `
    -Force | Out-Null
Write-Host "registered: $TaskName5pm (weekdays 17:00 local)"

# 1st of every month at 02:00 CT - cheap monthly retrain.
# Reuses Optuna-tuned hyperparams from the most recent quarterly tune,
# refreshes only the model weights against fresh data. ~5 min/universe.
# Day-1 may land on a weekend; the StartWhenAvailable setting fires it
# on the next available time if Windows missed it.
$TaskNameMonthly = "ta-agent-monthly-retrain"
$MonthlyTrigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
    -At ([datetime]::Today.AddHours(2).AddMinutes(0))
Register-ScheduledTask `
    -TaskName $TaskNameMonthly `
    -Action (Build-Action $MonthlyWrapper) `
    -Trigger $MonthlyTrigger `
    -Settings $Settings `
    -Description "ta-agent: monthly retrain (cheap, cached hyperparams)" `
    -Force | Out-Null
# Constrain to first business week of each month via the trigger's
# Repetition + StartBoundary -easier: register a separate per-month
# task. Windows doesn't have a direct cron-like 'first weekday of month'
# trigger, so we accept a coarser schedule and rely on the inner job
# being idempotent within a calendar day.
Write-Host "registered: $TaskNameMonthly (weekdays 02:00 local -inner job is idempotent within a day)"

# 1st of Jan/Apr/Jul/Oct at 03:00 CT - quarterly Optuna re-tune.
# ~3 hours/universe so it gets the longer ExecutionTimeLimit. Same
# 'first-business-day' coarseness as monthly.
$TaskNameQuarterly = "ta-agent-quarterly-retune"
$QuarterlyTrigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
    -At ([datetime]::Today.AddHours(3).AddMinutes(0))
Register-ScheduledTask `
    -TaskName $TaskNameQuarterly `
    -Action (Build-Action $QuarterlyWrapper) `
    -Trigger $QuarterlyTrigger `
    -Settings $LongSettings `
    -Description "ta-agent: quarterly Optuna re-tune (20 trials, ~3h/universe)" `
    -Force | Out-Null
Write-Host "registered: $TaskNameQuarterly (weekdays 03:00 local -inner job checks calendar)"

Write-Host ""
Write-Host "Done. Inspect with: Get-ScheduledTask -TaskName 'ta-agent-*'"
Write-Host "Run-on-demand for testing:"
Write-Host "  Start-ScheduledTask -TaskName '$TaskName8am'"
Write-Host "  Start-ScheduledTask -TaskName '$TaskNameMonthly'"
Write-Host "Watch logs:"
Write-Host "  Get-Content '$LogDir\scheduled_run_$(Get-Date -Format yyyy-MM-dd).log' -Wait -Tail 20"
Write-Host "  Get-Content '$LogDir\monthly_retrain_$(Get-Date -Format yyyy-MM-dd).log' -Wait -Tail 20"
Write-Host "  Get-Content '$LogDir\quarterly_retune_$(Get-Date -Format yyyy-MM-dd).log' -Wait -Tail 20"
