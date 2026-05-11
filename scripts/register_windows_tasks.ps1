# Register two Windows Scheduled Tasks that run the ta-agent pipeline at
# 08:00 CT and 17:00 CT each weekday. No third-party tools needed — this
# uses Windows' built-in Task Scheduler.
#
# Usage (PowerShell, run as your normal user — NOT admin required):
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

$ErrorActionPreference = "Stop"

$RepoRoot = "C:\dev\ta-agent"
$Python   = "$RepoRoot\.venv\Scripts\python.exe"
$LogDir   = "$RepoRoot\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# Sanity checks
if (-not (Test-Path $Python)) {
    Write-Error "Python interpreter not found at $Python — did you run uv venv?"
    exit 1
}
if (-not (Test-Path "$RepoRoot\jobs\run_pipeline.py")) {
    Write-Error "jobs\run_pipeline.py not found at $RepoRoot — wrong checkout?"
    exit 1
}

# Each task wraps the python invocation in a small powershell line so we
# can append stdout/stderr to a date-stamped log file.
function Build-WrappedAction {
    param([string]$TaskName)
    $cmd = @"
& '$Python' -m jobs.run_pipeline *>> '$LogDir\\$($TaskName)_$(Get-Date -Format yyyyMMdd).log'
"@
    return New-ScheduledTaskAction `
        -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -Command `"$cmd`"" `
        -WorkingDirectory $RepoRoot
}

# Trigger: weekdays at the requested local time (Windows respects the
# system's timezone setting; if you're not in CT, the trigger fires at
# 08:00 / 17:00 of YOUR local time. Set your Windows clock to CT or
# adjust below if needed).
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

# 8 AM CT — pre-market refresh
$TaskName8am = "ta-agent-pipeline-8am-ct"
$Action8am   = Build-WrappedAction -TaskName $TaskName8am
$Trigger8am  = Build-WeekdayTrigger -Hour 8 -Minute 0
Register-ScheduledTask `
    -TaskName $TaskName8am `
    -Action $Action8am `
    -Trigger $Trigger8am `
    -Settings $Settings `
    -Description "ta-agent: refresh OHLCV + predict + paper backtest at 08:00 CT pre-market" `
    -Force | Out-Null
Write-Host "registered: $TaskName8am (weekdays 08:00 local)"

# 5 PM CT — post-close refresh
$TaskName5pm = "ta-agent-pipeline-5pm-ct"
$Action5pm   = Build-WrappedAction -TaskName $TaskName5pm
$Trigger5pm  = Build-WeekdayTrigger -Hour 17 -Minute 0
Register-ScheduledTask `
    -TaskName $TaskName5pm `
    -Action $Action5pm `
    -Trigger $Trigger5pm `
    -Settings $Settings `
    -Description "ta-agent: refresh OHLCV + predict + paper backtest at 17:00 CT post-close" `
    -Force | Out-Null
Write-Host "registered: $TaskName5pm (weekdays 17:00 local)"

Write-Host ""
Write-Host "Done. Inspect with: Get-ScheduledTask -TaskName 'ta-agent-*'"
Write-Host "Run-on-demand for testing:"
Write-Host "  Start-ScheduledTask -TaskName '$TaskName8am'"
Write-Host "Logs land in: $LogDir\<task-name>_YYYYMMDD.log"
