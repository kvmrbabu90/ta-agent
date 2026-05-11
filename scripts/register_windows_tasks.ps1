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

$RepoRoot = "C:\dev\ta-agent"
$Wrapper  = "$RepoRoot\scripts\run_pipeline.cmd"
$LogDir   = "$RepoRoot\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# Sanity checks
if (-not (Test-Path $Wrapper)) {
    Write-Error "$Wrapper not found - wrong checkout?"
    exit 1
}

# Direct .cmd invocation. The .cmd handles cd to repo root, log capture,
# and exit-code propagation. Avoids both the PowerShell quoting hell of
# v1 and the WorkingDirectory-not-honored bug seen with direct python.exe
# invocation when the user session isn't foreground.
function Build-Action {
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

# 8 AM CT - pre-market refresh
$TaskName8am = "ta-agent-pipeline-8am-ct"
Register-ScheduledTask `
    -TaskName $TaskName8am `
    -Action (Build-Action) `
    -Trigger (Build-WeekdayTrigger -Hour 8 -Minute 0) `
    -Settings $Settings `
    -Description "ta-agent: refresh OHLCV + predict + paper backtest at 08:00 CT pre-market" `
    -Force | Out-Null
Write-Host "registered: $TaskName8am (weekdays 08:00 local)"

# 5 PM CT - post-close refresh
$TaskName5pm = "ta-agent-pipeline-5pm-ct"
Register-ScheduledTask `
    -TaskName $TaskName5pm `
    -Action (Build-Action) `
    -Trigger (Build-WeekdayTrigger -Hour 17 -Minute 0) `
    -Settings $Settings `
    -Description "ta-agent: refresh OHLCV + predict + paper backtest at 17:00 CT post-close" `
    -Force | Out-Null
Write-Host "registered: $TaskName5pm (weekdays 17:00 local)"

Write-Host ""
Write-Host "Done. Inspect with: Get-ScheduledTask -TaskName 'ta-agent-*'"
Write-Host "Run-on-demand for testing:"
Write-Host "  Start-ScheduledTask -TaskName '$TaskName8am'"
Write-Host "Watch the run log: Get-Content '$LogDir\scheduled_run_$(Get-Date -Format yyyy-MM-dd).log' -Wait -Tail 20"
