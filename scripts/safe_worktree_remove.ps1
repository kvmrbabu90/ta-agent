# Safely remove a git worktree on Windows, even when the worktree contains
# directory junctions (e.g. `data/` symlinked back to the main tree).
#
# Background: `git worktree remove` uses a recursive delete that, on Windows,
# RECURSES INTO directory junctions and deletes their TARGET contents — not
# just the junction itself. If you have a `data/` junction in the worktree
# pointing at the main tree's `data/`, `git worktree remove` will silently
# wipe out the main tree's data/ directory.
#
# This script:
#   1. Detects all directory junctions inside the worktree
#   2. Removes each junction with `cmd /c rmdir` (NOT following the link)
#   3. THEN runs `git worktree remove`
#
# Usage:
#   PowerShell -ExecutionPolicy Bypass -File .\scripts\safe_worktree_remove.ps1 <worktree-path>
#
# Example:
#   PowerShell -ExecutionPolicy Bypass -File .\scripts\safe_worktree_remove.ps1 ..\ta-agent-india-phase-a

param(
    [Parameter(Mandatory=$true)]
    [string]$WorktreePath
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $WorktreePath)) {
    Write-Error "Worktree path does not exist: $WorktreePath"
    exit 1
}

$WorktreeAbs = (Resolve-Path $WorktreePath).Path
Write-Host "Scanning $WorktreeAbs for directory junctions..."

# Find directory junctions / symlinks inside the worktree. ReparsePoint
# attribute covers both junctions and symlinks. We use cmd's dir /aL to be
# extra defensive, but Get-ChildItem -Attributes ReparsePoint should suffice.
$junctions = Get-ChildItem -Path $WorktreeAbs -Recurse -Force -Directory -Attributes ReparsePoint -ErrorAction SilentlyContinue

if ($junctions.Count -eq 0) {
    Write-Host "No junctions found. Proceeding with normal git worktree remove."
} else {
    Write-Host "Found $($junctions.Count) junction(s):"
    foreach ($j in $junctions) {
        $target = (Get-Item $j.FullName).Target
        Write-Host "  $($j.FullName)  ->  $target"
    }
    Write-Host ""
    Write-Host "Removing each junction (without following its target)..."
    foreach ($j in $junctions) {
        $path = $j.FullName
        Write-Host "  cmd /c rmdir `"$path`""
        # cmd's rmdir on a junction removes the junction itself, NOT its target.
        # PowerShell's Remove-Item -Recurse on a junction follows the target.
        $result = cmd /c "rmdir `"$path`"" 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to remove junction $path : $result"
            exit 1
        }
    }
    Write-Host "All junctions removed."
}

Write-Host ""
Write-Host "Running: git worktree remove `"$WorktreeAbs`""
git worktree remove $WorktreeAbs
if ($LASTEXITCODE -ne 0) {
    Write-Error "git worktree remove failed."
    exit 1
}

Write-Host "Worktree removed safely."
