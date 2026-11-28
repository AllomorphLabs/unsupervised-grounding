# Unattended GPU job queue: runs jobs one at a time, independent of any Claude session.
# Queue: runs/queue/jobs.txt, one job per line: "<name> | <command>". Lines starting with # are ignored.
# Each job writes runs/queue/<name>.log and, when finished, runs/queue/<name>.done (holding the exit code).
# A job already marked .done is skipped, so jobs can be appended while the runner is running.
# The runner stops when runs/queue/STOP exists, or after it has been idle for $IdleMinutes.
param([int]$IdleMinutes = 600)
$root = "C:\Sites\allomorphlabs\unsupervised-grounding"
$q = Join-Path $root "runs\queue"
Set-Location $root
$idleSince = Get-Date
while ($true) {
    if (Test-Path (Join-Path $q "STOP")) { break }
    $next = $null
    foreach ($line in Get-Content (Join-Path $q "jobs.txt") -ErrorAction SilentlyContinue) {
        if ($line -match '^\s*#' -or $line -notmatch '\|') { continue }
        $name, $cmd = $line.Split('|', 2) | ForEach-Object { $_.Trim() }
        if (-not (Test-Path (Join-Path $q "$name.done")) -and -not (Test-Path (Join-Path $q "$name.running"))) { $next = @($name, $cmd); break }
    }
    if ($null -eq $next) {
        if (((Get-Date) - $idleSince).TotalMinutes -gt $IdleMinutes) { break }
        Start-Sleep -Seconds 20; continue
    }
    $name, $cmd = $next
    $log = Join-Path $q "$name.log"
    Set-Content (Join-Path $q "$name.running") (Get-Date -Format o)
    Add-Content (Join-Path $q "runner.log") "$(Get-Date -Format o) start $name : $cmd"
    & cmd.exe /c "$cmd > `"$log`" 2>&1"
    $code = $LASTEXITCODE
    Set-Content (Join-Path $q "$name.done") $code
    Remove-Item (Join-Path $q "$name.running") -ErrorAction SilentlyContinue
    Add-Content (Join-Path $q "runner.log") "$(Get-Date -Format o) end $name exit $code"
    $idleSince = Get-Date
}
Add-Content (Join-Path $q "runner.log") "$(Get-Date -Format o) runner stopped"
