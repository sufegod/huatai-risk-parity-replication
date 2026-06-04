param(
    [string]$ProjectRoot,
    [string]$DataEndDate,
    [switch]$SkipDataUpdate,
    [switch]$ForceObservation,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
if (-not $ProjectRoot) {
    $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}
Set-Location -LiteralPath $ProjectRoot

$python = "python"
if (Test-Path -LiteralPath ".\.venv\Scripts\python.exe") {
    $python = (Resolve-Path -LiteralPath ".\.venv\Scripts\python.exe").Path
}

$dailyScript = Get-ChildItem -LiteralPath $ProjectRoot -Recurse -Filter "daily_update_strategy.py" |
    Select-Object -First 1
if (-not $dailyScript) {
    throw "Cannot find daily_update_strategy.py"
}

$scriptArgs = @($dailyScript.FullName)
if ($DataEndDate) {
    $scriptArgs += @("--data-end-date", $DataEndDate)
}
if ($SkipDataUpdate) {
    $scriptArgs += "--skip-data-update"
}
if ($ForceObservation) {
    $scriptArgs += "--force-observation"
}
if ($ExtraArgs) {
    $scriptArgs += $ExtraArgs
}

& $python @scriptArgs
exit $LASTEXITCODE
