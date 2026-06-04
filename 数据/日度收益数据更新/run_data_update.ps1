param(
    [string]$ProjectRoot,
    [string]$StartDate,
    [string]$EndDate,
    [int]$LookbackDays,
    [int]$CacheOverlapDays,
    [switch]$FullRefresh,
    [switch]$RebuildFromCache,
    [switch]$DryRun,
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

$updateMatch = Get-ChildItem -LiteralPath $ProjectRoot -Recurse -Filter "*.py" |
    Where-Object { $_.FullName -notmatch "\\tests\\" -and $_.FullName -notmatch "\\scripts\\" } |
    Select-String -Pattern "def connect_jydb" -List |
    Select-Object -First 1
if (-not $updateMatch) {
    throw "Cannot find data update script containing def connect_jydb"
}

$scriptArgs = @($updateMatch.Path)
if ($StartDate) {
    $scriptArgs += @("--start-date", $StartDate)
}
if ($EndDate) {
    $scriptArgs += @("--end-date", $EndDate)
}
if ($PSBoundParameters.ContainsKey("LookbackDays")) {
    $scriptArgs += @("--lookback-days", $LookbackDays)
}
if ($PSBoundParameters.ContainsKey("CacheOverlapDays")) {
    $scriptArgs += @("--cache-overlap-days", $CacheOverlapDays)
}
if ($FullRefresh) {
    $scriptArgs += "--full-refresh"
}
if ($RebuildFromCache) {
    $scriptArgs += "--rebuild-from-cache"
}
if ($DryRun) {
    $scriptArgs += "--dry-run"
}
if ($ExtraArgs) {
    $scriptArgs += $ExtraArgs
}

& $python @scriptArgs
exit $LASTEXITCODE
