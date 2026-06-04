param(
    [string]$ProjectRoot
)

$ErrorActionPreference = "Stop"
if (-not $ProjectRoot) {
    $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
Set-Location -LiteralPath $ProjectRoot

$python = "python"
if (Test-Path -LiteralPath ".\.venv\Scripts\python.exe") {
    $python = ".\.venv\Scripts\python.exe"
}

$updateMatch = Get-ChildItem -LiteralPath $ProjectRoot -Recurse -Filter "*.py" |
    Where-Object { $_.FullName -notmatch "\\tests\\" -and $_.FullName -notmatch "\\scripts\\" } |
    Select-String -Pattern "def connect_jydb" -List |
    Select-Object -First 1
if (-not $updateMatch) {
    throw "Cannot find data update script containing def connect_jydb"
}
$env:JYDB_UPDATE_SCRIPT = $updateMatch.Path

$diagnosticCode = @'
import argparse
import importlib.util
import os
import pathlib
import sys

p = pathlib.Path(os.environ["JYDB_UPDATE_SCRIPT"])
spec = importlib.util.spec_from_file_location("daily_return_update", p)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
m.load_project_env()

driver = os.environ.get("JYDB_DRIVER", "ODBC Driver 17 for SQL Server")
server = os.environ.get("JYDB_SERVER", "192.168.10.48")
database = os.environ.get("JYDB_DATABASE", "JYDB")
uid = os.environ.get("JYDB_UID", "tsreadonly")
pwd_set = bool(os.environ.get("JYDB_PWD"))

print(f"python={sys.executable}")
print(f"driver={driver}")
print(f"server={server}")
print(f"database={database}")
print(f"uid={uid}")
print(f"password_set={pwd_set}")

try:
    import pyodbc
    print(f"pyodbc={pyodbc.version}")
    print(f"drivers={pyodbc.drivers()}")
except Exception as exc:
    print(f"pyodbc_import_error={type(exc).__name__}: {exc}")
    raise

args = argparse.Namespace(
    jydb_password=None,
    jydb_driver=None,
    jydb_server=None,
    jydb_database=None,
    jydb_uid=None,
)

try:
    conn = m.connect_jydb(args)
    cur = conn.cursor()
    cur.execute("SELECT 1 AS ok")
    row = cur.fetchone()
    print(f"connect_result=ok value={row[0]}")
    conn.close()
except Exception as exc:
    print(f"connect_result=failed type={type(exc).__name__}")
    print(f"connect_error={exc}")
    raise
'@

$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("diagnose_jydb_connection_{0}.py" -f ([System.Guid]::NewGuid().ToString("N")))
try {
    Set-Content -LiteralPath $tmp -Value $diagnosticCode -Encoding UTF8
    & $python $tmp
}
finally {
    if (Test-Path -LiteralPath $tmp) {
        Remove-Item -LiteralPath $tmp -Force
    }
}
