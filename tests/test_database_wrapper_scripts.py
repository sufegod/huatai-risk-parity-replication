from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DAILY_UPDATE_DIR = PROJECT_ROOT / "策略复现与回测" / "每日更新策略"
DATA_UPDATE_DIR = PROJECT_ROOT / "数据" / "日度收益数据更新"
TESTS_DIR = PROJECT_ROOT / "tests"
OLD_SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def read_script(path: Path) -> str:
    return path.read_text(encoding="utf-8")



class DatabaseWrapperScriptTests(unittest.TestCase):
    def test_daily_update_wrapper_uses_approved_powershell_entrypoint(self) -> None:
        content = read_script(DAILY_UPDATE_DIR / "run_daily_update.ps1")

        self.assertNotIn("[string]$ProjectRoot =", content)
        self.assertIn("if (-not $ProjectRoot)", content)
        self.assertIn('Join-Path $PSScriptRoot "..\\.."', content)
        self.assertIn("daily_update_strategy.py", content)
        self.assertIn("--data-end-date", content)
        self.assertIn("--skip-data-update", content)
        self.assertNotIn("JYDB_PWD", content)

    def test_data_update_wrapper_discovers_jydb_script_without_chinese_literal_path(self) -> None:
        content = read_script(DATA_UPDATE_DIR / "run_data_update.ps1")

        self.assertNotIn("[string]$ProjectRoot =", content)
        self.assertIn("if (-not $ProjectRoot)", content)
        self.assertIn('Join-Path $PSScriptRoot "..\\.."', content)
        self.assertIn("def connect_jydb", content)
        self.assertIn("--end-date", content)
        self.assertIn("--rebuild-from-cache", content)
        self.assertIn("-notmatch", content)
        self.assertIn("\\tests\\", content)
        self.assertNotIn("JYDB_PWD", content)

    def test_diagnostic_script_lives_with_tests(self) -> None:
        content = read_script(TESTS_DIR / "diagnose_jydb_connection.ps1")

        self.assertNotIn("[string]$ProjectRoot =", content)
        self.assertIn("if (-not $ProjectRoot)", content)
        self.assertIn('Join-Path $PSScriptRoot ".."', content)
        self.assertIn("SELECT 1 AS ok", content)
        self.assertIn("password_set=", content)
        self.assertNotIn("PWD=", content)

    def test_old_scripts_directory_no_longer_contains_wrappers(self) -> None:
        old_wrappers = [
            OLD_SCRIPTS_DIR / "run_daily_update.ps1",
            OLD_SCRIPTS_DIR / "run_data_update.ps1",
            OLD_SCRIPTS_DIR / "diagnose_jydb_connection.ps1",
        ]

        self.assertFalse(any(path.exists() for path in old_wrappers))
