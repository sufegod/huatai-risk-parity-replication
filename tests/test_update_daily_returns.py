import importlib.util
import json
import os
import shutil
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "数据" / "JYDB数据替换" / "update_daily_returns.py"


def load_module():
    spec = importlib.util.spec_from_file_location("update_daily_returns", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class UpdateDailyReturnsTests(unittest.TestCase):
    def workspace_temp_dir(self, name: str) -> Path:
        root = PROJECT_ROOT / ".test_update_daily_returns"
        temp_dir = root / name
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        return temp_dir

    def test_load_env_file_parses_simple_key_values(self):
        module = load_module()
        env_path = self.workspace_temp_dir("env_parse") / ".env"
        env_path.write_text(
            "\n".join(
                [
                    "# local secrets",
                    "JYDB_SERVER=192.168.10.48",
                    "JYDB_UID=tsreadonly",
                    "IFIND_MCP_URL=\"https://ifind.example/mcp\"",
                    "IFIND_MCP_AUTHORIZATION='Bearer test-token'",
                    "EMPTY_VALUE=",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        with patch.dict(os.environ, {}, clear=True):
            module.load_env_file(env_path)

            self.assertEqual(os.environ["JYDB_SERVER"], "192.168.10.48")
            self.assertEqual(os.environ["JYDB_UID"], "tsreadonly")
            self.assertEqual(os.environ["IFIND_MCP_URL"], "https://ifind.example/mcp")
            self.assertEqual(os.environ["IFIND_MCP_AUTHORIZATION"], "Bearer test-token")
            self.assertEqual(os.environ["EMPTY_VALUE"], "")

    def test_load_env_file_does_not_override_existing_environment(self):
        module = load_module()
        env_path = self.workspace_temp_dir("env_no_override") / ".env"
        env_path.write_text(
            "\n".join(
                [
                    "JYDB_PWD=file-password",
                    "JYDB_SERVER=file-server",
                ]
            ),
            encoding="utf-8",
        )

        with patch.dict(os.environ, {"JYDB_PWD": "existing-password"}, clear=True):
            module.load_env_file(env_path)

            self.assertEqual(os.environ["JYDB_PWD"], "existing-password")
            self.assertEqual(os.environ["JYDB_SERVER"], "file-server")

    def test_read_ifind_mcp_config_prefers_environment_without_toml(self):
        module = load_module()

        with patch.dict(
            os.environ,
            {
                "IFIND_MCP_URL": "https://ifind.example/mcp",
                "IFIND_MCP_AUTHORIZATION": "Bearer env-token",
            },
            clear=True,
        ):
            url, headers = module.read_ifind_mcp_config(Path("does-not-exist.toml"))

        self.assertEqual(url, "https://ifind.example/mcp")
        self.assertEqual(headers["Authorization"], "Bearer env-token")
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(headers["Accept"], "application/json, text/event-stream")

    def test_read_ifind_mcp_config_falls_back_to_toml_when_env_missing(self):
        module = load_module()
        config_path = self.workspace_temp_dir("ifind_toml") / "config.toml"
        config_path.write_text(
            "\n".join(
                [
                    "[mcp_servers.hexin-ifind-ds-edb-mcp]",
                    'url = "https://ifind.example/toml"',
                    "[mcp_servers.hexin-ifind-ds-edb-mcp.http_headers]",
                    'Authorization = "Bearer toml-token"',
                ]
            ),
            encoding="utf-8",
        )

        with patch.dict(os.environ, {}, clear=True):
            url, headers = module.read_ifind_mcp_config(config_path)

        self.assertEqual(url, "https://ifind.example/toml")
        self.assertEqual(headers["Authorization"], "Bearer toml-token")

    def test_main_loads_project_env_before_connecting_to_jydb(self):
        module = load_module()
        env_path = self.workspace_temp_dir("main_env") / ".env"
        env_path.write_text("JYDB_PWD=file-password\n", encoding="utf-8")
        legacy_frame = pd.DataFrame(index=pd.to_datetime(["2026-05-28"]))

        with (
            patch.object(module, "ENV_FILE", env_path, create=True),
            patch.object(module, "find_existing_file", return_value=Path("returns.csv")),
            patch.object(module, "read_returns_csv", return_value=legacy_frame),
            patch.object(module, "connect_jydb", side_effect=RuntimeError("stop after env load")),
            patch.dict(os.environ, {"USERPROFILE": str(PROJECT_ROOT)}, clear=True),
        ):
            with self.assertRaisesRegex(RuntimeError, "stop after env load"):
                module.main(["--end-date", "2026-05-28"])

            self.assertEqual(os.environ["JYDB_PWD"], "file-password")

    def test_prune_output_columns_removes_unused_assets_and_preserves_order(self):
        module = load_module()
        df = pd.DataFrame(
            {
                "日期": ["2026-03-31"],
                "沪深300主连": [1.0],
                "红利低波ETF": [2.0],
                "有色ETF": [3.0],
                "能源化工ETF": [4.0],
                "一天期国债逆回购": [5.0],
                "布油连续": [6.0],
                "原油主连": [7.0],
            }
        )

        result = module.prune_output_columns(df)

        self.assertEqual(
            list(result.columns),
            ["日期", "沪深300主连", "红利低波ETF", "一天期国债逆回购", "原油主连"],
        )

    def test_compute_etf_return_uses_prev_close_on_adjustment_day(self):
        module = load_module()
        quotes = pd.DataFrame(
            {
                "日期": pd.to_datetime(["2021-10-22", "2021-10-25"]),
                "PrevClosePrice": [2.119, 1.036],
                "ClosePrice": [2.072, 1.036],
            }
        )

        result = module.compute_return_from_prev_close(quotes)

        self.assertAlmostEqual(result.loc[pd.Timestamp("2021-10-25")], 0.0, places=10)
        raw_close_return = (1.036 / 2.072 - 1.0) * 100.0
        self.assertLess(raw_close_return, -49.0)

    def test_parse_ifind_edb_response_unwraps_nested_json_text(self):
        module = load_module()
        inner = {
            "data": {
                "datas": [
                    {
                        "data": {
                            "columns": ["日期", "GC001(加权平均)"],
                            "attrs": {
                                "GC001(加权平均)": {
                                    "unit": "%",
                                    "dtype": "double",
                                    "index_id": "L004369613",
                                }
                            },
                            "data": [["2013-01-07", 3.553], ["2013-01-04", 5.256]],
                        }
                    }
                ]
            }
        }
        outer = {"result": {"content": [{"text": json.dumps(inner, ensure_ascii=False)}]}}

        result = module.parse_ifind_edb_response(outer)

        self.assertEqual(result.name, "一天期国债逆回购")
        self.assertEqual(list(result.index.strftime("%Y-%m-%d")), ["2013-01-04", "2013-01-07"])
        self.assertEqual(result.loc[pd.Timestamp("2013-01-04")], 5.256)


if __name__ == "__main__":
    unittest.main()
