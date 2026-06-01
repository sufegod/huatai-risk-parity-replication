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

    def test_merge_cache_frames_replaces_overlap_and_sorts(self):
        module = load_module()
        existing = pd.DataFrame(
            [
                ["沪深300主连", "financial", "2026-05-29", 1002, "IF2606", 4010.0, 1],
                ["沪深300主连", "financial", "2026-05-28", 1001, "IF2606", 4000.0, 1],
            ],
            columns=["资产", "来源", "日期", "合约内部编码", "合约代码", "收盘价", "主力标志"],
        )
        incoming = pd.DataFrame(
            [
                ["沪深300主连", "financial", "2026-05-30", 1002, "IF2606", 4020.0, 1],
                ["沪深300主连", "financial", "2026-05-29", 1002, "IF2606", 4011.0, 1],
            ],
            columns=existing.columns,
        )

        result = module.merge_cache_frames(
            existing,
            incoming,
            key_columns=["资产", "日期", "合约内部编码"],
            sort_columns=["资产", "日期", "合约内部编码"],
        )

        self.assertEqual(list(result["日期"].dt.strftime("%Y-%m-%d")), ["2026-05-28", "2026-05-29", "2026-05-30"])
        self.assertEqual(result.loc[result["日期"] == pd.Timestamp("2026-05-29"), "收盘价"].iloc[0], 4011.0)
        self.assertEqual(len(result), 3)

    def test_incremental_fetch_start_uses_cache_max_minus_overlap(self):
        module = load_module()
        cache = pd.DataFrame({"日期": pd.to_datetime(["2026-05-28", "2026-05-29"])})

        result = module.calculate_incremental_start(
            cache,
            fallback_start=pd.Timestamp("2013-01-04"),
            overlap_days=7,
            full_refresh=False,
        )

        self.assertEqual(result, pd.Timestamp("2026-05-22"))

    def test_incremental_fetch_start_uses_fallback_for_empty_or_full_refresh(self):
        module = load_module()
        cache = pd.DataFrame({"日期": pd.to_datetime(["2026-05-29"])})

        empty_result = module.calculate_incremental_start(
            pd.DataFrame(columns=["日期"]),
            fallback_start=pd.Timestamp("2013-01-04"),
            overlap_days=7,
            full_refresh=False,
        )
        refresh_result = module.calculate_incremental_start(
            cache,
            fallback_start=pd.Timestamp("2013-01-04"),
            overlap_days=7,
            full_refresh=True,
        )

        self.assertEqual(empty_result, pd.Timestamp("2013-01-04"))
        self.assertEqual(refresh_result, pd.Timestamp("2013-01-04"))

    def test_build_futures_outputs_from_cache_matches_adjusted_price_algorithm(self):
        module = load_module()
        cached_quotes = pd.DataFrame(
            [
                ["沪深300主连", "financial", "2026-05-27", 1, "IFOLD", 100.0, 1],
                ["沪深300主连", "financial", "2026-05-28", 1, "IFOLD", 110.0, 1],
                ["沪深300主连", "financial", "2026-05-28", 2, "IFNEW", 220.0, 0],
                ["沪深300主连", "financial", "2026-05-29", 1, "IFOLD", 120.0, 0],
                ["沪深300主连", "financial", "2026-05-29", 2, "IFNEW", 240.0, 1],
            ],
            columns=["资产", "来源", "日期", "合约内部编码", "合约代码", "收盘价", "主力标志"],
        )

        returns, prices, summary = module.build_futures_outputs_from_cache(
            cached_quotes,
            start_date=pd.Timestamp("2026-05-27"),
            end_date=pd.Timestamp("2026-05-29"),
            lookback_days=0,
        )

        self.assertAlmostEqual(prices.loc[pd.Timestamp("2026-05-27"), "沪深300主连"], 200.0)
        self.assertAlmostEqual(prices.loc[pd.Timestamp("2026-05-28"), "沪深300主连"], 220.0)
        self.assertAlmostEqual(prices.loc[pd.Timestamp("2026-05-29"), "沪深300主连"], 240.0)
        self.assertAlmostEqual(returns.loc[pd.Timestamp("2026-05-29"), "沪深300主连"], (240.0 / 220.0 - 1.0) * 100.0)
        self.assertEqual(summary[0]["行情行数"], 5)

    def test_main_rebuild_from_cache_does_not_call_external_sources(self):
        module = load_module()
        legacy_frame = pd.DataFrame(index=pd.to_datetime(["2026-05-27", "2026-05-28", "2026-05-29"]))
        futures_cache = pd.DataFrame(
            [
                ["沪深300主连", "financial", "2026-05-27", 1, "IF2606", 4000.0, 1],
                ["沪深300主连", "financial", "2026-05-28", 1, "IF2606", 4010.0, 1],
                ["沪深300主连", "financial", "2026-05-29", 1, "IF2606", 4020.0, 1],
            ],
            columns=["资产", "来源", "日期", "合约内部编码", "合约代码", "收盘价", "主力标志"],
        )
        etf_cache = pd.DataFrame(
            {
                "日期": pd.to_datetime(["2026-05-27", "2026-05-28", "2026-05-29"]),
                "PrevClosePrice": [1.0, 1.0, 1.0],
                "ClosePrice": [1.0, 1.01, 1.02],
            }
        )
        gc001_cache = pd.DataFrame(
            {
                "日期": pd.to_datetime(["2026-05-27", "2026-05-28", "2026-05-29"]),
                "一天期国债逆回购": [1.5, 1.6, 1.7],
            }
        )
        written = {"returns": 0, "prices": 0, "summary": 0}

        def count_returns(*args, **kwargs):
            written["returns"] += 1

        def count_prices(*args, **kwargs):
            written["prices"] += 1

        def count_summary(*args, **kwargs):
            written["summary"] += 1

        with (
            patch.object(module, "load_project_env"),
            patch.object(module, "find_existing_file", return_value=Path("returns.csv")),
            patch.object(module, "read_returns_csv", return_value=legacy_frame),
            patch.object(module, "read_futures_quote_cache", return_value=futures_cache),
            patch.object(module, "read_etf_quote_cache", return_value=etf_cache),
            patch.object(module, "read_gc001_cache", return_value=gc001_cache),
            patch.object(module, "connect_jydb", side_effect=AssertionError("JYDB should not be used")),
            patch.object(module, "fetch_gc001_weighted_average", side_effect=AssertionError("iFinD should not be used")),
            patch.object(module, "write_returns_csv", side_effect=count_returns),
            patch.object(module, "write_price_csv", side_effect=count_prices),
            patch.object(module, "write_summary", side_effect=count_summary),
        ):
            exit_code = module.main(["--rebuild-from-cache", "--end-date", "2026-05-29"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(written, {"returns": 2, "prices": 1, "summary": 1})

    def test_write_cache_csv_dry_run_does_not_create_file(self):
        module = load_module()
        target = self.workspace_temp_dir("cache_dry_run") / "cache.csv"
        df = pd.DataFrame({"日期": pd.to_datetime(["2026-05-29"]), "值": [1.0]})

        module.write_cache_csv(df, target, columns=["日期", "值"], dry_run=True)

        self.assertFalse(target.exists())

    def test_refresh_futures_quote_cache_skips_when_target_not_newer_than_cache(self):
        module = load_module()
        existing = pd.DataFrame(
            [
                ["沪深300主连", "financial", "2026-05-29", 1002, "IF2606", 4010.0, 1],
            ],
            columns=["资产", "来源", "日期", "合约内部编码", "合约代码", "收盘价", "主力标志"],
        )

        with (
            patch.object(module, "fetch_futures_cache_rows", side_effect=AssertionError("should not fetch")),
            patch.object(module, "write_cache_csv", side_effect=AssertionError("should not write")),
        ):
            result, stat = module.refresh_futures_quote_cache(
                conn=object(),
                existing=existing,
                fallback_start=pd.Timestamp("2013-01-04"),
                target_end=pd.Timestamp("2026-05-29"),
                overlap_days=7,
                full_refresh=False,
                dry_run=False,
            )

        self.assertEqual(len(result), 1)
        self.assertTrue(stat.skipped)
        self.assertEqual(stat.previous_end, "2026-05-29")

    def test_refresh_futures_quote_cache_initializes_empty_cache_and_writes_result(self):
        module = load_module()
        incoming = pd.DataFrame(
            [
                ["沪深300主连", "financial", "2013-01-04", 1001, "IF1301", 2500.0, 1],
            ],
            columns=["资产", "来源", "日期", "合约内部编码", "合约代码", "收盘价", "主力标志"],
        )
        captured = {}

        def capture_write(df, path, columns, dry_run):
            captured["rows"] = len(df)
            captured["path_name"] = path.name
            captured["columns"] = columns
            captured["dry_run"] = dry_run

        with (
            patch.object(module, "fetch_futures_cache_rows", return_value=incoming) as fetch_mock,
            patch.object(module, "write_cache_csv", side_effect=capture_write),
        ):
            result, stat = module.refresh_futures_quote_cache(
                conn=object(),
                existing=pd.DataFrame(columns=["日期"]),
                fallback_start=pd.Timestamp("2013-01-04"),
                target_end=pd.Timestamp("2013-01-04"),
                overlap_days=7,
                full_refresh=False,
                dry_run=False,
            )

        fetch_mock.assert_called_once()
        self.assertEqual(fetch_mock.call_args.args[1], pd.Timestamp("2013-01-04"))
        self.assertEqual(fetch_mock.call_args.args[2], pd.Timestamp("2013-01-04"))
        self.assertEqual(len(result), 1)
        self.assertEqual(stat.fetched_rows, 1)
        self.assertEqual(captured["rows"], 1)
        self.assertEqual(captured["path_name"], "期货行情.csv")
        self.assertEqual(captured["columns"], module.FUTURES_CACHE_COLUMNS)
        self.assertFalse(captured["dry_run"])

    def test_parse_args_no_longer_accepts_backup(self):
        module = load_module()

        with self.assertRaises(SystemExit):
            module.parse_args(["--backup"])


if __name__ == "__main__":
    unittest.main()
