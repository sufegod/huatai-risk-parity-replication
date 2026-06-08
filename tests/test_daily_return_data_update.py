import os
import shutil
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load script module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


module = load_script_module(
    "daily_returns_update_under_test",
    PROJECT_ROOT / "数据" / "日度收益数据更新" / "日度收益数据更新.py",
)


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
        env_path = self.workspace_temp_dir("env_parse") / ".env"
        env_path.write_text(
            "\n".join(
                [
                    "# local secrets",
                    "JYDB_SERVER=192.168.10.48",
                    "JYDB_UID=tsreadonly",
                    "CUSTOM_VALUE=\"quoted-value\"",
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
            self.assertEqual(os.environ["CUSTOM_VALUE"], "quoted-value")
            self.assertEqual(os.environ["EMPTY_VALUE"], "")

    def test_load_env_file_does_not_override_existing_environment(self):
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

    def test_main_loads_project_env_before_connecting_to_jydb(self):
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

    def test_futures_assets_include_csi500_contract_mapping(self):
        assets = {asset.name: asset for asset in module.FUTURES_ASSETS}

        self.assertIn("中证500主连", assets)
        csi500 = assets["中证500主连"]
        self.assertEqual(csi500.source, "financial")
        self.assertEqual(csi500.exchange_code, 20)
        self.assertEqual(csi500.option_code, 4978)
        self.assertEqual(
            module.OUTPUT_COLUMNS.index("中证500主连"),
            module.OUTPUT_COLUMNS.index("中证1000主连") + 1,
        )

    def test_fetch_csi1000_index_quote_rows_queries_qt_indexquote(self):
        captured = {}

        def fake_fetch_dataframe(conn, sql, params):
            captured["conn"] = conn
            captured["sql"] = sql
            captured["params"] = params
            return pd.DataFrame(
                {
                    "日期": ["2026-05-29", "2026-05-28"],
                    "PrevClosePrice": [100.0, 99.0],
                    "ClosePrice": [101.0, 100.0],
                    "ChangePCT": [1.0, 1.010101],
                }
            )

        conn = object()
        with patch.object(module, "fetch_dataframe", side_effect=fake_fetch_dataframe):
            result = module.fetch_csi1000_index_quote_rows(
                conn,
                pd.Timestamp("2026-05-28"),
                pd.Timestamp("2026-05-29"),
            )

        self.assertIs(captured["conn"], conn)
        self.assertIn("dbo.QT_IndexQuote", captured["sql"])
        self.assertIn("InnerCode = ?", captured["sql"])
        self.assertEqual(captured["params"], (39144, "2026-05-28", "2026-05-29"))
        self.assertEqual(list(result.columns), ["日期", "PrevClosePrice", "ClosePrice", "ChangePCT"])
        self.assertEqual(list(result["日期"].dt.strftime("%Y-%m-%d")), ["2026-05-28", "2026-05-29"])

    def test_build_outputs_fills_csi500_prelisting_with_csi1000_index_only_in_filled_output(self):
        dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
        official = pd.DataFrame(
            {"中证500主连": [pd.NA, 0.5, pd.NA]},
            index=dates,
        )
        legacy_filled = pd.DataFrame(
            {"中证500主连": [9.0, 9.0, 9.0]},
            index=dates,
        )
        csi1000_index_return = pd.Series([1.1, 1.2, 1.3], index=dates, name="中证1000指数")

        filled, unfilled = module.build_outputs(
            official,
            legacy_filled,
            start_date=dates[0],
            end_date=dates[-1],
            csi1000_index_return=csi1000_index_return,
        )

        self.assertAlmostEqual(filled.loc[dates[0], "中证500主连"], 1.1)
        self.assertAlmostEqual(filled.loc[dates[1], "中证500主连"], 0.5)
        self.assertAlmostEqual(filled.loc[dates[2], "中证500主连"], 9.0)
        self.assertTrue(pd.isna(unfilled.loc[dates[0], "中证500主连"]))
        self.assertAlmostEqual(unfilled.loc[dates[1], "中证500主连"], 0.5)
        self.assertTrue(pd.isna(unfilled.loc[dates[2], "中证500主连"]))

    def test_build_outputs_keeps_datetime_index_for_csv_writes(self):
        temp_dir = self.workspace_temp_dir("datetime_index_write")
        dates = pd.to_datetime(["2026-01-02", "2026-01-05"])
        official = pd.DataFrame({"中证500主连": [pd.NA, 0.5]}, index=dates)
        legacy_filled = pd.DataFrame({"中证500主连": [1.0, 1.1]}, index=dates)
        csi1000_index_return = pd.Series([2.0, 2.1], index=dates, name="中证1000指数")

        filled, _ = module.build_outputs(
            official,
            legacy_filled,
            start_date=dates[0],
            end_date=dates[-1],
            csi1000_index_return=csi1000_index_return,
        )

        self.assertIsInstance(filled.index, pd.DatetimeIndex)
        module.write_returns_csv(filled, temp_dir / "returns.csv", dry_run=True)

    def test_compute_etf_return_uses_prev_close_on_adjustment_day(self):
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

    def test_merge_cache_frames_replaces_overlap_and_sorts(self):
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
        cache = pd.DataFrame({"日期": pd.to_datetime(["2026-05-28", "2026-05-29"])})

        result = module.calculate_incremental_start(
            cache,
            fallback_start=pd.Timestamp("2013-01-04"),
            overlap_days=7,
            full_refresh=False,
        )

        self.assertEqual(result, pd.Timestamp("2026-05-22"))

    def test_incremental_fetch_start_uses_fallback_for_empty_or_full_refresh(self):
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

    def test_build_futures_outputs_from_cache_keeps_datetime_index_when_some_assets_have_no_quotes(self):
        cached_quotes = pd.DataFrame(
            [
                ["沪深300主连", "financial", "2026-05-27", 1, "IF2606", 4000.0, 1],
                ["沪深300主连", "financial", "2026-05-28", 1, "IF2606", 4010.0, 1],
                ["沪深300主连", "financial", "2026-05-29", 1, "IF2606", 4020.0, 1],
            ],
            columns=["资产", "来源", "日期", "合约内部编码", "合约代码", "收盘价", "主力标志"],
        )

        returns, prices, _ = module.build_futures_outputs_from_cache(
            cached_quotes,
            start_date=pd.Timestamp("2026-05-27"),
            end_date=pd.Timestamp("2026-05-29"),
            lookback_days=0,
        )

        self.assertIsInstance(returns.index, pd.DatetimeIndex)
        self.assertIsInstance(prices.index, pd.DatetimeIndex)
        module.write_returns_csv(returns, self.workspace_temp_dir("futures_index_write") / "returns.csv", dry_run=True)

    def test_main_rebuild_from_cache_does_not_call_external_sources(self):
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
        csi1000_index_cache = pd.DataFrame(
            {
                "日期": pd.to_datetime(["2026-05-27", "2026-05-28", "2026-05-29"]),
                "PrevClosePrice": [100.0, 101.0, 102.0],
                "ClosePrice": [101.0, 102.0, 103.0],
                "ChangePCT": [1.0, 0.990099, 0.980392],
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
            patch.object(module, "read_csi1000_index_quote_cache", return_value=csi1000_index_cache),
            patch.object(module, "connect_jydb", side_effect=AssertionError("JYDB should not be used")),
            patch.object(module, "write_returns_csv", side_effect=count_returns),
            patch.object(module, "write_price_csv", side_effect=count_prices),
            patch.object(module, "write_summary", side_effect=count_summary),
        ):
            exit_code = module.main(["--rebuild-from-cache", "--end-date", "2026-05-29"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(written, {"returns": 2, "prices": 1, "summary": 1})

    def test_fetch_gc001_quote_rows_queries_ftdb_ths_gc(self):
        captured = {}

        def fake_fetch_dataframe(conn, sql, params):
            captured["conn"] = conn
            captured["sql"] = sql
            captured["params"] = params
            return pd.DataFrame(
                {
                    "日期": ["2026-05-29", "2026-05-28"],
                    "一天期国债逆回购": [1.084, 1.324],
                }
            )

        conn = object()
        with patch.object(module, "fetch_dataframe", side_effect=fake_fetch_dataframe):
            result = module.fetch_gc001_quote_rows(
                conn,
                pd.Timestamp("2026-05-28"),
                pd.Timestamp("2026-05-29"),
            )

        self.assertIs(captured["conn"], conn)
        self.assertIn("FTDB.dbo.ths_GC", captured["sql"])
        self.assertIn("thscode = '204001.SH'", captured["sql"])
        self.assertIn("ths_wgt_avg_interest_bbond", captured["sql"])
        self.assertEqual(captured["params"], ("2026-05-28", "2026-05-29"))
        self.assertEqual(list(result.columns), ["日期", "一天期国债逆回购"])
        self.assertEqual(list(result["日期"].dt.strftime("%Y-%m-%d")), ["2026-05-28", "2026-05-29"])
        self.assertEqual(result["一天期国债逆回购"].tolist(), [1.324, 1.084])

    def test_refresh_gc001_cache_fetches_from_ftdb_and_replaces_overlap(self):
        existing = pd.DataFrame(
            {
                "日期": pd.to_datetime(["2026-05-28", "2026-05-29"]),
                "一天期国债逆回购": [1.324, 9.999],
            }
        )
        incoming = pd.DataFrame(
            {
                "日期": pd.to_datetime(["2026-05-29", "2026-06-01"]),
                "一天期国债逆回购": [1.084, 1.372],
            }
        )
        captured = {}

        def capture_write(df, path, columns, dry_run):
            captured["df"] = df.copy()
            captured["path_name"] = path.name
            captured["columns"] = columns
            captured["dry_run"] = dry_run

        conn = object()
        with (
            patch.object(module, "fetch_gc001_quote_rows", return_value=incoming) as fetch_mock,
            patch.object(module, "write_cache_csv", side_effect=capture_write),
        ):
            result, stat = module.refresh_gc001_cache(
                conn=conn,
                existing=existing,
                fallback_start=pd.Timestamp("2013-01-04"),
                target_end=pd.Timestamp("2026-06-01"),
                overlap_days=7,
                full_refresh=False,
                dry_run=False,
            )

        fetch_mock.assert_called_once()
        self.assertIs(fetch_mock.call_args.args[0], conn)
        self.assertEqual(fetch_mock.call_args.args[1], pd.Timestamp("2026-05-22"))
        self.assertEqual(fetch_mock.call_args.args[2], pd.Timestamp("2026-06-01"))
        self.assertEqual(list(result["日期"].dt.strftime("%Y-%m-%d")), ["2026-05-28", "2026-05-29", "2026-06-01"])
        self.assertEqual(result["一天期国债逆回购"].tolist(), [1.324, 1.084, 1.372])
        self.assertEqual(stat.fetched_rows, 2)
        self.assertEqual(stat.cache_rows, 3)
        self.assertEqual(captured["path_name"], "GC001.csv")
        self.assertEqual(captured["columns"], module.GC001_CACHE_COLUMNS)
        self.assertFalse(captured["dry_run"])

    def test_parse_args_no_longer_accepts_ifind_config(self):
        with self.assertRaises(SystemExit):
            module.parse_args(["--ifind-config", "config.toml"])

    def test_write_cache_csv_dry_run_does_not_create_file(self):
        target = self.workspace_temp_dir("cache_dry_run") / "cache.csv"
        df = pd.DataFrame({"日期": pd.to_datetime(["2026-05-29"]), "值": [1.0]})

        module.write_cache_csv(df, target, columns=["日期", "值"], dry_run=True)

        self.assertFalse(target.exists())

    def test_refresh_futures_quote_cache_skips_when_target_not_newer_than_cache(self):
        rows = [
            [asset.name, asset.source, "2026-05-29", idx + 1000, f"{idx}2606", 4010.0 + idx, 1]
            for idx, asset in enumerate(module.FUTURES_ASSETS)
        ]
        existing = pd.DataFrame(rows, columns=["资产", "来源", "日期", "合约内部编码", "合约代码", "收盘价", "主力标志"])

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

        self.assertEqual(len(result), len(module.FUTURES_ASSETS))
        self.assertTrue(stat.skipped)
        self.assertEqual(stat.previous_end, "2026-05-29")

    def test_refresh_futures_quote_cache_fetches_full_history_when_configured_asset_missing(self):
        existing = pd.DataFrame(
            [
                ["沪深300主连", "financial", "2026-05-29", 1002, "IF2606", 4010.0, 1],
            ],
            columns=["资产", "来源", "日期", "合约内部编码", "合约代码", "收盘价", "主力标志"],
        )
        incoming = pd.DataFrame(
            [
                ["中证500主连", "financial", "2015-04-16", 2001, "IC1505", 7000.0, 1],
            ],
            columns=existing.columns,
        )

        with (
            patch.object(module, "fetch_futures_cache_rows", return_value=incoming) as fetch_mock,
            patch.object(module, "write_cache_csv"),
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

        fetch_mock.assert_called_once()
        self.assertEqual(fetch_mock.call_args.args[1], pd.Timestamp("2013-01-04"))
        self.assertEqual(fetch_mock.call_args.args[2], pd.Timestamp("2026-05-29"))
        self.assertIn("中证500主连", set(result["资产"]))
        self.assertFalse(stat.skipped)
        self.assertEqual(stat.query_start, "2013-01-04")

    def test_refresh_futures_quote_cache_initializes_empty_cache_and_writes_result(self):
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
        with self.assertRaises(SystemExit):
            module.parse_args(["--backup"])


if __name__ == "__main__":
    unittest.main()
