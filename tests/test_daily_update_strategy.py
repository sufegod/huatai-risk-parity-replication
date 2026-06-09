import importlib.util
import subprocess
import sys
import unittest
from argparse import Namespace
from pathlib import Path
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
    "daily_update_strategy_under_test",
    PROJECT_ROOT / "策略复现与回测" / "每日更新策略" / "daily_update_strategy.py",
)


class DailyUpdateStrategyTests(unittest.TestCase):
    def test_daily_update_uses_v019_strategy_script_by_default(self):
        self.assertEqual(module.STRATEGY_SCRIPT.name, "资产风险平价策略0.19（IC替换IM+日频胜率）.py")

        strategy_module = module.load_strategy_module()

        self.assertEqual(strategy_module.VERSION, "0.19")
        self.assertEqual(strategy_module.INDEX_FUTURES, ["沪深300主连", "中证500主连"])

    def test_daily_weight_columns_append_margin_usage(self):
        columns = module.daily_weight_columns(["沪深300主连", "中证500主连", "10年国债主连"])

        self.assertEqual(
            columns,
            [
                "date",
                "策略名称",
                "股指期货信号",
                "股指期货仓位",
                "沪深300主连",
                "中证500主连",
                "10年国债主连",
                "资金占用比例",
            ],
        )

    def test_get_observation_dates_uses_all_trade_dates_for_daily_rebalance(self):
        dates = pd.to_datetime(["2026-05-18", "2026-05-19", "2026-05-21"])

        result = module.get_observation_dates(pd.DatetimeIndex(dates))

        self.assertEqual(result.tolist(), list(dates))

    def test_run_data_update_passes_end_date_to_update_script(self):
        calls = []

        def fake_runner(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return subprocess.CompletedProcess(cmd, 0)

        module.run_data_update(
            data_end_date="2026-05-28",
            runner=fake_runner,
            python_executable="python",
        )

        self.assertEqual(len(calls), 1)
        cmd, kwargs = calls[0]
        self.assertEqual(cmd[0], "python")
        self.assertEqual(Path(cmd[1]).name, "日度收益数据更新.py")
        self.assertEqual(cmd[2:], ["--end-date", "2026-05-28"])
        self.assertNotIn("--backup", cmd)
        self.assertEqual(Path(kwargs["cwd"]), PROJECT_ROOT)
        self.assertTrue(kwargs["check"])

    def test_parse_args_no_longer_accepts_data_backup(self):
        with self.assertRaises(SystemExit):
            module.parse_args(["--data-backup"])

    def test_run_stops_when_data_update_fails(self):
        def failing_runner(cmd, **kwargs):
            raise subprocess.CalledProcessError(1, cmd)

        with self.assertRaises(subprocess.CalledProcessError):
            module.run(["--data-end-date", "2026-05-28"], runner=failing_runner)

    def test_maybe_run_data_update_skips_runner_when_requested(self):
        args = Namespace(skip_data_update=True, data_end_date=None)

        def unexpected_runner(cmd, **kwargs):
            raise AssertionError("runner should not be called")

        result = module.maybe_run_data_update(args, runner=unexpected_runner)

        self.assertEqual(result, "skipped")

    def test_output_paths_use_classified_date_suffix_without_latest_word(self):
        output_dir = Path("out")

        paths = module.output_paths(output_dir, pd.Timestamp("2026-05-28"))

        expected = {
            "positions": output_dir / "仓位" / "仓位_2026-05-28.csv",
            "nav": output_dir / "净值" / "策略每日净值走势_2026-05-28.csv",
            "metrics": output_dir / "指标" / "年度及全局回测指标_2026-05-28.csv",
            "daily_weights": output_dir / "仓位明细" / "策略日度仓位明细_2026-05-28.csv",
            "chart": output_dir / "图表" / "回测图表_2026-05-28.png",
            "report": output_dir / "报告" / "回测报告_2026-05-28.md",
        }
        self.assertEqual(paths, expected)
        for path in paths.values():
            self.assertNotIn("最新", path.name)

    def test_latest_daily_return_formats_previous_trade_day_change(self):
        df_navs = pd.DataFrame(
            {"风险平价策略": [1.0, 1.02]},
            index=pd.to_datetime(["2026-05-27", "2026-05-28"]),
        )

        result = module._latest_daily_return(df_navs, "风险平价策略")

        self.assertEqual(result, "2.00%")

    def test_latest_daily_return_uses_zero_when_nav_history_insufficient(self):
        df_navs = pd.DataFrame(
            {"风险平价策略": [1.0]},
            index=pd.to_datetime(["2026-05-28"]),
        )

        result = module._latest_daily_return(df_navs, "风险平价策略")

        self.assertEqual(result, "0.00%")

    def test_write_report_creates_classified_full_report_outputs(self):
        target_report = module.TargetReport(
            as_of_date=pd.Timestamp("2026-05-28"),
            observation_date=pd.Timestamp("2026-05-22"),
            is_new_observation=False,
            raw_signal=0.5,
            index_weight=0.15,
            target_weights=pd.Series({"沪深300主连": 0.075, "10年国债主连": 0.925}),
            margin_ratios=pd.Series({"沪深300主连": 0.15, "10年国债主连": 0.03}),
        )
        backtest_result = module.BacktestResult(
            as_of_date=pd.Timestamp("2026-05-28"),
            first_date=pd.Timestamp("2018-01-08"),
            assets=["沪深300主连", "10年国债主连"],
            df_navs=pd.DataFrame(
                {"风险平价策略": [1.0, 1.02]},
                index=pd.to_datetime(["2026-05-27", "2026-05-28"]),
            ),
            df_metrics=pd.DataFrame(
                [
                    {
                        "回测区间": "全局 (Total)",
                        "组合/资产": "风险平价策略",
                        "年化收益": "10.00%",
                        "年化波动": "5.00%",
                        "夏普比率": "2.00",
                        "最大回撤": "-3.00%",
                        "日度胜率": "40.00%",
                        "平均资金占用": "10.00%",
                    }
                ]
            ),
            df_daily_weights=pd.DataFrame(
                {
                    "date": [pd.Timestamp("2026-05-22")],
                    "策略名称": ["风险平价策略"],
                    "股指期货信号": [0.5],
                    "股指期货仓位": [0.15],
                    "沪深300主连": [0.075],
                    "10年国债主连": [0.925],
                    "资金占用比例": [0.03825],
                }
            ),
        )

        output_dir = Path("out")
        with (
            patch.object(module.Path, "mkdir") as mkdir_mock,
            patch.object(pd.DataFrame, "to_csv") as to_csv_mock,
            patch.object(module.Path, "write_bytes") as write_bytes_mock,
            patch.object(module.Path, "write_text") as write_text_mock,
        ):
            paths = module.write_report(target_report, backtest_result, output_dir, render_chart=False)

            self.assertEqual(paths["positions"].parent, output_dir / "仓位")
            self.assertEqual(paths["nav"].parent, output_dir / "净值")
            self.assertEqual(paths["metrics"].parent, output_dir / "指标")
            self.assertEqual(paths["daily_weights"].parent, output_dir / "仓位明细")
            self.assertEqual(paths["chart"].parent, output_dir / "图表")
            self.assertEqual(paths["report"].parent, output_dir / "报告")
            self.assertGreaterEqual(mkdir_mock.call_count, 6)
            self.assertEqual(to_csv_mock.call_count, 4)
            write_bytes_mock.assert_called_once_with(b"")

            report_text = write_text_mock.call_args.args[0]
            self.assertIn("全局核心指标", report_text)
            self.assertIn("回测区间", report_text)
            self.assertIn("日报观察日：`2026-05-28`", report_text)
            self.assertIn("仓位来源观察日：`2026-05-22`", report_text)
            self.assertIn("当前有效仓位", report_text)
            self.assertIn("| 日度胜率 | 40.00% |", report_text)
            self.assertNotIn("月度胜率", report_text)
            self.assertIn("| 上一交易日 | 2.00% |", report_text)
            self.assertIn("回测图表_2026-05-28.png", report_text)

    def test_position_dataframe_labels_source_observation_date(self):
        target_report = module.TargetReport(
            as_of_date=pd.Timestamp("2026-06-03"),
            observation_date=pd.Timestamp("2026-05-29"),
            is_new_observation=False,
            raw_signal=0.5,
            index_weight=0.15,
            target_weights=pd.Series({"沪深300主连": 0.075}),
            margin_ratios=pd.Series({"沪深300主连": 0.15}),
        )

        positions = module.build_position_dataframe(target_report)

        self.assertIn("仓位来源观察日", positions.columns)
        self.assertNotIn("观察日", positions.columns)
        self.assertEqual(positions.loc[0, "日期"], "2026-06-03")
        self.assertEqual(positions.loc[0, "仓位来源观察日"], "2026-05-29")

    def test_select_observation_date_uses_latest_trade_date_for_daily_rebalance(self):
        dates = pd.to_datetime(
            [
                "2026-05-15",
                "2026-05-18",
                "2026-05-19",
                "2026-05-20",
                "2026-05-21",
            ]
        )

        result = module.select_observation_date(pd.DatetimeIndex(dates), pd.Timestamp("2026-05-21"), False)

        self.assertEqual(result.observation_date, pd.Timestamp("2026-05-21"))
        self.assertTrue(result.is_new_observation)

    def test_select_observation_date_uses_as_of_date_when_force_observation(self):
        dates = pd.to_datetime(["2026-05-15", "2026-05-18", "2026-05-21"])

        result = module.select_observation_date(pd.DatetimeIndex(dates), pd.Timestamp("2026-05-21"), True)

        self.assertEqual(result.observation_date, pd.Timestamp("2026-05-21"))
        self.assertTrue(result.is_new_observation)


if __name__ == "__main__":
    unittest.main()
