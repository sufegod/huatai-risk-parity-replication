import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "源码"
sys.path.insert(0, str(SOURCE_DIR))

from risk_parity import rebalance_frequency_experiment as module


class RebalanceFrequencyExperimentTests(unittest.TestCase):
    def test_daily_observation_dates_use_every_trade_date(self):
        dates = pd.to_datetime(["2026-05-18", "2026-05-19", "2026-05-21"])

        result = module.get_observation_dates(pd.DatetimeIndex(dates), "daily")

        self.assertEqual(list(result), list(dates))

    def test_weekly_observation_dates_use_last_actual_trade_date_in_each_week(self):
        dates = pd.to_datetime(
            ["2026-05-18", "2026-05-19", "2026-05-22", "2026-05-25", "2026-05-28"]
        )

        result = module.get_observation_dates(pd.DatetimeIndex(dates), "weekly")

        self.assertEqual(list(result), list(pd.to_datetime(["2026-05-22", "2026-05-28"])))

    def test_evaluate_optimization_prefers_sharpe_and_drawdown_over_return_only(self):
        weekly = {
            "期末净值": 2.0,
            "夏普比率": 1.6,
            "最大回撤": "-7.00%",
            "年化收益": "10.00%",
            "交易成本合计": "0.0100",
        }
        daily = {
            "期末净值": 2.1,
            "夏普比率": 1.7,
            "最大回撤": "-9.00%",
            "年化收益": "11.00%",
            "交易成本合计": "0.0200",
        }

        result = module.evaluate_optimization(weekly, daily)

        self.assertIn("不判定为明确优化", result)
        self.assertIn("最大回撤扩大", result)
        self.assertIn("交易成本显著增加", result)

    def test_evaluate_optimization_marks_clear_improvement_when_sharpe_and_drawdown_improve(self):
        weekly = {"期末净值": 2.0, "夏普比率": 1.6, "最大回撤": "-7.00%", "年化收益": "10.00%"}
        daily = {"期末净值": 2.1, "夏普比率": 1.7, "最大回撤": "-6.50%", "年化收益": "11.00%"}

        result = module.evaluate_optimization(weekly, daily)

        self.assertIn("判定为优化", result)

    def test_dataframe_to_markdown_does_not_require_optional_tabulate_dependency(self):
        df = pd.DataFrame([{"方案": "周频基准", "夏普比率": "1.74"}])

        result = module.dataframe_to_markdown(df)

        self.assertEqual(result.splitlines()[0], "| 方案 | 夏普比率 |")
        self.assertIn("| 周频基准 | 1.74 |", result)


if __name__ == "__main__":
    unittest.main()
