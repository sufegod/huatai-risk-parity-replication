import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "货币信用周期策略" / "货币信用周期策略.py"


def load_strategy_module():
    spec = importlib.util.spec_from_file_location("monetary_credit_cycle_strategy", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load strategy module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["monetary_credit_cycle_strategy"] = module
    spec.loader.exec_module(module)
    return module


class MonetaryCreditCycleStrategyTests(unittest.TestCase):
    def test_cycle_map_starts_from_2015_01_and_contains_four_cycles(self):
        module = load_strategy_module()

        cycle_map = module.load_monetary_credit_cycle_map(module.FILE_PATH_CYCLE)

        self.assertEqual(str(cycle_map.index.min()), "2015-01")
        self.assertEqual(cycle_map.loc[pd.Period("2015-01", freq="M")], "宽货币紧信用")
        self.assertEqual(cycle_map.loc[pd.Period("2015-07", freq="M")], "宽货币宽信用")
        self.assertEqual(set(cycle_map.unique()), {"宽货币紧信用", "宽货币宽信用", "紧货币紧信用", "紧货币宽信用"})

    def test_month_start_rebalance_dates_use_first_trade_day_in_cycle_months(self):
        module = load_strategy_module()
        trade_index = pd.DatetimeIndex(
            pd.to_datetime(["2014-12-31", "2015-01-05", "2015-01-06", "2015-02-02", "2015-02-27", "2015-03-02"])
        )
        cycle_months = pd.PeriodIndex(["2015-01", "2015-02"], freq="M")

        dates = module.get_month_start_rebalance_dates(trade_index, cycle_months)

        self.assertEqual(dates.tolist(), [pd.Timestamp("2015-01-05"), pd.Timestamp("2015-02-02")])

    def test_tight_money_wide_credit_uses_full_asset_pool(self):
        module = load_strategy_module()
        full_assets = ["沪深300主连", "中证500主连", "10年国债主连"]
        pool_df = pd.DataFrame(
            [
                {"周期划分": "宽货币紧信用", "资产池": "10年国债主连、沪深300主连"},
                {"周期划分": "宽货币宽信用", "资产池": "中证500主连"},
                {"周期划分": "紧货币紧信用", "资产池": "10年国债主连"},
            ]
        )

        pool_map = module.parse_asset_pool_table(pool_df, full_assets)

        self.assertEqual(pool_map["宽货币紧信用"], ["10年国债主连", "沪深300主连"])
        self.assertEqual(pool_map["宽货币宽信用"], ["中证500主连"])
        self.assertEqual(pool_map["紧货币紧信用"], ["10年国债主连"])
        self.assertEqual(pool_map["紧货币宽信用"], full_assets)

    def test_select_eligible_assets_filters_current_cycle_pool_and_listing_dates(self):
        module = load_strategy_module()
        pool_assets = ["沪深300主连", "中证500主连", "10年国债主连"]
        listing_dates = {
            "沪深300主连": pd.Timestamp("2010-01-01"),
            "中证500主连": pd.Timestamp("2020-01-01"),
            "10年国债主连": pd.Timestamp("2014-01-01"),
        }

        eligible = module.select_eligible_assets(
            pool_assets,
            listing_dates,
            pd.Timestamp("2015-01-05"),
            available_assets=["沪深300主连", "中证500主连", "10年国债主连"],
        )

        self.assertEqual(eligible, ["沪深300主连", "10年国债主连"])

    def test_last_cycle_holding_period_ends_at_last_trade_day_in_same_month(self):
        module = load_strategy_module()
        trade_index = pd.DatetimeIndex(
            pd.to_datetime(["2025-11-03", "2025-11-28", "2025-12-01", "2025-12-31", "2026-01-02"])
        )
        rebalance_dates = pd.DatetimeIndex(pd.to_datetime(["2025-11-03", "2025-12-01"]))

        first_end = module.get_holding_period_end_date(trade_index, rebalance_dates, 0)
        last_end = module.get_holding_period_end_date(trade_index, rebalance_dates, 1)

        self.assertEqual(first_end, pd.Timestamp("2025-11-28"))
        self.assertEqual(last_end, pd.Timestamp("2025-12-31"))

    def test_nav_frame_stops_at_last_active_holding_date(self):
        module = load_strategy_module()
        returns = pd.Series(
            [0.01, 0.02, 0.03, 0.04],
            index=pd.DatetimeIndex(pd.to_datetime(["2025-12-01", "2025-12-31", "2026-01-02", "2026-01-05"])),
        )

        nav = module.build_nav_frame(returns, pd.Timestamp("2025-12-01"), pd.Timestamp("2025-12-31"))

        self.assertEqual(nav.index.tolist(), [pd.Timestamp("2025-12-01"), pd.Timestamp("2025-12-31")])
        self.assertEqual(nav.index.name, "date")

    def test_strategy_removes_index_signal_interface(self):
        module = load_strategy_module()

        self.assertFalse(hasattr(module, "FILE_PATH_INDEX_SIGNAL"))
        self.assertFalse(hasattr(module, "load_index_signal"))
        self.assertFalse(hasattr(module, "allocate_index_futures"))


if __name__ == "__main__":
    unittest.main()
