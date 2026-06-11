import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "策略复现与回测"
    / "策略测试代码"
    / "资产风险平价策略0.19_1（TL删除测试）.py"
)


def load_script_module(test_case: unittest.TestCase):
    test_case.assertTrue(SCRIPT_PATH.exists(), f"Missing v0.19_1 strategy script: {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location("strategy_v019_1_under_test", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load script module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["strategy_v019_1_under_test"] = module
    spec.loader.exec_module(module)
    return module


class StrategyV0191TLDeletionTests(unittest.TestCase):
    def test_strategy_constants_remove_tl_from_risk_parity_universe(self):
        module = load_script_module(self)

        self.assertEqual(module.VERSION, "0.19_1")
        self.assertEqual(module.INDEX_FUTURES, ["沪深300主连", "中证500主连"])
        self.assertEqual(module.RISK_PARITY_ASSET_CLASSES["债券"], ["10年国债主连"])
        self.assertEqual(module.TEST_OUTPUT_DIR.name, "TL删除测试_v0.19_1")
        self.assertEqual(module.TEST_OUTPUT_DIR.parent.name, "策略测试结果")

        risk_parity_assets = {
            asset for assets in module.RISK_PARITY_ASSET_CLASSES.values() for asset in assets
        }
        plot_assets = {
            asset for assets in module.PLOT_ASSET_CLASSES.values() for asset in assets
        }
        self.assertNotIn("30年国债主连", risk_parity_assets)
        self.assertNotIn("30年国债主连", plot_assets)
        self.assertIn("10年国债主连", risk_parity_assets)

    def test_margin_tables_remove_tl(self):
        module = load_script_module(self)

        self.assertNotIn("30年国债主连", module.MARGIN_RATIOS_EXCHANGE_MIN)
        self.assertNotIn("30年国债主连", module.MARGIN_RATIOS_BROKER)
        self.assertNotIn("30年国债主连", module.MARGIN_RATIOS)
        self.assertEqual(module.MARGIN_RATIOS["10年国债主连"], 0.025)

    def test_calculate_metrics_keeps_daily_win_rate(self):
        module = load_script_module(self)
        returns = pd.Series(
            [0.01, -0.02, 0.00, 0.03, -0.01],
            index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]),
        )

        metrics = module.calculate_metrics(returns)

        self.assertEqual(metrics["日度胜率"], "40.00%")
        self.assertNotIn("月度胜率", metrics)

    def test_allocate_index_futures_still_uses_if_and_ic_only(self):
        module = load_script_module(self)
        assets = ["沪深300主连", "中证500主连", "中证1000主连", "红利低波ETF"]
        listing_dates = {
            "沪深300主连": pd.Timestamp("2013-01-04"),
            "中证500主连": pd.Timestamp("2015-04-17"),
            "中证1000主连": pd.Timestamp("2022-07-25"),
            "红利低波ETF": pd.Timestamp("2018-01-01"),
        }

        target = module.allocate_index_futures(1.0, assets, listing_dates, pd.Timestamp("2026-01-02"))

        self.assertAlmostEqual(target["沪深300主连"], 0.15)
        self.assertAlmostEqual(target["中证500主连"], 0.15)
        self.assertAlmostEqual(target["中证1000主连"], 0.0)
        self.assertAlmostEqual(target.sum(), 0.30)


if __name__ == "__main__":
    unittest.main()
