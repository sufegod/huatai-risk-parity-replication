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
    / "资产风险平价策略0.18_1（IC替换IM测试）.py"
)


def load_script_module():
    spec = importlib.util.spec_from_file_location("strategy_v018_1_under_test", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load script module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["strategy_v018_1_under_test"] = module
    spec.loader.exec_module(module)
    return module


class StrategyV0181ICReplacementTests(unittest.TestCase):
    def test_strategy_constants_replace_im_with_ic(self):
        self.assertTrue(SCRIPT_PATH.exists(), f"Missing v0.18_1 strategy script: {SCRIPT_PATH}")
        module = load_script_module()

        self.assertEqual(module.VERSION, "0.18_1")
        self.assertEqual(module.INDEX_FUTURES, ["沪深300主连", "中证500主连"])
        self.assertNotIn("中证1000主连", module.INDEX_FUTURES)
        self.assertEqual(module.MARGIN_RATIOS_EXCHANGE_MIN["中证500主连"], 0.08)
        self.assertEqual(module.MARGIN_RATIOS_BROKER["中证500主连"], 0.14)
        self.assertEqual(module.MARGIN_RATIOS["中证500主连"], 0.14)
        self.assertEqual(module.TEST_OUTPUT_DIR.name, "IC替换IM测试_v0.18_1")

    def test_allocate_index_futures_uses_if_and_ic_only(self):
        module = load_script_module()
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
