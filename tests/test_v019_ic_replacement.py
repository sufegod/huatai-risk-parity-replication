import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "策略复现与回测"
    / "策略代码"
    / "资产风险平价策略0.19（IC替换IM+日频胜率）.py"
)
VERSION_NOTE_PATH = PROJECT_ROOT / "策略复现与回测" / "策略版本说明" / "v0.19策略版本说明.md"


def load_script_module():
    spec = importlib.util.spec_from_file_location("strategy_v019_under_test", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load script module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["strategy_v019_under_test"] = module
    spec.loader.exec_module(module)
    return module


class StrategyV019ICReplacementTests(unittest.TestCase):
    def test_strategy_constants_replace_im_with_ic_in_official_version(self):
        self.assertTrue(SCRIPT_PATH.exists(), f"Missing v0.19 strategy script: {SCRIPT_PATH}")
        module = load_script_module()

        self.assertEqual(module.VERSION, "0.19")
        self.assertEqual(module.INDEX_FUTURES, ["沪深300主连", "中证500主连"])
        self.assertNotIn("中证1000主连", module.INDEX_FUTURES)
        self.assertEqual(module.MARGIN_RATIOS_EXCHANGE_MIN["中证500主连"], 0.08)
        self.assertEqual(module.MARGIN_RATIOS_BROKER["中证500主连"], 0.14)
        self.assertEqual(module.MARGIN_RATIOS["中证500主连"], 0.14)
        self.assertFalse(hasattr(module, "TEST_OUTPUT_DIR"))
        self.assertEqual(module.NAV_DIR.name, "净值")
        self.assertEqual(module.PERFORMANCE_DIR.name, "指标")
        self.assertEqual(module.WEIGHTS_DIR.name, "仓位明细")
        self.assertEqual(module.CHART_DIR.name, "回测图表")

    def test_calculate_metrics_reports_daily_win_rate(self):
        module = load_script_module()
        returns = pd.Series(
            [0.01, -0.02, 0.00, 0.03, -0.01],
            index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]),
        )

        metrics = module.calculate_metrics(returns)

        self.assertEqual(metrics["日度胜率"], "40.00%")
        self.assertNotIn("月度胜率", metrics)

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

    def test_version_note_documents_official_ic_replacement(self):
        self.assertTrue(VERSION_NOTE_PATH.exists(), f"Missing v0.19 version note: {VERSION_NOTE_PATH}")
        text = VERSION_NOTE_PATH.read_text(encoding="utf-8-sig")

        self.assertIn("IC替换IM", text)
        self.assertIn("日频胜率", text)
        self.assertIn("中证500主连", text)
        self.assertIn("日频调仓", text)
        self.assertIn("资产风险平价策略0.19（IC替换IM+日频胜率）.py", text)
        self.assertIn("策略每日净值走势_v0.19.csv", text)


if __name__ == "__main__":
    unittest.main()
