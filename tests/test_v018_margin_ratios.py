import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "策略复现与回测"
    / "策略代码"
    / "资产风险平价策略0.18（保证金修改）.py"
)
DOC_PATH = PROJECT_ROOT / "策略复现与回测" / "策略版本说明" / "v0.18策略版本说明.md"


EXCHANGE_MIN_RATIOS = {
    "沪深300主连": 0.08,
    "中证1000主连": 0.08,
    "红利低波ETF": 1.00,
    "10年国债主连": 0.02,
    "30年国债主连": 0.035,
    "沪铜主连": 0.05,
    "沪铝主连": 0.05,
    "PTA主连": 0.05,
    "原油主连": 0.05,
    "豆粕主连": 0.05,
    "沪金主连": 0.04,
}

BROKER_RATIOS = {
    "沪深300主连": 0.14,
    "中证1000主连": 0.14,
    "红利低波ETF": 1.00,
    "10年国债主连": 0.025,
    "30年国债主连": 0.05,
    "沪铜主连": 0.16,
    "沪铝主连": 0.16,
    "PTA主连": 0.17,
    "原油主连": 0.32,
    "豆粕主连": 0.13,
    "沪金主连": 0.28,
}


def load_script_module():
    spec = importlib.util.spec_from_file_location("strategy_v018_under_test", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load script module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["strategy_v018_under_test"] = module
    spec.loader.exec_module(module)
    return module


class StrategyV018MarginRatioTests(unittest.TestCase):
    def test_margin_ratio_sets_are_available_and_default_to_broker(self):
        self.assertTrue(SCRIPT_PATH.exists(), f"Missing v0.18 strategy script: {SCRIPT_PATH}")
        module = load_script_module()

        self.assertEqual(module.VERSION, "0.18")
        self.assertEqual(module.MARGIN_RATIOS_EXCHANGE_MIN, EXCHANGE_MIN_RATIOS)
        self.assertEqual(module.MARGIN_RATIOS_BROKER, BROKER_RATIOS)
        self.assertEqual(module.MARGIN_RATIOS, BROKER_RATIOS)

    def test_version_note_documents_broker_margin_assumption(self):
        self.assertTrue(DOC_PATH.exists(), f"Missing v0.18 version note: {DOC_PATH}")
        note = DOC_PATH.read_text(encoding="utf-8")

        self.assertIn("国泰君安期货当前的实际保证金", note)
        self.assertIn("过去的实际情况可能存在差异", note)


if __name__ == "__main__":
    unittest.main()
