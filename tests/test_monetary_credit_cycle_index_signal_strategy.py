import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "货币信用周期策略" / "货币信用周期策略0.2（股指信号）.py"


def load_strategy_module():
    spec = importlib.util.spec_from_file_location("monetary_credit_cycle_index_signal_strategy", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load strategy module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["monetary_credit_cycle_index_signal_strategy"] = module
    spec.loader.exec_module(module)
    return module


class MonetaryCreditCycleIndexSignalStrategyTests(unittest.TestCase):
    def test_strategy_identity_uses_v0_2_index_signal_name(self):
        module = load_strategy_module()

        self.assertEqual(module.VERSION, "monetary_credit_cycle_v0_2_index_signal")
        self.assertEqual(module.STRATEGY_NAME, "货币信用周期策略0.2（股指信号）")
        self.assertEqual(module.RESULT_DIR.name, "回测结果_0.2_股指信号")

    def test_load_index_signal_starts_from_2018_01_02(self):
        module = load_strategy_module()

        signal = module.load_index_signal(module.FILE_PATH_INDEX_SIGNAL)

        self.assertEqual(signal.first_valid_index(), pd.Timestamp("2018-01-02"))
        self.assertIn(1.0, set(signal.dropna().unique()))

    def test_normalize_index_signal_clips_to_long_only_range(self):
        module = load_strategy_module()

        cases = [
            (np.nan, 0.0),
            (-1.0, 0.0),
            (0.0, 0.0),
            (0.5, 0.5),
            (0.7, 0.7),
            (1.0, 1.0),
            (2.0, 1.0),
        ]

        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(module.normalize_index_signal(raw), expected)

    def test_allocate_index_futures_allocates_signal_to_three_index_futures(self):
        module = load_strategy_module()
        assets = ["沪深300主连", "中证1000主连", "中证500主连", "10年国债主连"]
        listing_dates = {asset: pd.Timestamp("2010-01-01") for asset in assets}

        target = module.allocate_index_futures(1.0, assets, listing_dates, pd.Timestamp("2020-01-02"))

        self.assertAlmostEqual(target[["沪深300主连", "中证1000主连", "中证500主连"]].sum(), 0.30)
        self.assertAlmostEqual(target["沪深300主连"], 0.10)
        self.assertAlmostEqual(target["中证1000主连"], 0.10)
        self.assertAlmostEqual(target["中证500主连"], 0.10)
        self.assertAlmostEqual(target["10年国债主连"], 0.0)

    def test_index_signal_independent_of_cycle_asset_pool(self):
        module = load_strategy_module()
        full_assets = ["沪深300主连", "中证1000主连", "中证500主连", "10年国债主连"]
        cycle_pool_without_index = ["10年国债主连"]
        listing_dates = {asset: pd.Timestamp("2010-01-01") for asset in full_assets}

        risk_parity_pool = module.get_risk_parity_pool_assets(cycle_pool_without_index)
        index_target = module.allocate_index_futures(1.0, full_assets, listing_dates, pd.Timestamp("2020-01-02"))

        self.assertEqual(risk_parity_pool, ["10年国债主连"])
        self.assertAlmostEqual(index_target[module.INDEX_FUTURES].sum(), 0.30)

    def test_combine_target_weights_scales_non_index_pool_by_remaining_weight(self):
        module = load_strategy_module()
        full_assets = ["沪深300主连", "中证1000主连", "中证500主连", "10年国债主连", "沪金主连"]
        index_target = pd.Series(
            {
                "沪深300主连": 0.10,
                "中证1000主连": 0.10,
                "中证500主连": 0.10,
                "10年国债主连": 0.0,
                "沪金主连": 0.0,
            }
        )

        target = module.combine_index_and_risk_parity_weights(
            index_target=index_target,
            risk_parity_assets=["10年国债主连", "沪金主连"],
            risk_parity_weights=np.array([0.25, 0.75]),
            full_asset_pool=full_assets,
        )

        self.assertAlmostEqual(target[module.INDEX_FUTURES].sum(), 0.30)
        self.assertAlmostEqual(target[["10年国债主连", "沪金主连"]].sum(), 0.70)
        self.assertAlmostEqual(target["10年国债主连"], 0.175)
        self.assertAlmostEqual(target["沪金主连"], 0.525)
        self.assertTrue(math.isclose(target.sum(), 1.0))

    def test_weight_record_contains_cycle_pool_signal_and_margin_fields(self):
        module = load_strategy_module()
        full_assets = ["沪深300主连", "中证1000主连", "中证500主连", "10年国债主连"]
        target = pd.Series({"沪深300主连": 0.1, "中证1000主连": 0.1, "中证500主连": 0.1, "10年国债主连": 0.7})

        record = module.build_weight_record(
            rebalance_date=pd.Timestamp("2020-01-02"),
            cycle="宽货币宽信用",
            pool_assets=["沪深300主连", "10年国债主连"],
            risk_parity_assets=["10年国债主连"],
            raw_signal=1.0,
            index_weight=0.30,
            target=target,
            full_asset_pool=full_assets,
        )

        expected_cols = {
            "date",
            "策略名称",
            "周期划分",
            "资产池",
            "风险平价资产池",
            "风险平价入选资产数",
            "股指期货信号",
            "股指期货仓位",
            "资金占用比例",
        }
        self.assertTrue(expected_cols.issubset(record.keys()))
        self.assertEqual(record["策略名称"], module.STRATEGY_NAME)
        self.assertEqual(record["股指期货仓位"], 0.30)


if __name__ == "__main__":
    unittest.main()
