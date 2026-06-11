import importlib.util
from pathlib import Path
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "货币信用周期策略" / "货币信用象限资产表现分析.py"


def load_analysis_module():
    spec = importlib.util.spec_from_file_location("monetary_credit_analysis", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MonetaryCreditAssetPoolTest(unittest.TestCase):
    def test_classifies_asset_cycles_by_obvious_underperformance(self):
        module = load_analysis_module()
        description = pd.DataFrame(
            [
                {"资产": "资产A", "周期划分": "宽货币紧信用", "年化收益(%)": 20.0, "月均收益(%)": 1.5, "胜率(%)": 60.0},
                {"资产": "资产A", "周期划分": "宽货币宽信用", "年化收益(%)": 8.0, "月均收益(%)": 0.7, "胜率(%)": 55.0},
                {"资产": "资产A", "周期划分": "紧货币紧信用", "年化收益(%)": 5.0, "月均收益(%)": 0.4, "胜率(%)": 50.0},
                {"资产": "资产B", "周期划分": "宽货币紧信用", "年化收益(%)": 4.0, "月均收益(%)": 0.3, "胜率(%)": 50.0},
                {"资产": "资产B", "周期划分": "宽货币宽信用", "年化收益(%)": 2.0, "月均收益(%)": 0.2, "胜率(%)": 48.0},
                {"资产": "资产B", "周期划分": "紧货币紧信用", "年化收益(%)": -3.0, "月均收益(%)": -0.2, "胜率(%)": 45.0},
                {"资产": "资产C", "周期划分": "宽货币紧信用", "年化收益(%)": 15.0, "月均收益(%)": 1.1, "胜率(%)": 62.0},
                {"资产": "资产C", "周期划分": "宽货币宽信用", "年化收益(%)": 14.0, "月均收益(%)": 1.0, "胜率(%)": 61.0},
                {"资产": "资产C", "周期划分": "紧货币紧信用", "年化收益(%)": 0.0, "月均收益(%)": 0.0, "胜率(%)": 40.0},
            ]
        )

        detail, pools = module.classify_asset_cycle_pools(description, threshold_pp=10.0)

        removed = {
            (row["资产"], row["周期划分"])
            for row in detail.loc[detail["是否剔除"], ["资产", "周期划分"]].to_dict("records")
        }
        self.assertEqual(
            removed,
            {
                ("资产A", "宽货币宽信用"),
                ("资产A", "紧货币紧信用"),
                ("资产C", "紧货币紧信用"),
            },
        )

        pool_map = dict(zip(pools["周期划分"], pools["资产池"]))
        self.assertEqual(pool_map["宽货币紧信用"], "资产A、资产B、资产C")
        self.assertEqual(pool_map["宽货币宽信用"], "资产B、资产C")
        self.assertEqual(pool_map["紧货币紧信用"], "资产B")


if __name__ == "__main__":
    unittest.main()
