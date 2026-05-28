import importlib.util
import json
from pathlib import Path
import sys
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "数据" / "JYDB数据替换" / "update_daily_returns.py"


def load_module():
    spec = importlib.util.spec_from_file_location("update_daily_returns", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class UpdateDailyReturnsTests(unittest.TestCase):
    def test_prune_output_columns_removes_unused_assets_and_preserves_order(self):
        module = load_module()
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

    def test_compute_etf_return_uses_prev_close_on_adjustment_day(self):
        module = load_module()
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

    def test_parse_ifind_edb_response_unwraps_nested_json_text(self):
        module = load_module()
        inner = {
            "data": {
                "datas": [
                    {
                        "data": {
                            "columns": ["日期", "GC001(加权平均)"],
                            "attrs": {
                                "GC001(加权平均)": {
                                    "unit": "%",
                                    "dtype": "double",
                                    "index_id": "L004369613",
                                }
                            },
                            "data": [["2013-01-07", 3.553], ["2013-01-04", 5.256]],
                        }
                    }
                ]
            }
        }
        outer = {"result": {"content": [{"text": json.dumps(inner, ensure_ascii=False)}]}}

        result = module.parse_ifind_edb_response(outer)

        self.assertEqual(result.name, "一天期国债逆回购")
        self.assertEqual(list(result.index.strftime("%Y-%m-%d")), ["2013-01-04", "2013-01-07"])
        self.assertEqual(result.loc[pd.Timestamp("2013-01-04")], 5.256)


if __name__ == "__main__":
    unittest.main()
