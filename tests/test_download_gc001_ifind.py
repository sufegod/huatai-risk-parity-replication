import importlib.util
import json
import shutil
import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "数据" / "download_gc001_ifind.py"


def load_module():
    spec = importlib.util.spec_from_file_location("download_gc001_ifind", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DownloadGC001IfindTests(unittest.TestCase):
    def workspace_temp_dir(self, name: str) -> Path:
        root = PROJECT_ROOT / ".test_download_gc001_ifind"
        temp_dir = root / name
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        return temp_dir

    def test_parse_gc001_response_returns_standard_dataframe(self):
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
                            "data": [
                                ["2013-01-07", 3.553],
                                ["2013-01-04", 5.256],
                                ["2013-01-07", 3.554],
                            ],
                        }
                    }
                ]
            }
        }
        response = {"result": {"content": [{"text": json.dumps(inner, ensure_ascii=False)}]}}

        df = module.parse_gc001_response(response)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(list(df.columns), ["date", "gc001_weighted_average_rate_pct"])
        self.assertEqual(df.to_dict("records"), [
            {"date": "2013-01-04", "gc001_weighted_average_rate_pct": 5.256},
            {"date": "2013-01-07", "gc001_weighted_average_rate_pct": 3.554},
        ])

    def test_write_csv_uses_dataframe_without_index(self):
        module = load_module()
        temp_dir = self.workspace_temp_dir("write_csv")
        output_path = temp_dir / "gc001.csv"
        df = pd.DataFrame(
            {
                "date": ["2013-01-04", "2013-01-07"],
                "gc001_weighted_average_rate_pct": [5.256, 3.553],
            }
        )

        module.write_csv(df, output_path, backup=False, dry_run=False)

        self.assertEqual(
            output_path.read_text(encoding="utf-8").splitlines(),
            [
                "date,gc001_weighted_average_rate_pct",
                "2013-01-04,5.256",
                "2013-01-07,3.553",
            ],
        )


if __name__ == "__main__":
    unittest.main()
