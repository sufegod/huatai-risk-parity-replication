import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OLD_PATH_PATTERNS = (
    "回测指标" + "/参数测试",
    "回测指标" + "\\参数测试",
    "回测指标" + "' / '参数测试",
    "回测指标" + '" / "参数测试',
)
SCAN_PATHS = (
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "策略复现与回测" / "策略测试代码",
    PROJECT_ROOT / "策略复现与回测" / "策略版本说明",
    PROJECT_ROOT / "策略复现与回测" / "回测结果对比说明",
    PROJECT_ROOT / "策略复现与回测" / "策略测试结果",
)


class StrategyTestResultsPathTests(unittest.TestCase):
    def test_legacy_parameter_test_path_is_not_referenced(self):
        offenders = []
        for scan_path in SCAN_PATHS:
            if not scan_path.exists():
                continue
            paths = [scan_path] if scan_path.is_file() else scan_path.rglob("*")
            for path in paths:
                if not path.is_file() or path.suffix.lower() not in {".py", ".md"}:
                    continue
                text = path.read_text(encoding="utf-8-sig")
                if any(pattern in text for pattern in OLD_PATH_PATTERNS):
                    offenders.append(str(path.relative_to(PROJECT_ROOT)))

        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
