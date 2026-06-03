from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_ROOT / "源码"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from risk_parity.strategy_v016 import *  # noqa: F401,F403
from risk_parity.strategy_v016 import main


if __name__ == "__main__":
    main()

