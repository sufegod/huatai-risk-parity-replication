from pathlib import Path
import sys


COURSE_DIR = Path(__file__).resolve().parent
if str(COURSE_DIR) not in sys.path:
    sys.path.insert(0, str(COURSE_DIR))

from src.experiment import generate_all_outputs


if __name__ == "__main__":
    generate_all_outputs(COURSE_DIR / "config.json")

