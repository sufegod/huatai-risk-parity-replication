from pathlib import Path


SOURCE_PACKAGE = Path(__file__).resolve().parents[1] / "源码" / "risk_parity"
__path__.append(str(SOURCE_PACKAGE))

