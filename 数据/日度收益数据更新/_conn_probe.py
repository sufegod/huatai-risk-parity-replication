import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "数据" / "日度收益数据更新"))
import importlib.util
spec = importlib.util.spec_from_file_location("du", str(ROOT / "数据" / "日度收益数据更新" / "日度收益数据更新.py"))
du = importlib.util.module_from_spec(spec)
sys.modules["du"] = du
spec.loader.exec_module(du)

du.load_project_env()
args = argparse.Namespace(jydb_server=None, jydb_database=None, jydb_uid=None, jydb_password=None, jydb_driver=None)
print("JYDB_SERVER =", os.environ.get("JYDB_SERVER"), "UID =", os.environ.get("JYDB_UID"))
print("JYDB_PWD present =", bool(os.environ.get("JYDB_PWD")))
try:
    conn = du.connect_jydb(args)
    print("CONNECT = OK")
    end = du.fetch_jydb_latest_end_date(conn)
    print("JYDB 各表共同最新交易日 =", end)
    for asset in du.FUTURES_ASSETS[:3]:
        print(f"  {asset.name} 最新 =", du.fetch_futures_latest_date(conn, asset))
    print("  红利低波ETF 最新 =", du.fetch_etf_latest_date(conn))
    print("  GC001 最新 =", du.fetch_gc001_latest_date(conn))
    conn.close()
except Exception as e:
    print("CONNECT FAILED:", type(e).__name__, str(e)[:300])
