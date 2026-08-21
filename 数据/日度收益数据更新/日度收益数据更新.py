from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _safe_replace(tmp: Path, target: Path) -> None:
    """os.replace with PermissionError fallback for locked files on Windows."""
    try:
        os.replace(tmp, target)
    except PermissionError:
        print(f"警告: 无法替换 {target.name}（文件被其他程序打开），正在直接覆盖写入...")
        # 直接写入目标文件
        import shutil
        shutil.copy2(tmp, target)
        try:
            tmp.unlink()
        except OSError:
            pass


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
SCRIPT_DIR = PROJECT_ROOT / "数据" / "日度收益数据更新"
LEGACY_DIR = PROJECT_ROOT / "数据" / "日度收益数据更新"
ENV_FILE = PROJECT_ROOT / ".env"

OUTPUT_FILLED = "日涨跌幅_填充.csv"
OUTPUT_UNFILLED = "日涨跌幅_未填充.csv"
LEGACY_FILLED = "日涨跌幅_填充.csv"
LEGACY_UNFILLED = "日涨跌幅_未填充.csv"
FUTURES_ADJUSTED_PRICE = "期货主力前复权收盘价.csv"
SUMMARY_FILE = "日度收益数据更新摘要.md"
CACHE_DIR = SCRIPT_DIR / "增量缓存"
FUTURES_QUOTE_CACHE = "期货行情.csv"
ETF_QUOTE_CACHE = "红利低波ETF行情.csv"
GC001_CACHE = "GC001.csv"
CSI1000_INDEX_QUOTE_CACHE = "中证1000指数行情.csv"

UNUSED_COLUMNS = {"有色ETF", "能源化工ETF", "布油连续"}
REPO_COLUMN = "一天期国债逆回购"
ETF_COLUMN = "红利低波ETF"
ETF_INNER_CODE = 201577
CSI500_FUTURES_COLUMN = "中证500主连"
CSI1000_INDEX_COLUMN = "中证1000指数"
CSI1000_INDEX_INNER_CODE = 39144

FUTURES_CACHE_COLUMNS = ["资产", "来源", "日期", "合约内部编码", "合约代码", "收盘价", "主力标志"]
ETF_CACHE_COLUMNS = ["日期", "PrevClosePrice", "ClosePrice"]
GC001_CACHE_COLUMNS = ["日期", REPO_COLUMN]
INDEX_QUOTE_CACHE_COLUMNS = ["日期", "PrevClosePrice", "ClosePrice", "ChangePCT"]


@dataclass(frozen=True)
class FuturesAsset:
    name: str
    source: str
    exchange_code: int
    option_code: int


@dataclass(frozen=True)
class CacheUpdateStat:
    source: str
    previous_end: str
    query_start: str
    query_end: str
    fetched_rows: int
    cache_rows: int
    skipped: bool = False


FUTURES_ASSETS = [
    FuturesAsset("沪深300主连", "financial", 20, 3145),
    FuturesAsset("10年国债主连", "financial", 20, 502),
    FuturesAsset("沪金主连", "commodity", 10, 313),
    FuturesAsset("豆粕主连", "commodity", 13, 345),
    FuturesAsset("中证1000主连", "financial", 20, 39144),
    FuturesAsset(CSI500_FUTURES_COLUMN, "financial", 20, 4978),
    FuturesAsset("30年国债主连", "financial", 20, 504),
    FuturesAsset("沪铜主连", "commodity", 10, 305),
    FuturesAsset("沪铝主连", "commodity", 10, 310),
    FuturesAsset("PTA主连", "commodity", 15, 322),
    FuturesAsset("原油主连", "commodity", 11, 319),
]

OUTPUT_COLUMNS = [
    "沪深300主连",
    "10年国债主连",
    "沪金主连",
    "豆粕主连",
    "中证1000主连",
    CSI500_FUTURES_COLUMN,
    "30年国债主连",
    ETF_COLUMN,
    REPO_COLUMN,
    "沪铜主连",
    "沪铝主连",
    "PTA主连",
    "原油主连",
]


class AdjustmentError(RuntimeError):
    pass


def _strip_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = _strip_env_value(value)


def load_project_env() -> None:
    load_env_file(ENV_FILE)


def prune_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    kept = [column for column in df.columns if column not in UNUSED_COLUMNS]
    return df.loc[:, kept].copy()


def compute_return_from_prev_close(quotes: pd.DataFrame, series_name: str = ETF_COLUMN) -> pd.Series:
    df = quotes.copy()
    df["日期"] = pd.to_datetime(df["日期"]).dt.normalize()
    close = pd.to_numeric(df["ClosePrice"], errors="coerce")
    prev_close = pd.to_numeric(df["PrevClosePrice"], errors="coerce")
    returns = (close / prev_close - 1.0) * 100.0
    returns[(prev_close == 0) | prev_close.isna()] = pd.NA
    returns.index = df["日期"]
    returns = returns.sort_index()
    returns.name = series_name
    return returns


def read_returns_csv(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig") as file:
        df = pd.read_csv(file)
    if "日期" not in df.columns:
        first = df.columns[0]
        df = df.rename(columns={first: "日期"})
    df["日期"] = pd.to_datetime(df["日期"]).dt.normalize()
    df = df.dropna(subset=["日期"]).set_index("日期").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return prune_output_columns(df.reset_index()).set_index("日期")


def find_existing_file(new_name: str, legacy_name: str) -> Path:
    candidates = [
        SCRIPT_DIR / new_name,
        SCRIPT_DIR / legacy_name,
        LEGACY_DIR / new_name,
        LEGACY_DIR / legacy_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"找不到历史数据文件: {new_name} / {legacy_name}")


def _cache_path(file_name: str) -> Path:
    return CACHE_DIR / file_name


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "日期" in result.columns:
        result["日期"] = pd.to_datetime(result["日期"]).dt.normalize()
        result = result.dropna(subset=["日期"])
    return result


def normalize_futures_quote_cache(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_frame(FUTURES_CACHE_COLUMNS)
    result = _normalize_date_column(df)
    for column in FUTURES_CACHE_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    result["收盘价"] = pd.to_numeric(result["收盘价"], errors="coerce")
    result["主力标志"] = pd.to_numeric(result["主力标志"], errors="coerce")
    result = result.dropna(subset=["资产", "日期", "合约内部编码", "收盘价"])
    return result.loc[:, FUTURES_CACHE_COLUMNS].sort_values(["资产", "日期", "合约内部编码"]).reset_index(drop=True)


def normalize_etf_quote_cache(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_frame(ETF_CACHE_COLUMNS)
    result = _normalize_date_column(df)
    for column in ETF_CACHE_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    result["PrevClosePrice"] = pd.to_numeric(result["PrevClosePrice"], errors="coerce")
    result["ClosePrice"] = pd.to_numeric(result["ClosePrice"], errors="coerce")
    result = result.dropna(subset=["日期"])
    return result.loc[:, ETF_CACHE_COLUMNS].sort_values("日期").reset_index(drop=True)


def normalize_gc001_cache(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_frame(GC001_CACHE_COLUMNS)
    result = _normalize_date_column(df)
    for column in GC001_CACHE_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    result[REPO_COLUMN] = pd.to_numeric(result[REPO_COLUMN], errors="coerce")
    result = result.dropna(subset=["日期"])
    return result.loc[:, GC001_CACHE_COLUMNS].sort_values("日期").reset_index(drop=True)


def normalize_index_quote_cache(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_frame(INDEX_QUOTE_CACHE_COLUMNS)
    result = _normalize_date_column(df)
    for column in INDEX_QUOTE_CACHE_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    result["PrevClosePrice"] = pd.to_numeric(result["PrevClosePrice"], errors="coerce")
    result["ClosePrice"] = pd.to_numeric(result["ClosePrice"], errors="coerce")
    result["ChangePCT"] = pd.to_numeric(result["ChangePCT"], errors="coerce")
    result = result.dropna(subset=["日期"])
    return result.loc[:, INDEX_QUOTE_CACHE_COLUMNS].sort_values("日期").reset_index(drop=True)


def read_futures_quote_cache(path: Path | None = None, missing_ok: bool = False) -> pd.DataFrame:
    cache_path = path or _cache_path(FUTURES_QUOTE_CACHE)
    if not cache_path.exists():
        if missing_ok:
            return _empty_frame(FUTURES_CACHE_COLUMNS)
        raise FileNotFoundError(f"找不到期货行情缓存: {cache_path}")
    return normalize_futures_quote_cache(pd.read_csv(cache_path, encoding="utf-8-sig"))


def read_etf_quote_cache(path: Path | None = None, missing_ok: bool = False) -> pd.DataFrame:
    cache_path = path or _cache_path(ETF_QUOTE_CACHE)
    if not cache_path.exists():
        if missing_ok:
            return _empty_frame(ETF_CACHE_COLUMNS)
        raise FileNotFoundError(f"找不到红利低波ETF行情缓存: {cache_path}")
    return normalize_etf_quote_cache(pd.read_csv(cache_path, encoding="utf-8-sig"))


def read_gc001_cache(path: Path | None = None, missing_ok: bool = False) -> pd.DataFrame:
    cache_path = path or _cache_path(GC001_CACHE)
    if not cache_path.exists():
        if missing_ok:
            return _empty_frame(GC001_CACHE_COLUMNS)
        raise FileNotFoundError(f"找不到GC001缓存: {cache_path}")
    return normalize_gc001_cache(pd.read_csv(cache_path, encoding="utf-8-sig"))


def read_csi1000_index_quote_cache(path: Path | None = None, missing_ok: bool = False) -> pd.DataFrame:
    cache_path = path or _cache_path(CSI1000_INDEX_QUOTE_CACHE)
    if not cache_path.exists():
        if missing_ok:
            return _empty_frame(INDEX_QUOTE_CACHE_COLUMNS)
        raise FileNotFoundError(f"找不到中证1000指数行情缓存: {cache_path}")
    return normalize_index_quote_cache(pd.read_csv(cache_path, encoding="utf-8-sig"))


def merge_cache_frames(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    key_columns: list[str],
    sort_columns: list[str],
) -> pd.DataFrame:
    existing_norm = _normalize_date_column(existing)
    incoming_norm = _normalize_date_column(incoming)
    if existing_norm.empty:
        result = incoming_norm.copy()
    elif incoming_norm.empty:
        result = existing_norm.copy()
    else:
        result = pd.concat([existing_norm, incoming_norm], ignore_index=True)
        result = result.drop_duplicates(subset=key_columns, keep="last")
    if "日期" in result.columns:
        result["日期"] = pd.to_datetime(result["日期"]).dt.normalize()
    if not result.empty:
        result = result.sort_values(sort_columns)
    return result.reset_index(drop=True)


def normalize_datetime_index_frame(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if len(result.index) == 0:
        result.index = pd.DatetimeIndex([])
        return result
    dates = pd.to_datetime(result.index, errors="coerce")
    valid = ~pd.isna(dates)
    result = result.loc[valid].copy()
    result.index = pd.DatetimeIndex(dates[valid]).normalize()
    return result.sort_index()


def write_cache_csv(
    df: pd.DataFrame,
    path: Path,
    columns: list[str],
    dry_run: bool,
) -> None:
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = pd.NA
    out = out.loc[:, columns]
    if "日期" in out.columns:
        out["日期"] = pd.to_datetime(out["日期"]).dt.strftime("%Y-%m-%d")
    if dry_run:
        print(f"[dry-run] would write {path} rows={len(out)}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    out.to_csv(tmp, index=False, encoding="utf-8-sig", float_format="%.10f")
    _safe_replace(tmp, path)


def cache_max_date(df: pd.DataFrame) -> pd.Timestamp | None:
    if df.empty or "日期" not in df.columns:
        return None
    dates = pd.to_datetime(df["日期"], errors="coerce").dropna()
    if dates.empty:
        return None
    return pd.Timestamp(dates.max()).normalize()


def calculate_incremental_start(
    cache: pd.DataFrame,
    fallback_start: pd.Timestamp,
    overlap_days: int,
    full_refresh: bool,
) -> pd.Timestamp:
    fallback = pd.Timestamp(fallback_start).normalize()
    if full_refresh:
        return fallback
    latest = cache_max_date(cache)
    if latest is None:
        return fallback
    incremental = latest - pd.Timedelta(days=max(overlap_days, 0))
    return max(fallback, incremental)


def fetch_dataframe(conn: Any, sql: str, params: tuple[Any, ...]) -> pd.DataFrame:
    cursor = conn.cursor()
    cursor.execute(sql, params)
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    return pd.DataFrame.from_records(rows, columns=columns)


def connect_jydb(args: argparse.Namespace) -> Any:
    try:
        import pyodbc
    except ImportError as exc:
        raise RuntimeError("缺少 pyodbc，请先运行: python -m pip install pyodbc") from exc

    password = args.jydb_password or os.environ.get("JYDB_PWD")
    if not password:
        raise RuntimeError("缺少 JYDB_PWD 环境变量；为避免泄露密码，脚本不在项目文件中保存数据库密码。")

    driver = args.jydb_driver or os.environ.get("JYDB_DRIVER", "ODBC Driver 17 for SQL Server")
    server = args.jydb_server or os.environ.get("JYDB_SERVER", "192.168.10.48")
    database = args.jydb_database or os.environ.get("JYDB_DATABASE", "JYDB")
    uid = args.jydb_uid or os.environ.get("JYDB_UID", "tsreadonly")
    conn_str = (
        f"DRIVER={{{driver}}};SERVER={server};DATABASE={database};UID={uid};PWD={password};"
        "Encrypt=no;TrustServerCertificate=yes;"
    )
    return pyodbc.connect(conn_str, timeout=30)


def fetch_futures_quotes(conn: Any, asset: FuturesAsset, start_date: str, end_date: str) -> pd.DataFrame:
    if asset.source == "financial":
        sql = """
SELECT
    TradingDay AS 日期,
    ContractInnerCode AS 合约内部编码,
    ContractCode AS 合约代码,
    CAST(ClosePrice AS float) AS 收盘价,
    CAST(MainContractMark AS int) AS 主力标志
FROM dbo.Fut_TradingQuote
WHERE ExchangeCode = ?
  AND OptionCode = ?
  AND TradingDay BETWEEN ? AND ?
  AND ClosePrice IS NOT NULL
ORDER BY TradingDay, ContractInnerCode
"""
    else:
        sql = """
SELECT
    q.EndDate AS 日期,
    q.InnerCode AS 合约内部编码,
    COALESCE(cm.ContractCode, q.ContractName, CONVERT(varchar(50), q.InnerCode)) AS 合约代码,
    CAST(q.ClosePrice AS float) AS 收盘价,
    CAST(q.MainContractMark AS int) AS 主力标志
FROM dbo.Fut_DailyQuote AS q
LEFT JOIN dbo.Fut_ContractMain AS cm
  ON cm.ContractInnerCode = q.InnerCode
WHERE q.Exchange = ?
  AND q.OptionCode = ?
  AND q.EndDate BETWEEN ? AND ?
  AND q.ClosePrice IS NOT NULL
ORDER BY q.EndDate, q.InnerCode
"""
    df = fetch_dataframe(conn, sql, (asset.exchange_code, asset.option_code, start_date, end_date))
    if df.empty:
        return df
    df["日期"] = pd.to_datetime(df["日期"]).dt.normalize()
    df["收盘价"] = pd.to_numeric(df["收盘价"], errors="coerce")
    df["主力标志"] = pd.to_numeric(df["主力标志"], errors="coerce")
    return df.dropna(subset=["日期", "合约内部编码", "收盘价"]).sort_values(["日期", "合约内部编码"])


def fetch_scalar(conn: Any, sql: str, params: tuple[Any, ...]) -> Any:
    cursor = conn.cursor()
    cursor.execute(sql, params)
    row = cursor.fetchone()
    if row is None:
        return None
    return row[0]


def fetch_futures_latest_date(conn: Any, asset: FuturesAsset) -> pd.Timestamp | None:
    if asset.source == "financial":
        sql = """
SELECT MAX(TradingDay)
FROM dbo.Fut_TradingQuote
WHERE ExchangeCode = ?
  AND OptionCode = ?
  AND ClosePrice IS NOT NULL
"""
    else:
        sql = """
SELECT MAX(EndDate)
FROM dbo.Fut_DailyQuote
WHERE Exchange = ?
  AND OptionCode = ?
  AND ClosePrice IS NOT NULL
"""
    value = fetch_scalar(conn, sql, (asset.exchange_code, asset.option_code))
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).normalize()


def fetch_etf_latest_date(conn: Any) -> pd.Timestamp | None:
    sql = """
SELECT MAX(TradingDay)
FROM dbo.DZ_DailyQuote
WHERE InnerCode = ?
  AND ClosePrice IS NOT NULL
  AND PrevClosePrice IS NOT NULL
"""
    value = fetch_scalar(conn, sql, (ETF_INNER_CODE,))
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).normalize()


def fetch_gc001_latest_date(conn: Any) -> pd.Timestamp | None:
    sql = """
SELECT MAX(time)
FROM FTDB.dbo.ths_GC
WHERE thscode = '204001.SH'
  AND ths_wgt_avg_interest_bbond IS NOT NULL
"""
    value = fetch_scalar(conn, sql, ())
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).normalize()


def fetch_csi1000_index_latest_date(conn: Any) -> pd.Timestamp | None:
    sql = """
SELECT MAX(TradingDay)
FROM dbo.QT_IndexQuote
WHERE InnerCode = ?
  AND ClosePrice IS NOT NULL
  AND PrevClosePrice IS NOT NULL
"""
    value = fetch_scalar(conn, sql, (CSI1000_INDEX_INNER_CODE,))
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).normalize()


def fetch_jydb_latest_end_date(conn: Any) -> pd.Timestamp:
    dates = [fetch_futures_latest_date(conn, asset) for asset in FUTURES_ASSETS]
    dates.append(fetch_etf_latest_date(conn))
    dates.append(fetch_gc001_latest_date(conn))
    dates.append(fetch_csi1000_index_latest_date(conn))
    valid_dates = [date for date in dates if date is not None]
    if not valid_dates:
        raise RuntimeError("数据库没有可用最新日期")
    return min(valid_dates)


def fetch_futures_cache_rows(conn: Any, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    frames = []
    start_text = start_date.strftime("%Y-%m-%d")
    end_text = end_date.strftime("%Y-%m-%d")
    for asset in FUTURES_ASSETS:
        quotes = fetch_futures_quotes(conn, asset, start_text, end_text)
        if quotes.empty:
            continue
        cached = quotes.copy()
        cached.insert(0, "来源", asset.source)
        cached.insert(0, "资产", asset.name)
        frames.append(cached.loc[:, FUTURES_CACHE_COLUMNS])
    if not frames:
        return _empty_frame(FUTURES_CACHE_COLUMNS)
    return normalize_futures_quote_cache(pd.concat(frames, ignore_index=True))


def build_adjusted_main_price(quotes: pd.DataFrame, asset_name: str) -> pd.Series:
    if quotes.empty:
        return pd.Series(dtype="float64", name=asset_name)

    main = quotes.loc[quotes["主力标志"] == 1].copy()
    if main.empty:
        return pd.Series(dtype="float64", name=asset_name)

    main = main.sort_values(["日期", "合约内部编码"])
    main = main.drop_duplicates(subset=["日期"], keep="last")
    main["segment"] = (main["合约内部编码"] != main["合约内部编码"].shift()).cumsum()

    lookup = (
        quotes.sort_values(["日期", "合约内部编码"])
        .drop_duplicates(subset=["日期", "合约内部编码"], keep="last")
        .set_index(["日期", "合约内部编码"])["收盘价"]
    )

    segments: list[dict[str, Any]] = []
    for segment_id, group in main.groupby("segment", sort=True):
        ordered = group.sort_values("日期")
        segments.append(
            {
                "id": int(segment_id),
                "contract": ordered["合约内部编码"].iloc[0],
                "start": ordered["日期"].iloc[0],
                "end": ordered["日期"].iloc[-1],
            }
        )

    factors = {segments[-1]["id"]: 1.0}
    for idx in range(len(segments) - 2, -1, -1):
        old = segments[idx]
        new = segments[idx + 1]
        next_factor = factors[new["id"]]
        switch_date = new["start"]
        fallback_date = old["end"]
        ratio = _switch_ratio(lookup, old["contract"], new["contract"], switch_date)
        if ratio is None:
            ratio = _switch_ratio(lookup, old["contract"], new["contract"], fallback_date)
        if ratio is None:
            raise AdjustmentError(
                f"{asset_name} 主力切换无法计算复权比例: {old['contract']} -> {new['contract']} @ {switch_date.date()}"
            )
        factors[old["id"]] = next_factor * ratio

    adjusted = main["收盘价"] * main["segment"].map(factors)
    adjusted.index = main["日期"]
    adjusted = adjusted.sort_index()
    adjusted.name = asset_name
    return adjusted


def _switch_ratio(
    lookup: pd.Series,
    old_contract: Any,
    new_contract: Any,
    date: pd.Timestamp,
) -> float | None:
    old_price = lookup.get((date, old_contract))
    new_price = lookup.get((date, new_contract))
    if old_price is None or new_price is None:
        return None
    old_price = float(old_price)
    new_price = float(new_price)
    if old_price == 0:
        return None
    return new_price / old_price


def fetch_futures_returns(
    conn: Any,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    lookback_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    query_start = (start_date - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    query_end = end_date.strftime("%Y-%m-%d")
    returns_by_asset: dict[str, pd.Series] = {}
    prices_by_asset: dict[str, pd.Series] = {}
    summary: list[dict[str, Any]] = []

    for asset in FUTURES_ASSETS:
        quotes = fetch_futures_quotes(conn, asset, query_start, query_end)
        price = build_adjusted_main_price(quotes, asset.name)
        returns = price.pct_change() * 100.0
        returns_by_asset[asset.name] = returns.loc[(returns.index >= start_date) & (returns.index <= end_date)]
        prices_by_asset[asset.name] = price.loc[(price.index >= start_date) & (price.index <= end_date)]
        summary.append(
            {
                "资产": asset.name,
                "行情行数": int(len(quotes)),
                "主力价格开始": _date_or_blank(price.index.min() if len(price) else None),
                "主力价格结束": _date_or_blank(price.index.max() if len(price) else None),
                "收益率非空": int(returns_by_asset[asset.name].notna().sum()),
            }
        )

    returns_df = normalize_datetime_index_frame(pd.DataFrame(returns_by_asset))
    price_df = normalize_datetime_index_frame(pd.DataFrame(prices_by_asset))
    return returns_df, price_df, summary


def build_futures_outputs_from_cache(
    cached_quotes: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    lookback_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    cache = normalize_futures_quote_cache(cached_quotes)
    query_start = (start_date - pd.Timedelta(days=lookback_days)).normalize()
    returns_by_asset: dict[str, pd.Series] = {}
    prices_by_asset: dict[str, pd.Series] = {}
    summary: list[dict[str, Any]] = []

    for asset in FUTURES_ASSETS:
        asset_quotes = cache.loc[
            (cache["资产"] == asset.name)
            & (cache["日期"] >= query_start)
            & (cache["日期"] <= end_date),
            ["日期", "合约内部编码", "合约代码", "收盘价", "主力标志"],
        ].copy()
        price = build_adjusted_main_price(asset_quotes, asset.name)
        if price.empty:
            returns_by_asset[asset.name] = pd.Series(dtype="float64", name=asset.name)
            prices_by_asset[asset.name] = pd.Series(dtype="float64", name=asset.name)
        else:
            returns = price.pct_change() * 100.0
            returns_by_asset[asset.name] = returns.loc[(returns.index >= start_date) & (returns.index <= end_date)]
            prices_by_asset[asset.name] = price.loc[(price.index >= start_date) & (price.index <= end_date)]
        summary.append(
            {
                "资产": asset.name,
                "行情行数": int(len(asset_quotes)),
                "主力价格开始": _date_or_blank(price.index.min() if len(price) else None),
                "主力价格结束": _date_or_blank(price.index.max() if len(price) else None),
                "收益率非空": int(returns_by_asset[asset.name].notna().sum()),
            }
        )

    returns_df = normalize_datetime_index_frame(pd.DataFrame(returns_by_asset))
    price_df = normalize_datetime_index_frame(pd.DataFrame(prices_by_asset))
    return returns_df, price_df, summary


def build_etf_return_from_cache(cached_quotes: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series:
    cache = normalize_etf_quote_cache(cached_quotes)
    cache = cache.loc[(cache["日期"] >= start_date) & (cache["日期"] <= end_date)]
    if cache.empty:
        return pd.Series(dtype="float64", name=ETF_COLUMN)
    return compute_return_from_prev_close(cache)


def build_gc001_from_cache(cached_gc001: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series:
    cache = normalize_gc001_cache(cached_gc001)
    cache = cache.loc[(cache["日期"] >= start_date) & (cache["日期"] <= end_date)]
    if cache.empty:
        return pd.Series(dtype="float64", name=REPO_COLUMN)
    series = cache.set_index("日期")[REPO_COLUMN].sort_index()
    series = series[~series.index.duplicated(keep="last")]
    series.name = REPO_COLUMN
    return series


def build_csi1000_index_return_from_cache(
    cached_quotes: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    cache = normalize_index_quote_cache(cached_quotes)
    cache = cache.loc[(cache["日期"] >= start_date) & (cache["日期"] <= end_date)]
    if cache.empty:
        return pd.Series(dtype="float64", name=CSI1000_INDEX_COLUMN)
    return compute_return_from_prev_close(cache, CSI1000_INDEX_COLUMN)


def fetch_etf_return(conn: Any, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series:
    df = fetch_etf_quote_rows(conn, start_date, end_date)
    if df.empty:
        return pd.Series(dtype="float64", name=ETF_COLUMN)
    return compute_return_from_prev_close(df)


def fetch_etf_quote_rows(conn: Any, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    sql = """
SELECT
    TradingDay AS 日期,
    CAST(PrevClosePrice AS float) AS PrevClosePrice,
    CAST(ClosePrice AS float) AS ClosePrice
FROM dbo.DZ_DailyQuote
WHERE InnerCode = ?
  AND TradingDay BETWEEN ? AND ?
ORDER BY TradingDay
"""
    df = fetch_dataframe(conn, sql, (ETF_INNER_CODE, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")))
    if df.empty:
        return _empty_frame(ETF_CACHE_COLUMNS)
    return normalize_etf_quote_cache(df)


def fetch_gc001_quote_rows(conn: Any, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    sql = """
SELECT
    CAST(time AS date) AS 日期,
    CAST(ths_wgt_avg_interest_bbond AS float) AS 一天期国债逆回购
FROM FTDB.dbo.ths_GC
WHERE thscode = '204001.SH'
  AND time BETWEEN ? AND ?
  AND ths_wgt_avg_interest_bbond IS NOT NULL
ORDER BY time
"""
    df = fetch_dataframe(conn, sql, (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")))
    if df.empty:
        return _empty_frame(GC001_CACHE_COLUMNS)
    return normalize_gc001_cache(df)


def fetch_csi1000_index_quote_rows(conn: Any, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    sql = """
SELECT
    TradingDay AS 日期,
    CAST(PrevClosePrice AS float) AS PrevClosePrice,
    CAST(ClosePrice AS float) AS ClosePrice,
    CAST(ChangePCT AS float) AS ChangePCT
FROM dbo.QT_IndexQuote
WHERE InnerCode = ?
  AND TradingDay BETWEEN ? AND ?
ORDER BY TradingDay
"""
    df = fetch_dataframe(
        conn,
        sql,
        (CSI1000_INDEX_INNER_CODE, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")),
    )
    if df.empty:
        return _empty_frame(INDEX_QUOTE_CACHE_COLUMNS)
    return normalize_index_quote_cache(df)


def _date_text(value: pd.Timestamp | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _skipped_stat(source: str, cache: pd.DataFrame, target_end: pd.Timestamp) -> CacheUpdateStat:
    return CacheUpdateStat(
        source=source,
        previous_end=_date_text(cache_max_date(cache)),
        query_start="",
        query_end=target_end.strftime("%Y-%m-%d"),
        fetched_rows=0,
        cache_rows=int(len(cache)),
        skipped=True,
    )


def should_skip_fetch(cache: pd.DataFrame, target_end: pd.Timestamp, full_refresh: bool) -> bool:
    if full_refresh:
        return False
    latest = cache_max_date(cache)
    return latest is not None and target_end <= latest


def futures_cache_has_all_configured_assets(cache: pd.DataFrame) -> bool:
    if cache.empty or "资产" not in cache.columns:
        return False
    cached_assets = set(cache["资产"].dropna().astype(str))
    required_assets = {asset.name for asset in FUTURES_ASSETS}
    return required_assets.issubset(cached_assets)


def refresh_futures_quote_cache(
    conn: Any,
    existing: pd.DataFrame,
    fallback_start: pd.Timestamp,
    target_end: pd.Timestamp,
    overlap_days: int,
    full_refresh: bool,
    dry_run: bool,
) -> tuple[pd.DataFrame, CacheUpdateStat]:
    existing = _empty_frame(FUTURES_CACHE_COLUMNS) if full_refresh else normalize_futures_quote_cache(existing)
    has_all_assets = futures_cache_has_all_configured_assets(existing)
    if has_all_assets and should_skip_fetch(existing, target_end, full_refresh):
        return existing, _skipped_stat("期货行情", existing, target_end)

    query_start = (
        calculate_incremental_start(existing, fallback_start, overlap_days, full_refresh)
        if has_all_assets
        else pd.Timestamp(fallback_start).normalize()
    )
    incoming = fetch_futures_cache_rows(conn, query_start, target_end)
    merged = merge_cache_frames(
        existing,
        incoming,
        key_columns=["资产", "日期", "合约内部编码"],
        sort_columns=["资产", "日期", "合约内部编码"],
    )
    merged = normalize_futures_quote_cache(merged)
    write_cache_csv(merged, _cache_path(FUTURES_QUOTE_CACHE), FUTURES_CACHE_COLUMNS, dry_run)
    return merged, CacheUpdateStat(
        source="期货行情",
        previous_end=_date_text(cache_max_date(existing)),
        query_start=query_start.strftime("%Y-%m-%d"),
        query_end=target_end.strftime("%Y-%m-%d"),
        fetched_rows=int(len(incoming)),
        cache_rows=int(len(merged)),
    )


def refresh_etf_quote_cache(
    conn: Any,
    existing: pd.DataFrame,
    fallback_start: pd.Timestamp,
    target_end: pd.Timestamp,
    overlap_days: int,
    full_refresh: bool,
    dry_run: bool,
) -> tuple[pd.DataFrame, CacheUpdateStat]:
    existing = _empty_frame(ETF_CACHE_COLUMNS) if full_refresh else normalize_etf_quote_cache(existing)
    if should_skip_fetch(existing, target_end, full_refresh):
        return existing, _skipped_stat("红利低波ETF行情", existing, target_end)

    query_start = calculate_incremental_start(existing, fallback_start, overlap_days, full_refresh)
    incoming = fetch_etf_quote_rows(conn, query_start, target_end)
    merged = merge_cache_frames(existing, incoming, key_columns=["日期"], sort_columns=["日期"])
    merged = normalize_etf_quote_cache(merged)
    write_cache_csv(merged, _cache_path(ETF_QUOTE_CACHE), ETF_CACHE_COLUMNS, dry_run)
    return merged, CacheUpdateStat(
        source="红利低波ETF行情",
        previous_end=_date_text(cache_max_date(existing)),
        query_start=query_start.strftime("%Y-%m-%d"),
        query_end=target_end.strftime("%Y-%m-%d"),
        fetched_rows=int(len(incoming)),
        cache_rows=int(len(merged)),
    )


def refresh_gc001_cache(
    conn: Any,
    existing: pd.DataFrame,
    fallback_start: pd.Timestamp,
    target_end: pd.Timestamp,
    overlap_days: int,
    full_refresh: bool,
    dry_run: bool,
) -> tuple[pd.DataFrame, CacheUpdateStat]:
    existing = _empty_frame(GC001_CACHE_COLUMNS) if full_refresh else normalize_gc001_cache(existing)
    if should_skip_fetch(existing, target_end, full_refresh):
        return existing, _skipped_stat("GC001", existing, target_end)

    query_start = calculate_incremental_start(existing, fallback_start, overlap_days, full_refresh)
    incoming = fetch_gc001_quote_rows(conn, query_start, target_end)
    incoming = normalize_gc001_cache(incoming)
    merged = merge_cache_frames(existing, incoming, key_columns=["日期"], sort_columns=["日期"])
    merged = normalize_gc001_cache(merged)
    write_cache_csv(merged, _cache_path(GC001_CACHE), GC001_CACHE_COLUMNS, dry_run)
    return merged, CacheUpdateStat(
        source="GC001",
        previous_end=_date_text(cache_max_date(existing)),
        query_start=query_start.strftime("%Y-%m-%d"),
        query_end=target_end.strftime("%Y-%m-%d"),
        fetched_rows=int(len(incoming)),
        cache_rows=int(len(merged)),
    )


def refresh_csi1000_index_quote_cache(
    conn: Any,
    existing: pd.DataFrame,
    fallback_start: pd.Timestamp,
    target_end: pd.Timestamp,
    overlap_days: int,
    full_refresh: bool,
    dry_run: bool,
) -> tuple[pd.DataFrame, CacheUpdateStat]:
    existing = _empty_frame(INDEX_QUOTE_CACHE_COLUMNS) if full_refresh else normalize_index_quote_cache(existing)
    if should_skip_fetch(existing, target_end, full_refresh):
        return existing, _skipped_stat("中证1000指数行情", existing, target_end)

    query_start = calculate_incremental_start(existing, fallback_start, overlap_days, full_refresh)
    incoming = fetch_csi1000_index_quote_rows(conn, query_start, target_end)
    merged = merge_cache_frames(existing, incoming, key_columns=["日期"], sort_columns=["日期"])
    merged = normalize_index_quote_cache(merged)
    write_cache_csv(merged, _cache_path(CSI1000_INDEX_QUOTE_CACHE), INDEX_QUOTE_CACHE_COLUMNS, dry_run)
    return merged, CacheUpdateStat(
        source="中证1000指数行情",
        previous_end=_date_text(cache_max_date(existing)),
        query_start=query_start.strftime("%Y-%m-%d"),
        query_end=target_end.strftime("%Y-%m-%d"),
        fetched_rows=int(len(incoming)),
        cache_rows=int(len(merged)),
    )


def build_outputs(
    official: pd.DataFrame,
    legacy_filled: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    csi1000_index_return: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    calendar = official.index.union(legacy_filled.index)
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar, errors="coerce"))
    calendar = calendar[~pd.isna(calendar)]
    calendar = calendar[(calendar >= start_date) & (calendar <= end_date)].sort_values()
    official = official.reindex(calendar)
    official = official.reindex(columns=OUTPUT_COLUMNS)

    filled = official.copy()
    if csi1000_index_return is not None and CSI500_FUTURES_COLUMN in filled.columns:
        fallback = csi1000_index_return.reindex(calendar)
        first_csi500_return_date = filled[CSI500_FUTURES_COLUMN].first_valid_index()
        pre_listing_missing = filled[CSI500_FUTURES_COLUMN].isna()
        if first_csi500_return_date is not None:
            pre_listing_missing &= filled.index < first_csi500_return_date
        filled.loc[pre_listing_missing, CSI500_FUTURES_COLUMN] = fallback.loc[pre_listing_missing]

    legacy_aligned = legacy_filled.reindex(calendar).reindex(columns=OUTPUT_COLUMNS)
    filled = filled.combine_first(legacy_aligned)
    return filled, official


def write_returns_csv(df: pd.DataFrame, path: Path, dry_run: bool) -> None:
    out = df.copy()
    out = out.reindex(columns=OUTPUT_COLUMNS)
    out.insert(0, "日期", out.index.strftime("%Y-%m-%d"))
    if dry_run:
        print(f"[dry-run] would write {path} rows={len(out)}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    out.to_csv(tmp, index=False, encoding="utf-8-sig", float_format="%.10f")
    _safe_replace(tmp, path)


def write_price_csv(df: pd.DataFrame, path: Path, dry_run: bool) -> None:
    out = df.copy()
    out.insert(0, "日期", out.index.strftime("%Y-%m-%d"))
    if dry_run:
        print(f"[dry-run] would write {path} rows={len(out)}")
        return
    tmp = path.with_name(f"{path.name}.tmp")
    out.to_csv(tmp, index=False, encoding="utf-8-sig", float_format="%.10f")
    _safe_replace(tmp, path)


def write_summary(
    path: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    filled: pd.DataFrame,
    unfilled: pd.DataFrame,
    futures_summary: list[dict[str, Any]],
    dry_run: bool,
    cache_stats: list[CacheUpdateStat] | None = None,
) -> None:
    lines = [
        "# 日度收益数据更新摘要",
        "",
        f"- 更新时间：`{datetime.now():%Y-%m-%d %H:%M:%S}`",
        f"- 日期范围：`{start_date:%Y-%m-%d}` 至 `{end_date:%Y-%m-%d}`",
        f"- 填充版行数：`{len(filled)}`",
        f"- 未填充版行数：`{len(unfilled)}`",
        f"- 输出列：`{', '.join(OUTPUT_COLUMNS)}`",
        "- 红利低波ETF：JYDB `512890.SH`，按 `ClosePrice / PrevClosePrice - 1` 计算百分比涨跌幅。",
        "- 一天期国债逆回购：FTDB `dbo.ths_GC / 204001.SH / ths_wgt_avg_interest_bbond`，单位 `%`，直接作为年化利率。",
        "- 中证500主连：JYDB `Fut_TradingQuote / ExchangeCode=20 / OptionCode=4978`；填充版在期货收益上市前用 `QT_IndexQuote / InnerCode=39144` 的中证1000指数日涨跌幅补齐。",
        "",
        "## 缺失值",
        "",
        "| 数据集 | 缺失值合计 |",
        "| --- | ---: |",
        f"| 填充版 | {int(filled.isna().sum().sum())} |",
        f"| 未填充版 | {int(unfilled.isna().sum().sum())} |",
        "",
    ]
    if cache_stats is not None:
        lines.extend(
            [
                "## 增量缓存",
                "",
                "| 数据源 | 原缓存最新日 | 查询开始 | 查询结束 | 拉取行数 | 缓存行数 | 状态 |",
                "| --- | --- | --- | --- | ---: | ---: | --- |",
            ]
        )
        for stat in cache_stats:
            status = "跳过" if stat.skipped else "更新"
            lines.append(
                f"| {stat.source} | {stat.previous_end} | {stat.query_start} | {stat.query_end} | "
                f"{stat.fetched_rows} | {stat.cache_rows} | {status} |"
            )
        lines.append("")
    lines.extend(
        [
            "## 期货覆盖",
            "",
            "| 资产 | 行情行数 | 主力价格开始 | 主力价格结束 | 收益率非空 |",
            "| --- | ---: | --- | --- | ---: |",
        ]
    )
    for item in futures_summary:
        lines.append(
            f"| {item['资产']} | {item['行情行数']} | {item['主力价格开始']} | {item['主力价格结束']} | {item['收益率非空']} |"
        )
    text = "\n".join(lines) + "\n"
    if dry_run:
        print(f"[dry-run] would write {path}")
        return
    path.write_text(text, encoding="utf-8-sig")


def _date_or_blank(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def resolve_end_date(series_list: list[pd.Series | pd.DataFrame], requested: pd.Timestamp | None) -> pd.Timestamp:
    max_dates = []
    for item in series_list:
        if len(item.index) == 0:
            continue
        max_dates.append(pd.Timestamp(item.index.max()).normalize())
    if not max_dates:
        raise RuntimeError("没有可用数据用于确定结束日期")
    available_end = min(max_dates)
    if requested is None:
        return available_end
    return min(requested.normalize(), available_end)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="更新日涨跌幅_填充.csv和日涨跌幅_未填充.csv")
    parser.add_argument("--start-date", help="起始日期，默认使用历史填充版首日")
    parser.add_argument("--end-date", help="结束日期，默认使用所有数据源共同最新日期")
    parser.add_argument("--lookback-days", type=int, default=60, help="期货复权查询向前回看天数")
    parser.add_argument("--cache-overlap-days", type=int, default=7, help="增量缓存更新时向前覆盖的天数")
    parser.add_argument("--full-refresh", action="store_true", help="忽略现有缓存，从数据源重新初始化缓存和输出")
    parser.add_argument("--rebuild-from-cache", action="store_true", help="不访问数据库，只用现有缓存重算输出")
    parser.add_argument("--dry-run", action="store_true", help="只执行查询和校验，不写文件")
    parser.add_argument("--jydb-server")
    parser.add_argument("--jydb-database")
    parser.add_argument("--jydb-uid")
    parser.add_argument("--jydb-password")
    parser.add_argument("--jydb-driver")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    load_project_env()
    legacy_filled_path = find_existing_file(OUTPUT_FILLED, LEGACY_FILLED)
    legacy_filled = read_returns_csv(legacy_filled_path)
    start_date = pd.Timestamp(args.start_date).normalize() if args.start_date else legacy_filled.index.min().normalize()
    requested_end = pd.Timestamp(args.end_date).normalize() if args.end_date else None

    futures_cache_start = (start_date - pd.Timedelta(days=args.lookback_days)).normalize()
    cache_stats: list[CacheUpdateStat] = []
    if args.rebuild_from_cache:
        futures_cache = read_futures_quote_cache()
        etf_cache = read_etf_quote_cache()
        gc001_cache = read_gc001_cache()
        csi1000_index_cache = read_csi1000_index_quote_cache()
        cache_dates = [
            cache_max_date(futures_cache),
            cache_max_date(etf_cache),
            cache_max_date(gc001_cache),
            cache_max_date(csi1000_index_cache),
        ]
        valid_cache_dates = [date for date in cache_dates if date is not None]
        if not valid_cache_dates:
            raise RuntimeError("缓存为空，无法执行 --rebuild-from-cache")
        initial_end = min(requested_end, min(valid_cache_dates)) if requested_end is not None else min(valid_cache_dates)
        cache_stats = [
            _skipped_stat("期货行情", futures_cache, initial_end),
            _skipped_stat("红利低波ETF行情", etf_cache, initial_end),
            _skipped_stat("GC001", gc001_cache, initial_end),
            _skipped_stat("中证1000指数行情", csi1000_index_cache, initial_end),
        ]
    else:
        conn = connect_jydb(args)
        try:
            jydb_latest_end = fetch_jydb_latest_end_date(conn)
            initial_end = min(requested_end, jydb_latest_end) if requested_end is not None else jydb_latest_end
            futures_cache_existing = read_futures_quote_cache(missing_ok=True)
            etf_cache_existing = read_etf_quote_cache(missing_ok=True)
            futures_cache, futures_stat = refresh_futures_quote_cache(
                conn,
                futures_cache_existing,
                futures_cache_start,
                initial_end,
                args.cache_overlap_days,
                args.full_refresh,
                args.dry_run,
            )
            etf_cache, etf_stat = refresh_etf_quote_cache(
                conn,
                etf_cache_existing,
                start_date,
                initial_end,
                args.cache_overlap_days,
                args.full_refresh,
                args.dry_run,
            )
            gc001_cache_existing = read_gc001_cache(missing_ok=True)
            gc001_cache, gc001_stat = refresh_gc001_cache(
                conn,
                gc001_cache_existing,
                start_date,
                initial_end,
                args.cache_overlap_days,
                args.full_refresh,
                args.dry_run,
            )
            csi1000_index_cache_existing = read_csi1000_index_quote_cache(missing_ok=True)
            csi1000_index_cache, csi1000_index_stat = refresh_csi1000_index_quote_cache(
                conn,
                csi1000_index_cache_existing,
                start_date,
                initial_end,
                args.cache_overlap_days,
                args.full_refresh,
                args.dry_run,
            )
        finally:
            conn.close()

        cache_stats = [futures_stat, etf_stat, gc001_stat, csi1000_index_stat]

    futures_returns, futures_prices, futures_summary = build_futures_outputs_from_cache(
        futures_cache, start_date, initial_end, args.lookback_days
    )
    etf_return = build_etf_return_from_cache(etf_cache, start_date, initial_end)
    gc001 = build_gc001_from_cache(gc001_cache, start_date, initial_end)
    csi1000_index_return = build_csi1000_index_return_from_cache(csi1000_index_cache, start_date, initial_end)
    end_date = resolve_end_date([futures_returns, etf_return, gc001, csi1000_index_return], requested_end)

    official = futures_returns.loc[futures_returns.index <= end_date].copy()
    official[ETF_COLUMN] = etf_return.loc[etf_return.index <= end_date]
    official[REPO_COLUMN] = gc001.loc[gc001.index <= end_date]
    official = official.reindex(columns=OUTPUT_COLUMNS)

    filled, unfilled = build_outputs(
        official,
        legacy_filled,
        start_date,
        end_date,
        csi1000_index_return=csi1000_index_return.loc[csi1000_index_return.index <= end_date],
    )
    futures_prices = futures_prices.loc[(futures_prices.index >= start_date) & (futures_prices.index <= end_date)]

    write_returns_csv(filled, SCRIPT_DIR / OUTPUT_FILLED, args.dry_run)
    write_returns_csv(unfilled, SCRIPT_DIR / OUTPUT_UNFILLED, args.dry_run)
    write_price_csv(futures_prices, SCRIPT_DIR / FUTURES_ADJUSTED_PRICE, args.dry_run)
    write_summary(SCRIPT_DIR / SUMMARY_FILE, start_date, end_date, filled, unfilled, futures_summary, args.dry_run, cache_stats)

    print(f"updated_range={start_date:%Y-%m-%d}:{end_date:%Y-%m-%d}")
    print(f"rows_filled={len(filled)} rows_unfilled={len(unfilled)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

