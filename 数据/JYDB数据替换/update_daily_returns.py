from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tomllib
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
LEGACY_DIR = PROJECT_ROOT / "数据" / "JYDB数据替换"

OUTPUT_FILLED = "日涨跌幅_填充.csv"
OUTPUT_UNFILLED = "日涨跌幅_未填充.csv"
LEGACY_FILLED = "日涨跌幅_填充.csv"
LEGACY_UNFILLED = "日涨跌幅_未填充.csv"
FUTURES_ADJUSTED_PRICE = "期货主力前复权收盘价.csv"
SUMMARY_FILE = "日涨跌幅更新摘要.md"

UNUSED_COLUMNS = {"有色ETF", "能源化工ETF", "布油连续"}
IFIND_SERVER_NAME = "hexin-ifind-ds-edb-mcp"
IFIND_GC001_INDEX_ID = "L004369613"
IFIND_GC001_FIELD = "GC001(加权平均)"
REPO_COLUMN = "一天期国债逆回购"
ETF_COLUMN = "红利低波ETF"
ETF_INNER_CODE = 201577


@dataclass(frozen=True)
class FuturesAsset:
    name: str
    source: str
    exchange_code: int
    option_code: int


FUTURES_ASSETS = [
    FuturesAsset("沪深300主连", "financial", 20, 3145),
    FuturesAsset("10年国债主连", "financial", 20, 502),
    FuturesAsset("沪金主连", "commodity", 10, 313),
    FuturesAsset("豆粕主连", "commodity", 13, 345),
    FuturesAsset("中证1000主连", "financial", 20, 39144),
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


def prune_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    kept = [column for column in df.columns if column not in UNUSED_COLUMNS]
    return df.loc[:, kept].copy()


def compute_return_from_prev_close(quotes: pd.DataFrame) -> pd.Series:
    df = quotes.copy()
    df["日期"] = pd.to_datetime(df["日期"]).dt.normalize()
    close = pd.to_numeric(df["ClosePrice"], errors="coerce")
    prev_close = pd.to_numeric(df["PrevClosePrice"], errors="coerce")
    returns = (close / prev_close - 1.0) * 100.0
    returns[(prev_close == 0) | prev_close.isna()] = pd.NA
    returns.index = df["日期"]
    returns = returns.sort_index()
    returns.name = ETF_COLUMN
    return returns


def parse_ifind_edb_response(outer: dict[str, Any]) -> pd.Series:
    text = outer["result"]["content"][0]["text"]
    inner = json.loads(text)
    data_block = inner["data"]["datas"][0]["data"]
    columns = data_block["columns"]
    attrs = data_block.get("attrs", {})
    rows = data_block["data"]

    date_col = "日期"
    value_col = IFIND_GC001_FIELD if IFIND_GC001_FIELD in columns else next(
        column for column in columns if column != date_col
    )
    index_id = attrs.get(value_col, {}).get("index_id")
    if index_id is not None and index_id != IFIND_GC001_INDEX_ID:
        raise ValueError(f"iFinD GC001指标ID不匹配: {index_id}")

    frame = pd.DataFrame(rows, columns=columns)
    frame[date_col] = pd.to_datetime(frame[date_col]).dt.normalize()
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
    series = frame.dropna(subset=[date_col]).set_index(date_col)[value_col].sort_index()
    series = series[~series.index.duplicated(keep="last")]
    series.name = REPO_COLUMN
    return series


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

    returns_df = pd.DataFrame(returns_by_asset)
    price_df = pd.DataFrame(prices_by_asset)
    return returns_df, price_df, summary


def fetch_etf_return(conn: Any, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series:
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
        return pd.Series(dtype="float64", name=ETF_COLUMN)
    return compute_return_from_prev_close(df)


def read_ifind_mcp_config(config_path: Path, server_name: str = IFIND_SERVER_NAME) -> tuple[str, dict[str, str]]:
    with config_path.open("rb") as file:
        config = tomllib.load(file)
    server = config["mcp_servers"][server_name]
    url = server["url"]
    headers = dict(server.get("http_headers", {}))
    if "Authorization" not in headers:
        raise RuntimeError(f"{config_path} 中 {server_name} 缺少 Authorization")
    headers.update({"Content-Type": "application/json", "Accept": "application/json, text/event-stream"})
    return url, headers


def _decode_json_or_sse(body: str) -> dict[str, Any]:
    stripped = body.strip()
    if stripped.startswith("{"):
        return json.loads(stripped)
    data_lines = [line[5:].strip() for line in stripped.splitlines() if line.startswith("data:")]
    if not data_lines:
        raise ValueError("MCP响应不是JSON或SSE data")
    return json.loads(data_lines[-1])


def _mcp_post(url: str, headers: dict[str, str], payload: dict[str, Any], session_id: str | None = None) -> tuple[dict[str, str], dict[str, Any]]:
    request_headers = dict(headers)
    if session_id:
        request_headers["mcp-session-id"] = session_id
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        body = response.read().decode("utf-8")
        return dict(response.headers), _decode_json_or_sse(body)


def fetch_gc001_weighted_average(start_date: pd.Timestamp, end_date: pd.Timestamp, config_path: Path) -> pd.Series:
    url, headers = read_ifind_mcp_config(config_path)
    init_headers, _ = _mcp_post(
        url,
        headers,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "jydb-daily-return-update", "version": "1.0"},
            },
        },
    )
    session_id = init_headers.get("mcp-session-id") or init_headers.get("Mcp-Session-Id")
    if not session_id:
        raise RuntimeError("iFinD MCP initialize响应缺少mcp-session-id")
    _mcp_post(url, headers, {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}, session_id)
    _, result = _mcp_post(
        url,
        headers,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "get_edb_data",
                "arguments": {
                    "query": (
                        f"查询GC001(加权平均)从{start_date:%Y-%m-%d}至{end_date:%Y-%m-%d}的日度数据"
                    )
                },
            },
        },
        session_id,
    )
    return parse_ifind_edb_response(result)


def build_outputs(
    official: pd.DataFrame,
    legacy_filled: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    calendar = official.index.union(legacy_filled.index)
    calendar = calendar[(calendar >= start_date) & (calendar <= end_date)].sort_values()
    official = official.reindex(calendar)
    official = official.reindex(columns=OUTPUT_COLUMNS)

    filled = official.copy()
    legacy_aligned = legacy_filled.reindex(calendar).reindex(columns=OUTPUT_COLUMNS)
    filled = filled.combine_first(legacy_aligned)
    return filled, official


def write_returns_csv(df: pd.DataFrame, path: Path, backup: bool, dry_run: bool) -> None:
    out = df.copy()
    out = out.reindex(columns=OUTPUT_COLUMNS)
    out.insert(0, "日期", out.index.strftime("%Y-%m-%d"))
    if dry_run:
        print(f"[dry-run] would write {path} rows={len(out)}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if backup and path.exists():
        backup_dir = path.parent / "backup"
        backup_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(path, backup_dir / f"{path.stem}_{timestamp}{path.suffix}")
    tmp = path.with_name(f"{path.name}.tmp")
    out.to_csv(tmp, index=False, encoding="utf-8-sig", float_format="%.10f")
    os.replace(tmp, path)


def write_price_csv(df: pd.DataFrame, path: Path, backup: bool, dry_run: bool) -> None:
    out = df.copy()
    out.insert(0, "日期", out.index.strftime("%Y-%m-%d"))
    if dry_run:
        print(f"[dry-run] would write {path} rows={len(out)}")
        return
    if backup and path.exists():
        backup_dir = path.parent / "backup"
        backup_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(path, backup_dir / f"{path.stem}_{timestamp}{path.suffix}")
    tmp = path.with_name(f"{path.name}.tmp")
    out.to_csv(tmp, index=False, encoding="utf-8-sig", float_format="%.10f")
    os.replace(tmp, path)


def write_summary(
    path: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    filled: pd.DataFrame,
    unfilled: pd.DataFrame,
    futures_summary: list[dict[str, Any]],
    dry_run: bool,
) -> None:
    lines = [
        "# 日涨跌幅更新摘要",
        "",
        f"- 更新时间：`{datetime.now():%Y-%m-%d %H:%M:%S}`",
        f"- 日期范围：`{start_date:%Y-%m-%d}` 至 `{end_date:%Y-%m-%d}`",
        f"- 填充版行数：`{len(filled)}`",
        f"- 未填充版行数：`{len(unfilled)}`",
        f"- 输出列：`{', '.join(OUTPUT_COLUMNS)}`",
        "- 红利低波ETF：JYDB `512890.SH`，按 `ClosePrice / PrevClosePrice - 1` 计算百分比涨跌幅。",
        "- 一天期国债逆回购：iFinD EDB `L004369613 / GC001(加权平均)`，单位 `%`，直接作为年化利率。",
        "",
        "## 缺失值",
        "",
        "| 数据集 | 缺失值合计 |",
        "| --- | ---: |",
        f"| 填充版 | {int(filled.isna().sum().sum())} |",
        f"| 未填充版 | {int(unfilled.isna().sum().sum())} |",
        "",
        "## 期货覆盖",
        "",
        "| 资产 | 行情行数 | 主力价格开始 | 主力价格结束 | 收益率非空 |",
        "| --- | ---: | --- | --- | ---: |",
    ]
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
    parser.add_argument("--backup", action="store_true", help="覆盖输出前备份旧文件")
    parser.add_argument("--dry-run", action="store_true", help="只执行查询和校验，不写文件")
    parser.add_argument("--ifind-config", default=str(Path.home() / ".codex" / "config.toml"), help="Codex config.toml路径")
    parser.add_argument("--jydb-server")
    parser.add_argument("--jydb-database")
    parser.add_argument("--jydb-uid")
    parser.add_argument("--jydb-password")
    parser.add_argument("--jydb-driver")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    legacy_filled_path = find_existing_file(OUTPUT_FILLED, LEGACY_FILLED)
    legacy_filled = read_returns_csv(legacy_filled_path)
    start_date = pd.Timestamp(args.start_date).normalize() if args.start_date else legacy_filled.index.min().normalize()
    requested_end = pd.Timestamp(args.end_date).normalize() if args.end_date else None

    initial_end = requested_end or pd.Timestamp.today().normalize()
    conn = connect_jydb(args)
    try:
        futures_returns, futures_prices, futures_summary = fetch_futures_returns(
            conn, start_date, initial_end, args.lookback_days
        )
        etf_return = fetch_etf_return(conn, start_date, initial_end)
    finally:
        conn.close()

    gc001 = fetch_gc001_weighted_average(start_date, initial_end, Path(args.ifind_config))
    end_date = resolve_end_date([futures_returns, etf_return, gc001], requested_end)

    official = futures_returns.loc[futures_returns.index <= end_date].copy()
    official[ETF_COLUMN] = etf_return.loc[etf_return.index <= end_date]
    official[REPO_COLUMN] = gc001.loc[gc001.index <= end_date]
    official = official.reindex(columns=OUTPUT_COLUMNS)

    filled, unfilled = build_outputs(official, legacy_filled, start_date, end_date)
    futures_prices = futures_prices.loc[(futures_prices.index >= start_date) & (futures_prices.index <= end_date)]

    write_returns_csv(filled, SCRIPT_DIR / OUTPUT_FILLED, args.backup, args.dry_run)
    write_returns_csv(unfilled, SCRIPT_DIR / OUTPUT_UNFILLED, args.backup, args.dry_run)
    write_price_csv(futures_prices, SCRIPT_DIR / FUTURES_ADJUSTED_PRICE, args.backup, args.dry_run)
    write_summary(SCRIPT_DIR / SUMMARY_FILE, start_date, end_date, filled, unfilled, futures_summary, args.dry_run)

    print(f"updated_range={start_date:%Y-%m-%d}:{end_date:%Y-%m-%d}")
    print(f"rows_filled={len(filled)} rows_unfilled={len(unfilled)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

