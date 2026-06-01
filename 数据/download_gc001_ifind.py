from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tomllib
import urllib.request
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd


# 默认路径指向当前脚本所在的数据目录。
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = Path.home() / ".codex" / "config.toml"
DEFAULT_OUTPUT = SCRIPT_DIR / "GC001_daily_weighted_average_ifind.csv"

IFIND_SERVER_NAME = "hexin-ifind-ds-edb-mcp"
IFIND_MCP_URL_ENV = "IFIND_MCP_URL"
IFIND_MCP_AUTH_ENV = "IFIND_MCP_AUTHORIZATION"
HARDCODED_IFIND_MCP_URL = "https://api-mcp.51ifind.com:8643/ds-mcp-servers/hexin-ifind-ds-edb-mcp"
HARDCODED_IFIND_MCP_AUTHORIZATION = "eyJhbGciOiJSU0EtT0FFUC0yNTYiLCJlbmMiOiJBMjU2R0NNIn0.Vwea0AZKcEnpsyY-AyMn5xm0NJwQbbz_HJF_D6Hn1AgNeDi3gUSEAUqFulvQZWVRVJNRYEqEEyzWf5XnAiIShlJXWqRKpmmqXGiIDVqx0qzSyvXQQvDcPJAt8wB-NbkOMMQAjZcURCYLuLS-DGaaMwblWCcbKUKDWkUKpM9huTPx63wqXToGBl-cPoaKXQpzS9xlyvwgOmZQXaVG33W2i7MadpBsYFuFiwvfuFkPWcpSfhc-cnodKNkr4S37NgpmJ7c-NpeKnLFD2uedIWtrLLefKbBYu6eEkJ-SZM1wivosuGsLha11sKlwZLbWuQ61MLGzrvukmlxd1dGZpEIgIA.N3bLFV1GsqrBH4nl.m97nKgKU_ux-5hDvyhlcuk6qyIUEQBoiMph6352Vt7ype_ogtvFO5XXNsuy0ZFVGmIXuwRJsC5PLwe8nJ8fhVWnnul7zO3BBiq_pzcT4DbJzq5CLCjzRcTc_JDI69Jgb5kRVKInUg92H-q4nYOYAfe-1XlQgtXAqRPlri8LqSDM3ejXtHf1Y1O4q_dRfkkTJlj3999Z83cL-xEyB2bRgP05s24IF-rlZkOVqw29w0bHbVECXAshojXaTV_Hz32Z5stigEsij_PwN8e0hfkf7s6p16Vfkozsxom4gX51EwmiReNQ1QjkGaF_VWFsUjx54BZCCLC0U5DfNcYjs77YO4yxNYLAnWJ8TMx26JLYARXlECwtCt6k2PtFil_wdlgAsmll9yTWHGMWcs8H5EbIMv9urpozxNw.UlOjybdd2Ec0bBVHXP_uhw"
GC001_FIELD = "GC001(加权平均)"
GC001_INDEX_ID = "L004369613"


def parse_yyyy_mm_dd(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"日期格式应为 YYYY-MM-DD: {value}") from exc


def read_ifind_mcp_config(config_path: Path) -> tuple[str, dict[str, str]]:
    """按常量、环境变量、TOML配置的顺序返回 MCP URL 和 headers。"""

    if (
        HARDCODED_IFIND_MCP_URL
        and HARDCODED_IFIND_MCP_AUTHORIZATION
        and HARDCODED_IFIND_MCP_AUTHORIZATION != "__IFIND_MCP_AUTHORIZATION__"
    ):
        return HARDCODED_IFIND_MCP_URL, build_headers(HARDCODED_IFIND_MCP_AUTHORIZATION)

    env_url = os.environ.get(IFIND_MCP_URL_ENV)
    env_authorization = os.environ.get(IFIND_MCP_AUTH_ENV)
    if env_url and env_authorization:
        return env_url, build_headers(env_authorization)

    with config_path.open("rb") as file:
        config = tomllib.load(file)

    server = config["mcp_servers"][IFIND_SERVER_NAME]
    url = server["url"]
    headers = dict(server.get("http_headers", {}))
    authorization = headers.get("Authorization")
    if not authorization:
        raise RuntimeError(f"{config_path} 中 {IFIND_SERVER_NAME} 缺少 Authorization")
    return url, build_headers(authorization)


def build_headers(authorization: str) -> dict[str, str]:
    return {
        "Authorization": authorization,
        "Content-Type": "application/json",
        # 与 decode_json_or_sse 的两种解析分支保持一致。
        "Accept": "application/json, text/event-stream",
    }


def decode_json_or_sse(body: str) -> dict[str, Any]:
    """把 JSON 或 SSE 文本统一解析为 dict。"""

    stripped = body.strip()
    if not stripped:
        return {}
    if stripped.startswith("{"):
        return json.loads(stripped)

    data_lines = [line[5:].strip() for line in stripped.splitlines() if line.startswith("data:")]
    if not data_lines:
        raise ValueError("iFinD MCP 响应既不是 JSON，也没有 SSE data 行")
    return json.loads(data_lines[-1])


def mcp_post(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    session_id: str | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """发送一次 MCP JSON-RPC POST 请求。"""

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
        return dict(response.headers), decode_json_or_sse(body)


def initialize_mcp_session(url: str, headers: dict[str, str]) -> str:
    """执行 MCP initialize，并从响应头取得 mcp-session-id。"""

    response_headers, response_body = mcp_post(
        url,
        headers,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "gc001-ifind-downloader", "version": "1.0"},
            },
        },
    )
    if "error" in response_body:
        raise RuntimeError(f"iFinD MCP initialize 失败: {response_body['error']}")

    session_id = response_headers.get("mcp-session-id") or response_headers.get("Mcp-Session-Id")
    if not session_id:
        raise RuntimeError("iFinD MCP initialize 响应缺少 mcp-session-id")

    # initialize 完成后发送 initialized 通知。
    _, initialized_body = mcp_post(
        url,
        headers,
        {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
        session_id=session_id,
    )
    if "error" in initialized_body:
        raise RuntimeError(f"iFinD MCP initialized 通知失败: {initialized_body['error']}")
    return session_id


def call_get_edb_data(
    url: str,
    headers: dict[str, str],
    session_id: str,
    start_date: date,
    end_date: date,
) -> dict[str, Any]:
    """按日期区间构造查询语句并调用 get_edb_data。"""

    query = f"查询GC001(加权平均)从{start_date:%Y-%m-%d}至{end_date:%Y-%m-%d}的日度数据"
    _, response_body = mcp_post(
        url,
        headers,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "get_edb_data", "arguments": {"query": query}},
        },
        session_id=session_id,
    )
    if "error" in response_body:
        raise RuntimeError(f"iFinD get_edb_data 失败: {response_body['error']}")
    return response_body


def parse_gc001_response(response_body: dict[str, Any]) -> pd.DataFrame:
    """解析 tools/call 响应中 content[0].text 里的 JSON 字符串。"""

    text = response_body["result"]["content"][0]["text"]
    inner = json.loads(text)
    data_block = inner["data"]["datas"][0]["data"]
    columns = data_block["columns"]
    attrs = data_block.get("attrs", {})
    rows = data_block["data"]

    if "日期" not in columns:
        raise RuntimeError(f"iFinD 返回表头缺少 日期: {columns}")
    value_column = GC001_FIELD if GC001_FIELD in columns else next(column for column in columns if column != "日期")

    # 校验指标 ID，确保 value_column 对应 GC001。
    index_id = attrs.get(value_column, {}).get("index_id")
    if index_id is not None and index_id != GC001_INDEX_ID:
        raise RuntimeError(f"GC001 指标 ID 不匹配: 期望 {GC001_INDEX_ID}, 实际 {index_id}")

    raw_df = pd.DataFrame(rows, columns=columns)
    output_df = pd.DataFrame(
        {
            "date": pd.to_datetime(raw_df["日期"]).dt.strftime("%Y-%m-%d"),
            "gc001_weighted_average_rate_pct": pd.to_numeric(raw_df[value_column], errors="coerce"),
        }
    )
    output_df = output_df.dropna(subset=["date"])
    output_df = output_df.drop_duplicates(subset=["date"], keep="last")
    output_df = output_df.sort_values("date").reset_index(drop=True)
    return output_df


def write_csv(df: pd.DataFrame, output_path: Path, backup: bool, dry_run: bool) -> None:
    """把 DataFrame 写成 date 和 gc001_weighted_average_rate_pct 两列。"""

    if dry_run:
        print(f"[dry-run] records={len(df)} output={output_path}")
        if len(df) > 0:
            print(f"[dry-run] range={df['date'].iloc[0]}:{df['date'].iloc[-1]}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if backup and output_path.exists():
        backup_dir = output_path.parent / "backup"
        backup_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(output_path, backup_dir / f"{output_path.stem}_{timestamp}{output_path.suffix}")

    temp_path = output_path.with_name(f"{output_path.name}.tmp")
    df.to_csv(temp_path, index=False, encoding="utf-8")
    os.replace(temp_path, output_path)
    print(f"wrote {output_path} rows={len(df)}")


def download_gc001(start_date: date, end_date: date, config_path: Path) -> pd.DataFrame:
    if end_date < start_date:
        raise ValueError("end-date 不能早于 start-date")

    url, headers = read_ifind_mcp_config(config_path)
    session_id = initialize_mcp_session(url, headers)
    response_body = call_get_edb_data(url, headers, session_id, start_date, end_date)
    return parse_gc001_response(response_body)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="通过 iFinD MCP 下载 GC001(加权平均)日度数据，并写出 CSV。"
    )
    parser.add_argument("--start-date", required=True, type=parse_yyyy_mm_dd, help="起始日期，格式 YYYY-MM-DD")
    parser.add_argument("--end-date", type=parse_yyyy_mm_dd, default=date.today(), help="结束日期，默认今天")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"输出 CSV，默认 {DEFAULT_OUTPUT}")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help=f"Codex config.toml，默认 {DEFAULT_CONFIG}")
    parser.add_argument("--no-backup", action="store_true", help="覆盖输出文件前不创建备份")
    parser.add_argument("--dry-run", action="store_true", help="只请求并解析数据，不写出 CSV")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    df = download_gc001(args.start_date, args.end_date, args.config)
    write_csv(df, args.output, backup=not args.no_backup, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
