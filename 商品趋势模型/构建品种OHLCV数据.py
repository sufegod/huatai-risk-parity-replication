# -*- coding: utf-8 -*-
"""
构建《商品趋势模型》品种 OHLCV 数据 Excel。
- 数据来源：新浪财经主力连续日线（akshare.futures_main_sina）+ ETF 日线（akshare.fund_etf_hist_em）
- 同时并入全天候策略原有的 JYDB 前复权收盘价作为对照列，确保"全天候品种数据"也在表中。
- 严格隔离：仅读取原数据，结果写入独立文件夹 商品趋势模型/，绝不改动原全天候数据文件。
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import akshare as ak
import pandas as pd

ROOT = Path(r"C:/Users/aa/WorkBuddy/2026-05-28-14-41-45/huatai-risk-parity-replication")
OUTDIR = ROOT / "商品趋势模型"
OUTDIR.mkdir(parents=True, exist_ok=True)
JYDB_CSV = ROOT / "数据" / "日度收益数据更新" / "期货主力前复权收盘价.csv"
TODAY = dt.date.today().strftime("%Y%m%d")

# 全天候 v0.19 策略的 10 个品种及其新浪主连符号 / ETF 代码
FUTURES = {
    "沪深300主连": "IF0",
    "中证500主连": "IC0",
    "10年国债主连": "T0",
    "沪铜主连": "CU0",
    "PTA主连": "TA0",
    "原油主连": "SC0",
    "豆粕主连": "M0",
    "沪金主连": "AU0",
}
ETF = {"红利低波ETF": "512890"}  # 华泰柏瑞中证红利低波动ETF（红利低波ETF）

ALL_ASSETS = {**FUTURES, **ETF}


def load_jydb() -> pd.DataFrame:
    df = pd.read_csv(JYDB_CSV, encoding="utf-8-sig")
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_futures_ohlc(symbol: str) -> pd.DataFrame:
    df = ak.futures_main_sina(symbol=symbol)
    df = df.rename(
        columns={
            "日期": "date",
            "开盘价": "开盘价",
            "最高价": "最高价",
            "最低价": "最低价",
            "收盘价": "收盘价_新浪主连",
            "成交量": "成交量",
            "持仓量": "持仓量",
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    cols = ["date", "开盘价", "最高价", "最低价", "收盘价_新浪主连", "成交量", "持仓量"]
    return df[[c for c in cols if c in df.columns]]


def fetch_etf_ohlc(code: str) -> pd.DataFrame:
    # 新浪 ETF 历史接口需带市场前缀 sh/sz；返回英文列名（date/open/high/low/close/volume/amount）
    df = ak.fund_etf_hist_sina(symbol="sh" + code)
    df = df.rename(
        columns={
            "date": "date",
            "open": "开盘价",
            "high": "最高价",
            "low": "最低价",
            "close": "收盘价_新浪主连",
            "volume": "成交量",
            "amount": "成交额",
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    cols = ["date", "开盘价", "最高价", "最低价", "收盘价_新浪主连", "成交量", "成交额"]
    return df[[c for c in cols if c in df.columns]]


def build_sheet(jydb: pd.DataFrame, name: str, sina_df: pd.DataFrame) -> pd.DataFrame:
    if name in jydb.columns:
        # 期货：并入全天候原 JYDB 前复权收盘价作为对照列
        base = jydb[["date", name]].rename(columns={name: "前复权收盘价_JYDB"})
        merged = pd.merge(base, sina_df, on="date", how="outer").sort_values("date").reset_index(drop=True)
    else:
        # ETF 不在 JYDB 期货表中，仅以新浪 OHLCV 为准（无 JYDB 对照列，不产生空行）
        merged = sina_df.copy()
    # 整数化成交量更直观
    for c in ("成交量", "持仓量", "成交额"):
        if c in merged.columns:
            merged[c] = merged[c].astype("float")
    return merged


def main() -> None:
    jydb = load_jydb()
    coverage = []
    sheets: dict[str, pd.DataFrame] = {}

    for name, sym in FUTURES.items():
        print(f"抓取期货主连 OHLCV: {name} ({sym}) ...")
        sina = fetch_futures_ohlc(sym)
        sheet = build_sheet(jydb, name, sina)
        sheets[name] = sheet
        coverage.append({
            "品种": name, "类型": "期货主连", "新浪符号": sym,
            "OHLCV起始": sina["date"].min().date(), "OHLCV截止": sina["date"].max().date(),
            "JYDB收盘起始": (jydb.loc[jydb[name].notna(), "date"].min() if jydb[name].notna().any() else pd.NaT),
            "行数": len(sheet),
            "备注": "",
        })

    for name, code in ETF.items():
        print(f"抓取 ETF 日线: {name} ({code}) ...")
        sina = fetch_etf_ohlc(code)
        sheet = build_sheet(jydb, name, sina)
        sheets[name] = sheet
        coverage.append({
            "品种": name, "类型": "ETF", "新浪符号": code,
            "OHLCV起始": sina["date"].min().date(), "OHLCV截止": sina["date"].max().date(),
            "JYDB收盘起始": (jydb.loc[jydb[name].notna(), "date"].min() if name in jydb.columns and jydb[name].notna().any() else pd.NaT),
            "行数": len(sheet),
            "备注": "",
        })

    cov_df = pd.DataFrame(coverage)

    # 说明表
    notes = pd.DataFrame({
        "项目": [
            "生成时间", "数据用途", "OHLCV 来源", "全天候对照列来源",
            "隔离声明", "字段说明", "缺口说明-商品", "缺口说明-金融/ETF",
            "原油主连", "股指期货/国债主连", "红利低波ETF",
        ],
        "内容": [
            dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "商品趋势模型（原全天候策略品种，独立副本）",
            "新浪财经主力连续日线，经 akshare 获取（开盘价/最高价/最低价/收盘价/成交量/持仓量）",
            "原全天候策略 JYDB 期货主力前复权收盘价.csv（仅读取，未改动）",
            "本工作簿位于 商品趋势模型/ ，与 策略复现与回测/ 物理隔离；原全天候数据文件未被任何修改、覆盖或删除。",
            "收盘价_新浪主连=新浪主力连续；前复权收盘价_JYDB=全天候原数据对照；成交量/持仓量单位依新浪原始口径。",
            "沪铜/豆粕(2005)、PTA(2006)、沪金(2008) 均完整覆盖 2013 年至今。",
            "沪深300/中证500/10年国债主连在新浪仅自 2017-01-17 起有 OHLCV；红利低波ETF 自 2019-01 起。",
            "原油期货 2018-03 才上市，2013–2018 无数据（合约不存在）。",
            "2013–2016 区间仅有 JYDB 前复权收盘价（无开高低量）。如需该区间 OHLC，可用沪深300/中证500指数日线替代，可另行补充。",
            "ETF 2018 年底成立，2019-01 前无数据。",
        ],
    })

    # 目录表
    toc = pd.DataFrame({
        "工作表": [f"{n}" for n in sheets.keys()] + ["说明", "数据覆盖"],
        "说明": ([f"{n} 日线 OHLCV" for n in sheets.keys()]
                 + ["数据来源与字段、缺口说明", "各品种 OHLCV 起止与行数"]),
    })

    out_path = OUTDIR / f"商品趋势模型_品种数据_{TODAY}.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl", datetime_format="yyyy-mm-dd") as xw:
        toc.to_excel(xw, sheet_name="目录", index=False)
        notes.to_excel(xw, sheet_name="说明", index=False)
        cov_df.to_excel(xw, sheet_name="数据覆盖", index=False)
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)

    print(f"\n已生成: {out_path}")
    print(cov_df[["品种", "类型", "OHLCV起始", "JYDB收盘起始", "行数"]].to_string(index=False))


if __name__ == "__main__":
    main()
