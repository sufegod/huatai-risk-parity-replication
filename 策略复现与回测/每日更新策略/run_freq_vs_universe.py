# -*- coding: utf-8 -*-
"""品种 vs 调仓频率：哪个对全天候（风险平价）策略更重要。
两个维度各跑一组同口径回测（复用 daily_update_strategy.build_backtest_result）：
- 频率维度（资产集固定 = v0.20 完整 6 资产）：日频 / 周频 / 月频 / 季频
- 品种维度（频率固定 = 日频）：完整 v0.20 / +沪铝+PTA(=v0.19) / 残缺股债版 / 无债版
所有情景在共同窗口（各情景最晚起点 → 数据末日）上用同一套指标口径重算，保证可比。
换手率由日度仓位明细的权重变动绝对值合计年化得到。
"""
import importlib.util
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DUS_PATH = HERE / "daily_update_strategy.py"
V20_PATH = REPO / "策略复现与回测" / "策略代码" / "资产风险平价策略0.20（剔除沪铝PTA）.py"
OUT_CSV = HERE / "输出" / "指标" / "品种vs调仓频率对照_2026-10-13.csv"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


dus = load_module("dus_mod", DUS_PATH)
v20_src = V20_PATH.read_text(encoding="utf-8")
base_strat = dus.load_strategy_module(V20_PATH)
base_classes = {k: list(v) for k, v in base_strat.RISK_PARITY_ASSET_CLASSES.items()}
STRAT_NAME = base_strat.STRATEGY_NAME

V19_CLASSES = {
    "股票": ["红利低波ETF"],
    "债券": ["10年国债主连"],
    "商品": ["沪铜主连", "沪铝主连", "PTA主连", "原油主连", "豆粕主连"],
    "黄金": ["沪金主连"],
}
STOCK_BOND_ONLY = {"股票": ["红利低波ETF"], "债券": ["10年国债主连"]}
NO_BOND = {k: v for k, v in base_classes.items() if k != "债券"}


def make_block(classes):
    lines = ["RISK_PARITY_ASSET_CLASSES = {"]
    for k, v in classes.items():
        if v:
            lines.append(f"    '{k}': {v},")
    lines.append("}")
    return "\n".join(lines)


def variant_src(classes):
    return re.sub(
        r"RISK_PARITY_ASSET_CLASSES = \{.*?\n\}",
        make_block(classes),
        v20_src,
        count=1,
        flags=re.S,
    )


def make_obs_getter(freq):
    """返回观察日（调仓日）选择函数：daily=每日；否则取每周/月/季最后一个交易日。"""

    def getter(index):
        idx = pd.DatetimeIndex(index)
        if freq == "daily":
            return idx
        p = idx.to_period({"weekly": "W", "monthly": "M", "quarterly": "Q"}[freq])
        s = pd.Series(idx, index=idx)
        last = s.groupby(p).last()
        return pd.DatetimeIndex(last.values)

    return getter


def window_metrics(nav: pd.Series, start):
    nav = nav.dropna()
    nav = nav.loc[nav.index >= start]
    ret = nav.pct_change().dropna()
    n = len(ret)
    ann = nav.iloc[-1] ** (252 / n) - 1
    vol = float(ret.std() * np.sqrt(252))
    sharpe = float(ret.mean() * 252 / vol) if vol > 0 else np.nan
    mdd = float((nav / nav.cummax() - 1).min())
    win = float((ret > 0).mean())
    return ann, vol, sharpe, mdd, win


# (维度, 情景标签, 资产类, 频率)
scenarios = [
    ("调仓频率", "日频（生产口径）", base_classes, "daily"),
    ("调仓频率", "周频", base_classes, "weekly"),
    ("调仓频率", "月频", base_classes, "monthly"),
    ("调仓频率", "季频", base_classes, "quarterly"),
    ("品种选择", "完整 v0.20（6资产）", base_classes, "daily"),
    ("品种选择", "+沪铝+PTA（=v0.19，8资产）", V19_CLASSES, "daily"),
    ("品种选择", "残缺：仅股票+债券", STOCK_BOND_ONLY, "daily"),
    ("品种选择", "残缺：无债券（去压舱石）", NO_BOND, "daily"),
]

rows = []
for dim, label, classes, freq in scenarios:
    tmp_path = None
    if classes == base_classes:
        dus.STRATEGY_SCRIPT = V20_PATH
    else:
        src2 = variant_src(classes)
        safe = re.sub(r"[^\w]", "", label)
        tmp_path = V20_PATH.parent / f"_fv_tmp_{safe}.py"
        tmp_path.write_text(src2, encoding="utf-8")
        dus.STRATEGY_SCRIPT = tmp_path
    orig_getter = dus.get_observation_dates
    dus.get_observation_dates = make_obs_getter(freq)
    try:
        bt = dus.build_backtest_result()
        nav = bt.df_navs[STRAT_NAME]
        n_assets = len([a for a in bt.assets if a not in base_strat.INDEX_FUTURES])
        dfw = bt.df_daily_weights
        wcols = [c for c in bt.assets if c in dfw.columns]
        turnover_total = dfw[wcols].diff().abs().sum(axis=1).sum()
        years = (nav.index[-1] - nav.index[0]).days / 365.25
        rows.append(
            {
                "维度": dim,
                "情景": label,
                "非股指资产数": n_assets,
                "频率": freq,
                "回测起点": nav.index[0].date(),
                "回测终点": nav.index[-1].date(),
                "年化换手": turnover_total / years,
                "_nav": nav,
            }
        )
        print(f"[done] {dim} | {label}: 起点={nav.index[0].date()} 资产数={n_assets}", flush=True)
    finally:
        dus.get_observation_dates = orig_getter
        if tmp_path is not None and tmp_path.exists():
            os.remove(tmp_path)

common_start = max(r["_nav"].index[0] for r in rows)
print(f"\n共同对比窗口起点: {common_start.date()}", flush=True)

for r in rows:
    ann, vol, sharpe, mdd, win = window_metrics(r.pop("_nav"), common_start)
    r.update({"年化收益": ann, "年化波动": vol, "夏普比率": sharpe, "最大回撤": mdd, "日度胜率": win})

df = pd.DataFrame(rows)
df = df[["维度", "情景", "非股指资产数", "频率", "回测起点", "回测终点", "年化收益", "年化波动", "夏普比率", "最大回撤", "日度胜率", "年化换手"]]

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig", float_format="%.6f")

base_daily = df[(df["维度"] == "调仓频率") & (df["情景"].str.startswith("日频"))].iloc[0]
base_full = df[(df["维度"] == "品种选择") & (df["情景"].str.startswith("完整"))].iloc[0]

print("\n===== 品种 vs 调仓频率 对照（共同窗口同口径）=====")
for _, r in df.iterrows():
    print(
        f"{r['维度']} | {r['情景']:24s} 夏普={r['夏普比率']:.2f}  年化={r['年化收益']:+.2%}  "
        f"波动={r['年化波动']:.2%}  回撤={r['最大回撤']:.2%}  胜率={r['日度胜率']:.2%}  年化换手={r['年化换手']:.1f}x"
    )

print("\n===== 效应幅度对比（夏普）=====")
f_sharpes = df[df["维度"] == "调仓频率"]["夏普比率"]
s_sharpes = df[df["维度"] == "品种选择"]["夏普比率"]
print(f"调仓频率维度：夏普区间 {f_sharpes.min():.2f} ~ {f_sharpes.max():.2f}，极差 {f_sharpes.max()-f_sharpes.min():.2f}")
print(f"品种选择维度：夏普区间 {s_sharpes.min():.2f} ~ {s_sharpes.max():.2f}，极差 {s_sharpes.max()-s_sharpes.min():.2f}")
f_mdd = df[df["维度"] == "调仓频率"]["最大回撤"]
s_mdd = df[df["维度"] == "品种选择"]["最大回撤"]
print(f"调仓频率维度：回撤区间 {f_mdd.min():.2%} ~ {f_mdd.max():.2%}")
print(f"品种选择维度：回撤区间 {s_mdd.min():.2%} ~ {s_mdd.max():.2%}")
print(f"\n对照表已保存: {OUT_CSV}")
