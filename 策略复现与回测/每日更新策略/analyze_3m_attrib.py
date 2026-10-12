# -*- coding: utf-8 -*-
"""近三个月策略收益按品种归因（价格口径，与策略实际实现一致）。
贡献近似 = 区间平均权重 × 品种区间几何涨跌。
价格源：期货/国债/金/商品 用 期货主力前复权收盘价.csv；
红利低波ETF 用 增量缓存/红利低波ETF行情.csv(ClosePrice)。
权重用 策略日度仓位明细 的区间均值。
"""
import pandas as pd
import numpy as np

BASE = r"C:\Users\aa\WorkBuddy\2026-05-28-14-41-45\huatai-risk-parity-replication"
FUT = f"{BASE}\\数据\\日度收益数据更新\\期货主力前复权收盘价.csv"
ETF = f"{BASE}\\数据\\日度收益数据更新\\增量缓存\\红利低波ETF行情.csv"
WGT = f"{BASE}\\策略复现与回测\\每日更新策略\\输出\\仓位明细\\策略日度仓位明细_2026-10-08.csv"
NAV = f"{BASE}\\策略复现与回测\\每日更新策略\\输出\\净值\\策略每日净值走势_2026-10-08.csv"

S = pd.Timestamp("2026-07-08")
E = pd.Timestamp("2026-10-08")

fut = pd.read_csv(FUT, index_col=0, parse_dates=True, encoding="utf-8-sig").sort_index()
etf = pd.read_csv(ETF, index_col=0, parse_dates=True, encoding="utf-8-sig").sort_index()
wgt = pd.read_csv(WGT, index_col=0, parse_dates=True, encoding="utf-8-sig").sort_index()
nav = pd.read_csv(NAV, index_col=0, parse_dates=True, encoding="utf-8-sig").sort_index()
nav_c = nav.columns[0]

# 9 个品种来自前复权价；红利低波ETF 来自 ETF 文件
PX = {
    "沪深300主连": fut["沪深300主连"], "10年国债主连": fut["10年国债主连"],
    "沪金主连": fut["沪金主连"], "豆粕主连": fut["豆粕主连"],
    "中证500主连": fut["中证500主连"], "沪铜主连": fut["沪铜主连"],
    "沪铝主连": fut["沪铝主连"], "PTA主连": fut["PTA主连"], "原油主连": fut["原油主连"],
    "红利低波ETF": etf["ClosePrice"],
}

A = list(PX.keys())
rows = []
for a in A:
    s = PX[a]
    p0, p1 = s.asof(S), s.asof(E)
    ret = p1 / p0 - 1.0
    w = wgt[a].loc[(wgt.index >= S) & (wgt.index <= E)].mean()
    rows.append((a, ret, w, ret * w))

df = pd.DataFrame(rows, columns=["asset", "ret", "avg_w", "contrib"]).sort_values("contrib")

# 净值实际区间收益
ns = nav.loc[S:E, nav_c]
geo_nav = ns.iloc[-1] / ns.iloc[0] - 1.0

print(f"区间 {S.date()} ~ {E.date()}  价格涨跌幅 × 区间平均权重 的贡献分解")
print(f"{'资产':>12s} {'区间涨跌':>10s} {'平均权重':>9s} {'贡献(pct)':>10s}")
for _, r in df.iterrows():
    flag = "  <<< 亏损来源" if r.contrib < 0 else ""
    print(f"{r.asset:>12s} {r.ret*100:>+9.2f}% {r.avg_w*100:>8.2f}% {r.contrib*100:>+9.3f}%{flag}")
asset_sum = df.contrib.sum()
print(f"{'资产端合计':>12s} {'':>10s} {'':>9s} {asset_sum*100:>+9.3f}%")
print(f"\n策略净值区间几何收益(含现金管理): {geo_nav*100:+.3f}%")
neg = df[df.contrib < 0]
print(f"负贡献(亏损来源)合计: {neg.contrib.sum()*100:+.3f}%")
print("亏损来源明细(由大到小):")
for _, r in neg.iterrows():
    print(f"  {r.asset:>12s} 贡献 {r.contrib*100:+.3f}%  占全部负贡献 {r.contrib/neg.contrib.sum()*100:5.1f}%")
print(f"  => 最大单一亏损来源: {neg.asset.iloc[0]} ({neg.contrib.iloc[0]*100:+.3f}%)")
