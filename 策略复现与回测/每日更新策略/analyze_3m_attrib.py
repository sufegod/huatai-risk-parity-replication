# -*- coding: utf-8 -*-
"""近三个月策略收益按品种归因（贡献度分解）。
贡献口径：第 i 个品种在交易日 t 的收益贡献 = 持有权重 w_i,t × 品种日收益 r_i,t，
区间累计 = Σ_t (w_i,t * r_i,t)，所有品种合计 ≈ 策略区间日收益之和。
"""
import pandas as pd
import numpy as np

BASE = r"C:\Users\aa\WorkBuddy\2026-05-28-14-41-45\huatai-risk-parity-replication"
RET_F = f"{BASE}\\数据\\日度收益数据更新\\日涨跌幅_填充.csv"
WGT_F = f"{BASE}\\策略复现与回测\\每日更新策略\\输出\\仓位明细\\策略日度仓位明细_2026-10-08.csv"
NAV_F = f"{BASE}\\策略复现与回测\\每日更新策略\\输出\\净值\\策略每日净值走势_2026-10-08.csv"

# 策略 10 个品种（资产层，不含股指信号覆盖外的替代品种、不含 GC001 现金）
ASSETS = ["沪深300主连", "中证500主连", "红利低波ETF", "10年国债主连",
          "沪铜主连", "沪铝主连", "PTA主连", "原油主连", "豆粕主连", "沪金主连"]

# 1) 收益
ret = pd.read_csv(RET_F, parse_dates=["日期"], encoding="utf-8-sig")
ret = ret.sort_values("日期").set_index("日期")
# 收益率文件存的是"百分比数值"(0.0438 => 0.0438%)，转成小数分数再计算
ret = ret[ASSETS].ffill() / 100.0

# 2) 权重（日度持有权重，ffill 补齐到最新一日）
wgt = pd.read_csv(WGT_F, parse_dates=["date"], encoding="utf-8-sig")
wgt = wgt.sort_values("date").set_index("date")
wcols = [c for c in wgt.columns if c in ASSETS]
wgt = wgt[wcols].ffill()

# 3) 净值
nav = pd.read_csv(NAV_F, index_col=0, parse_dates=True, encoding="utf-8-sig")
nav = nav.sort_index()
nav_col = nav.columns[0]
nav_series = nav[nav_col]

# 对齐区间：最近三个月（自然日 2026-07-08 -> 2026-10-08）
END = pd.Timestamp("2026-10-08")
START = pd.Timestamp("2026-07-08")
idx = ret.index[(ret.index >= START) & (ret.index <= END)]
idx = idx.union([END]) if END not in idx else idx

r = ret.loc[idx, ASSETS]
w = wgt.reindex(idx).ffill().loc[idx, ASSETS]

# 逐品种贡献（权重 × 收益），区间累计
contrib = (w * r).sum(axis=0)  # Series: 品种 -> 累计贡献(pct, 小数)

# 策略区间收益：两种口径
nav_start = nav_series.asof(START)
nav_end = nav_series.asof(END)
geo_ret = nav_end / nav_start - 1.0          # 几何（净值法）
daily = (w * r).sum(axis=1)                   # 每日策略收益(算术加和)
arith_ret = daily.sum()                       # 算术累计

print("=" * 60)
print(f"区间 {START.date()} ~ {END.date()}  共 {len(idx)} 个交易日")
print(f"净值法几何收益 : {geo_ret*100:+.4f}%")
print(f"日收益算术累计 : {arith_ret*100:+.4f}%")
print(f"贡献合计(Σ w*r): {contrib.sum()*100:+.4f}%")
print("=" * 60)
print("\n各品种收益贡献（由大到小，单位 %）：")
contrib_pct = (contrib * 100).sort_values(ascending=False)
for a, v in contrib_pct.items():
    tag = "  <<< 负贡献(亏损来源)" if v < 0 else ""
    print(f"  {a:>12s} : {v:+.4f}%{tag}")

neg = contrib_pct[contrib_pct < 0]
print("\n负贡献品种合计 : {:.4f}%".format(neg.sum()))
if len(neg):
    worst = neg.idxmin()
    print(f"最大亏损来源   : {worst}  贡献 {neg.min()*100:+.4f}%  "
          f"(占全部负贡献 {neg.min()/neg.sum()*100:.1f}%)")
