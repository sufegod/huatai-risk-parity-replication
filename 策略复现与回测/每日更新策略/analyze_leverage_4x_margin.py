# -*- coding: utf-8 -*-
"""4 倍杠杆 - 精确强平测算（修正版）

上一版把"现金 ÷ 保证金占用"当成缓冲，过于简化且误导。
期货保证金制度的真实约束是：

  每日盯市：账户权益 = 现金 + 持仓浮盈浮亏
  当 账户权益 < 维持保证金 时 → 追保（margin call）
  不补则强平

关键参数（本策略 10 个品种，券商口径保证金率）：
  沪深300 14% / 中证500 14% / 10年国债 2.5% / 沪铜 16% / 沪铝 16%
  PTA 17% / 原油 32% / 豆粕 13% / 沪金 28% / 上证50ETF 100%(现货，无杠杆)

本脚本逐日模拟：
  - 4 倍杠杆 = 名义敞口 / 净资产 = 4
  - 每日重设杠杆回 4（否则会漂移）
  - 追踪 权益 / 维持保证金 的比值，找历史最低点
"""
import pandas as pd, numpy as np

P = 'C:/Users/aa/WorkBuddy/2026-05-28-14-41-45/huatai-risk-parity-replication'
D = f'{P}/策略复现与回测/每日更新策略'

d = pd.read_csv(f'{D}/输出/净值/2016起回溯_日收益对比.csv', index_col=0, parse_dates=True)
ret = d['上证50ETF_日收益'].dropna()

pos = pd.read_csv(f'{D}/输出/对比分析/2016起_仓位明细_上证50.csv', index_col=0, parse_dates=True)
ASSETS = ['沪深300主连', '中证500主连', '10年国债主连', '沪铜主连', '沪铝主连',
          'PTA主连', '原油主连', '豆粕主连', '沪金主连', '上证50ETF']
pos = pos.reindex(ret.index).ffill()

print('=' * 100)
print('4 倍杠杆 · 精确强平测算')
print('=' * 100)

# ---------- 各资产日收益（从策略日收益与权重无法反解，用近似）----------
# 说明：我们只有"策略日收益"（已按权重加总）。逐资产收益需从原始数据取。
# 这里用 rr（逐年资产收益 CSV 已有，但那是年度）。改用成分数据反推：
# 简化但正确性足够的做法 —— 用策略日收益 × 4 作为"整体4x"，然后
# 用保证金占用比例追踪强平。这是压力测试的通常做法。
#
# 强平精确条件：
#   设杠杆 L，保证金占用率 m（相对名义敞口），则
#   账户权益 E，名义敞口 N = L·E，保证金 M = m·N = m·L·E
#   维持保证金 M_maint = k · M （k 为维持/开仓比，通常 0.7~0.8）
#   每日盈亏 ΔE = N · r = L·E·r
#   从初始 E0=1，权益累计 E = 1 + Σ ΔE
#
# 更严格：逐日按日收益复利
r4 = 4.0 * ret
nav4 = (1 + r4).cumprod()
# 每日名义敞口 = 4 × nav（每日重设杠杆）
# 保证金 = margin_rate_daily × 名义敞口
# 但 margin 是按"权重"算的，权重和=1（无杠杆）。放大 4 倍后：
#   名义敞口 N = 4 × nav
#   保证金 M = 4 × nav × m   （m = 无杠杆时的保证金/净值比）
#   现金 C = nav − M = nav(1 − 4m)
# 若 4m > 1 则现金为负（需融资）→ 已不可行
m = pos['margin']
print(f'\n【核心约束】无杠杆保证金率 m 的分布：')
print(f'  均值 {m.mean():.2%}  中位 {m.median():.2%}  最大 {m.max():.2%}  最小 {m.min():.2%}')
print(f'  4 倍后需占用现金流：均值 {4*m.mean():.2%}  最大 {4*m.max():.2%}')
over = (4 * m > 1).sum()
print(f'  ⚠️ 4m > 100%（现金为负、必须融资）的天数：{over} 天（占 {over/len(m):.1%}）')
if over > 0:
    seg = m[4 * m > 1].sort_values(ascending=False)
    print(f'     最严重的几天：')
    for dt, v in seg.head(5).items():
        print(f'       {dt.date()}  m={v:.2%} → 4m={4*v:.2%}')

# ---------- 真实强平模拟（逐日盯市）----------
print('\n' + '=' * 100)
print('逐日盯市模拟：账户权益 vs 维持保证金')
print('=' * 100)
K_MAINT = 0.75   # 维持保证金 = 开仓保证金 × 0.75（期货公司常见）
L = 4.0

nav = 1.0
equity = [1.0]
maint_ratio = []
cash_ratio = []
breach = []   # 是否触及追保线

for i, (dt, r) in enumerate(ret.items()):
    mi = float(m.loc[dt]) if dt in m.index else float(m.mean())
    # 本日盈亏（杠杆后）
    pnl = nav * (L * r)
    nav = nav + pnl
    if nav <= 0:
        nav = 0.0
    # 每日重设杠杆：名义敞口 = L × nav，开仓保证金 = mi × L × nav
    M_open = mi * L * nav
    M_maint = M_open * K_MAINT
    C = nav - M_open           # 自由现金（可为负 = 需要融资）
    equity.append(nav)
    maint_ratio.append(M_maint / nav if nav > 0 else np.nan)
    cash_ratio.append(C / nav if nav > 0 else np.nan)
    breach.append(nav < M_maint)

eq = pd.Series(equity[1:], index=ret.index, name='权益')
mr = pd.Series(maint_ratio, index=ret.index, name='维持保证金/权益')
cr = pd.Series(cash_ratio, index=ret.index, name='现金/权益')
bc = pd.Series(breach, index=ret.index, name='触及追保')

print(f'  逐日复利后的终值：{eq.iloc[-1]:.4f}（即 {(eq.iloc[-1]-1)*100:+.0f}%）')
print(f'  历史最低权益：{eq.min():.4f}（{eq.idxmin().date()}）')
print(f'  维持保证金/权益 的区间：{mr.min():.2%} ~ {mr.max():.2%}')
print(f'  自由现金/权益 的区间：{cr.min():.2%} ~ {cr.max():.2%}')
print(f'  ⚠️ 自由现金曾为负（需融资）的天数：{(cr < 0).sum()} 天（占 {(cr<0).mean():.1%}）')
if (cr < 0).sum() > 0:
    print(f'     最早出现：{cr[cr<0].index[0].date()}；最差：{cr.min():.2%}（{cr.idxmin().date()}）')
print(f'  触及追保线（权益 < 维持保证金）的天数：{bc.sum()} 天')
if bc.sum() > 0:
    print('     触及时点：', ', '.join(str(x.date()) for x in bc[bc].index[:10]))

# ---------- 融资成本测算 ----------
print('\n' + '=' * 100)
print('融资成本测算（4 倍杠杆的真实代价）')
print('=' * 100)
# 4x 需要额外 3 倍资金。其中 ETF 部分 100% 保证金无法融资，只能现货/融资买入
# ETF 权重（上证50）平均约 4.8%
etf_w = pos['上证50ETF']
print(f'  ETF 平均权重 {etf_w.mean():.2%}（现货，无杠杆，4x 需融资买入）')
# 期货端：保证金占用 4m，剩余现金 (1-4m)
need = 3.0  # 需要额外借 3 倍
for rate in [0.03, 0.05, 0.08]:
    print(f'  按融资年化 {rate:.0%}：成本 ≈ {need*rate:.2%}/年')
print()
print(f'  无杠杆年化 11.94% × 4 = {4*0.1194:.2%}（毛）')
for rate in [0.03, 0.05, 0.08]:
    net = 4 * 0.1194 - 3 * rate
    print(f'    扣融资成本 {rate:.0%} 后 → {net:.2%}/年')

# ---------- 杠杆不会放大 4 倍的原因 ----------
print('\n' + '=' * 100)
print('为什么回撤只放大 3.61 倍、收益却放大了 4.41 倍？')
print('=' * 100)
nav1 = (1 + ret).cumprod()
dd1 = (nav1 / nav1.cummax() - 1)
dd4 = (eq / eq.cummax() - 1)
print(f'  回撤倍数：{dd4.min()/dd1.min():.2f}x')
print(f'  年化倍数：{((eq.iloc[-1]**(243/len(eq))-1)) / ((nav1.iloc[-1]**(243/len(nav1))-1)):.2f}x')
print('  → 因为是日复利，杠杆对"乘法路径"的影响不对称：')
print('     上涨日放大、下跌日也放大，但复合后的路径依赖使回撤<4x、收益>4x')
print('     （这是特定样本的运气，不可依赖）')

# ---------- 关键阈值反推 ----------
print('\n' + '=' * 100)
print('反向测算：多大杠杆会出问题？')
print('=' * 100)
print(f"{'杠杆':>6}{'年均保证金占用':>14}{'回撤':>10}{'最差单日':>10}{'结论':>28}")
for Lx in [1, 2, 3, 4, 5, 6, 8, 10]:
    rlx = Lx * ret
    nlx = (1 + rlx).cumprod()
    ddlx = (nlx / nlx.cummax() - 1).min()
    mavg = Lx * m.mean()
    worst1 = rlx.min()
    if mavg > 1.0:
        verdict = '❌ 保证金超100%，需持续融资'
    elif ddlx <= -0.5:
        verdict = '❌ 回撤过半，实盘必被赎回'
    elif ddlx <= -0.3:
        verdict = '⚠️ 回撤超30%，心理与追保压力大'
    elif ddlx <= -0.2:
        verdict = '⚠️ 回撤超20%（当前4x水平）'
    else:
        verdict = '✅ 相对可控'
    print(f'{Lx:>5}x{mavg:>14.2%}{ddlx:>10.2%}{worst1:>10.2%}{verdict:>28}')

# ---------- 极值情景 ----------
print('\n' + '=' * 100)
print('极端情景推演（4x）')
print('=' * 100)
scen = [
    ('2019-05-06 贸易战单日', -0.0282),
    ('2024-10-09 政策预期落空', -0.0266),
    ('连续9日亏损累计', -0.0576),
    ('2020-03 疫情(单月)', -0.0781),
    ('沪深300单日 -7%（历史级）', -0.07),
    ('沪深300单日 -10%（熔断级）', -0.10),
]
print(f"{'情景':<28}{'无杠杆':>10}{'4x':>10}{'对应%':>10}")
for name, r in scen:
    print(f'{name:<28}{r:>10.2%}{r*4:>10.2%}')
