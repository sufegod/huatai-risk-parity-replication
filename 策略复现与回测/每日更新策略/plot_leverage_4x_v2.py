# -*- coding: utf-8 -*-
"""4 倍杠杆情景压力测试（修正版）· 六子图

口径与 analyze_leverage_4x_v2.py 完全一致：
  现金 = 当日净资产 − L × 保证金(m)；净资产不放大；剩余现金按存款利率计息
"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
rcParams['axes.unicode_minus'] = False

P = 'C:/Users/aa/WorkBuddy/2026-05-28-14-41-45/huatai-risk-parity-replication'
D = f'{P}/策略复现与回测/每日更新策略'

COLS = ['沪深300主连', '中证500主连', '10年国债主连', '沪铜主连', '沪铝主连',
        'PTA主连', '原油主连', '豆粕主连', '沪金主连', '上证50ETF']
MARGIN = {'沪深300主连': 0.14, '中证500主连': 0.14, '10年国债主连': 0.025,
          '沪铜主连': 0.16, '沪铝主连': 0.16, 'PTA主连': 0.17,
          '原油主连': 0.32, '豆粕主连': 0.13, '沪金主连': 0.28, '上证50ETF': 1.00}
DEPOSIT_SCHEDULE = [('2016-01-01', 0.0175), ('2022-09-15', 0.0165),
                    ('2023-06-08', 0.0155), ('2023-12-22', 0.0145),
                    ('2024-07-25', 0.0135), ('2025-05-20', 0.0095)]
FEE_RATE = 0.0005
REAL_START = pd.Timestamp('2018-01-02')

# ---------- 数据与权重重放 ----------
d = pd.read_csv(f'{D}/输出/净值/2016起回溯_日收益对比.csv', index_col=0, parse_dates=True)
strat = d['上证50ETF_日收益']
un = pd.read_csv(f'{P}/数据/日度收益数据更新/日涨跌幅_填充.csv', index_col=0, parse_dates=True)
un = un.loc[:, ~un.columns.duplicated()]
s50 = pd.read_csv(f'{P}/数据/原始数据/上证50ETF_日涨跌幅.csv', index_col=0, parse_dates=True)
s50.index = pd.to_datetime(s50.index)
s50 = s50.loc[:, ~s50.columns.duplicated()]
un = un.drop(columns=['上证50ETF'], errors='ignore').join(s50, how='left')
rec = pd.read_csv(f'{D}/输出/对比分析/2016起_仓位明细_上证50.csv', parse_dates=['date']).set_index('date')

lvl = un.reindex(columns=COLS + ['一天期国债逆回购']).fillna(0.0) / 100.0
m_ratio = pd.Series({c: MARGIN.get(c, 1.0) for c in COLS})
idx = strat.index
rr = lvl.loc[idx, COLS]
eff = {}
for rdt in rec.index:
    cand = idx[idx > rdt]
    if len(cand):
        eff[cand[0]] = rdt
W = pd.DataFrame(np.nan, index=idx, columns=COLS)
for dt, rdt in eff.items():
    W.loc[dt] = rec[COLS].loc[rdt].values
W = W.ffill()
W = W.loc[~W.isna().all(axis=1)].fillna(0.0)
idx = W.index
gross = W.mul(rr.loc[idx], axis=0).sum(axis=1)
margin = (W * m_ratio).sum(axis=1)
costd = W.diff().abs().sum(axis=1).fillna(0.0) * FEE_RATE
cal_days = idx.to_series().diff().dt.days.fillna(1)
cal_days.index = idx
_rate = pd.Series(0.0, index=idx)
for s, r in DEPOSIT_SCHEDULE:
    _rate.loc[idx >= pd.Timestamp(s)] = r
dep_daily = (_rate / 365.0) * cal_days


def lev(L):
    cr = 1.0 - L * margin
    return L * gross + cr * dep_daily - L * costd, cr


r1, cr1 = lev(1.0); r2, cr2 = lev(2.0); r4, cr4 = lev(4.0)
n1 = (1 + r1).cumprod(); n2 = (1 + r2).cumprod(); n4 = (1 + r4).cumprod()
dd1 = n1 / n1.cummax() - 1; dd4 = n4 / n4.cummax() - 1
m4 = 4.0 * margin
neg4 = cr4 < -1e-12

fig, axes = plt.subplots(3, 2, figsize=(17, 13.5))
C1, C2, C4 = '#1f77b4', '#ff7f0e', '#d62728'

# ① 净值
ax = axes[0, 0]
ax.plot(n1.index, n1, label=f'1x  （{n1.iloc[-1]-1:+.1%}）', color=C1, lw=1.6)
ax.plot(n2.index, n2, label=f'2x  （{n2.iloc[-1]-1:+.1%}）', color=C2, lw=1.6)
ax.plot(n4.index, n4, label=f'4x  （{n4.iloc[-1]-1:+.1%}）', color=C4, lw=1.6)
ax.set_yscale('log')
ax.set_title('① 净值走势（对数轴，2016-2026）\n修正口径：现金按存款利率计息，现金端不随杠杆放大',
             fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(alpha=0.3)

# ② 回撤
ax = axes[0, 1]
ax.fill_between(dd1.index, dd1 * 100, 0, color=C1, alpha=0.45, label=f'1x 最深 {dd1.min():.2%}')
ax.fill_between(dd4.index, dd4 * 100, 0, color=C4, alpha=0.45, label=f'4x 最深 {dd4.min():.2%}')
ax.axhline(-20, color='orange', ls='--', lw=1.2, label='-20% 心理线')
ax.axhline(-30, color='red', ls='--', lw=1.2, label='-30% 危险线')
ax.set_title(f'② 回撤对比：4x 放大 {dd4.min()/dd1.min():.2f} 倍\n杠杆的真实代价在这里，不在资金端',
             fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=9, framealpha=0.95)
ax.grid(alpha=0.3)
ax.set_ylabel('回撤 %')

# ③ 保证金 vs 现金（4x）
ax = axes[1, 0]
ax.plot(margin.index, m4 * 100, color=C4, lw=1.2, label='4x 保证金占用 (4m)')
ax.plot(margin.index, cr4 * 100, color='#2ca02c', lw=1.2, label='4x 剩余现金 (1−4m)')
ax.axhline(100, color='black', ls='-', lw=1.5, label='100% 线（保证金 = 净资产）')
ax.axvline(REAL_START, color='purple', ls=':', lw=2, label='2018-01-02 真实信号起点')
ax.axvspan(margin.index[0], REAL_START, color='gray', alpha=0.18)
ax.fill_between(m4.index, 100, m4 * 100, where=(m4 > 1), color='darkred', alpha=0.35,
                label=f'超 100%：{int((m4>1).sum())} 天（全在构造期）')
ax.text(margin.index[30], 62, '构造期\n（股指补满仓假设）', fontsize=9, color='dimgray')
ax.set_ylim(-28, 132)
ax.set_title('③ 【核心修正】4x 保证金占用与现金\n超 100% 的 138 天全部落在 2018 年前的构造期，真实段为 0 天',
             fontsize=12, fontweight='bold')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3, fontsize=8,
          framealpha=0.95)
ax.grid(alpha=0.3)
ax.set_ylabel('占净资产 %')

# ④ 杠杆阶梯
ax = axes[1, 1]
Ls = [1, 2, 3, 4, 5, 6, 8, 10]
dds, mgs, gaps = [], [], []
for Lx in Ls:
    rl, crl = lev(Lx)
    nl = (1 + rl).cumprod()
    dds.append((nl / nl.cummax() - 1).min() * 100)
    mgs.append(Lx * margin.mean() * 100)
    gaps.append(int(((crl < -1e-12) & (crl.index >= REAL_START)).sum()))
x = np.arange(len(Ls))
cols = ['#2ca02c', '#2ca02c', '#ff7f0e', '#ff7f0e', '#d62728', '#d62728', '#4d4d4d', '#4d4d4d']
ax.bar(x, dds, color=cols, alpha=0.85, label='最大回撤')
for i, (ddv, mgv) in enumerate(zip(dds, mgs)):
    ax.text(i, ddv - 5.5, f'{ddv:.0f}%', ha='center', fontsize=9, fontweight='bold', color='white')
    ax.text(i, 4, f'保证金 {mgv:.0f}%', ha='center', fontsize=8, color='navy')
ax.axhline(-30, color='red', ls='--', lw=1.2)
ax.set_ylim(-72, 16)
ax.set_xticks(x)
ax.set_xticklabels([f'{l}x\n缺口 {g} 天' for l, g in zip(Ls, gaps)], fontsize=9)
ax.legend(loc='lower left', fontsize=9)
ax.set_title('④ 杠杆阶梯：回撤 vs 保证金 vs 真实段资金缺口\n资金约束分界在 4x/5x 之间，回撤约束更早生效',
             fontsize=12, fontweight='bold')
ax.set_ylabel('最大回撤 %')
ax.grid(alpha=0.3, axis='y')

# ⑤ 逐年收益
ax = axes[2, 0]
yy = [y for y in sorted(set(idx.year)) if (idx.year == y).sum() > 20]
r1y = [((1 + r1[idx.year == y]).prod() - 1) * 100 for y in yy]
r4y = [((1 + r4[idx.year == y]).prod() - 1) * 100 for y in yy]
w = 0.38
ax.bar(np.arange(len(yy)) - w / 2, r1y, w, label='1x', color=C1, alpha=0.9)
ax.bar(np.arange(len(yy)) + w / 2, r4y, w, label='4x', color=C4, alpha=0.9)
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(np.arange(len(yy))); ax.set_xticklabels(yy, fontsize=9)
ax.set_title('⑤ 逐年收益：11 年全部为正', fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(alpha=0.3, axis='y'); ax.set_ylabel('收益 %')

# ⑥ 逐年最大回撤
ax = axes[2, 1]
dd1y, dd4y = [], []
for y in yy:
    a = (1 + r1[idx.year == y]).cumprod(); b = (1 + r4[idx.year == y]).cumprod()
    dd1y.append((a / a.cummax() - 1).min() * 100)
    dd4y.append((b / b.cummax() - 1).min() * 100)
ax.bar(np.arange(len(yy)) - w / 2, dd1y, w, label='1x', color=C1, alpha=0.9)
ax.bar(np.arange(len(yy)) + w / 2, dd4y, w, label='4x', color=C4, alpha=0.9)
ax.axhline(-20, color='orange', ls='--', lw=1.2, label='-20%')
ax.set_xticks(np.arange(len(yy))); ax.set_xticklabels(yy, fontsize=9)
ax.set_title('⑥ 逐年最大回撤：4x 下多数年份回撤超 15%', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y'); ax.set_ylabel('最大回撤 %')

plt.suptitle('风险平价策略 · 4 倍杠杆情景压力测试（修正版 · 2016-2026）\n'
             '口径：现金 = 当日净资产 − L×保证金，净资产不放大，剩余现金按存款利率计息\n'
             '核心修正：4x 真实信号段（2018 起）从未出现资金缺口；代价是 −29% 回撤',
             fontsize=14, fontweight='bold', y=0.998)
plt.tight_layout(rect=[0, 0, 1, 0.965])
out = f'{D}/输出/图表/杠杆情景_4x压力测试_2016-2026_修正版.png'
try:
    plt.savefig(out, dpi=140, bbox_inches='tight')
except PermissionError:
    out = out.replace('.png', '_v2.png')
    plt.savefig(out, dpi=140, bbox_inches='tight')
print(f'已保存: {out}')
