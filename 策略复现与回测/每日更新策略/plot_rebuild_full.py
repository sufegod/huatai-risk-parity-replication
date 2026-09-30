# -*- coding: utf-8 -*-
"""策略完整回溯重制 · 图表（2026-09-29）

数据源：rebuild_full_backtest.py 导出的日度/逐年 CSV
产出两张图：
  A. 回测全景_现金管理与风险特征_2016-2026.png   —— 1x 口径（现金管理口径对比）
  B. 杠杆对比_1x至4x_2016-2026.png              —— 1x/2x/3x/4x 杠杆情景
"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
rcParams['axes.unicode_minus'] = False

D = 'C:/Users/aa/WorkBuddy/2026-05-28-14-41-45/huatai-risk-parity-replication/策略复现与回测/每日更新策略'
REAL_START = pd.Timestamp('2018-01-02')
CHART = f'{D}/输出/图表'

daily = pd.read_csv(f'{D}/输出/对比分析/回测全景_现金管理与杠杆_日度_2016-2026.csv',
                    index_col=0, parse_dates=True)
yr = pd.read_csv(f'{D}/输出/对比分析/回测全景_逐年_1x至4x_2016-2026.csv')

idx = daily.index
# 三条 1x 净值
nav_asset = (1 + daily['策略_不含现金']).cumprod()
nav_dep = (1 + daily['策略_存款口径']).cumprod()
nav_repo = (1 + daily['策略_GC001口径']).cumprod()
dd_asset = nav_asset / nav_asset.cummax() - 1
dd_dep = nav_dep / nav_dep.cummax() - 1
dd_repo = nav_repo / nav_repo.cummax() - 1

CA, CD, CR = '#7f7f7f', '#1f77b4', '#d62728'


def savefig(fig, name):
    out = f'{CHART}/{name}'
    try:
        fig.savefig(out, dpi=140, bbox_inches='tight')
    except PermissionError:
        out = out.replace('.png', '_v2.png')
        fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f'已保存: {out}')


# ============================ 图 A：现金管理与风险特征 ============================
fig, axes = plt.subplots(3, 2, figsize=(17, 14))

# ① 净值
ax = axes[0, 0]
ax.plot(idx, nav_asset, color=CA, lw=1.5, label=f'① 纯资产端（不含现金收益）{nav_asset.iloc[-1]-1:+.1%}')
ax.plot(idx, nav_dep, color=CD, lw=1.7, label=f'② +现金管理·银行存款 {nav_dep.iloc[-1]-1:+.1%}')
ax.plot(idx, nav_repo, color=CR, lw=1.7, label=f'③ +现金管理·GC001逆回购 {nav_repo.iloc[-1]-1:+.1%}')
ax.set_yscale('log')
ax.axvline(REAL_START, color='purple', ls=':', lw=1.6)
ax.axvspan(idx[0], REAL_START, color='gray', alpha=0.13)
ax.text(0.055, 0.42, '构造期\n(股指补满仓)', transform=ax.transAxes, fontsize=8.5,
        color='dimgray', va='center')
ax.set_title('① 历史净值：现金管理贡献了约 1/6 的累计收益\n（对数轴；灰底为 2016-2017 股指补满仓构造期）',
             fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(alpha=0.3, which='both')
ax.set_ylabel('净值（起点=1）')

# ② 回撤
ax = axes[0, 1]
ax.fill_between(idx, dd_asset * 100, 0, color=CA, alpha=0.35, label=f'① 资产端 最深 {dd_asset.min():.2%}')
ax.fill_between(idx, dd_dep * 100, 0, color=CD, alpha=0.50, label=f'② 含存款 最深 {dd_dep.min():.2%}')
ax.plot(idx, dd_repo * 100, color=CR, lw=1.3, label=f'③ 含GC001 最深 {dd_repo.min():.2%}')
ax.axvline(REAL_START, color='purple', ls=':', lw=1.6)
ax.axhline(-5, color='gray', ls='--', lw=1)
ax.set_title('② 回撤：现金管理不改变回撤形态\n三条曲线几乎重合 —— 现金收益是"稳定抬升"，不是"平滑波动"',
             fontsize=12, fontweight='bold')
ax.legend(loc='lower left', fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylabel('回撤 %')

# ③ 保证金占用比例
ax = axes[1, 0]
m = daily['margin'] * 100
ax.plot(idx, m, color='#2ca02c', lw=1.1, label='保证金占用 m（占净资产）')
ax.axhline(m.mean(), color='darkgreen', ls='--', lw=1.3, label=f'均值 {m.mean():.2f}%')
ax.axhline(m.max(), color='red', ls=':', lw=1.2, label=f'最大 {m.max():.2f}%（{m.idxmax().date()}）')
ax.fill_between(idx, 0, m, color='#2ca02c', alpha=0.18)
ax.axvline(REAL_START, color='purple', ls=':', lw=1.6)
ax.axvspan(idx[0], REAL_START, color='gray', alpha=0.13)
ax.set_title(f'③ 保证金占用比例：常年仅 {m.mean():.1f}%\n低占用 = 大量闲置资金 = 现金管理有真金白银的空间',
             fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylabel('占净资产 %')
ax.set_ylim(0, 34)

# ④ 逐年贡献分解
ax = axes[1, 1]
yy = yr['年份'].astype(int).values
cash_ann = yr['现金收益_存款'].values * 100
cost_ann = yr['交易成本'].values * 100
# 资产端年度贡献 = 1x收益 - 现金 + 成本
asset_ann = yr['1x收益'].values * 100 - cash_ann + cost_ann
x = np.arange(len(yy))
w = 0.62
ax.bar(x, asset_ann, w, label='资产端贡献', color='#1f77b4', alpha=0.9)
ax.bar(x, cash_ann, w, bottom=asset_ann, label='现金管理贡献（存款口径）', color='#2ca02c', alpha=0.9)
ax.bar(x, -cost_ann, w, bottom=asset_ann + cash_ann, label='交易成本', color='#d62728', alpha=0.9)
ax.plot(x, yr['1x收益'].values * 100, color='black', marker='o', ms=5, lw=1.3, label='合计年收益')
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(yy, fontsize=8.5)
ax.set_title('④ 逐年收益拆解：现金贡献稳定在 +0.6 ~ +1.5 pct\n利率下行令 2025-2026 的现金贡献明显收窄',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=8.5, loc='lower left')
ax.grid(alpha=0.3, axis='y')
ax.set_ylabel('收益 %')

# ⑤ 逐年三档口径
ax = axes[2, 0]
wid = 0.27
a1 = yr['1x收益'].values * 100 - cash_ann + cost_ann
a2 = yr['1x收益'].values * 100
a3 = a2 - cash_ann + yr['现金收益_GC001'].values * 100
ax.bar(x - wid, a1, wid, label='① 纯资产端', color=CA, alpha=0.9)
ax.bar(x, a2, wid, label='② 含存款现金', color=CD, alpha=0.9)
ax.bar(x + wid, a3, wid, label='③ 含GC001现金', color=CR, alpha=0.9)
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(yy, fontsize=8.5)
ax.set_title('⑤ 逐年收益：11 年全部为正\n现金管理每年把"资产端"再往上垫 1~3 个百分点',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')
ax.set_ylabel('收益 %')

# ⑥ 滚动一年夏普
ax = axes[2, 1]
roll = 252
for lab, r, c in [('① 纯资产端', daily['策略_不含现金'], CA),
                  ('② 含存款现金', daily['策略_存款口径'], CD),
                  ('③ 含GC001现金', daily['策略_GC001口径'], CR)]:
    s = (r.rolling(roll).mean() * 252) / (r.rolling(roll).std() * np.sqrt(252))
    ax.plot(idx, s, lw=1.2, color=c, label=lab)
ax.axhline(1, color='gray', ls='--', lw=1)
ax.axvline(REAL_START, color='purple', ls=':', lw=1.6)
ax.set_title('⑥ 滚动 252 日夏普：现金管理稳定抬升夏普约 0.2\n且从未把夏普从正拖到负',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='lower left')
ax.grid(alpha=0.3)
ax.set_ylabel('滚动夏普')

plt.suptitle('风险平价策略 · 加入现金管理收益后的完整回溯（2016-01-05 ~ 2026-09-28 · 2608 个交易日）\n'
             '现金口径：现金 = 净资产 − 保证金占用，按银行存款挂牌利率 / GC001 逆回购利率计息\n'
             '结论：保证金占用常年仅 16.6%，闲置资金贡献年均 +1.4（存款）~ +2.2（GC001）个百分点，'
             '且完全不改变回撤形态',
             fontsize=13.5, fontweight='bold', y=0.997)
plt.tight_layout(rect=[0, 0, 1, 0.962])
savefig(fig, '回测全景_现金管理与风险特征_2016-2026.png')
plt.close(fig)


# ============================ 图 B：1x~4x 杠杆对比 ============================
fig, axes = plt.subplots(3, 2, figsize=(17, 14))
LEVS = [1, 2, 3, 4]
COL = {1: '#1f77b4', 2: '#2ca02c', 3: '#ff7f0e', 4: '#d62728'}

# ① 净值走势
ax = axes[0, 0]
for L in LEVS:
    n = daily[f'{L}x_净值']
    ax.plot(idx, n, color=COL[L], lw=1.6, label=f'{L}x  {n.iloc[-1]-1:+.0%}')
ax.set_yscale('log')
ax.axvline(REAL_START, color='purple', ls=':', lw=1.6)
ax.axvspan(idx[0], REAL_START, color='gray', alpha=0.13)
ax.set_title('① 净值走势（对数轴）：4x 终值 +3752%\n倍数看似诱人，代价在回撤（见②）',
             fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(alpha=0.3, which='both')
ax.set_ylabel('净值（起点=1）')

# ② 回撤曲线
ax = axes[0, 1]
for L in LEVS:
    dd = daily[f'{L}x_回撤'] * 100
    ax.fill_between(idx, dd, 0, color=COL[L], alpha=0.32, label=f'{L}x 最深 {dd.min():.2f}%')
ax.axvline(REAL_START, color='purple', ls=':', lw=1.6)
ax.axhline(-20, color='orange', ls='--', lw=1.2, label='-20% 心理线')
ax.axhline(-30, color='red', ls='--', lw=1.2, label='-30% 危险线')
ax.set_title('② 回撤：4x 最大 -29.05%（1x 仅 -7.89%，放大 3.68 倍）\n杠杆的真实代价全在这里',
             fontsize=12, fontweight='bold')
ax.legend(loc='lower left', fontsize=8.5, ncol=2)
ax.grid(alpha=0.3)
ax.set_ylabel('回撤 %')

# ③ 逐年收益
ax = axes[1, 0]
wid = 0.2
for k, L in enumerate(LEVS):
    ax.bar(x + (k - 1.5) * wid, yr[f'{L}x收益'].values * 100, wid,
           label=f'{L}x', color=COL[L], alpha=0.9)
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(yy, fontsize=8.5)
ax.set_title('③ 逐年收益：4 档杠杆 11 年全部为正\n4x 的年度收益约是 1x 的 2.4~4.4 倍（弱市年份倍数最低）',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9, ncol=4)
ax.grid(alpha=0.3, axis='y')
ax.set_ylabel('收益 %')

# ④ 逐年最大回撤
ax = axes[1, 1]
for k, L in enumerate(LEVS):
    ax.bar(x + (k - 1.5) * wid, yr[f'{L}x回撤'].values * 100, wid,
           label=f'{L}x', color=COL[L], alpha=0.9)
ax.axhline(-20, color='orange', ls='--', lw=1.2, label='-20%')
ax.axhline(-30, color='red', ls='--', lw=1.2, label='-30%')
ax.set_xticks(x)
ax.set_xticklabels(yy, fontsize=8.5)
ax.set_title('④ 逐年最大回撤：4x 下多数年份回撤 10~29%\n2020 年是 4x 最难受的一年（-29.05%）',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=8.5, ncol=2)
ax.grid(alpha=0.3, axis='y')
ax.set_ylabel('最大回撤 %')

# ⑤ 保证金占用 vs 现金比例
ax = axes[2, 0]
mgm = [daily[f'{L}x_保证金'].mean() * 100 for L in LEVS]
mgx = [daily[f'{L}x_保证金'].max() * 100 for L in LEVS]
csh = [daily[f'{L}x_现金比例'].mean() * 100 for L in LEVS]
xt = np.arange(len(LEVS))
ax.bar(xt - 0.2, mgm, 0.38, label='保证金占用·均值', color='#8c564b', alpha=0.9)
ax.bar(xt + 0.2, csh, 0.38, label='剩余现金·均值', color='#2ca02c', alpha=0.9)
ax.plot(xt, mgx, color='red', marker='D', ls='--', ms=7, lw=1.4, label='保证金占用·历史最大（含构造期）')
ax.axhline(100, color='black', ls='-', lw=1.4, label='100% 线（保证金=净资产）')
for i, (a, b) in enumerate(zip(mgm, csh)):
    ax.text(xt[i] - 0.2, a + 2.5, f'{a:.1f}%', ha='center', fontsize=9, fontweight='bold')
    ax.text(xt[i] + 0.2, b + 2.5, f'{b:.1f}%', ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(xt)
ax.set_xticklabels([f'{L}x' for L in LEVS], fontsize=10)
ax.set_ylim(0, 125)
ax.set_title('⑤ 保证金占用与剩余现金：杠杆吃掉的不是本金，是利息\n只有 4x 在构造期出现过 138 天资金缺口（真实段 0 天）',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=8.5, loc='upper center', ncol=2)
ax.grid(alpha=0.3, axis='y')
ax.set_ylabel('占净资产 %')

# ⑥ 风险收益定位
ax = axes[2, 1]
anns, mdds = [], []
for L in LEVS:
    r = daily[f'{L}x_日收益']
    nav = (1 + r).cumprod()
    anns.append((nav.iloc[-1] ** (252 / len(r)) - 1) * 100)
    mdds.append(((nav / nav.cummax()) - 1).min() * 100)
ax.plot(mdds, anns, 'o-', color='#333333', lw=1.6, ms=9, zorder=3)
for L, xs, ys in zip(LEVS, mdds, anns):
    ax.annotate(f'{L}x\n年化{ys:.1f}%\n回撤{xs:.1f}%', (xs, ys), textcoords='offset points',
                xytext=(12, -12), fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc=COL[L], alpha=0.22, ec=COL[L]))
ax.axhline(0, color='black', lw=0.8)
ax.set_xlim(-40, 2)
ax.set_ylim(0, 52)
ax.set_title('⑥ 风险收益定位：年化随杠杆近线性上升，\n回撤也近线性放大 —— 夏普只从 1.59 微降到 1.41',
             fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)
ax.set_xlabel('最大回撤 %')
ax.set_ylabel('年化收益 %')

plt.suptitle('风险平价策略 · 1x / 2x / 3x / 4x 杠杆情景（2016-01-05 ~ 2026-09-28）\n'
             '口径：现金 = 当日净资产 − L×保证金规模；净资产不放大；剩余现金按银行存款利率计息\n'
             '核心：杠杆不制造资金缺口（真实信号段），它只把回撤等比放大，并把年年为正变成"年年为正但要扛得住"',
             fontsize=13.5, fontweight='bold', y=0.997)
plt.tight_layout(rect=[0, 0, 1, 0.962])
savefig(fig, '杠杆对比_1x至4x_2016-2026.png')
plt.close(fig)
print('完成。')
