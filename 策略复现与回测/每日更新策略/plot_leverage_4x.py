# -*- coding: utf-8 -*-
"""4 倍杠杆风险可视化"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
rcParams['axes.unicode_minus'] = False

P = 'C:/Users/aa/WorkBuddy/2026-05-28-14-41-45/huatai-risk-parity-replication'
D = f'{P}/策略复现与回测/每日更新策略'

d = pd.read_csv(f'{D}/输出/净值/2016起回溯_日收益对比.csv', index_col=0, parse_dates=True)
ret = d['上证50ETF_日收益'].dropna()
pos = pd.read_csv(f'{D}/输出/对比分析/2016起_仓位明细_上证50.csv', index_col=0, parse_dates=True)
pos = pos.reindex(ret.index).ffill()
m = pos['margin']

L = 4.0
r4 = L * ret
nav1 = (1 + ret).cumprod()
nav4 = (1 + r4).cumprod()
dd1 = nav1 / nav1.cummax() - 1
dd4 = nav4 / nav4.cummax() - 1

fig, axes = plt.subplots(3, 2, figsize=(17, 13))

# ① 净值对比（对数）
ax = axes[0, 0]
ax.plot(nav1.index, nav1, label='无杠杆（1x）', color='#1f77b4', lw=1.6)
ax.plot(nav4.index, nav4, label='4x 杠杆', color='#d62728', lw=1.6)
ax.set_yscale('log')
ax.set_title('① 净值走势（对数，2016-2026）\n4x 终值 +9255% vs 无杠杆 +235%', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(alpha=0.3)

# ② 回撤对比
ax = axes[0, 1]
ax.fill_between(dd1.index, dd1 * 100, 0, color='#1f77b4', alpha=0.45, label='无杠杆')
ax.fill_between(dd4.index, dd4 * 100, 0, color='#d62728', alpha=0.45, label='4x')
ax.axhline(-20, color='orange', ls='--', lw=1.2, label='-20% 心理线')
ax.axhline(-30, color='red', ls='--', lw=1.2, label='-30% 危险线')
ax.set_title(f'② 回撤对比\n无杠杆 {dd1.min():.2%} → 4x {dd4.min():.2%}（放大 3.61 倍）',
             fontsize=12, fontweight='bold')
ax.legend(loc='lower left', fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylabel('回撤 %')

# ③ 保证金占用（无杠杆 vs 4x）
ax = axes[1, 0]
ax.plot(m.index, m * 100, color='#1f77b4', lw=1.2, label='无杠杆保证金占用')
ax.plot(m.index, 4 * m * 100, color='#d62728', lw=1.2, label='4x 保证金占用')
ax.axhline(100, color='red', ls='-', lw=1.8, label='100% 现金耗尽线')
mg4 = 4 * m
ax.fill_between(mg4.index, 100, mg4 * 100, where=(mg4 > 1),
                color='darkred', alpha=0.35, label=f'现金为负 {int((mg4>1).sum())}天')
ax.set_title('③ 【关键】保证金占用：4x 下有 138 天超过 100%\n（现金耗尽，必须持续融资）',
             fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylabel('占净资产 %')

# ④ 杠杆阶梯
ax = axes[1, 1]
Ls = [1, 2, 3, 4, 5, 6, 8, 10]
dds, mgs = [], []
for Lx in Ls:
    n = (1 + Lx * ret).cumprod()
    dds.append((n / n.cummax() - 1).min() * 100)
    mgs.append(Lx * m.mean() * 100)
x = np.arange(len(Ls))
ax.bar(x, dds, color=['#2ca02c', '#2ca02c', '#ff7f0e', '#d62728', '#d62728', '#8c564b', '#4d4d4d', '#4d4d4d'],
       alpha=0.85, label='最大回撤')
for i, (dd, mg) in enumerate(zip(dds, mgs)):
    ax.text(i, dd - 4, f'{dd:.0f}%', ha='center', fontsize=9, fontweight='bold')
    ax.text(i, 3, f'{mg:.0f}%', ha='center', fontsize=8, color='navy')
ax.axhline(-30, color='red', ls='--', lw=1.2)
ax.set_xticks(x)
ax.set_xticklabels([f'{l}x' for l in Ls])
ax.set_title('④ 杠杆阶梯：回撤 vs 保证金占用（蓝字=保证金占比）\n8x 起保证金超100%', fontsize=12, fontweight='bold')
ax.set_ylabel('最大回撤 %')
ax.grid(alpha=0.3, axis='y')

# ⑤ 逐年对比
ax = axes[2, 0]
yrs = sorted(set(ret.index.year))
r1y = [((1 + ret[ret.index.year == y]).prod() - 1) * 100 for y in yrs if (ret.index.year == y).sum() > 20]
r4y = [((1 + r4[ret.index.year == y]).prod() - 1) * 100 for y in yrs if (ret.index.year == y).sum() > 20]
yy = [y for y in yrs if (ret.index.year == y).sum() > 20]
w = 0.38
ax.bar(np.arange(len(yy)) - w / 2, r1y, w, label='无杠杆', color='#1f77b4', alpha=0.9)
ax.bar(np.arange(len(yy)) + w / 2, r4y, w, label='4x', color='#d62728', alpha=0.9)
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(np.arange(len(yy)))
ax.set_xticklabels(yy, fontsize=9)
ax.set_title('⑤ 逐年收益：无杠杆 11 年全正，4x 全面提升', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')
ax.set_ylabel('收益 %')

# ⑥ 4x 逐年最大回撤
ax = axes[2, 1]
dd4y, dd1y = [], []
for y in yy:
    n1 = (1 + ret[ret.index.year == y]).cumprod()
    n4 = (1 + r4[ret.index.year == y]).cumprod()
    dd1y.append((n1 / n1.cummax() - 1).min() * 100)
    dd4y.append((n4 / n4.cummax() - 1).min() * 100)
w = 0.38
ax.bar(np.arange(len(yy)) - w / 2, dd1y, w, label='无杠杆', color='#1f77b4', alpha=0.9)
ax.bar(np.arange(len(yy)) + w / 2, dd4y, w, label='4x', color='#d62728', alpha=0.9)
ax.axhline(-20, color='orange', ls='--', lw=1.2, label='-20%')
ax.set_xticks(np.arange(len(yy)))
ax.set_xticklabels(yy, fontsize=9)
ax.set_title('⑥ 逐年最大回撤：4x 下 7 年回撤超 15%', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')
ax.set_ylabel('最大回撤 %')

plt.suptitle('风险平价策略 · 4 倍杠杆情景压力测试（2016-2026）\n'
             '核心矛盾：收益放大很诱人，但现金常年为负、需持续融资',
             fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.975])
out = f'{D}/输出/图表/杠杆情景_4x压力测试_2016-2026.png'
plt.savefig(out, dpi=140, bbox_inches='tight')
print(f'已保存: {out}')
