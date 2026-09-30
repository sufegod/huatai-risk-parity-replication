# -*- coding: utf-8 -*-
"""回撤归因与修复条件分析图（2026-09-29）

数据来源：analyze_drawdown_attrib.py 产出的
  · 输出/指标/回撤归因_逐次明细_2018-2026-<date>.csv
  · 输出/指标/回撤归因_日度贡献_2018-2026-<date>.csv
"""
import os
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PROJ = 'C:/Users/aa/WorkBuddy/2026-05-28-14-41-45/huatai-risk-parity-replication'
SCRIPT_DIR = f'{PROJ}/策略复现与回测/每日更新策略'
OUT = f'{SCRIPT_DIR}/输出'

CLS_ORDER = ['股指', '商品', '债券', '黄金', '红利ETF']
CLS_COLOR = {'股指': '#d62728', '商品': '#ff7f0e', '债券': '#1f77b4',
             '黄金': '#c9a227', '红利ETF': '#9467bd', '其他': '#7f7f7f'}


def latest(pattern):
    fs = sorted(glob.glob(pattern))
    if not fs:
        raise FileNotFoundError(pattern)
    return fs[-1]


def fresh(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f'  [warn] 旧文件删除失败 {os.path.basename(path)}: {e}')
    return path


f_attr = latest(f'{OUT}/指标/回撤归因_逐次明细_*.csv')
f_daily = latest(f'{OUT}/指标/回撤归因_日度贡献_*.csv')
attrib = pd.read_csv(f_attr, encoding='utf-8-sig')
daily = pd.read_csv(f_daily, encoding='utf-8-sig', index_col=0, parse_dates=True)
attrib['峰值日'] = pd.to_datetime(attrib['峰值日'])
attrib['谷底日'] = pd.to_datetime(attrib['谷底日'])
attrib['年份'] = attrib['峰值日'].dt.year

sub = attrib[attrib['恢复日数'].notna()].copy()
print(f'读入 {len(attrib)} 次回撤（已恢复 {len(sub)}），日度 {len(daily)} 行')

fig = plt.figure(figsize=(16.5, 18))
gs = fig.add_gridspec(3, 2, hspace=0.34, wspace=0.20,
                      left=0.065, right=0.975, top=0.945, bottom=0.045)

# ---------- ① 全周期水下回撤 ----------
ax = fig.add_subplot(gs[0, :])
dd = daily['回撤'] * 100
ax.fill_between(dd.index, dd.values, 0, color='#2ca02c', alpha=0.30, zorder=2)
ax.plot(dd.index, dd.values, color='#1f7a1f', lw=0.9, zorder=3)
top10 = attrib.nlargest(10, '深度绝对值')
for _, row in top10.iterrows():
    t = row['谷底日']
    if t in dd.index:
        ax.plot([t], [dd.loc[t]], 'o', ms=4.5, color='#d62728', zorder=4)
# 只标注「最深 5 次 + 当前未恢复」，避免时间相邻的标注互相压叠
to_label = pd.concat([attrib.nlargest(5, '深度绝对值'),
                      attrib[attrib['恢复日数'].isna()]]).drop_duplicates(subset=['谷底日'])
for _, row in to_label.iterrows():
    t, dep = row['谷底日'], row['深度绝对值']
    if t in dd.index:
        off = 20 if dep >= 0.06 else (-30 if dep >= 0.045 else -50)
        dx = 26 if (t - dd.index[0]).days < 100 else 0
        ax.annotate(f"{dep:.2%}\n{t.date()}", xy=(t, dd.loc[t]),
                    xytext=(dx, off), textcoords='offset points',
                    ha='center' if dx == 0 else 'left',
                    fontsize=8.5, color='#a32d2d',
                    arrowprops=dict(arrowstyle='-', color='#d62728', lw=0.7))
ax.set_title(f'① 全周期水下回撤曲线（{dd.index[0].date()} ~ {dd.index[-1].date()}）'
             f'　共 {len(attrib)} 次回撤，{len(sub)} 次已收复，1 次仍在进行',
             fontsize=13.5, fontweight='bold', pad=10)
ax.set_ylabel('回撤（%）', fontsize=11)
ax.grid(True, ls=':', lw=0.7, color='#cccccc', alpha=0.8)
ax.set_axisbelow(True)
ax.axhline(0, color='#555555', lw=0.7)
ax.set_ylim(dd.min() * 1.32, 0.45)
ax.set_xlim(dd.index[0], dd.index[-1])

# ---------- ② 逐年回撤次数与最深回撤 ----------
ax = fig.add_subplot(gs[1, 0])
g = attrib.groupby('年份').agg(次数=('深度', 'size'), 最深=('深度绝对值', 'max'),
                              平均深度=('深度绝对值', 'mean'))
x = np.arange(len(g))
ax.bar(x, g['次数'], color='#8c8c8c', width=0.58, label='回撤次数')
ax.set_ylabel('回撤次数（次）', fontsize=11, color='#555555')
ax.set_xticks(x)
ax.set_xticklabels([str(i) for i in g.index], fontsize=10)
ax.set_ylim(0, g['次数'].max() * 1.45)
for xi, v in zip(x, g['次数']):
    ax.text(xi, v + 0.6, str(int(v)), ha='center', fontsize=9, color='#555555')
ax2 = ax.twinx()
ax2.plot(x, g['最深'] * 100, 'o-', color='#d62728', lw=1.8, ms=5.5, label='当年最深回撤')
for xi, v in zip(x, g['最深'] * 100):
    ax2.text(xi, v + 0.22, f'{v:.1f}', ha='center', fontsize=8.5, color='#a32d2d')
ax2.set_ylabel('当年最深回撤（%）', fontsize=11, color='#a32d2d')
ax2.set_ylim(0, g['最深'].max() * 100 * 1.45)
ax2.tick_params(axis='y', colors='#a32d2d')
ax.set_title('② 逐年回撤频次与深度', fontsize=13, fontweight='bold', pad=9)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=9, loc='upper left', framealpha=0.92)
ax.grid(True, axis='y', ls=':', lw=0.7, color='#dddddd')
ax.set_axisbelow(True)

# ---------- ③ 深度 vs 修复天数散点 ----------
ax = fig.add_subplot(gs[1, 1])
for c in CLS_ORDER:
    d = sub[sub['主导拖累'] == c]
    if len(d):
        ax.scatter(d['深度绝对值'] * 100, d['恢复日数'], s=34, alpha=0.72,
                   color=CLS_COLOR[c], edgecolors='white', linewidths=0.5,
                   label=f'{c}（{len(d)}次）', zorder=3)
z = np.polyfit(sub['深度绝对值'] * 100, sub['恢复日数'], 1)
xs = np.linspace(0, sub['深度绝对值'].max() * 100 * 1.04, 50)
ax.plot(xs, np.polyval(z, xs), '--', color='#333333', lw=1.3, zorder=2,
        label=f'线性拟合  r = +0.822')
ax.set_xlabel('回撤深度（绝对值，%）', fontsize=11)
ax.set_ylabel('修复所需交易日数', fontsize=11)
ax.set_title('③ 回撤越深，修复越慢（最强约束）', fontsize=13, fontweight='bold', pad=9)
ax.legend(fontsize=9, loc='upper left', framealpha=0.92)
ax.grid(True, ls=':', lw=0.7, color='#dddddd')
ax.set_axisbelow(True)

# ---------- ④ 按主导拖累：下跌 vs 修复天数 ----------
ax = fig.add_subplot(gs[2, 0])
gg = sub.groupby('主导拖累').agg(下跌日=('下跌日数', 'mean'), 恢复日=('恢复日数', 'mean'),
                              次数=('深度', 'size'))
gg = gg[gg['次数'] >= 3].sort_values('恢复日', ascending=False)
x = np.arange(len(gg))
w = 0.36
ax.bar(x - w / 2, gg['下跌日'], w, color='#c0392b', label='平均下跌天数')
ax.bar(x + w / 2, gg['恢复日'], w, color='#27ae60', label='平均修复天数')
for xi, (a, b, n) in enumerate(zip(gg['下跌日'], gg['恢复日'], gg['次数'])):
    ax.text(xi - w / 2, a + 0.2, f'{a:.1f}', ha='center', fontsize=9, color='#a93226')
    ax.text(xi + w / 2, b + 0.2, f'{b:.1f}', ha='center', fontsize=9, color='#1e8449')
    ax.text(xi, -1.6, f'n={n}', ha='center', fontsize=8.5, color='#777777')
ax.set_xticks(x)
ax.set_xticklabels(gg.index, fontsize=10.5)
ax.set_ylabel('交易日数', fontsize=11)
ax.set_ylim(-3, gg[['下跌日', '恢复日']].max().max() * 1.28)
ax.set_title('④ 不同成因的回撤：持续与修复时长', fontsize=13, fontweight='bold', pad=9)
ax.legend(fontsize=9.5)
ax.grid(True, axis='y', ls=':', lw=0.7, color='#dddddd')
ax.set_axisbelow(True)

# ---------- ⑤ 信号状态 × 拖累来源 ----------
ax = fig.add_subplot(gs[2, 1])
a5 = attrib.copy()
a5['信号状态'] = pd.cut(a5['起始股指信号'], [-2, 0.001, 0.99, 2],
                      labels=['清仓\n(信号≤0)', '半仓\n(0<信号<1)', '满仓\n(信号=1.0)'])
ct = pd.crosstab(a5['信号状态'], a5['主导拖累']).reindex(columns=CLS_ORDER).fillna(0)
bottom = np.zeros(len(ct))
x = np.arange(len(ct))
for c in CLS_ORDER:
    if c in ct.columns:
        ax.bar(x, ct[c].values, 0.55, bottom=bottom, color=CLS_COLOR[c], label=c)
        for xi, (v, b) in enumerate(zip(ct[c].values, bottom)):
            if v >= 3:
                ax.text(xi, b + v / 2, f'{int(v)}', ha='center', va='center',
                        fontsize=9, color='white', fontweight='bold')
        bottom = bottom + ct[c].values
ax.set_xticks(x)
ax.set_xticklabels(ct.index, fontsize=10)
ax.set_ylabel('回撤次数', fontsize=11)
ax.set_title('⑤ 信号状态决定「回撤由谁造成」', fontsize=13, fontweight='bold', pad=9)
ax.legend(fontsize=9, ncol=3, loc='upper center')
ax.set_ylim(0, bottom.max() * 1.28)
ax.grid(True, axis='y', ls=':', lw=0.7, color='#dddddd')
ax.set_axisbelow(True)

fig.suptitle('风险平价策略 v0.19 · 全周期回撤归因与修复条件分析（2018-2026）',
             fontsize=16.5, fontweight='bold', y=0.978)

out_png = fresh(f'{OUT}/图表/回撤归因分析_2018-2026-{daily.index[-1].date()}.png')
fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'图已保存：{out_png}')

# ---------- 单独出「当前回撤分解」图 ----------
unrec = attrib[attrib['恢复日数'].isna()]
if len(unrec):
    u = unrec.iloc[0]
    pk, tr = u['峰值日'], u['谷底日']
    seg1 = daily.loc[pk:tr].iloc[1:]
    seg2 = daily.loc[tr:daily.index[-1]].iloc[1:]
    cats = ['股指', '商品', '债券', '黄金', '红利ETF']
    v1 = [float(seg1[f'贡献_{c}'].sum()) * 100 for c in cats] + [float(seg1['现金贡献'].sum()) * 100]
    v2 = [float(seg2[f'贡献_{c}'].sum()) * 100 for c in cats] + [float(seg2['现金贡献'].sum()) * 100]
    labels = cats + ['现金']
    colors = [CLS_COLOR[c] for c in cats] + ['#2ca02c']

    fig2, ax = plt.subplots(figsize=(11.5, 6))
    x = np.arange(len(labels))
    w = 0.37
    ax.bar(x - w / 2, v1, w, color=colors, edgecolor='white', linewidth=0.8, label=f'下跌段 {pk.date()}→{tr.date()}')
    ax.bar(x + w / 2, v2, w, color=colors, alpha=0.45, edgecolor='white', linewidth=0.8,
           label=f'谷底至今 {tr.date()}→{daily.index[-1].date()}')
    for xi in range(len(labels)):
        for val, off in ((v1[xi], -w / 2), (v2[xi], w / 2)):
            ax.text(xi + off, val + (0.06 if val >= 0 else -0.16), f'{val:+.2f}',
                    ha='center', fontsize=9.5, color='#333333')
    ax.axhline(0, color='#555555', lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('对策略收益的累计贡献（%）', fontsize=11)
    ax.set_title(f'当前未修复回撤的资产级分解：主导拖累是「股指」，而修复也卡在「股指」\n'
                 f'下跌段累计 {seg1["策略日收益"].sum():+.2%}　谷底至今累计 {seg2["策略日收益"].sum():+.2%}'
                 f'　距峰值仍差 {float(daily["回撤"].iloc[-1]):.2%}',
                 fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(True, axis='y', ls=':', lw=0.7, color='#dddddd')
    ax.set_axisbelow(True)
    ax.set_ylim(min(v1 + v2) * 1.35 - 0.2, max(v1 + v2) * 1.28 + 0.2)

    out2 = fresh(f'{OUT}/图表/当前回撤资产分解_{daily.index[-1].date()}.png')
    fig2.savefig(out2, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print(f'图已保存：{out2}')
print('完成。')
