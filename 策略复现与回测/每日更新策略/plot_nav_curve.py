# -*- coding: utf-8 -*-
"""累计净值曲线图（生产 v0.19）

从每日净值序列绘制：
  上半：累计净值走势（红色主线 + 1.0 基准线 + 历史峰值/最大回撤区间标注）
  下半：水下回撤曲线（绿色填充）

输入：输出/净值/策略每日净值走势_<最新日期>.csv（自动取最新）
输出：输出/图表/净值曲线_<起始>-<最新>.png
      输出/净值/净值与回撤_<起始>-<最新>.csv

运行：C:/Users/aa/.workbuddy/binaries/python/envs/default/Scripts/python.exe plot_nav_curve.py
"""
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

P = 'C:/Users/aa/WorkBuddy/2026-05-28-14-41-45/huatai-risk-parity-replication'
BASE = f'{P}/策略复现与回测/每日更新策略/输出'

# ---------- 1. 读取最新净值序列 ----------
files = sorted(glob.glob(f'{BASE}/净值/策略每日净值走势_*.csv'))
net_file = files[-1]
latest_date = os.path.basename(net_file)[-14:-4]
start_year = '2018'

nav = pd.read_csv(net_file, encoding='utf-8-sig', parse_dates=['日期'])
nav = nav.set_index('日期').iloc[:, 0].astype(float).sort_index()
nav.name = 'nav'

n = len(nav)
ret = nav.pct_change()
ret.iloc[0] = nav.iloc[0] - 1.0
ret = ret.dropna()

# ---------- 2. 指标（与生产/逐年图同口径） ----------
cum = float(nav.iloc[-1] - 1.0)                       # 累计收益（基准 1.0）
ann = float((1 + cum) ** (252.0 / n) - 1)             # 年化收益（几何）
vol = float(ret.std(ddof=1) * np.sqrt(252))           # 年化波动
sharpe = float(ret.mean() * 252 / vol) if vol > 0 else np.nan  # 生产口径：算术年化/波动

run_max = nav.cummax()
dd = nav / run_max - 1.0
mdd = float(dd.min())
calmar = ann / abs(mdd) if mdd < 0 else np.nan

trough_d = dd.idxmin()
peak_d = nav.loc[:trough_d].idxmax()
hi_d, hi_v = nav.idxmax(), float(nav.max())

# ---------- 3. 出图 ----------
def _fresh(path):
    """先尝试删除旧文件以复用文件名；被沙箱拦截时退化为 _vN 后缀。"""
    if not os.path.exists(path):
        return path
    try:
        os.remove(path)
        return path
    except Exception as e:
        print(f'  [注] 无法删除旧文件({e.__class__.__name__})，改用带序号的新文件')
        root, ext = os.path.splitext(path)
        i = 2
        while os.path.exists(f'{root}_v{i}{ext}'):
            i += 1
        return f'{root}_v{i}{ext}'


C_NAV = '#c0392b'    # 净值主线：中国习惯红
C_DD = '#2e7d32'     # 回撤：绿

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(14, 9), sharex=True,
    gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08})

# ---- 上：累计净值 ----
ax1.fill_between(nav.index, 1.0, nav.values, color='#e74c3c', alpha=0.07, zorder=1)
ax1.plot(nav.index, nav.values, color=C_NAV, lw=2.0, zorder=3,
         label='风险平价策略 v0.19 累计净值')
ax1.axhline(1.0, color='#777', ls='--', lw=1.0, zorder=2)

# 最大回撤区间高亮
ax1.axvspan(peak_d, trough_d, color='#7f8c8d', alpha=0.13, zorder=0,
            label=f'最大回撤区间（{peak_d.date()} ~ {trough_d.date()}）')

# 起点 / 峰值 / 终点标注
ax1.scatter([nav.index[0]], [nav.iloc[0]], s=28, color='#34495e', zorder=5)
ax1.scatter([hi_d], [hi_v], s=42, color='#e67e22', zorder=5)
ax1.scatter([nav.index[-1]], [nav.iloc[-1]], s=42, color=C_NAV, zorder=5,
            edgecolors='white', linewidths=0.8)

ax1.annotate(f'最新 {nav.index[-1].date()}\n{nav.iloc[-1]:.4f}',
             xy=(nav.index[-1], nav.iloc[-1]),
             xytext=(-8, -46), textcoords='offset points',
             fontsize=10.5, fontweight='bold', color=C_NAV, ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='none', alpha=0.88),
             arrowprops=dict(arrowstyle='-', color=C_NAV, lw=1.0))

ax1.annotate(f'峰值 {hi_v:.4f}\n({hi_d.date()})',
             xy=(hi_d, hi_v), xytext=(-6, 16), textcoords='offset points',
             fontsize=9.5, color='#b9600a', ha='right',
             arrowprops=dict(arrowstyle='-', color='#e67e22', lw=0.9))

# 统计文本框
stats = (f'累计收益   {cum:+.2%}      历史峰值  {hi_v:.4f}\n'
         f'年化收益   {ann:+.2%}      峰值日期  {hi_d.date()}\n'
         f'年化波动   {vol:.2%}\n'
         f'夏普比率   {sharpe:.2f}\n'
         f'最大回撤   {mdd:.2%}\n'
         f'卡玛比     {calmar:.2f}')
ax1.text(0.012, 0.975, stats, transform=ax1.transAxes, va='top', ha='left',
         fontsize=10.5, color='#2c3e50', linespacing=1.55,
         bbox=dict(boxstyle='round,pad=0.65', fc='white', ec=C_NAV, lw=1.1, alpha=0.92))

ax1.set_ylim(bottom=0.95, top=max(hi_v, nav.iloc[-1]) * 1.05)
ax1.set_ylabel('累计净值（初始 = 1.0）', fontsize=11.5)
ax1.set_title(
    f'风险平价策略 v0.19 · 累计净值走势（{nav.index[0].date()} ~ {nav.index[-1].date()}）\n'
    f'{n} 个交易日 · 累计 {cum:+.1%} · 年化 {ann:.2%} · 最大回撤 {mdd:.2%}',
    fontsize=14.5, fontweight='bold', pad=14)
ax1.legend(loc='lower right', fontsize=9.5, framealpha=0.9)
ax1.grid(True, ls=':', lw=0.7, color='#cccccc', alpha=0.85)
ax1.set_axisbelow(True)

# ---- 下：水下回撤 ----
ax2.fill_between(dd.index, dd.values, 0, color=C_DD, alpha=0.22, zorder=1)
ax2.plot(dd.index, dd.values, color=C_DD, lw=1.1, zorder=3)
ax2.axhline(0, color='#777', lw=0.8)
ax2.axhline(mdd, color='#b71c1c', ls='--', lw=0.9, zorder=2)
ax2.annotate(f'{mdd:.2%}', xy=(trough_d, mdd), xytext=(8, 6),
             textcoords='offset points', fontsize=9.5, color='#b71c1c',
             fontweight='bold')

ax2.set_ylabel('回撤', fontsize=11.5)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:.0%}'))
ax2.grid(True, ls=':', lw=0.7, color='#cccccc', alpha=0.85)
ax2.set_axisbelow(True)

ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.set_xlim(nav.index[0], nav.index[-1])

fig.align_ylabels([ax1, ax2])

out_png = _fresh(f'{BASE}/图表/净值曲线_{start_year}-{latest_date}.png')
fig.savefig(out_png, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'图已保存: {out_png}')

# ---------- 4. 导出净值与回撤序列 ----------
out_csv = _fresh(f'{BASE}/净值/净值与回撤_{start_year}-{latest_date}.csv')
out = pd.DataFrame({'日期': nav.index, '净值': nav.values, '回撤': dd.values})
out.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f'数据已导出: {out_csv}')

# ---------- 5. 控制台摘要 ----------
print(f'\n样本: {nav.index[0].date()} ~ {nav.index[-1].date()}（{n} 交易日）')
print(f'期末净值 {nav.iloc[-1]:.4f} | 累计 {cum:+.2%} | 年化 {ann:+.2%} | '
      f'波动 {vol:.2%} | 夏普 {sharpe:.2f} | 最大回撤 {mdd:.2%} | 卡玛 {calmar:.2f}')
print(f'历史峰值 {hi_v:.4f}（{hi_d.date()}）| 最大回撤区间 {peak_d.date()} ~ {trough_d.date()}')
