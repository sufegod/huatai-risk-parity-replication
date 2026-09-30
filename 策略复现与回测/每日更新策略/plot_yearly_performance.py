# -*- coding: utf-8 -*-
"""逐年业绩指标对比图（生产 v0.19）

从每日净值序列重算逐年指标并出图：
  ① 逐年收益率     ② 逐年夏普比率
  ③ 逐年最大回撤   ④ 逐年卡玛比（年化收益 / |最大回撤|）
  ⑤ 年化收益 vs 年化波动   ⑥ 累计净值走势

输入：输出/净值/策略每日净值走势_<最新日期>.csv（自动取最新）
输出：输出/图表/逐年业绩指标对比_<起始>-<最新>.png
      输出/指标/逐年业绩指标_<起始>-<最新>.csv

运行：C:/Users/aa/.workbuddy/binaries/python/envs/default/Scripts/python.exe plot_yearly_performance.py
"""
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

# 日收益：首日相对 1.0 基准
ret = nav.pct_change()
ret.iloc[0] = nav.iloc[0] - 1.0
ret = ret.dropna()

# ---------- 2. 逐年指标 ----------
rows = []
for y, g in ret.groupby(ret.index.year):
    cum = float((1 + g).prod() - 1)                 # 当年累计收益
    n = len(g)
    ann = float((1 + cum) ** (252.0 / n) - 1)       # 年化收益（几何，252 折）
    vol = float(g.std(ddof=1) * np.sqrt(252))       # 年化波动
    # 夏普与生产 v0.19 口径一致：算术年化收益 / 年化波动
    sharpe = float(g.mean() * 252 / vol) if vol > 0 else np.nan
    # 年内最大回撤：年初基准 1.0 纳入峰值
    cnav = (1 + g).cumprod().values
    peaks = np.maximum.accumulate(np.concatenate([[1.0], cnav]))[1:]
    mdd = float(np.min(cnav / peaks - 1))
    calmar = ann / abs(mdd) if mdd < 0 else np.nan
    rows.append({
        '年份': int(y), '收益': cum, '年化收益': ann, '年化波动': vol,
        '夏普比率': sharpe, '最大回撤': mdd, '卡玛比': calmar, '交易日数': n,
    })

df = pd.DataFrame(rows)
yrs = df['年份'].tolist()

# 全局（全样本）
tot_cum = float((1 + ret).prod() - 1)
tot_n = len(ret)
tot_ann = float((1 + tot_cum) ** (252.0 / tot_n) - 1)
tot_vol = float(ret.std(ddof=1) * np.sqrt(252))
tot_sharpe = float(ret.mean() * 252 / tot_vol)   # 生产口径：算术年化 / 年化波动
cnav_all = (1 + ret).cumprod().values
peaks_all = np.maximum.accumulate(np.concatenate([[1.0], cnav_all]))[1:]
tot_mdd = float(np.min(cnav_all / peaks_all - 1))
tot_calmar = tot_ann / abs(tot_mdd)

print(f'净值文件: {net_file}')
print(f'样本: {ret.index[0].date()} ~ {ret.index[-1].date()}  ({tot_n} 日)')
print(f'全局: 累计 {tot_cum*100:.2f}%  年化 {tot_ann*100:.2f}%  波动 {tot_vol*100:.2f}%  '
      f'夏普 {tot_sharpe:.2f}  回撤 {tot_mdd*100:.2f}%  卡玛 {tot_calmar:.2f}')

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


C_UP, C_DOWN = '#d62728', '#2ca02c'   # 中国习惯：红涨绿跌
fig, axes = plt.subplots(2, 3, figsize=(19, 11))

# ① 逐年收益率
ax = axes[0, 0]
v = df['收益'].values * 100
cols = [C_UP if x >= 0 else C_DOWN for x in v]
ax.bar(yrs, v, color=cols, edgecolor='black', linewidth=0.6)
for x, y in zip(yrs, v):
    ax.text(x, y + 0.35, f'{y:.2f}', ha='center', fontsize=9, fontweight='bold')
ax.axhline(tot_ann * 100, color='gray', ls='--', lw=1.2,
           label=f'全期年化 {tot_ann*100:.2f}%')
ax.set_title('① 逐年收益率（%）', fontsize=14, fontweight='bold')
ax.set_ylabel('%')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')
ax.set_ylim(0, max(v) * 1.18)

# ② 逐年夏普
ax = axes[0, 1]
v = df['夏普比率'].values
cols = [C_UP if x >= 0 else C_DOWN for x in v]
ax.bar(yrs, v, color=cols, edgecolor='black', linewidth=0.6)
for x, y in zip(yrs, v):
    ax.text(x, y + 0.05, f'{y:.2f}', ha='center', fontsize=9, fontweight='bold')
ax.axhline(tot_sharpe, color='gray', ls='--', lw=1.2, label=f'全期夏普 {tot_sharpe:.2f}')
ax.axhline(1.0, color='#1f77b4', ls=':', lw=1, label='夏普=1')
ax.set_title('② 逐年夏普比率（算术年化 / 年化波动，生产口径）', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')
ax.set_ylim(0, max(v) * 1.18)

# ③ 逐年最大回撤
ax = axes[0, 2]
v = df['最大回撤'].values * 100
ax.bar(yrs, v, color=C_DOWN, edgecolor='black', linewidth=0.6)
for x, y in zip(yrs, v):
    ax.text(x, y - 0.55, f'{y:.2f}', ha='center', va='top', fontsize=9, fontweight='bold')
ax.axhline(tot_mdd * 100, color='gray', ls='--', lw=1.2, label=f'全期回撤 {tot_mdd*100:.2f}%')
ax.set_title('③ 逐年最大回撤（%）', fontsize=14, fontweight='bold')
ax.set_ylabel('%')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')
ax.set_ylim(min(v) * 1.25, 0)

# ④ 逐年卡玛比
ax = axes[1, 0]
v = df['卡玛比'].values
cols = [C_UP if x >= 0 else C_DOWN for x in v]
ax.bar(yrs, v, color=cols, edgecolor='black', linewidth=0.6)
for x, y in zip(yrs, v):
    ax.text(x, y + 0.12, f'{y:.2f}', ha='center', fontsize=9, fontweight='bold')
ax.axhline(tot_calmar, color='gray', ls='--', lw=1.2, label=f'全期卡玛 {tot_calmar:.2f}')
ax.axhline(1.0, color='#1f77b4', ls=':', lw=1, label='卡玛=1')
ax.set_title('④ 逐年卡玛比（年化收益 / |最大回撤|）', fontsize=14, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')
ax.set_ylim(0, max(v) * 1.18)

# ⑤ 年化收益 vs 年化波动
ax = axes[1, 1]
w = 0.38
x = np.arange(len(yrs))
ax.bar(x - w / 2, df['年化收益'].values * 100, w, color='#d62728',
       edgecolor='black', linewidth=0.6, label='年化收益(%)')
ax.bar(x + w / 2, df['年化波动'].values * 100, w, color='#9467bd',
       edgecolor='black', linewidth=0.6, label='年化波动(%)')
ax.set_xticks(x)
ax.set_xticklabels(yrs)
ax.set_title('⑤ 逐年年化收益 vs 年化波动（%）', fontsize=14, fontweight='bold')
ax.set_ylabel('%')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

# ⑥ 累计净值走势
ax = axes[1, 2]
ax.plot(nav.index, nav.values, color='#d62728', lw=1.8, label='风险平价策略')
for y in yrs:
    yy = pd.Timestamp(f'{y}-01-01')
    ax.axvline(yy, color='gray', ls=':', lw=0.8, alpha=0.6)
for y in yrs:
    seg = nav[nav.index.year == y]
    if len(seg):
        ax.annotate(f'{y}', xy=(seg.index[len(seg) // 2], nav.max() * 0.995),
                    ha='center', fontsize=8, color='gray')
ax.set_title('⑥ 累计净值走势（起点 2018-01-03 = 1.0）', fontsize=14, fontweight='bold')
ax.set_ylabel('净值')
ax.grid(alpha=0.3)
ax.legend(fontsize=9)

fig.suptitle(
    f'风险平价策略 v0.19 · 逐年业绩指标对比（{ret.index[0].date()} ~ {ret.index[-1].date()}）\n'
    f'全期累计 {tot_cum*100:.1f}% / 年化 {tot_ann*100:.2f}% / 夏普 {tot_sharpe:.2f} / '
    f'最大回撤 {tot_mdd*100:.2f}% / 卡玛比 {tot_calmar:.2f}',
    fontsize=15, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])

out_png = _fresh(f'{BASE}/图表/逐年业绩指标对比_{start_year}-{latest_date}.png')
fig.savefig(out_png, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'图已保存: {out_png}')

# ---------- 4. 导出 CSV ----------
out_csv = _fresh(f'{BASE}/指标/逐年业绩指标_{start_year}-{latest_date}.csv')
df.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f'数据已导出: {out_csv}')
print(df.to_string(index=False))
