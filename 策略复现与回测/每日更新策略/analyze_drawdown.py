# -*- coding: utf-8 -*-
"""全周期回撤专项分析（生产 v0.19）

从最新净值序列提取所有回撤区间并统计：
  - 每个回撤区间：峰值/谷底/恢复日期、深度、下跌与恢复用时
  - 汇总：最大回撤、水下天数占比、Ulcer Index、平均回撤深度、最长水下期
  - 逐年最大回撤
  - 当前回撤状态

输入：输出/净值/策略每日净值走势_<最新日期>.csv（自动取最新）
输出：输出/指标/全周期回撤区间明细_<起始>-<最新>.csv
      输出/指标/全周期回撤统计_<起始>-<最新>.csv
      输出/图表/回撤专项分析_<起始>-<最新>.png

运行：C:/Users/aa/.workbuddy/binaries/python/envs/default/Scripts/python.exe analyze_drawdown.py
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

# ---------- 1. 读取最新净值 ----------
files = sorted(glob.glob(f'{BASE}/净值/策略每日净值走势_*.csv'))
net_file = files[-1]
latest_date = os.path.basename(net_file)[-14:-4]
start_year = '2018'

nav = pd.read_csv(net_file, encoding='utf-8-sig', parse_dates=['日期'])
nav = nav.set_index('日期').iloc[:, 0].astype(float).sort_index()

idx = nav.index
vals = nav.values
n = len(vals)

run_max = nav.cummax().values
dd = vals / run_max - 1.0

# ---------- 2. 提取回撤区间 ----------
eps = []
i = 0
while i < n:
    if dd[i] < 0:
        peak_pos = i - 1 if i > 0 else i
        j = i
        while j < n and dd[j] < 0:
            j += 1
        seg = dd[peak_pos:j]
        trough_pos = peak_pos + int(np.argmin(seg))
        rec_pos = j if j < n else None
        eps.append({
            '峰值日期': idx[peak_pos],
            '谷底日期': idx[trough_pos],
            '恢复日期': idx[rec_pos] if rec_pos is not None else pd.NaT,
            '深度': dd[trough_pos],
            '下跌交易日': trough_pos - peak_pos,
            '恢复交易日': (rec_pos - trough_pos) if rec_pos is not None else np.nan,
            '总交易日': (rec_pos if rec_pos is not None else n - 1) - peak_pos,
            '是否已恢复': rec_pos is not None,
        })
        i = j
    else:
        i += 1

dd_df = pd.DataFrame(eps)
dd_df['深度'] = dd_df['深度'].astype(float)

# ---------- 3. 汇总统计 ----------
mdd = float(dd.min())
mdd_date = idx[int(np.argmin(dd))]
underwater_ratio = float((dd < 0).mean())
ulcer = float(np.sqrt(np.mean(dd ** 2)))          # Ulcer Index
pain = float(np.mean(np.abs(dd)))                 # Pain Index（平均回撤深度）
avg_rec = float(dd_df.loc[dd_df['是否已恢复'], '恢复交易日'].mean())
max_rec = float(dd_df.loc[dd_df['是否已恢复'], '恢复交易日'].max())
longest = dd_df.loc[dd_df['总交易日'].idxmax()]
cur_dd = float(dd[-1])

# 距最近一次峰值
last_peak_pos = int(nav.values[:].argmax()) if cur_dd >= 0 else None

summary = [
    ('样本区间', f'{idx[0].date()} ~ {idx[-1].date()}'),
    ('交易日数', n),
    ('期末净值', f'{vals[-1]:.4f}'),
    ('全周期最大回撤', f'{mdd:.2%}'),
    ('最大回撤谷底日', f'{mdd_date.date()}'),
    ('水下天数占比', f'{underwater_ratio:.2%}'),
    ('Ulcer Index', f'{ulcer:.4%}'),
    ('Pain Index（平均回撤深度）', f'{pain:.4%}'),
    ('回撤区间总数', len(dd_df)),
    ('已恢复区间数', int(dd_df['是否已恢复'].sum())),
    ('单次回撤平均恢复用时（交易日）', f'{avg_rec:.1f}'),
    ('单次回撤最长恢复用时（交易日）', f'{max_rec:.0f}'),
    ('最长水下期', f"{longest['峰值日期'].date()} ~ "
                   f"{longest['恢复日期'].date() if longest['是否已恢复'] else idx[-1].date()}"),
    ('最深一次回撤区间', f"{dd_df['深度'].idxmin()}: "
                         f"{dd_df.loc[dd_df['深度'].idxmin(), '峰值日期'].date()} ~ "
                         f"{dd_df.loc[dd_df['深度'].idxmin(), '谷底日期'].date()}"),
]

# 逐年最大回撤
yr_rows = []
for y, g in nav.groupby(nav.index.year):
    rmax = g.cummax()
    d = g / rmax - 1.0
    yr_rows.append({'年份': int(y), '年内最大回撤': float(d.min()),
                    '年内水下占比': float((d < 0).mean()),
                    '年末净值': float(g.iloc[-1])})
yr_df = pd.DataFrame(yr_rows)

# ---------- 4. 出图 ----------
def _fresh(path):
    if not os.path.exists(path):
        return path
    try:
        os.remove(path)
        return path
    except Exception as e:
        print(f'  [注] 无法删除旧文件({e.__class__.__name__})，改用带序号的新文件')
        root, ext = os.path.splitext(path)
        k = 2
        while os.path.exists(f'{root}_v{k}{ext}'):
            k += 1
        return f'{root}_v{k}{ext}'


C_DD = '#2e7d32'
top5 = dd_df.nsmallest(5, '深度').reset_index(drop=True)

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(14, 9),
    gridspec_kw={'height_ratios': [3, 1.15], 'hspace': 0.30})

# ---- 上：水下回撤曲线 + top5 区间高亮 ----
ax1.fill_between(idx, dd, 0, color=C_DD, alpha=0.20, zorder=1)
ax1.plot(idx, dd, color=C_DD, lw=1.1, zorder=3)

palette = ['#b71c1c', '#e65100', '#f9a825', '#6a1b9a', '#0277bd']
for k, r in top5.iterrows():
    ax1.axvspan(r['峰值日期'], r['恢复日期'] if r['是否已恢复'] else idx[-1],
                color=palette[k], alpha=0.10, zorder=0)
    ax1.annotate(f"#{k+1} {r['深度']:.2%}\n{r['谷底日期'].date()}",
                 xy=(r['谷底日期'], r['深度']),
                 xytext=(0, -30 if k % 2 == 0 else -48), textcoords='offset points',
                 fontsize=9, color=palette[k], ha='center',
                 bbox=dict(boxstyle='round,pad=0.28', fc='white', ec=palette[k],
                           lw=0.8, alpha=0.9),
                 arrowprops=dict(arrowstyle='-', color=palette[k], lw=0.8))

ax1.axhline(mdd, color='#b71c1c', ls='--', lw=1.0, zorder=2)
ax1.set_ylim(mdd * 1.35, 0.004)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:.0%}'))
ax1.set_ylabel('回撤', fontsize=11.5)

stats = (f'最大回撤        {mdd:.2%}\n'
         f'水下天数占比    {underwater_ratio:.2%}\n'
         f'Ulcer Index     {ulcer:.2%}\n'
         f'Pain Index      {pain:.2%}\n'
         f'单次平均恢复    {avg_rec:.1f} 交易日\n'
         f'当前回撤        {cur_dd:.2%}')
ax1.text(0.985, 0.06, stats, transform=ax1.transAxes, va='bottom', ha='right',
         fontsize=10.5, color='#1b5e20', linespacing=1.55,
         bbox=dict(boxstyle='round,pad=0.65', fc='white', ec=C_DD, lw=1.1, alpha=0.93))

ax1.set_title(
    f'风险平价策略 v0.19 · 全周期回撤专项分析（{idx[0].date()} ~ {idx[-1].date()}）\n'
    f'最大回撤 {mdd:.2%}（{mdd_date.date()}） · 水下占比 {underwater_ratio:.1%} · '
    f'Ulcer Index {ulcer:.2%}',
    fontsize=14.5, fontweight='bold', pad=14)
ax1.grid(True, ls=':', lw=0.7, color='#cccccc', alpha=0.85)
ax1.set_axisbelow(True)
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.set_xlim(idx[0], idx[-1])

# ---- 下：逐年最大回撤柱状 ----
yrs = yr_df['年份'].astype(str).tolist()
bar = ax2.bar(yrs, yr_df['年内最大回撤'], color=C_DD, alpha=0.75, width=0.55, zorder=3)
for b, v in zip(bar, yr_df['年内最大回撤']):
    ax2.text(b.get_x() + b.get_width() / 2, v - 0.006, f'{v:.2%}',
             ha='center', va='top', fontsize=9.5, color='#1b5e20', fontweight='bold')
ax2.axhline(mdd, color='#b71c1c', ls='--', lw=0.9, zorder=2)
ax2.set_ylim(mdd * 1.30, 0.004)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:.0%}'))
ax2.set_ylabel('逐年最大回撤', fontsize=11.5)
ax2.grid(True, ls=':', lw=0.7, color='#cccccc', alpha=0.85)
ax2.set_axisbelow(True)

fig.align_ylabels([ax1, ax2])

out_png = _fresh(f'{BASE}/图表/回撤专项分析_{start_year}-{latest_date}.png')
fig.savefig(out_png, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'图已保存: {out_png}')

# ---------- 5. 导出 ----------
out_ep = _fresh(f'{BASE}/指标/全周期回撤区间明细_{start_year}-{latest_date}.csv')
exp = dd_df.copy()
for c in ['峰值日期', '谷底日期', '恢复日期']:
    exp[c] = exp[c].apply(lambda d: d.date().isoformat() if pd.notna(d) else '')
exp.sort_values('深度').to_csv(out_ep, index=False, encoding='utf-8-sig')
print(f'回撤区间明细已导出: {out_ep}')

out_sm = _fresh(f'{BASE}/指标/全周期回撤统计_{start_year}-{latest_date}.csv')
sm_rows = [{'项目': k, '数值': v} for k, v in summary]
sm_rows += [{'项目': '——逐年最大回撤——', '数值': ''}]
sm_rows += [{'项目': f"{int(r['年份'])}", '数值': f"{r['年内最大回撤']:.4%}"}
            for _, r in yr_df.iterrows()]
pd.DataFrame(sm_rows).to_csv(out_sm, index=False, encoding='utf-8-sig')
print(f'回撤统计已导出: {out_sm}')

# ---------- 6. 控制台 ----------
print(f'\n样本: {idx[0].date()} ~ {idx[-1].date()}（{n} 交易日）')
for k, v in summary:
    print(f'  {k}: {v}')
print('\n—— 最深 10 次回撤 ——')
for k, r in dd_df.nsmallest(10, '深度').reset_index(drop=True).iterrows():
    rec = r['恢复日期'].date() if pd.notna(r['恢复日期']) else '未恢复'
    print(f"  #{k+1} {r['深度']:.2%} | 峰 {r['峰值日期'].date()} -> 谷 {r['谷底日期'].date()} "
          f"-> 恢复 {rec} | 下跌 {int(r['下跌交易日'])} 日 / 恢复 "
          f"{'' if pd.isna(r['恢复交易日']) else int(r['恢复交易日'])} 日")
print('\n—— 逐年最大回撤 ——')
for _, r in yr_df.iterrows():
    print(f"  {int(r['年份'])}: {r['年内最大回撤']:.2%}（水下占比 {r['年内水下占比']:.1%}，"
          f"年末净值 {r['年末净值']:.4f}）")
