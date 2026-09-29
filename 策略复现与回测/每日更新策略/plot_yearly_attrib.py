"""逐年归因与相关性分析图表（现金口径：资产净规模 − 保证金消耗 = 现金余额）

现金收益 = 现金余额 × 年化利率/365 × 自然日
  · 存款口径：国有大行一年期挂牌利率，日度分段
  · GC001口径：逆回购真实市场利率（对照）

产出：策略复现与回测/每日更新策略/输出/图表/逐年归因与相关性分析_2016-2026.png
运行：C:/Users/aa/.workbuddy/binaries/python/envs/default/Scripts/python.exe plot_yearly_attrib.py
"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

P = 'C:/Users/aa/WorkBuddy/2026-05-28-14-41-45/huatai-risk-parity-replication'
O = f'{P}/策略复现与回测/每日更新策略/输出/对比分析'

at = pd.read_csv(f'{O}/逐年归因_2016-2026.csv')
cr = pd.read_csv(f'{O}/逐年相关性_2016-2026.csv')
dr = pd.read_csv(f'{O}/逐年分散化比率_2016-2026.csv')
yrs = at['年份'].astype(int).tolist()

fig, axes = plt.subplots(3, 2, figsize=(16, 15))

# ---- 1. 逐年策略收益 ----
ax = axes[0, 0]
v = at['策略收益'].values * 100
cols = ['#d62728' if x >= 10 else '#ff7f0e' for x in v]
ax.bar(yrs, v, color=cols, edgecolor='black', linewidth=0.6)
ax.axhline(10, color='gray', ls='--', lw=1, label='10% 参考线')
for x, y in zip(yrs, v):
    ax.text(x, y + 0.4, f'{y:.1f}', ha='center', fontsize=9, fontweight='bold')
ax.set_title('① 逐年策略收益（%）—— 2022/2018/2026/2023 偏弱', fontsize=13, fontweight='bold')
ax.set_ylabel('%'); ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')

# ---- 2. 类别贡献堆叠 ----
ax = axes[0, 1]
classes = ['股指', '债券', '商品', '黄金', '权益ETF']
colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']
bottom_pos = np.zeros(len(yrs)); bottom_neg = np.zeros(len(yrs))
for c, col in zip(classes, colors):
    vv = at[c].values * 100
    pos = np.where(vv > 0, vv, 0); neg = np.where(vv < 0, vv, 0)
    ax.bar(yrs, pos, bottom=bottom_pos, color=col, label=c, edgecolor='white', linewidth=0.5)
    ax.bar(yrs, neg, bottom=bottom_neg, color=col, edgecolor='white', linewidth=0.5)
    bottom_pos += pos; bottom_neg += neg
ax.plot(yrs, at['策略收益'].values * 100, 'k.-', lw=1.5, ms=7, label='策略收益')
ax.axhline(0, color='black', lw=1)
ax.set_title('② 逐年类别贡献拆解（pct）', fontsize=13, fontweight='bold')
ax.set_ylabel('pct'); ax.legend(fontsize=8, ncol=3); ax.grid(alpha=0.3, axis='y')

# ---- 3. 现金收益两口径 vs 机会成本 ----
ax = axes[1, 0]
x = np.arange(len(yrs)); wd = 0.27
ax.bar(x - wd, at['现金收益_存款'].values * 100, wd,
       label='现金收益_存款（分段挂牌利率）', color='#17becf', edgecolor='black', linewidth=0.5)
ax.bar(x, at['现金收益_GC001'].values * 100, wd,
       label='现金收益_GC001（逆回购实际）', color='#2ca02c', edgecolor='black', linewidth=0.5)
ax.bar(x + wd, at['机会成本'].values * 100, wd,
       label='机会成本（拖累项）', color='#8c564b', edgecolor='black', linewidth=0.5)
ax.axhline(0, color='black', lw=1)
ax.set_xticks(x); ax.set_xticklabels(yrs, fontsize=9)
ax.set_title('③ 现金收益：两口径 vs 机会成本（pct）\n现金收益恒为正，机会成本才是拖累',
             fontsize=13, fontweight='bold')
ax.set_ylabel('pct'); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

# ---- 4. 交易成本 ----
ax = axes[1, 1]
v = at['交易成本'].values * 100
ax.bar(yrs, v, color='#7f7f7f', edgecolor='black', linewidth=0.6)
for xx, yy in zip(yrs, v):
    ax.text(xx, yy - 0.025, f'{yy:.2f}', ha='center', fontsize=8, color='white', fontweight='bold')
ax.set_title('④ 逐年交易成本（pct）—— 稳定在 −0.24 ~ −0.37', fontsize=13, fontweight='bold')
ax.set_ylabel('pct'); ax.grid(alpha=0.3, axis='y')

# ---- 5. 平均相关性 & 股债相关性 ----
ax = axes[2, 0]
ax.plot(yrs, cr['平均相关'].values, 'o-', color='#1f77b4', lw=2, ms=7, label='资产平均相关性')
ax.plot(yrs, cr['股债相关'].values, 's--', color='#d62728', lw=2, ms=7, label='股债相关性')
ax.axhline(0, color='black', lw=1)
ax.set_title('⑤ 逐年相关性 —— 弱势年分散化并未失效', fontsize=13, fontweight='bold')
ax.set_ylabel('相关系数'); ax.legend(fontsize=9); ax.grid(alpha=0.3)
# 标注弱势年
for y in [2018, 2022, 2023, 2026]:
    ax.axvspan(y - 0.35, y + 0.35, color='orange', alpha=0.12)

# ---- 6. 分散化比率 ----
ax = axes[2, 1]
v = dr['分散化比率'].values
cols = ['#d62728' if x < 1.5 else ('#ff7f0e' if x < 2.0 else '#2ca02c') for x in v]
ax.bar(yrs, v, color=cols, edgecolor='black', linewidth=0.6)
ax.axhline(1.5, color='red', ls='--', lw=1.2, label='警戒线 1.5')
for xx, yy in zip(yrs, v):
    ax.text(xx, yy + 0.03, f'{yy:.2f}', ha='center', fontsize=8.5, fontweight='bold')
ax.set_title('⑥ 逐年分散化比率（加权波动 ÷ 组合波动）', fontsize=13, fontweight='bold')
ax.set_ylabel('比率'); ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')

fig.suptitle('风险平价策略 · 逐年业绩归因与相关性分析（2016-2026）\n'
             '现金口径：资产净规模 − 保证金消耗 = 现金余额；现金收益 = 现金余额 × 利率',
             fontsize=16, fontweight='bold', y=0.997)
plt.tight_layout(rect=[0, 0, 1, 0.98])
out = f'{P}/策略复现与回测/每日更新策略/输出/图表/逐年归因与相关性分析_2016-2026.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print('已保存:', out)
