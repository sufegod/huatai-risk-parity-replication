# -*- coding: utf-8 -*-
"""4 倍杠杆情景压力测试（v2 · 现金口径修正版）

【修正说明 2026-09-29（用户指定口径）】
  现金(剩余资金) = 当日净资产 − 保证金规模
    · 净资产【不随杠杆倍数放大】—— 杠杆只放大仓位，不改变自有资金规模
    · 保证金规模 = L × (无杠杆保证金/净值比 m)
    · 即 现金比例 = 1 − L·m；4x 下保证金占用 = 4m
  剩余现金按【银行存款利率】计息（国有大行一年期挂牌，分段）
  现金收益恒 ≥ 0 的前提是 L·m ≤ 1（保证金不超过净资产）

  与旧版 analyze_leverage_4x.py 的差异：
    · 旧版 = 4 × ret 简单复利，完全忽略现金收益，且用"现金为负 138 天"作一级结论
    · 新版把【资产端 / 现金端 / 成本端】拆开，现金收益按存款利率计入总收益，
      并区分"真实信号段(2018-01-02 起)"与"股指补满仓构造期(2016-2017)"

【研究性质，不接生产链路，不改动任何既有数据】
"""
import pandas as pd, numpy as np

P = 'C:/Users/aa/WorkBuddy/2026-05-28-14-41-45/huatai-risk-parity-replication'
D = f'{P}/策略复现与回测/每日更新策略'

COLS = ['沪深300主连', '中证500主连', '10年国债主连', '沪铜主连', '沪铝主连',
        'PTA主连', '原油主连', '豆粕主连', '沪金主连', '上证50ETF']
MARGIN = {'沪深300主连': 0.14, '中证500主连': 0.14, '10年国债主连': 0.025,
          '沪铜主连': 0.16, '沪铝主连': 0.16, 'PTA主连': 0.17,
          '原油主连': 0.32, '豆粕主连': 0.13, '沪金主连': 0.28,
          '上证50ETF': 1.00}
DEPOSIT_SCHEDULE = [
    ('2016-01-01', 0.0175),
    ('2022-09-15', 0.0165),
    ('2023-06-08', 0.0155),
    ('2023-12-22', 0.0145),
    ('2024-07-25', 0.0135),
    ('2025-05-20', 0.0095),
]
FEE_RATE = 0.0005
REAL_START = pd.Timestamp('2018-01-02')   # 真实股指信号起点

# ---------------- 读数据 & 重放每日权重（与 attrib_yearly.py 完全一致） ----------------
d = pd.read_csv(f'{D}/输出/净值/2016起回溯_日收益对比.csv', index_col=0, parse_dates=True)
strat = d['上证50ETF_日收益']          # 列名误导：实为【策略日收益】

un = pd.read_csv(f'{P}/数据/日度收益数据更新/日涨跌幅_填充.csv', index_col=0, parse_dates=True)
un = un.loc[:, ~un.columns.duplicated()]
s50 = pd.read_csv(f'{P}/数据/原始数据/上证50ETF_日涨跌幅.csv', index_col=0, parse_dates=True)
s50.index = pd.to_datetime(s50.index)
s50 = s50.loc[:, ~s50.columns.duplicated()]
un = un.drop(columns=['上证50ETF'], errors='ignore').join(s50, how='left')

rec = pd.read_csv(f'{D}/输出/对比分析/2016起_仓位明细_上证50.csv', parse_dates=['date']).set_index('date')

lvl = un.reindex(columns=COLS + ['一天期国债逆回购']).fillna(0.0) / 100.0
repo = lvl['一天期国债逆回购']
m_ratio = pd.Series({c: MARGIN.get(c, 1.0) for c in COLS})

idx = strat.index
rr = lvl.loc[idx, COLS]
recw = rec[COLS]

eff = {}
for rdt in rec.index:
    cand = idx[idx > rdt]
    if len(cand):
        eff[cand[0]] = rdt

W = pd.DataFrame(np.nan, index=idx, columns=COLS)
for dt, rdt in eff.items():
    W.loc[dt] = recw.loc[rdt].values
W = W.ffill()
valid = ~W.isna().all(axis=1)
W = W.loc[valid].fillna(0.0)
idx = W.index

gross = W.mul(rr.loc[idx], axis=0).sum(axis=1)        # 资产端（名义加权收益，权重和=1）
margin = (W * m_ratio).sum(axis=1)                    # 无杠杆保证金占用/净资产
costd = W.diff().abs().sum(axis=1).fillna(0.0) * FEE_RATE   # 单边万五

cal_days = idx.to_series().diff().dt.days.fillna(1)
cal_days.index = idx
_rate = pd.Series(0.0, index=idx)
for s, r in DEPOSIT_SCHEDULE:
    _rate.loc[idx >= pd.Timestamp(s)] = r
dep_daily = (_rate / 365.0) * cal_days                       # 存款日利率（小数）

repo_ann = repo.reindex(idx).fillna(0.0).shift(1).fillna(0.0)
repo_daily = np.maximum((repo_ann / 365.0) * cal_days - 1e-6, 0.0)   # GC001 日利率（对照）

# ================= 新口径：杠杆组合日收益 =================
def lev_ret(L, rate_daily):
    """杠杆 L 倍的组合日收益：资产端 ×L，现金端 (1−L·m) 计息，成本端 ×L"""
    cash_ratio = 1.0 - L * margin
    return L * gross + cash_ratio * rate_daily - L * costd, cash_ratio

LEVS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]

print('=' * 108)
print('4 倍杠杆情景压力测试 v2（现金口径修正版）')
print('=' * 108)
print('口径：现金 = 当日净资产 − 保证金规模；净资产不随杠杆放大；')
print('      保证金规模 = L × m（m=无杠杆保证金/净值）；剩余现金按银行存款利率计息')
print(f'样本：{idx[0].date()} ~ {idx[-1].date()}（{len(idx)} 个交易日 / {len(idx)/243:.2f} 年）')
print(f'  · 2016-2017 段为"股指补满仓(信号=1.0)"构造期，共 {int((idx<REAL_START).sum())} 日')
print(f'  · 2018-01-02 起为真实信号段，共 {int((idx>=REAL_START).sum())} 日')
print(f'无杠杆保证金占用 m：均值 {margin.mean():.2%}  中位 {margin.median():.2%}  '
      f'最大 {margin.max():.2%}（{margin.idxmax().date()}）')


def perf(r, label):
    nav = (1 + r).cumprod()
    yrs = len(r) / 243
    tot = nav.iloc[-1] - 1
    ann = nav.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(243)
    dd = (nav / nav.cummax() - 1)
    mdd = dd.min()
    return dict(口径=label, 累计=tot, 年化=ann, 波动=vol,
                夏普=(ann / vol if vol > 0 else np.nan),
                最大回撤=mdd, 卡玛=(ann / abs(mdd) if mdd != 0 else np.nan),
                最差单日=r.min(), 日胜率=(r > 0).mean())


# ================= 一、全期表现（存款口径） =================
print('\n' + '=' * 108)
print('一、全期表现（现金按【存款利率】计息）')
print('=' * 108)
rows, navs, cash_ratios = [], {}, {}
for L in LEVS:
    r, cr = lev_ret(L, dep_daily)
    navs[L] = (1 + r).cumprod()
    cash_ratios[L] = cr
    rows.append(perf(r, f'{L:.0f}x'))
res = pd.DataFrame(rows)
show = res.copy()
for c in ['累计', '年化', '波动', '最大回撤', '最差单日', '日胜率']:
    show[c] = show[c].map(lambda x: f'{x:+.2%}')
for c in ['夏普', '卡玛']:
    show[c] = show[c].map(lambda x: f'{x:.2f}')
print(show.to_string(index=False))

print('\n【对照】现金按 GC001 逆回购利率计息：')
rows2 = [perf(lev_ret(L, repo_daily)[0], f'{L:.0f}x') for L in LEVS]
show2 = pd.DataFrame(rows2)
for c in ['累计', '年化', '波动', '最大回撤', '最差单日', '日胜率']:
    show2[c] = show2[c].map(lambda x: f'{x:+.2%}')
for c in ['夏普', '卡玛']:
    show2[c] = show2[c].map(lambda x: f'{x:.2f}')
print(show2.to_string(index=False))

# ================= 二、现金与保证金（核心修正点） =================
print('\n' + '=' * 108)
print('二、现金与保证金占用（核心修正点）')
print('=' * 108)
print(f"{'杠杆':>5}{'保证金占用':>12}{'现金(剩余)':>12}{'现金为负天数':>14}{'其中2018+':>12}{'现金日均':>10}")
for L in LEVS:
    cr = cash_ratios[L]
    neg = cr < -1e-12
    print(f'{L:>4.0f}x{L*margin.mean():>12.2%}{cr.mean():>12.2%}{int(neg.sum()):>14}'
          f'{int(neg[idx>=REAL_START].sum()):>12}{cr.mean():>10.2%}')

print('\n【4x 专项】')
cr4 = cash_ratios[4.0]
m4 = 4.0 * margin
neg4 = cr4 < -1e-12
print(f'  保证金占用 4m：均值 {m4.mean():.2%}  最大 {m4.max():.2%}（{m4.idxmax().date()}）')
print(f'  剩余现金 1−4m：均值 {cr4.mean():.2%}  最小 {cr4.min():.2%}')
print(f'  ⚠️ 现金为负的天数：{int(neg4.sum())} 天（占 {neg4.mean():.1%}）')
if neg4.sum() > 0:
    yb = neg4.groupby(neg4.index.year).sum()
    yb = yb[yb > 0]
    print('     年份分布：' + '、'.join(f'{y}年 {int(n)}天' for y, n in yb.items()))
    print(f'     区间：{neg4[neg4].index[0].date()} ~ {neg4[neg4].index[-1].date()}')
    print(f'  ✅ 真实信号段（2018-01-02 起）现金为负天数：{int(neg4[idx>=REAL_START].sum())} 天')
    print(f'     真实段 4m 最大值：{m4[idx>=REAL_START].max():.2%}（{m4[idx>=REAL_START].idxmax().date()}）'
          f' → 现金最低仍为正 {1-m4[idx>=REAL_START].max():.2%}')
print('\n  说明：现金为负仅出现在 2016-11 ~ 2017-11 的股指补满仓构造期，')
print('        主因是当时上证50ETF 现货权重一度达 18.13%（100% 保证金），放大 4 倍即 72.5%；')
print('        2018 年起 ETF 权重回落至约 6.1%，4x 保证金占用再未超过 100%。')

# ================= 三、收益拆解（4x 的收益从哪来） =================
print('\n' + '=' * 108)
print('三、收益拆解：4x 的收益从哪里来（存款口径，括号内为 1x 对照）')
print('=' * 108)
yrs_n = len(idx) / 243
for L in [2.0, 4.0]:
    cr = cash_ratios[L]
    asset_c = (1 + L * gross - L * costd).prod() ** (1 / yrs_n) - 1     # 资产端 − 成本端
    cash_c = (cr * dep_daily).sum() / yrs_n                             # 现金端年均贡献
    r, _ = lev_ret(L, dep_daily)
    tot_ann = ((1 + r).prod() ** (243 / len(r)) - 1)
    print(f'  {L:.0f}x：资产端年化贡献 {asset_c:+.2%}（1x {(1+gross-costd).prod()**(1/yrs_n)-1:+.2%}）')
    print(f'      现金端年化贡献 {cash_c:+.2%}（1x {((1-margin)*dep_daily).sum()/yrs_n:+.2%}）')
    print(f'      合计年化     {tot_ann:+.2%}（1x {((1+1*gross+(1-margin)*dep_daily-costd).prod()**(243/len(idx))-1):+.2%}）')
    print()

# ================= 四、回撤与水下期（4x） =================
r4, _ = lev_ret(4.0, dep_daily)
nav4 = (1 + r4).cumprod()
dd4 = nav4 / nav4.cummax() - 1
r1, _ = lev_ret(1.0, dep_daily)
nav1 = (1 + r1).cumprod()
dd1 = nav1 / nav1.cummax() - 1
print('=' * 108)
print('四、4x 回撤路径（这一项与旧版一致，是杠杆的真实代价）')
print('=' * 108)
print(f'  最大回撤   {dd4.min():+.2%}（无杠杆 {dd1.min():+.2%}，放大 {dd4.min()/dd1.min():.2f} 倍）')
print(f'  最深日期   {dd4.idxmin().date()}')
underwater = dd4 < -1e-9
groups = (underwater != underwater.shift()).cumsum()
uw = underwater.groupby(groups).sum()
uw = uw[underwater.groupby(groups).first()]
if len(uw) > 0:
    print('  最长的 5 段水下期：')
    for gid, n in uw.sort_values(ascending=False).head(5).items():
        seg = underwater[groups == gid]
        print(f'    {seg.index[0].date()} ~ {seg.index[-1].date()}  {int(n):>4} 个交易日（约 {n/243:.2f} 年）')
for thr in [0.20, 0.30, 0.40]:
    n = (dd4 <= -thr).sum()
    print(f'  回撤曾触及 -{thr:.0%}：{"是" if n > 0 else "否"}（{n} 个交易日）')

# ================= 五、逐年分解 =================
print('\n' + '=' * 108)
print('五、逐年分解（存款口径）')
print('=' * 108)
yr_rows = []
for y in sorted(set(idx.year)):
    msk = idx.year == y
    if msk.sum() < 20:
        continue
    rr1, cr1 = lev_ret(1.0, dep_daily)
    rr4, cr4g = lev_ret(4.0, dep_daily)
    n1 = (1 + rr1[msk]).cumprod(); n4 = (1 + rr4[msk]).cumprod()
    yr_rows.append({
        '年份': y,
        '1x收益': (1 + rr1[msk]).prod() - 1,
        '4x收益': (1 + rr4[msk]).prod() - 1,
        '1x最大回撤': (n1 / n1.cummax() - 1).min(),
        '4x最大回撤': (n4 / n4.cummax() - 1).min(),
        '4x最差单日': rr4[msk].min(),
        '4x现金均值': cr4g[msk].mean(),
        '4x现金为负天数': int((cr4g[msk] < -1e-12).sum()),
    })
yr = pd.DataFrame(yr_rows)
yshow = yr.copy()
for c in yshow.columns:
    if c not in ['年份', '4x现金为负天数']:
        yshow[c] = yshow[c].map(lambda x: f'{x:+.2%}')
print(yshow.to_string(index=False))

# ================= 六、极端情景 =================
print('\n' + '=' * 108)
print('六、极端情景（4x）')
print('=' * 108)
worst = r4.nsmallest(8)
print(f"{'日期':<14}{'1x':>10}{'4x':>10}")
for dt, v in worst.items():
    print(f'{str(dt.date()):<14}{r1[dt]:>10.2%}{v:>10.2%}')
print(f'\n  4x 单日跌幅 >5%：{int((r4<=-0.05).sum())} 天；>8%：{int((r4<=-0.08).sum())} 天；'
      f'>10%：{int((r4<=-0.10).sum())} 天')
print('  情景外推：沪深300 单日 -7% → 4x ≈ -28%；单日 -10%（2016 熔断级）→ 4x ≈ -40%')

# ================= 七、保存 =================
out_csv = f'{D}/输出/对比分析/杠杆情景_4x压力测试_2016-2026_修正版.csv'
yr.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f'\n已导出：{out_csv}')

navdf = pd.DataFrame({
    '1x净值': nav1, '4x净值': nav4,
    '1x回撤': dd1, '4x回撤': dd4,
    '1x现金比例': cash_ratios[1.0], '4x现金比例': cr4,
    '无杠杆保证金m': margin, '4x保证金占用': m4,
    '4x现金为负': neg4.astype(int),
})
out_nav = f'{D}/输出/净值/杠杆情景_4x净值序列_修正版.csv'
navdf.to_csv(out_nav, encoding='utf-8-sig')
print(f'已导出：{out_nav}')
