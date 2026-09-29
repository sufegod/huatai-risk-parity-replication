"""逐年业绩归因分析：把策略收益拆解到每个资产

方法：
  1. 用每日持仓权重（按当日各资产涨跌幅漂移后的实际权重）× 当日资产收益 = 该资产当日贡献
  2. 逐年/逐月汇总，得到每个资产对策略收益的贡献（百分点）
  3. 计算逐年资产相关性矩阵，判断"分散化是否失效"
  4. 计算分散化比率（加权波动 ÷ 组合波动）

现金口径（三个独立概念，前两个恒为正）：
  [概念① 存款]    cash_ret = (1 − 保证金消耗) × 存款年化利率/365 × 自然日
                  存款利率按国有大行一年期挂牌利率**日度分段**（用户指定，2026-09-28）：
                    2016-01-01 ~ 2022-09-14 : 1.75%
                    2022-09-15 ~ 2023-06-07 : 1.65%
                    2023-06-08 ~ 2023-12-21 : 1.55%
                    2023-12-22 ~ 2024-07-24 : 1.45%
                    2024-07-25 ~ 2025-05-19 : 1.35%
                    2025-05-20 ~ 至今        : 0.95%
                  （2015-10-24 后央行不再调整存款基准利率，此处用商业银行挂牌利率）
  [概念② GC001]  cash_ret = (1 − 保证金消耗) × GC001实际利率/365 × 自然日
                  逆回购真实市场利率，作为对照口径
  [概念③ 机会成本] opportunity = 保证金消耗 × gross
                  唯一可正可负的"拖累"项

  三者相互独立，① ② 恒 ≥ 0，绝不出现负数。

现金余额 = 资产净规模(1.0) − 保证金消耗（券商实际口径 MARGIN_RATIOS_BROKER）

运行：C:/Users/aa/.workbuddy/binaries/python/envs/default/Scripts/python.exe attrib_yearly.py
"""
import pandas as pd, numpy as np

P = 'C:/Users/aa/WorkBuddy/2026-05-28-14-41-45/huatai-risk-parity-replication'
COLS = ['沪深300主连', '中证500主连', '10年国债主连', '沪铜主连', '沪铝主连',
        'PTA主连', '原油主连', '豆粕主连', '沪金主连', '上证50ETF']
CLASS = {'沪深300主连': '股指', '中证500主连': '股指', '10年国债主连': '债券',
         '沪铜主连': '商品', '沪铝主连': '商品', 'PTA主连': '商品',
         '原油主连': '商品', '豆粕主连': '商品', '沪金主连': '黄金',
         '上证50ETF': '权益ETF'}

# ---------- 存款利率分段（国有大行一年期挂牌利率，用户指定 2026-09-28） ----------
# 2015-10-24 后央行不再调整存款基准利率，故采用商业银行挂牌利率
DEPOSIT_SCHEDULE = [
    ('2016-01-01', 0.0175),   # 央行基准 1.50%，大行上浮至 1.75%（长期稳定期）
    ('2022-09-15', 0.0165),   # 第一次集体降息
    ('2023-06-08', 0.0155),   # 第二次（6/8 降后 9/1 续降至 1.55）
    ('2023-12-22', 0.0145),   # 第三次
    ('2024-07-25', 0.0135),   # 第四次（10/18 后实际挂牌 1.10%）
    ('2025-05-20', 0.0095),   # 第五次，进入"1时代"
]

FEE_RATE = 0.0005                 # 单边交易费率（与策略代码一致）

# ---------- 读数据 ----------
d = pd.read_csv(f'{P}/策略复现与回测/每日更新策略/输出/净值/2016起回溯_日收益对比.csv',
                index_col=0, parse_dates=True)
# 注意：文件列名有误导性！'上证50ETF_日收益' 实际是【策略日收益】（bt_2016_full.py 的 retB）
# 见 bt_2016_full.py line 229：pd.DataFrame({'红利低波ETF_日收益': retA, '上证50ETF_日收益': retB})
strat = d['上证50ETF_日收益']

un = pd.read_csv(f'{P}/数据/日度收益数据更新/日涨跌幅_填充.csv', index_col=0, parse_dates=True)
un = un.loc[:, ~un.columns.duplicated()]
s50 = pd.read_csv(f'{P}/数据/原始数据/上证50ETF_日涨跌幅.csv', index_col=0, parse_dates=True)
s50.index = pd.to_datetime(s50.index)
s50 = s50.loc[:, ~s50.columns.duplicated()]
un = un.drop(columns=['上证50ETF'], errors='ignore').join(s50, how='left')

rec = pd.read_csv(f'{P}/策略复现与回测/每日更新策略/输出/对比分析/2016起_仓位明细_上证50.csv',
                  parse_dates=['date']).set_index('date')

lvl = un.reindex(columns=COLS + ['一天期国债逆回购']).fillna(0.0) / 100.0
repo = lvl['一天期国债逆回购']

# ---------- 重放每日持仓权重（与 bt_2016_full.py 严格一致） ----------
# 关键：bt_2016_full.py 的 obs = 全部交易日 → **每日都是调仓日**（2607 次）
#   各日权重 = rec 中"次一交易日起生效"的目标权重，当日不漂移
#   contrib = 当日权重 × 当日资产收益，行和 ≡ 1
#   margin  = Σ(当日权重 × 券商保证金率)
#
# 保证金率必须用策略代码的 MARGIN_RATIOS_BROKER（券商实际口径，含加收），
# 不能用交易所标准值——两者相差约 4.5 pct（占用 0.1210 vs 0.1659）。
MARGIN = {'沪深300主连': 0.14, '中证500主连': 0.14, '10年国债主连': 0.025,
          '沪铜主连': 0.16, '沪铝主连': 0.16, 'PTA主连': 0.17,
          '原油主连': 0.32, '豆粕主连': 0.13, '沪金主连': 0.28,
          '上证50ETF': 1.00}
m_ratio = pd.Series({c: MARGIN.get(c, 1.0) for c in COLS})

idx = strat.index
rr = lvl.loc[idx, COLS]
recw = rec[COLS]

# 把 rebalance_date 映射到"次一交易日"（权重生效日）
eff = {}
for rdt in rec.index:
    cand = idx[idx > rdt]
    if len(cand):
        eff[cand[0]] = rdt

# 构造每日权重矩阵
W = pd.DataFrame(np.nan, index=idx, columns=COLS)
for dt, rdt in eff.items():
    W.loc[dt] = recw.loc[rdt].values
W = W.ffill()

# 只保留策略有值的区间（起始日之前无持仓）
valid = ~W.isna().all(axis=1)
W = W.loc[valid].fillna(0.0)
idx = W.index

daily_w = W
contrib = daily_w.mul(rr.loc[idx], axis=0)
gross = contrib.sum(axis=1)
margin_series = (daily_w * m_ratio).sum(axis=1)

# 交易成本：按相邻交易日权重变动计算
cost_series = (daily_w.diff().abs().sum(axis=1).fillna(0.0) * FEE_RATE)
cost_daily = cost_series

# ================= 现金口径（存款利率）=================
# 【用户指定口径 2026-09-28】
#   资产净规模 = 1.0（100%）
#   保证金消耗 = Σ(持仓权重 × 券商保证金率)          → margin_daily
#   现金余额   = 1.0 − 保证金消耗                    → idle_ratio_daily
#   现金收益   = 现金余额 × 存款年化利率 / 365 × 自然日间隔
#
# 三者相互独立：
#   ① 现金收益_存款 = 现金余额 × 存款利率（分段）    → 恒 ≥ 0
#   ② 现金收益_GC001 = 现金余额 × GC001实际利率      → 恒 ≥ 0（对照口径）
#   ③ 机会成本       = 保证金消耗 × gross            → 唯一可正可负
#
# 会计恒等式：
#   策略收益 = gross(名义加权收益) + 现金收益_GC001 − 交易成本
#   （存款口径仅作对照，不参与该恒等式）

# ---------- 资金占用与现金余额 ----------
# 资金占用直接取重放时记录的每日 Σ(权重 × 券商保证金率)
margin_daily = margin_series
idle_ratio_daily = (1.0 - margin_daily).clip(lower=0)  # 每日现金（闲置）比例

cal_days = idx.to_series().diff().dt.days.fillna(1)
cal_days.index = idx

# ---------- ① 现金收益_存款：现金余额 × 存款年化利率（分段） ----------
_rate_ser = pd.Series(0.0, index=idx)
for _start, _r in DEPOSIT_SCHEDULE:
    _rate_ser.loc[idx >= pd.Timestamp(_start)] = _r
deposit_ann = _rate_ser.copy()
deposit_ann.name = '存款年化'
deposit_daily = (deposit_ann / 365.0) * cal_days
idle_ret_deposit = idle_ratio_daily * deposit_daily            # 恒 ≥ 0

# ---------- ② 现金收益_GC001：现金余额 × GC001 实际利率（对照） ----------
# repo 已由 lvl 给出（已 /100，是小数形式）
# GC001 按自然日折日，T-1 可得
repo_ann = repo.reindex(idx).fillna(0.0).shift(1).fillna(0.0)   # 小数形式年化利率
# 与策略一致：扣除手续费并保底为 0
repo_daily = np.maximum((repo_ann / 365.0) * cal_days - 1e-6, 0.0)
idle_ret_gc = idle_ratio_daily * repo_daily                     # 恒 ≥ 0

# 保留变量别名（供对账段落引用）
idle_ret_real = idle_ret_gc

# ---------- ③ 机会成本 ----------
# 口径：现金余额若按【同等保证金率】加杠杆配置风险平价资产，本可获得的收益
#      = 保证金消耗比例 × gross
# 注：这是理论上限口径，用来量化"资金效率损失"；实际现金受流动性约束，
#     真实可获收益介于存款/GC001 与机会成本之间。
opportunity = margin_daily * gross
# 资产净贡献 = gross − 机会成本 = gross × 现金余额比例（即可用资金真正能赚的部分）
net_asset_contrib = gross - opportunity

# ---------- 对账校验 ----------
# 恒等式：gross(资产贡献) + 现金收益_GC001 − 交易成本 = 策略收益
resid = strat - gross - idle_ret_real + cost_daily
opp_cost = opportunity

print('=' * 100)
print('=== 口径说明（现金收益永远是正收益，负数只可能出现在"机会成本"列）===')
print('   资产净规模 1.0 − 保证金消耗 = 现金余额；现金收益 = 现金余额 × 年化利率/365 × 自然日')
print(f'① 现金收益_存款：现金余额（均值 {idle_ratio_daily.mean():.2%}）× 存款年化利率（分段）')
for _s, _r in DEPOSIT_SCHEDULE:
    print(f'     自 {_s} 起：{_r*100:.2f}%')
print(f'② 现金收益_GC001：现金余额 × GC001 实际利率（区间均值 {repo_ann.mean()*100:.2f}%）')
print(f'③ 机会成本      ：保证金消耗 × gross（若该笔现金也按同等保证金率加杠杆配置）')
print(f'   平均实际保证金消耗 {margin_daily.mean():.2%}，平均现金余额 {idle_ratio_daily.mean():.2%}')
print('-' * 100)
print(f'[对账] 资产贡献(gross) + 现金收益_GC001 − 交易成本 = 策略收益，残差均值 {abs(resid).mean():.2e}')
print(f'  全期合计：资产贡献(gross) {gross.sum()*100:+.2f} pct'
      f'（年均 {gross.sum()/(len(idx)/243)*100:+.2f}）')
print(f'            减:机会成本    {-opportunity.sum()*100:+.2f} pct'
      f'（年均 {-opportunity.sum()/(len(idx)/243)*100:+.2f}）')
print(f'            = 资产净贡献   {net_asset_contrib.sum()*100:+.2f} pct'
      f'（年均 {net_asset_contrib.sum()/(len(idx)/243)*100:+.2f}）')
print(f'            现金收益_存款  {idle_ret_deposit.sum()*100:+.2f} pct'
      f'（年均 {idle_ret_deposit.sum()/(len(idx)/243)*100:+.2f}）')
print(f'            现金收益_GC001 {idle_ret_gc.sum()*100:+.2f} pct'
      f'（年均 {idle_ret_gc.sum()/(len(idx)/243)*100:+.2f}）')
print(f'            交易成本      {-cost_daily.sum()*100:+.2f} pct'
      f'（年均 {-cost_daily.sum()/(len(idx)/243)*100:+.2f}）')
print(f'            = 策略收益     {strat.sum()*100:+.2f} pct'
      f'（年均 {strat.sum()/(len(idx)/243)*100:+.2f}）')
print(f'  两口径差异（存款−GC001）年均 {(idle_ret_deposit.sum()-idle_ret_gc.sum())/(len(idx)/243)*100:+.2f} pct')

# ---------- 逐年归因 ----------
years = sorted(set(idx.year))
rows = []
for y in years:
    m = idx.year == y
    if m.sum() < 20:
        continue
    tot = strat[m].sum()
    r = {'年份': y, '策略收益': tot, '交易日': int(m.sum())}
    for c in COLS:
        r[c] = contrib.loc[m, c].sum()
    r['现金收益_存款'] = idle_ret_deposit[m].sum()
    r['现金收益_GC001'] = idle_ret_gc[m].sum()
    r['机会成本'] = opp_cost[m].sum()
    r['交易成本'] = -cost_daily[m].sum()
    r['实际闲置比例'] = idle_ratio_daily[m].mean()
    r['实际资金占用'] = margin_daily[m].mean()
    r['GC001年化'] = repo_ann[m].mean() * 100        # 转成百分点
    r['存款年化'] = deposit_ann[m].mean() * 100      # 转成百分点
    for cls in ['股指', '债券', '商品', '黄金', '权益ETF']:
        r[cls] = sum(r[c] for c in COLS if CLASS[c] == cls)
    rows.append(r)

at = pd.DataFrame(rows)
at.to_csv(f'{P}/策略复现与回测/每日更新策略/输出/对比分析/逐年归因_2016-2026.csv',
          index=False, encoding='utf-8-sig')

print('\n' + '=' * 100)
print('逐年归因（单位：pct）—— 现金收益全部为正，机会成本才是拖累')
print('=' * 100)
hdr = (f"{'年份':<6}{'策略':>8}{'股指':>8}{'债券':>8}{'商品':>8}{'黄金':>8}{'ETF':>7}"
       f"{'现金_存款':>10}{'现金_GC001':>11}{'机会成本':>10}{'交易成本':>10}")
print(hdr)
print('-' * 116)
for _, r in at.iterrows():
    print(f"{int(r['年份']):<6}{r['策略收益']*100:>8.2f}{r['股指']*100:>8.2f}{r['债券']*100:>8.2f}"
          f"{r['商品']*100:>8.2f}{r['黄金']*100:>8.2f}{r['权益ETF']*100:>7.2f}"
          f"{r['现金收益_存款']*100:>10.2f}{r['现金收益_GC001']*100:>11.2f}"
          f"{r['机会成本']*100:>10.2f}{r['交易成本']*100:>10.2f}")

# ---------- 逐年相关性 ----------
print('\n' + '=' * 100)
print('逐年资产相关性（等权平均相关系数 = 分散化程度，越低越好）')
print('=' * 100)
corr_rows = []
core = COLS
print(f"{'年份':<6}{'平均相关':>10}{'股债相关':>10}{'股商相关':>10}{'债金相关':>10}{'最低对':>22}")
for y in years:
    m = idx.year == y
    if m.sum() < 20:
        continue
    cm = rr.loc[m, core].corr()
    vals = cm.where(np.triu(np.ones(cm.shape), 1).astype(bool)).stack()
    avg = vals.mean()
    eq = cm.loc['沪深300主连', '10年国债主连']
    st = cm.loc['沪深300主连', ['沪铜主连', '沪铝主连', 'PTA主连', '原油主连', '豆粕主连']].mean()
    bg = cm.loc['10年国债主连', '沪金主连']
    mn = vals.idxmin()
    corr_rows.append({'年份': y, '平均相关': avg, '股债相关': eq, '股商相关': st,
                      '债金相关': bg, '最低相关对': f'{mn[0]}-{mn[1]}', '最低值': vals.min()})
    print(f"{y:<6}{avg:>10.3f}{eq:>10.3f}{st:>10.3f}{bg:>10.3f}"
          f"{mn[0]+'-'+mn[1]:>22}")
pd.DataFrame(corr_rows).to_csv(
    f'{P}/策略复现与回测/每日更新策略/输出/对比分析/逐年相关性_2016-2026.csv',
    index=False, encoding='utf-8-sig')

# ---------- 分散化比率 ----------
print('\n' + '=' * 100)
print('分散化比率（各资产加权波动 ÷ 组合实际波动；>1 说明分散化有效，越大约好）')
print('=' * 100)
print(f"{'年份':<6}{'加权波动':>12}{'组合波动':>12}{'分散化比率':>12}{'评价':>14}")
dr_rows = []
for y in years:
    m = idx.year == y
    if m.sum() < 20:
        continue
    avg_w = daily_w.loc[m].mean()
    asset_vol = rr.loc[m, core].std() * np.sqrt(252)
    wavg = float((avg_w[core] * asset_vol).sum())
    port = float(strat[m].std() * np.sqrt(252))
    ratio = wavg / port if port > 0 else np.nan
    lab = '优秀' if ratio > 2.5 else ('良好' if ratio > 2.0 else ('一般' if ratio > 1.5 else '失效'))
    dr_rows.append({'年份': y, '加权波动': wavg, '组合波动': port, '分散化比率': ratio, '评价': lab})
    print(f"{y:<6}{wavg:>12.2%}{port:>12.2%}{ratio:>12.2f}{lab:>14}")
pd.DataFrame(dr_rows).to_csv(
    f'{P}/策略复现与回测/每日更新策略/输出/对比分析/逐年分散化比率_2016-2026.csv',
    index=False, encoding='utf-8-sig')

# ---------- 逐年资产自身收益 ----------
print('\n' + '=' * 100)
print('逐年各资产自身收益（%）')
print('=' * 100)
ar = pd.DataFrame(index=years, columns=core, dtype=float)
for y in years:
    m = idx.year == y
    if m.sum() < 20:
        continue
    for c in core:
        ar.loc[y, c] = ((1 + rr.loc[m, c]).prod() - 1) * 100
pd.set_option('display.width', 250)
print(ar.round(2).to_string())
ar.round(4).to_csv(f'{P}/策略复现与回测/每日更新策略/输出/对比分析/逐年资产收益_2016-2026.csv',
                   encoding='utf-8-sig')

# ---------- 逐年平均权重 ----------
print('\n' + '=' * 100)
print('逐年平均持仓权重（%）')
print('=' * 100)
aw = pd.DataFrame(index=years, columns=core, dtype=float)
for y in years:
    m = idx.year == y
    if m.sum() < 20:
        continue
    for c in core:
        aw.loc[y, c] = daily_w.loc[m, c].mean() * 100
print(aw.round(2).to_string())
aw.round(4).to_csv(f'{P}/策略复现与回测/每日更新策略/输出/对比分析/逐年平均权重_2016-2026.csv',
                   encoding='utf-8-sig')

# ---------- 现金收益三口径对照表 ----------
print('\n' + '=' * 100)
print('现金收益三口径对照（pct）—— 前两列恒为正，第三列才是拖累')
print('=' * 100)
cmp = at[['年份', '策略收益', '现金收益_存款', '现金收益_GC001', '机会成本',
          '交易成本', '实际资金占用', '实际闲置比例', '存款年化', 'GC001年化']].copy()
cmp.columns = ['年份', '策略收益', '现金收益_存款(分段)', '现金收益_GC001',
               '机会成本(拖累)', '交易成本', '保证金消耗', '现金余额比例',
               '存款年化%', 'GC001年化%']
pd.set_option('display.width', 250)
print(cmp.round(4).to_string(index=False))
print('-' * 130)
print(f"{'全期合计':<10}{cmp['策略收益'].sum()*100:>10.2f}"
      f"{cmp['现金收益_存款(分段)'].sum()*100:>20.2f}{cmp['现金收益_GC001'].sum()*100:>16.2f}"
      f"{cmp['机会成本(拖累)'].sum()*100:>16.2f}{cmp['交易成本'].sum()*100:>12.2f}"
      f"{cmp['保证金消耗'].mean():>12.2%}{cmp['现金余额比例'].mean():>12.2%}")
cmp.round(4).to_csv(
    f'{P}/策略复现与回测/每日更新策略/输出/对比分析/现金收益三口径对照_2016-2026.csv',
    index=False, encoding='utf-8-sig')

print('\n已导出 6 份归因文件 + 1 份现金收益三口径对照')
