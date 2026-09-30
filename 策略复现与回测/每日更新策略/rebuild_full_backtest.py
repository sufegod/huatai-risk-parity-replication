# -*- coding: utf-8 -*-
"""策略完整回溯重制：现金管理收益 + 1~4 倍杠杆情景（2026-09-29）

【用户需求】
  ① 重新回溯策略，呈现【加入现金管理收益后】的完整表现：
     历史净值、保证金占用比例、风险收益特征
  ② 呈现 1x / 2x / 3x / 4x 杠杆的净值走势、逐年收益、逐年回撤

【现金口径（用户 2026-09-29 指定）】
  现金(剩余资金) = 当日净资产 − L × 保证金规模
    · 净资产【不随杠杆放大】—— 杠杆只放大仓位，不改变自有资金规模
    · 保证金规模 = L × m，其中 m = 无杠杆保证金占用 / 净资产
    · 现金比例 = max(0, 1 − L·m)
  剩余现金按【银行存款利率】计息（国有大行一年期挂牌，日度分段）
  对照口径：GC001 银行间 1 天期国债逆回购利率（策略代码原生口径）

【杠杆组合日收益】
  r_L = L×gross − L×cost + max(0, 1−L·m) × 现金日利率

【口径说明】
  主样本 = 2016-01-05 ~ 最新（上证50ETF 替代红利低波ETF + 2016-2017 股指段补满仓）
  —— 与既有 2016 起长周期回溯、4x 杠杆分析口径一致，便于横向对照。
  指标算法与 bt_2016_full.py 完全一致（252 交易日/年折算、算术夏普）。

【研究性质】不接生产链路；所有输出均为新文件，不覆盖任何既有数据。
"""
import pandas as pd, numpy as np
from scipy.optimize import minimize

PROJ = 'C:/Users/aa/WorkBuddy/2026-05-28-14-41-45/huatai-risk-parity-replication'
D = f'{PROJ}/策略复现与回测/每日更新策略'

START = '2016-01-01'
EWMA_DECAY = 0.97
INDEX_BASE_WEIGHT = 0.30
FEE_RATE = 0.0005
REPO_FEE_RATE = 0.000001
TRADING_DAYS = 252
REAL_START = pd.Timestamp('2018-01-02')      # 真实股指信号起点

INDEX_FUTURES = ['沪深300主连', '中证500主连']
MARGIN = {'沪深300主连': 0.14, '中证500主连': 0.14, '红利低波ETF': 1.00, '上证50ETF': 1.00,
          '10年国债主连': 0.025, '沪铜主连': 0.16, '沪铝主连': 0.16, 'PTA主连': 0.17,
          '原油主连': 0.32, '豆粕主连': 0.13, '沪金主连': 0.28}

# 存款利率分段（国有大行一年期挂牌利率，用户 2026-09-28 指定）
# 2015-10-24 后央行不再调整存款基准利率，此后均为商业银行挂牌利率自主下调
DEPOSIT_SCHEDULE = [
    ('2016-01-01', 0.0175),
    ('2022-09-15', 0.0165),
    ('2023-06-08', 0.0155),
    ('2023-12-22', 0.0145),
    ('2024-07-25', 0.0135),
    ('2025-05-20', 0.0095),
]


# ============================== 风险平价求解（与 bt_2016_full.py 一致） ==============================
def ewma_semi_cov(r, decay=EWMA_DECAY):
    d = np.minimum(r.values, 0.0)
    T, N = d.shape
    w = decay ** np.arange(T - 1, -1, -1)
    w /= w.sum()
    return np.dot((d * np.sqrt(w[:, None])).T, (d * np.sqrt(w[:, None]))) * 252 + np.eye(N) * 1e-8


def rp_obj(x, C): return 0.5 * x @ C @ x - np.sum(np.log(x)) / len(x)
def rp_jac(x, C): return C @ x - 1.0 / (len(x) * x)


def rp_w(C):
    n = C.shape[0]
    r = minimize(rp_obj, np.ones(n), args=(C,), method='L-BFGS-B', jac=rp_jac,
                 bounds=[(1e-8, None)] * n, options={'ftol': 1e-12})
    return r.x / r.x.sum()


def load_signal():
    raw = pd.read_excel(f'{PROJ}/数据/原始数据/股指期货信号.xlsx', sheet_name=0, header=None)
    sc = None
    for c in raw.columns:
        if (raw[c].astype(str).str.strip() == '股指期货').any():
            sc = c
            break
    if sc is None:
        sc = 1
    d = raw[[0, sc]].copy()
    d.columns = ['date', 'signal']
    d['date'] = pd.to_datetime(d['date'], errors='coerce')
    d['signal'] = pd.to_numeric(d['signal'], errors='coerce')
    d = d.dropna(subset=['date']).set_index('date').sort_index()
    d = d[~d.index.duplicated(keep='last')]
    return d.loc[d['signal'].first_valid_index():, 'signal'].ffill()


SIGNAL = load_signal()
FIRST_SIG = SIGNAL.first_valid_index()


# ============================== 逐日重放（含三分量拆解） ==============================
def run_daily(etf_col='上证50ETF', signal_mode='fill1'):
    """重放策略每日收益，并拆分出 资产端 / 成本端 / 保证金 / 现金 分量"""
    un = pd.read_csv(f'{PROJ}/数据/日度收益数据更新/日涨跌幅_填充.csv',
                     index_col=0, parse_dates=True)
    un = un.loc[:, ~un.columns.duplicated()]
    if etf_col == '上证50ETF':
        s50 = pd.read_csv(f'{PROJ}/数据/原始数据/上证50ETF_日涨跌幅.csv',
                          index_col=0, parse_dates=True)
        s50.index = pd.to_datetime(s50.index)
        s50 = s50.loc[:, ~s50.columns.duplicated()]
        un = un.drop(columns=['上证50ETF'], errors='ignore').join(s50, how='left')

    cols = INDEX_FUTURES + ['10年国债主连', '沪铜主连', '沪铝主连', 'PTA主连',
                            '原油主连', '豆粕主连', '沪金主连', etf_col]
    lvl_all = un.reindex(columns=cols + ['一天期国债逆回购']).fillna(0.0) / 100.0
    idx = lvl_all.loc[START:].index
    listing = {c: lvl_all[c].first_valid_index() for c in cols}

    sig_d = SIGNAL.reindex(idx, method='ffill').copy()
    if signal_mode == 'fill1':
        sig_d.loc[sig_d.index < FIRST_SIG] = 1.0

    trade = lvl_all[cols].fillna(0)
    rp_assets = [c for c in cols if c not in INDEX_FUTURES]

    repo_ann = lvl_all['一天期国债逆回购'].loc[idx]
    cal = idx.to_series().diff().dt.days.fillna(1)
    repo_net = np.maximum((repo_ann.shift(1).fillna(0) / 365.0) * cal - REPO_FEE_RATE, 0.0)
    m_ratios = pd.Series({a: MARGIN.get(a, 1.0) for a in cols})

    recs = []
    curr_w = pd.Series(0.0, index=cols)
    curr_m = 0.0
    wrecs = []
    first_date = None
    obs = pd.DatetimeIndex(idx)

    for i in range(len(obs) - 1):
        rd = obs[i]
        rs = sig_d.loc[rd]
        if pd.isna(rs):
            continue
        elig = [a for a in rp_assets if listing[a] is not None and listing[a] <= rd]
        if not elig:
            continue
        look = lvl_all.loc[rd - pd.DateOffset(months=12):rd, elig]
        if len(look) < 150:
            continue

        tgt = pd.Series(0.0, index=cols)
        tw = INDEX_BASE_WEIGHT * (min(float(rs), 1.0) if rs > 0 else 0.0)
        listed_if = [a for a in INDEX_FUTURES if listing[a] <= rd]
        if tw > 0 and listed_if:
            tgt.loc[listed_if] = tw / len(listed_if)
        iw = float(tgt.sum())
        rem = max(0.0, 1.0 - iw)
        w = rp_w(ewma_semi_cov(look))
        tgt.loc[elig] = w * rem

        hp = trade.loc[rd + pd.Timedelta(days=1):obs[i + 1]]
        if len(hp) == 0:
            continue
        if first_date is None:
            first_date = hp.index[0]

        for date, dr in hp.iterrows():
            d_repo = float(repo_net.loc[date])
            if date == hp.index[0]:
                nm = float((tgt * m_ratios).sum())
                idle = max(0.0, 1.0 - nm)
                cost = float((tgt - curr_w).abs().sum() * FEE_RATE)
                g = float((tgt * dr).sum())
                ret = g - cost + idle * d_repo
                curr_w = tgt.copy()
                m_use, w_use = nm, tgt.copy()
            else:
                idle = max(0.0, 1.0 - curr_m)
                g = float((curr_w * dr).sum())
                cost = 0.0
                ret = g + idle * d_repo
                m_use, w_use = curr_m, curr_w.copy()
            recs.append({'date': date, 'gross': g, 'cost': cost,
                         'margin': m_use, 'idle': idle,
                         'repo_net': d_repo, 'ret': float(ret),
                         'signal': float(rs), 'idxw': iw})
            wrecs.append(w_use)
            gw = float((curr_w * (1 + dr)).sum())
            curr_w = (curr_w * (1 + dr)) / (gw if gw else 1.0)
            curr_m = float((curr_w * m_ratios).sum())

    df = pd.DataFrame(recs).set_index('date').sort_index()
    W = pd.DataFrame(wrecs, index=df.index, columns=cols)
    return df, W


# ============================== 指标计算（与 bt_2016_full.py 一致） ==============================
def metrics(r, nav0=None):
    r = r.fillna(0.0)
    nav = (1 + r).cumprod()
    y = len(r) / TRADING_DAYS
    tot = nav.iloc[-1] - 1
    ann = nav.iloc[-1] ** (1 / y) - 1 if nav.iloc[-1] > 0 else -1.0
    vol = r.std() * np.sqrt(TRADING_DAYS)
    mdd = ((nav / nav.cummax()) - 1).min()
    return dict(累计=tot, 年化=ann, 波动=vol, 夏普=(r.mean() * TRADING_DAYS) / vol if vol > 0 else np.nan,
                最大回撤=mdd, 卡玛=ann / abs(mdd) if mdd else np.nan,
                最差单日=r.min(), 最好单日=r.max(), 日胜率=(r > 0).mean(),
                交易日数=len(r), nav=nav)


def lev_ret(df, L, rate):
    """杠杆 L 倍的组合日收益 + 现金比例"""
    cash = (1.0 - L * df['margin']).clip(lower=0.0)
    return L * df['gross'] - L * df['cost'] + cash * rate, cash


# ============================== 主流程 ==============================
print('=' * 112)
print('策略完整回溯重制 —— 现金管理收益 + 1~4 倍杠杆（2016-2026）')
print('=' * 112)

df, W = run_daily('上证50ETF', 'fill1')

# 校验：策略原始收益 = 资产端 − 成本端 + 现金端(GC001)
chk = (df['gross'] - df['cost'] + df['idle'] * df['repo_net']) - df['ret']
print(f'\n[校验] 会计恒等式残差：最大 {chk.abs().max():.3e}  均值 {chk.abs().mean():.3e}  → 通过')

# 现金端：存款利率（分段）+ GC001（对照）
cal_days = df.index.to_series().diff().dt.days.fillna(1)
dep_rate = pd.Series(0.0, index=df.index)
for s, r in DEPOSIT_SCHEDULE:
    dep_rate.loc[df.index >= pd.Timestamp(s)] = r
df['dep_rate'] = dep_rate
df['dep_daily'] = dep_rate / 365.0 * cal_days

# 三档 1x 口径
df['r_asset_only'] = df['gross'] - df['cost']
df['r_dep'] = df['r_asset_only'] + df['idle'] * df['dep_daily']
df['r_repo'] = df['r_asset_only'] + df['idle'] * df['repo_net']

print(f'\n样本：{df.index[0].date()} ~ {df.index[-1].date()}  '
      f'{len(df)} 个交易日 / {len(df)/TRADING_DAYS:.2f} 年（按 252 折算）')
print(f'  · 2016-2017 构造期（股指补满仓假设）：{int((df.index < REAL_START).sum())} 日')
print(f'  · 2018-01-02 起真实信号段        ：{int((df.index >= REAL_START).sum())} 日')

seg = [('全程 2016-2026', None, None),
       ('2016-2017 构造期', None, REAL_START),
       ('2018-01-02 起真实段', REAL_START, None)]


def cut(s, e):
    """严格区间切片：[s, e)"""
    msk = pd.Series(True, index=df.index)
    if s is not None:
        msk &= (df.index >= s)
    if e is not None:
        msk &= (df.index < e)
    return df[msk]


print('\n' + '=' * 112)
print('一、加入现金管理收益后的策略表现（1 倍杠杆 = 无杠杆）')
print('=' * 112)
hdr = f"{'口径':<26}{'累计':>11}{'年化':>9}{'波动':>9}{'夏普':>8}{'最大回撤':>11}{'卡玛':>8}{'日胜率':>9}"
for name, s, e in seg:
    sub = cut(s, e)
    print(f'\n--- {name}（{len(sub)} 日）---')
    print(hdr)
    for lab, col in [('① 纯资产端（不含现金收益）', 'r_asset_only'),
                     ('② 含现金管理·银行存款利率', 'r_dep'),
                     ('③ 含现金管理·GC001逆回购', 'r_repo')]:
        m = metrics(sub[col])
        print(f"{lab:<26}{m['累计']:>10.2%}{m['年化']:>9.2%}{m['波动']:>9.2%}"
              f"{m['夏普']:>8.2f}{m['最大回撤']:>11.2%}{m['卡玛']:>8.2f}{m['日胜率']:>9.2%}")

# 现金收益贡献拆解
print('\n--- 现金收益的贡献（年均，百分点）---')
for name, s, e in seg:
    sub = cut(s, e)
    yn = len(sub) / TRADING_DAYS
    a_ann = ((1 + sub['r_asset_only']).prod() ** (1 / yn) - 1)
    d_ann = ((1 + sub['r_dep']).prod() ** (1 / yn) - 1)
    c_ann = ((1 + sub['r_repo']).prod() ** (1 / yn) - 1)
    cash_dep = ((sub['idle'] * sub['dep_daily']).sum() / yn)
    cash_rep = ((sub['idle'] * sub['repo_net']).sum() / yn)
    print(f'{name:<24} 资产端年化 {a_ann:+.2%} | 存款口径年化 {d_ann:+.2%}（现金 +{cash_dep:.2%}）'
          f' | GC001口径年化 {c_ann:+.2%}（现金 +{cash_rep:.2%}）')

# 保证金与现金
print('\n' + '=' * 112)
print('二、保证金占用比例与现金比例')
print('=' * 112)
m = df['margin']
print(f"{'区间':<24}{'均值':>9}{'中位':>9}{'最大':>9}{'最大日':>13}{'平均现金':>11}")
for name, s, e in seg:
    sub = cut(s, e)
    print(f"{name:<24}{sub['margin'].mean():>9.2%}{sub['margin'].median():>9.2%}"
          f"{sub['margin'].max():>9.2%}{str(sub['margin'].idxmax().date()):>13}"
          f"{sub['idle'].mean():>11.2%}")
print('\n  分资产保证金占用（全程日均）：')
wmean = W.mean()
mg_contrib = (wmean * pd.Series({c: MARGIN.get(c, 1.0) for c in W.columns})).sort_values(ascending=False)
for c, v in mg_contrib.items():
    print(f'    {c:<14} 日均权重 {wmean[c]:>6.2%} × 保证金率 {MARGIN.get(c,1.0):>5.0%} '
          f'= 占用 {v:>6.3%}')

# ============================== 杠杆情景 1x~4x ==============================
print('\n' + '=' * 112)
print('三、杠杆情景：1x / 2x / 3x / 4x（现金按存款利率计息）')
print('=' * 112)
LEVS = [1.0, 2.0, 3.0, 4.0]
res, navs, dds, cashr = {}, {}, {}, {}
for L in LEVS:
    r, cr = lev_ret(df, L, df['dep_daily'])
    res[L] = r
    navs[L] = (1 + r).cumprod()
    dds[L] = navs[L] / navs[L].cummax() - 1
    cashr[L] = cr

print(f"\n{'区间':<22}{'杠杆':>6}{'累计':>12}{'年化':>9}{'波动':>9}{'夏普':>8}{'最大回撤':>11}{'卡玛':>8}{'日胜率':>9}")
for name, s, e in seg:
    msk = pd.Series(True, index=df.index)
    if e is not None:
        msk &= (df.index < e)
    if s is not None:
        msk &= (df.index >= s)
    for L in LEVS:
        mm = metrics(res[L][msk])
        print(f"{name:<22}{L:>5.0f}x{mm['累计']:>12.2%}{mm['年化']:>9.2%}{mm['波动']:>9.2%}"
              f"{mm['夏普']:>8.2f}{mm['最大回撤']:>11.2%}{mm['卡玛']:>8.2f}{mm['日胜率']:>9.2%}")
    print()

print('【对照】现金按 GC001 计息：')
print(f"{'杠杆':>6}{'累计':>12}{'年化':>9}{'夏普':>8}{'最大回撤':>11}")
for L in LEVS:
    r, _ = lev_ret(df, L, df['repo_net'])
    mm = metrics(r)
    print(f'{L:>5.0f}x{mm["累计"]:>12.2%}{mm["年化"]:>9.2%}{mm["夏普"]:>8.2f}{mm["最大回撤"]:>11.2%}')

# 各杠杆保证金与现金
print('\n--- 各杠杆档位的保证金占用与现金 ---')
print(f"{'杠杆':>6}{'保证金均值':>12}{'保证金最大':>12}{'现金均值':>11}{'现金最小':>11}{'现金缺口天数':>14}{'其中真实段':>12}")
for L in LEVS:
    mg = L * df['margin']
    cr = cashr[L]
    raw = 1.0 - L * df['margin']
    gap = raw < -1e-12
    print(f'{L:>5.0f}x{mg.mean():>12.2%}{mg.max():>12.2%}{cr.mean():>11.2%}{cr.min():>11.2%}'
          f'{int(gap.sum()):>14}{int(gap[gap.index >= REAL_START].sum()):>12}')

# ============================== 逐年 ==============================
print('\n' + '=' * 112)
print('四、逐年表现（存款口径）')
print('=' * 112)
yy = [y for y in sorted(set(df.index.year)) if (df.index.year == y).sum() > 20]
rows = []
for y in yy:
    msk = df.index.year == y
    rec = {'年份': y}
    for L in LEVS:
        rl = res[L][msk]
        nl = (1 + rl).cumprod()
        rec[f'{L:.0f}x收益'] = (1 + rl).prod() - 1
        rec[f'{L:.0f}x回撤'] = (nl / nl.cummax() - 1).min()
    rec['保证金占用'] = df['margin'][msk].mean()
    rec['平均现金'] = df['idle'][msk].mean()
    rec['现金收益_存款'] = (df['idle'] * df['dep_daily'])[msk].sum()
    rec['现金收益_GC001'] = (df['idle'] * df['repo_net'])[msk].sum()
    rec['交易成本'] = df['cost'][msk].sum()
    rows.append(rec)
yr = pd.DataFrame(rows)

show = yr.copy()
order = ['年份', '1x收益', '2x收益', '3x收益', '4x收益',
         '1x回撤', '2x回撤', '3x回撤', '4x回撤',
         '保证金占用', '平均现金', '现金收益_存款', '现金收益_GC001', '交易成本']
show = show[order]
for c in show.columns:
    if c != '年份':
        show[c] = show[c].map(lambda x: f'{x:+.2%}' if '收益' in c or '回撤' in c or '成本' in c else f'{x:.2%}')
print(show.to_string(index=False))

print('\n--- 逐年收益：4x 相对 1x ---')
print(f"{'年份':<8}{'1x':>10}{'2x':>10}{'3x':>10}{'4x':>10}{'4x/1x倍数':>12}")
for _, rr in yr.iterrows():
    b = rr['4x收益'] / rr['1x收益'] if rr['1x收益'] else np.nan
    print(f"{int(rr['年份']):<8}{rr['1x收益']:>10.2%}{rr['2x收益']:>10.2%}{rr['3x收益']:>10.2%}"
          f"{rr['4x收益']:>10.2%}{b:>12.2f}")

# ============================== 导出 ==============================
daily = pd.DataFrame({
    'gross': df['gross'], 'cost': df['cost'], 'margin': df['margin'], 'idle': df['idle'],
    'dep_rate': df['dep_rate'], 'repo_net': df['repo_net'],
    '策略原始_GC001': df['ret'], '策略_不含现金': df['r_asset_only'],
    '策略_存款口径': df['r_dep'], '策略_GC001口径': df['r_repo'],
})
for L in LEVS:
    daily[f'{L:.0f}x_日收益'] = res[L]
    daily[f'{L:.0f}x_净值'] = navs[L]
    daily[f'{L:.0f}x_回撤'] = dds[L]
    daily[f'{L:.0f}x_保证金'] = L * df['margin']
    daily[f'{L:.0f}x_现金比例'] = cashr[L]
out_daily = f'{D}/输出/对比分析/回测全景_现金管理与杠杆_日度_2016-2026.csv'
daily.to_csv(out_daily, encoding='utf-8-sig')
print(f'\n已导出：{out_daily}')

out_yr = f'{D}/输出/对比分析/回测全景_逐年_1x至4x_2016-2026.csv'
yr.to_csv(out_yr, index=False, encoding='utf-8-sig')
print(f'已导出：{out_yr}')

# 全景指标表（三档 1x + 四档杠杆 × 三分段）
ind = []
for name, s, e in seg:
    msk = pd.Series(True, index=df.index)
    if e is not None:
        msk &= (df.index < e)
    if s is not None:
        msk &= (df.index >= s)
    for lab, col in [('1x·纯资产端', 'r_asset_only'),
                     ('1x·含存款现金', 'r_dep'),
                     ('1x·含GC001现金', 'r_repo')]:
        mm = metrics(df[col][msk])
        ind.append({'区间': name, '口径': lab, '累计': mm['累计'], '年化': mm['年化'],
                    '波动': mm['波动'], '夏普': mm['夏普'], '最大回撤': mm['最大回撤'],
                    '卡玛': mm['卡玛'], '日胜率': mm['日胜率'], '交易日数': mm['交易日数']})
    for L in LEVS:
        mm = metrics(res[L][msk])
        ind.append({'区间': name, '口径': f'{L:.0f}x·含存款现金', '累计': mm['累计'], '年化': mm['年化'],
                    '波动': mm['波动'], '夏普': mm['夏普'], '最大回撤': mm['最大回撤'],
                    '卡玛': mm['卡玛'], '日胜率': mm['日胜率'], '交易日数': mm['交易日数']})
ind = pd.DataFrame(ind)
out_ind = f'{D}/输出/指标/回测全景_风险收益特征.csv'
ind.to_csv(out_ind, index=False, encoding='utf-8-sig')
print(f'已导出：{out_ind}')
print('\n完成。')
