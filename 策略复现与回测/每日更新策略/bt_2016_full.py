"""2016 起长周期回溯回测（上证50ETF 替代红利低波ETF + 股指段补全）

口径设定（用户 2026-09-28 确认）：
  1. 资产池：上证50ETF（510050.SH）替代 红利低波ETF（512890.SH）
  2. 数据口径：用「填充表」日涨跌幅_填充.csv，使 2016 年全资产齐备
  3. 股指信号缺口（2016-01 ~ 2017-12 无信号，信号表 2018-01-02 才开始）：
     视为「满仓」→ 信号 = 1.0 → 股指仓位 = INDEX_BASE_WEIGHT(0.30)，IF/IC 各半
  4. 回测区间：2016-01-04 起（受 12 个月回看窗口约束，最早可再平衡日约 2016-01-04）

运行：C:/Users/aa/.workbuddy/binaries/python/envs/default/Scripts/python.exe bt_2016_full.py
"""
import pandas as pd, numpy as np
from scipy.optimize import minimize

PROJ = 'C:/Users/aa/WorkBuddy/2026-05-28-14-41-45/huatai-risk-parity-replication'
START = '2016-01-01'
EWMA_DECAY = 0.97
INDEX_BASE_WEIGHT = 0.30
FEE_RATE = 0.0005
REPO_FEE_RATE = 0.000001
INDEX_FUTURES = ['沪深300主连', '中证500主连']
MARGIN = {'沪深300主连': 0.14, '中证500主连': 0.14, '红利低波ETF': 1.00, '上证50ETF': 1.00,
          '10年国债主连': 0.025, '沪铜主连': 0.16, '沪铝主连': 0.16, 'PTA主连': 0.17,
          '原油主连': 0.32, '豆粕主连': 0.13, '沪金主连': 0.28}


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


def calc_metrics(r):
    r = r.fillna(0)
    if len(r) < 5:
        return {}
    nav = (1 + r).cumprod()
    y = len(r) / 252
    ann = nav.iloc[-1] ** (1 / y) - 1 if nav.iloc[-1] > 0 else -1.0
    vol = r.std() * np.sqrt(252)
    return dict(累计=nav.iloc[-1] - 1, 年化=ann, 波动=vol,
                夏普=(r.mean() * 252) / vol if vol > 0 else 0,
                回撤=((nav / nav.cummax()) - 1).min(),
                胜率=(r > 0).mean(), nav=nav)


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


def run(etf_col, label='', signal_mode='actual'):
    """signal_mode: 'actual'=用真实信号（2018前跳过）; 'fill1'=2018前补满仓(1.0)"""
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
    # 关键：先保留全量历史用于12个月回看窗口，再单独标记回测起始索引
    lvl_all = un.reindex(columns=cols + ['一天期国债逆回购']).fillna(0.0) / 100.0
    idx = lvl_all.loc[START:].index
    lvl = lvl_all  # 回看窗口需访问 2015 年及更早数据
    listing = {c: lvl_all[c].first_valid_index() for c in cols}

    sig_d = SIGNAL.reindex(idx, method='ffill')
    if signal_mode == 'fill1':
        # 2018-01-02（真实信号起点）之前补为满仓 1.0
        base = pd.Series(1.0, index=idx)
        sig_d = sig_d.copy()
        sig_d.loc[sig_d.index < FIRST_SIG] = 1.0

    trade = lvl[cols].fillna(0)
    rp_assets = [c for c in cols if c not in INDEX_FUTURES]

    repo_ann = lvl['一天期国债逆回购'].loc[idx]
    cal = idx.to_series().diff().dt.days.fillna(1)
    repo_net = np.maximum((repo_ann.shift(1).fillna(0) / 365.0) * cal - REPO_FEE_RATE, 0.0)
    m_ratios = pd.Series({a: MARGIN.get(a, 1.0) for a in cols})

    ret = pd.Series(0.0, index=idx)
    marg = pd.Series(0.0, index=idx)
    idxw = pd.Series(0.0, index=idx)
    recs = []
    curr_w = pd.Series(0.0, index=cols)
    curr_m = 0.0
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
        look = lvl.loc[rd - pd.DateOffset(months=12):rd, elig]
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
            d_repo = repo_net.loc[date]
            if date == hp.index[0]:
                nm = (tgt * m_ratios).sum()
                idle = max(0.0, 1.0 - nm)
                cost = (tgt - curr_w).abs().sum() * FEE_RATE
                ret.loc[date] = (tgt * dr).sum() - cost + idle * d_repo
                curr_w = tgt.copy()
                idxw.loc[date] = iw
                recs.append({'date': rd, 'signal': float(rs), 'idxw': iw,
                             'margin': float(nm), **{a: tgt.loc[a] for a in cols}})
            else:
                idle = max(0.0, 1.0 - curr_m)
                ret.loc[date] = (curr_w * dr).sum() + idle * d_repo
            gw = (curr_w * (1 + dr)).sum()
            curr_w = (curr_w * (1 + dr)) / (gw or 1)
            curr_m = (curr_w * m_ratios).sum()
            marg.loc[date] = curr_m

    out = ret.loc[first_date:]
    m = calc_metrics(out)
    m['平均资金占用'] = marg.loc[first_date:].mean()
    m['起始日'] = first_date.date()
    m['结束日'] = idx[-1].date()
    m['交易日数'] = len(out)
    print(f'\n=== {label} ===')
    print(f"区间: {m['起始日']} ~ {m['结束日']}  ({m['交易日数']} 日 / {m['交易日数']/252:.1f} 年)")
    print(f"累计={m['累计']:+.2%}  年化={m['年化']:+.2%}  波动={m['波动']:.2%}  "
          f"夏普={m['夏普']:.2f}  回撤={m['回撤']:.2%}  日胜率={m['胜率']:.2%}  "
          f"平均资金占用={m['平均资金占用']:.2%}")
    return out, m, pd.DataFrame(recs), marg.loc[first_date:], idxw.loc[first_date:]


print('#' * 72)
print('# 2016 起长周期回溯 —— 上证50ETF替代红利低波ETF + 股指段2018前补满仓')
print('#' * 72)
retB, mB, recB, mgB, iwB = run('上证50ETF', '主口径：上证50ETF替代 + 股指段补满仓', signal_mode='fill1')

print('\n' + '#' * 72)
print('# 对照：保留红利低波ETF（填充口径，2016起）')
print('#' * 72)
retA, mA, recA, mgA, iwA = run('红利低波ETF', '对照：红利低波ETF（填充口径）', signal_mode='fill1')

# 逐年对比
print('\n=== 逐年收益对比（2016起）===')
print(f"{'年份':<6}{'A 红利低波':>12}{'B 上证50':>12}{'差异':>11}")
rows = []
common = retA.index.intersection(retB.index)
for y in sorted(set(common.year)):
    sa = retA[retA.index.year == y]
    sb = retB[retB.index.year == y]
    if len(sa) < 20:
        continue
    a = (1 + sa).prod() - 1
    b = (1 + sb).prod() - 1
    rows.append((y, a, b, b - a))
    print(f"{y:<6}{a:>11.2%}{b:>12.2%}{b-a:>11.2%}")
pd.DataFrame(rows, columns=['年份', '红利低波ETF', '上证50ETF', '差异']).to_csv(
    f'{PROJ}/策略复现与回测/每日更新策略/输出/对比分析/2016起_逐年对比.csv',
    index=False, encoding='utf-8-sig')

# 分段统计：2016-2017（无真实信号期）vs 2018+（有真实信号期）
print('\n=== 分段统计（B 上证50替代版）===')
for lab, s, e in [('2016-2017（股指段补满仓）', '2016-01-01', '2017-12-31'),
                  ('2018-2026（真实信号）', '2018-01-01', '2026-12-31'),
                  ('全程 2016-2026', '2016-01-01', '2026-12-31')]:
    x = retB.loc[s:e]
    if len(x) < 20:
        continue
    mm = calc_metrics(x)
    print(f"{lab:26s} {len(x):4d}日  累计={mm['累计']:+8.2%}  年化={mm['年化']:+7.2%}  "
          f"波动={mm['波动']:6.2%}  夏普={mm['夏普']:.2f}  回撤={mm['回撤']:7.2%}")

# 导出
pd.DataFrame({'红利低波ETF_日收益': retA, '上证50ETF_日收益': retB}).to_csv(
    f'{PROJ}/策略复现与回测/每日更新策略/输出/净值/2016起回溯_日收益对比.csv', encoding='utf-8-sig')
recB.to_csv(f'{PROJ}/策略复现与回测/每日更新策略/输出/对比分析/2016起_仓位明细_上证50.csv',
            index=False, encoding='utf-8-sig')
pd.DataFrame({'资金占用': mgB, '股指仓位': iwB}).to_csv(
    f'{PROJ}/策略复现与回测/每日更新策略/输出/指标/2016起_资金占用与股指仓位.csv', encoding='utf-8-sig')
print('\n已导出净值/仓位明细/资金占用')
