"""上证50ETF 替代 红利低波ETF 对比回测

三种口径：
  口径一（filled=False, synth=False）：真实上市口径，与 v0.19 生产版完全一致
  口径二（filled=True,  synth=False）：填充口径
  口径三（filled=True,  synth=True） ：假设信号全程=1，剥离信号约束测纯资产层可回溯性

结论摘要：上证50ETF 数据起自 2005-02-23（远早于红利低波ETF的2019-01-18），
        但策略起点仍被股指期货信号（2018-01-02）约束；替代后年化 -0.60pct、夏普 -0.07。

运行：C:/Users/aa/.workbuddy/binaries/python/envs/default/Scripts/python.exe compare_hs50_vs_dividend_etf.py
"""
import pandas as pd, numpy as np
from scipy.optimize import minimize

PROJ = 'C:/Users/aa/WorkBuddy/2026-05-28-14-41-45/huatai-risk-parity-replication'
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
                胜率=(r > 0).mean())


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

# 假设模式：全程信号=1（用于测试纯资产层面能回溯多远）
import pandas as _pd
_full_un = _pd.read_csv(f'{PROJ}/数据/日度收益数据更新/日涨跌幅_填充.csv',
                        index_col=0, parse_dates=True)
SIGNAL_SYNTH = _pd.Series(1.0, index=_full_un.index)


def run(etf_col, filled, label='', synth_signal=False):
    """filled=True 用填充版（历史可回溯至2013），False 用未填充版（真实上市口径）"""
    fn = '日涨跌幅_填充.csv' if filled else '日涨跌幅_未填充.csv'
    un = pd.read_csv(f'{PROJ}/数据/日度收益数据更新/{fn}', index_col=0, parse_dates=True)
    un = un.loc[:, ~un.columns.duplicated()]

    if etf_col == '上证50ETF':
        s50 = pd.read_csv(f'{PROJ}/数据/原始数据/上证50ETF_日涨跌幅.csv',
                          index_col=0, parse_dates=True)
        s50.index = pd.to_datetime(s50.index)
        s50 = s50.loc[:, ~s50.columns.duplicated()]
        un = un.drop(columns=['上证50ETF'], errors='ignore').join(s50, how='left')

    cols = INDEX_FUTURES + ['10年国债主连', '沪铜主连', '沪铝主连', 'PTA主连',
                            '原油主连', '豆粕主连', '沪金主连', etf_col]
    lvl = un.reindex(columns=cols + ['一天期国债逆回购']) / 100.0

    if filled:
        # 填充版：缺失值视为无收益（口径：上市前按0处理）
        lvl = lvl.fillna(0.0)
        listing = {c: lvl.index[0] for c in cols}
    else:
        listing = {c: lvl[c].first_valid_index() for c in cols}

    trade = lvl[cols].fillna(0)
    idx = trade.index
    signal_d = (SIGNAL_SYNTH if synth_signal else SIGNAL).reindex(idx, method='ffill')
    rp_assets = [c for c in cols if c not in INDEX_FUTURES]

    repo_ann = lvl.get('一天期国债逆回购', pd.Series(0.0, index=idx))
    cal = lvl.index.to_series().diff().dt.days.fillna(1)
    repo_net = np.maximum((repo_ann.shift(1).fillna(0) / 365.0) * cal - REPO_FEE_RATE, 0.0)
    m_ratios = pd.Series({a: MARGIN.get(a, 1.0) for a in cols})

    ret = pd.Series(0.0, index=idx)
    marg = pd.Series(0.0, index=idx)
    recs = []
    curr_w = pd.Series(0.0, index=cols)
    curr_m = 0.0
    first_date = None

    obs = pd.DatetimeIndex(idx)
    for i in range(len(obs) - 1):
        rd = obs[i]
        if not synth_signal and rd < FIRST_SIG:
            continue
        rs = signal_d.loc[rd]
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
        listed_if = [a for a in INDEX_FUTURES if listing[a] is not None and listing[a] <= rd]
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
    print(f"区间: {m['起始日']} ~ {m['结束日']}  ({m['交易日数']} 日)")
    print(f"累计={m['累计']:+.2%}  年化={m['年化']:+.2%}  波动={m['波动']:.2%}  "
          f"夏普={m['夏普']:.2f}  回撤={m['回撤']:.2%}  日胜率={m['胜率']:.2%}  "
          f"平均资金占用={m['平均资金占用']:.2%}")
    return out, m, pd.DataFrame(recs), marg.loc[first_date:]


print('#' * 70)
print('# 口径一：真实上市口径（未填充）—— 与 v0.19 生产版完全一致')
print('#' * 70)
retA1, mA1, recA1, mgA1 = run('红利低波ETF', False, 'A1 原版 红利低波ETF（生产版基准）')
retB1, mB1, recB1, mgB1 = run('上证50ETF', False, 'B1 替代 上证50ETF')

print('\n' + '#' * 70)
print('# 口径二：填充口径（2013起可回溯）')
print('#' * 70)
retA2, mA2, recA2, mgA2 = run('红利低波ETF', True, 'A2 原版 红利低波ETF（填充口径）')
retB2, mB2, recB2, mgB2 = run('上证50ETF', True, 'B2 替代 上证50ETF（填充口径，可回溯2013）')

print('\n' + '#' * 70)
print('# 口径三：假设信号全程=1（剥离信号约束，测纯资产层可回溯性）')
print('#' * 70)
retA3, mA3, recA3, mgA3 = run('红利低波ETF', True, 'A3 原版（假设信号=1）', synth_signal=True)
retB3, mB3, recB3, mgB3 = run('上证50ETF', True, 'B3 上证50（假设信号=1）', synth_signal=True)

print('\n=== 【假设信号口径】逐年收益对比 ===')
print(f"{'年份':<6}{'A 红利低波':>12}{'B 上证50':>12}{'差异':>11}")
common3 = retA3.index.intersection(retB3.index)
for y in sorted(set(common3.year)):
    sa = retA3[retA3.index.year == y]; sb = retB3[retB3.index.year == y]
    if len(sa) < 20: continue
    a = (1+sa).prod()-1; b = (1+sb).prod()-1
    print(f"{y:<6}{a:>11.2%}{b:>12.2%}{b-a:>11.2%}")

pd.DataFrame({'红利低波ETF_日收益': retA3, '上证50ETF_日收益': retB3}).to_csv(
    f'{PROJ}/策略复现与回测/每日更新策略/输出/净值/上证50替代_假设信号_日收益.csv',
    encoding='utf-8-sig')
print('\n已导出 A3/B3')
