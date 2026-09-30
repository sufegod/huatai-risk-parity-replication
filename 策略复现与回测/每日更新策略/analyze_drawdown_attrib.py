# -*- coding: utf-8 -*-
"""全周期回撤的资产级归因 + 修复条件分析（2026-09-29）

【目标】回答两个问题：
  1) 全周期每一次显著回撤，究竟是谁把净值拖下来的？
  2) 什么条件下回撤会修复？（按成因分类，统计修复时长规律）

【方法】导入生产策略模块（资产风险平价策略0.19），复刻 daily_update_strategy 的逐日重放，
  额外记录每日「各资产持仓权重 + 各资产收益贡献 + 现金贡献 + 交易成本 + 保证金占用 + 股指信号」，
  再按回撤区间（峰值→谷底→恢复）分段累加归因。
  非破坏性：不改生产脚本、不写回任何生产数据。

【校验】GC001 口径净值应复现生产报告（2026-09-28：净值 2.5766 / 年化 11.91% / 夏普 1.75）。
"""
import os
import sys

import numpy as np
import pandas as pd

PROJ = 'C:/Users/aa/WorkBuddy/2026-05-28-14-41-45/huatai-risk-parity-replication'
SCRIPT_DIR = f'{PROJ}/策略复现与回测/每日更新策略'
OUT_METRICS = f'{SCRIPT_DIR}/输出/指标'

sys.path.insert(0, SCRIPT_DIR)
import daily_update_strategy as D          # noqa: E402

strategy = D.load_strategy_module()
TD = 252

# 资产 -> 大类。用「子串 in 资产名」匹配，兼容日后改名
CLASS_MAP = {
    '沪深300': '股指', '中证500': '股指',
    '10年国债': '债券',
    '沪金': '黄金',
    '红利低波': '红利ETF',
    '豆粕': '商品', '沪铜': '商品', '沪铝': '商品', 'PTA': '商品', '原油': '商品',
}
CLASS_ORDER = ['股指', '债券', '商品', '黄金', '红利ETF', '其他']
# 决定性成因阈值：某类拖累占区间总跌幅的比例
DOMINANT_SHARE = 0.30


def classify(asset):
    for key, cls in CLASS_MAP.items():
        if key in asset:
            return cls
    return '其他'


def fresh(path):
    """复用干净文件名：能删就删，删不掉保留（沙箱可能拦截覆盖写）。"""
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f'  [warn] 旧文件删除失败 {os.path.basename(path)}: {e}')
    return path


# ==================== 逐日重放 ====================
def replay_daily():
    df_weight_raw, df_trade_raw, index_signal, active_assets = D.load_strategy_inputs(strategy)

    df_weight_all = df_weight_raw / 100.0
    df_trade_all_raw = df_trade_raw / 100.0
    df_trade_all = df_trade_all_raw.fillna(0)

    as_of = min(df_weight_all.index.max(), df_trade_all.index.max()).normalize()
    df_weight_all = df_weight_all.loc[:as_of]
    df_trade_all_raw = df_trade_all_raw.loc[:as_of]
    df_trade_all = df_trade_all.loc[:as_of]

    assets = [a for a in active_assets if a in df_weight_all.columns and a in df_trade_all.columns]
    rp_assets = [a for a in assets if a not in strategy.INDEX_FUTURES]

    df_weight = df_weight_all[assets].fillna(0)
    df_trade = df_trade_all[assets]
    listing = {a: df_trade_all_raw[a].first_valid_index() for a in assets}

    sig = index_signal.reindex(df_trade.index, method='ffill')
    first_sig = index_signal.first_valid_index()

    cal_days = df_trade_all.index.to_series().diff().dt.days.fillna(1)
    repo_ann = df_trade_all.get('一天期国债逆回购', pd.Series(0.0, index=df_trade_all.index))
    repo_net = np.maximum((repo_ann.shift(1).fillna(0) / 365.0) * cal_days - strategy.REPO_FEE_RATE, 0.0)

    m_ratios = pd.Series({a: strategy.MARGIN_RATIOS.get(a, 1.0) for a in assets})
    obs = D.get_observation_dates(df_trade.index)

    dates, rec_w, rec_c = [], [], []
    rec_cash, rec_cost, rec_total, rec_margin, rec_sig = [], [], [], [], []

    curr_w = pd.Series(0.0, index=assets)
    first_date = None

    for i in range(len(obs) - 1):
        rb = obs[i]
        if rb < first_sig:
            continue
        raw_signal = sig.loc[rb]
        if pd.isna(raw_signal):
            continue

        elig = [a for a in rp_assets if listing.get(a) is not None and listing[a] <= rb]
        if not elig:
            continue
        look = df_weight.loc[rb - pd.DateOffset(months=12):rb, elig]
        if len(look) < 150:
            continue

        idx_t = strategy.allocate_index_futures(raw_signal, assets, listing, rb)
        idx_w = float(idx_t.sum())
        rem = max(0.0, 1.0 - idx_w)
        rp = strategy.get_risk_parity_weights(
            strategy.calculate_ewma_semi_cov(look, strategy.EWMA_DECAY))
        target = pd.Series(0.0, index=assets)
        target.loc[idx_t.index] = idx_t
        target.loc[elig] = rp * rem

        holding = df_trade.loc[rb + pd.Timedelta(days=1):obs[i + 1]]
        if len(holding) == 0:
            continue
        if first_date is None:
            first_date = holding.index[0]

        for date, dr in holding.iterrows():
            is_first = (date == holding.index[0])
            w_used = target.copy() if is_first else curr_w.copy()
            contrib = w_used * dr
            gross = float(contrib.sum())
            c = float((target - curr_w).abs().sum() * strategy.FEE_RATE) if is_first else 0.0
            m_use = float((w_used * m_ratios).sum())
            idle = max(0.0, 1.0 - m_use)
            ca = idle * float(repo_net.loc[date])

            dates.append(date)
            rec_w.append(w_used.values)
            rec_c.append(contrib.values)
            rec_cash.append(ca)
            rec_cost.append(c)
            rec_total.append(gross - c + ca)
            rec_margin.append(m_use)
            rec_sig.append(float(raw_signal))

            if is_first:
                curr_w = target.copy()
            else:
                gw = float((curr_w * (1 + dr)).sum())
                curr_w = (curr_w * (1 + dr)) / (gw if gw else 1.0)

    idx = pd.DatetimeIndex(dates)
    return dict(
        assets=assets,
        W=pd.DataFrame(rec_w, index=idx, columns=assets),
        C=pd.DataFrame(rec_c, index=idx, columns=assets),
        cash=pd.Series(rec_cash, index=idx),
        cost=pd.Series(rec_cost, index=idx),
        total=pd.Series(rec_total, index=idx),
        margin=pd.Series(rec_margin, index=idx),
        signal=pd.Series(rec_sig, index=idx),
        as_of=as_of,
    )


# ==================== 回撤区间识别 ====================
def find_episodes(nav):
    """识别 (峰值日, 谷底日, 恢复日, 深度)；未恢复的恢复日为 None。"""
    eps = []
    pk_d, pk_v = nav.index[0], float(nav.iloc[0])
    tr_d, tr_v = pk_d, pk_v
    in_dd = False
    for dt, v in nav.items():
        v = float(v)
        if v >= pk_v:
            if in_dd:
                eps.append((pk_d, tr_d, dt, tr_v / pk_v - 1.0))
                in_dd = False
            pk_d, pk_v = dt, v
            tr_d, tr_v = dt, v
        else:
            in_dd = True
            if v < tr_v:
                tr_d, tr_v = dt, v
    if in_dd:
        eps.append((pk_d, tr_d, None, tr_v / pk_v - 1.0))
    return eps


def by_class(contrib_df):
    out = {}
    for a in contrib_df.columns:
        k = classify(a)
        out[k] = out.get(k, 0.0) + float(contrib_df[a].sum())
    return out


def seg_stats(d, a, b):
    """区间 (a, b] 的归因。a=None 表示从头。"""
    c = d['C'].loc[a:b] if a is not None else d['C'].loc[:b]
    if a is not None:
        c = c.iloc[1:]
    if len(c) == 0:
        return None
    cls = by_class(c)
    tot = float(d['total'].loc[c.index[0]:c.index[-1]].sum())
    return dict(
        n=len(c),
        cls=cls,
        total=tot,
        cash=float(d['cash'].loc[c.index[0]:c.index[-1]].sum()),
        cost=float(d['cost'].loc[c.index[0]:c.index[-1]].sum()),
        signal_start=float(d['signal'].loc[c.index[0]]),
        signal_min=float(d['signal'].loc[c.index[0]:c.index[-1]].min()),
        signal_max=float(d['signal'].loc[c.index[0]:c.index[-1]].max()),
        index_w_start=float(d['W'].loc[c.index[0]].reindex(strategy.INDEX_FUTURES).fillna(0).sum()),
        index_w_min=float(d['W'].loc[c.index[0]:c.index[-1]][strategy.INDEX_FUTURES].sum(axis=1).min()),
        index_w_max=float(d['W'].loc[c.index[0]:c.index[-1]][strategy.INDEX_FUTURES].sum(axis=1).max()),
        margin_avg=float(d['margin'].loc[c.index[0]:c.index[-1]].mean()),
    )


def dominant(cls):
    """返回拖累最大的类别及其实占比。"""
    if not cls:
        return '无', 0.0
    worst = min(cls.items(), key=lambda kv: kv[1])
    return worst[0], worst[1]


# ==================== 主流程 ====================
print('=' * 118)
print('生产 v0.19 全周期回撤：资产级归因 + 修复条件分析')
print('=' * 118)

d = replay_daily()
r = d['total']
nav = (1 + r).cumprod()
n = len(r)
years = n / TD

ann = nav.iloc[-1] ** (1 / years) - 1
vol = float(r.std() * np.sqrt(TD))
sharpe = float(r.mean() * TD / vol)
mdd = float((nav / nav.cummax() - 1).min())

print(f"\n样本：{r.index[0].date()} ~ {r.index[-1].date()}   共 {n} 交易日 / {years:.2f} 年")
print(f"期末净值 {nav.iloc[-1]:.4f} | 年化 {ann:.2%} | 波动 {vol:.2%} | 夏普 {sharpe:.2f} | 最大回撤 {mdd:.2%}")
print(f"校验：应≈生产报告 2.5766 / 11.91% / 6.54% / 1.75 / -7.79%")

eps = find_episodes(nav)
recovered = [e for e in eps if e[2] is not None]
print(f"\n共识别 {len(eps)} 次回撤，其中 {len(recovered)} 次已恢复、{len(eps) - len(recovered)} 次未恢复。")

# ---------- 逐次归因 ----------
rows = []
detail = {}
for pk, tr, rc, dep in eps:
    dn = seg_stats(d, pk, tr)                       # 下跌段 (峰值, 谷底]
    n_dn = dn['n'] if dn else 0
    n_rc = None
    up = seg_stats(d, tr, rc) if rc is not None else None
    if rc is not None:
        n_rc = up['n'] if up else 0

    dom_cls, dom_val = dominant(dn['cls']) if dn else ('无', 0.0)
    dom2 = ''
    if dn:
        srt = sorted(dn['cls'].items(), key=lambda kv: kv[1])
        dom2 = ' + '.join(f'{k}{v:.2%}' for k, v in srt[:2])

    rec_cls = up['cls'] if up else {}
    rec_srt = sorted(rec_cls.items(), key=lambda kv: -kv[1]) if rec_cls else []
    rec_driver = ' + '.join(f'{k}{v:+.2%}' for k, v in rec_srt[:2]) if rec_srt else ('—未恢复—')
    rec_top = max(rec_cls.items(), key=lambda kv: kv[1])[0] if rec_cls else None
    same_dir = '—' if rc is None else ('是' if (rec_top and rec_top == dom_cls) else '否')

    rows.append(dict(
        峰值日=pk.date(), 谷底日=tr.date(), 恢复日=rc.date() if rc is not None else '',
        深度=dep, 下跌日数=n_dn, 恢复日数=n_rc,
        总日数=(n_dn + n_rc) if n_rc is not None else n_dn,
        主导拖累=dom_cls,
        修复主导=rec_top or '', 同向修复=same_dir,
        起始股指信号=dn['signal_start'] if dn else np.nan,
        起始股指仓位=dn['index_w_start'] if dn else np.nan,
        深度绝对值=abs(dep),
        下跌段归因=dom2,
        下跌速度=f'{abs(dep) / n_dn * 100:.3f}' if n_dn else '',
        修复驱动=rec_driver,
        修复速度=f'{(up["total"] / n_rc * 100):.3f}' if (up and n_rc) else '',
        区间股指信号=f'{dn["signal_min"]:.1f}~{dn["signal_max"]:.1f}' if dn else '',
        区间股指仓位=f'{dn["index_w_min"]:.1%}~{dn["index_w_max"]:.1%}' if dn else '',
    ))
    detail[(pk, tr)] = dict(dn=dn, up=up, dep=dep)

attrib_df = pd.DataFrame(rows).sort_values('深度').reset_index(drop=True)

print('\n' + '-' * 118)
print('全周期回撤明细（按深度排序，前 20 次）')
print('-' * 118)
show = attrib_df.copy()
show['深度'] = show['深度'].map(lambda x: f'{x:.2%}')
print(show.head(20).to_string(index=False, max_colwidth=34))

# ---------- 按主导拖累类别统计修复规律 ----------
print('\n' + '=' * 118)
print('① 按「主导拖累类别」分组：修复时长规律')
print('=' * 118)

sub = attrib_df[attrib_df['恢复日数'].notna()].copy()
sub['深度_'] = sub['深度'].astype(float)
grp = sub.groupby('主导拖累')
print(f"\n{'主导拖累':<10}{'次数':>5}{'平均深度':>11}{'平均下跌日':>11}{'平均恢复日':>11}"
      f"{'平均总日':>10}{'平均修复速度':>13}{'最慢一次':>12}")
for k, g in sorted(grp, key=lambda kv: kv[1]['恢复日数'].mean()):
    print(f"{k:<10}{len(g):>5}{g['深度_'].mean():>10.2%}{g['下跌日数'].mean():>11.1f}"
          f"{g['恢复日数'].mean():>11.1f}{(g['下跌日数'] + g['恢复日数']).mean():>10.1f}"
          f"{g['修复速度'].astype(float).mean():>12.3f}%{g['恢复日数'].max():>11.0f}日")

unrec = attrib_df[attrib_df['恢复日数'].isna()]
if len(unrec):
    print(f"\n未恢复：{'、'.join(str(x) for x in unrec['峰值日'])}（{len(unrec)} 次）")

# ---------- 急跌 vs 阴跌 ----------
print('\n' + '=' * 118)
print('② 急跌 vs 阴跌：下跌快慢与修复时长的关系')
print('=' * 118)
sub['下跌速度_'] = sub['下跌速度'].astype(float)
sub['形态'] = np.where(sub['下跌速度_'] >= sub['下跌速度_'].median(), '急跌', '阴跌')
for k, g in sub.groupby('形态'):
    print(f"{k}：{len(g):>3} 次 | 平均下跌 {g['下跌日数'].mean():>5.1f} 日 | "
          f"平均深度 {g['深度_'].mean():>6.2%} | 平均恢复 {g['恢复日数'].mean():>5.1f} 日")

# ---------- 相关性 ----------
print('\n' + '=' * 118)
print('③ 修复时长 与 各因素的相关性（Pearson，样本 = 已恢复的 128 次回撤）')
print('=' * 118)
for col, lab in [('深度绝对值', '回撤深度（绝对值）'), ('下跌日数', '下跌持续天数'),
                 ('下跌速度_', '下跌速度(%/日)'), ('起始股指仓位', '下跌段起始股指仓位')]:
    cc = sub[['恢复日数', col]].corr().iloc[0, 1]
    print(f"  修复日数 vs {lab:<20}  r = {cc:+.3f}")

# ---------- 谁拖的靠谁修 ----------
print('\n' + '=' * 118)
print('④ 谁拖的，靠谁修？——下跌主导拖累 vs 修复段最大驱动')
print('=' * 118)
same = int((sub['同向修复'] == '是').sum())
print(f"\n  整体：{same}/{len(sub)} = {same / len(sub):.1%} 的回撤，修复段最大驱动正是当初的拖累项")
print(f"\n  {'主导拖累':<10}{'次数':>5}{'同向修复':>10}{'同向占比':>10}{'平均恢复日':>11}")
for k, g in sorted(sub.groupby('主导拖累'), key=lambda kv: -kv[1]['恢复日数'].mean()):
    if len(g) >= 3:
        s = int((g['同向修复'] == '是').sum())
        print(f"  {k:<10}{len(g):>5}{s:>10}{s / len(g):>10.0%}{g['恢复日数'].mean():>11.1f}")

# ---------- 信号状态与拖累来源 ----------
print('\n' + '=' * 118)
print('⑤ 股指信号状态，决定「回撤由谁造成」')
print('=' * 118)
sub['信号状态'] = pd.cut(sub['起始股指信号'], [-2, 0.001, 0.99, 2],
                       labels=['清仓(信号≤0)', '半仓(0<信号<1)', '满仓(信号=1.0)'])
ct = pd.crosstab(sub['信号状态'], sub['主导拖累'])
print('\n' + ct.to_string())
print('\n  同一分组下的平均回撤深度与恢复时长：')
for k, g in sub.groupby('信号状态'):
    if len(g):
        print(f"    {str(k):<16}{len(g):>4} 次 | 平均深度 {g['深度绝对值'].mean():>6.2%} | "
              f"平均下跌 {g['下跌日数'].mean():>5.1f} 日 | 平均恢复 {g['恢复日数'].mean():>5.1f} 日")

# ---------- 当前未恢复回撤专项 ----------
if len(unrec):
    pk, tr = eps[-1][0], eps[-1][1]
    dn = detail[(pk, tr)]['dn']
    post = seg_stats(d, tr, r.index[-1])
    print('\n' + '=' * 118)
    print(f'⑥ 当前未恢复回撤专项：{pk.date()} → {tr.date()}，深度 {eps[-1][3]:.2%}')
    print('=' * 118)
    print(f"\n[下跌段] {pk.date()} → {tr.date()}（{dn['n']} 交易日，累加 {dn['total']:+.2%}）")
    for k in CLASS_ORDER:
        if k in dn['cls']:
            print(f"    {k:<8}{dn['cls'][k]:>+9.2%}")
    print(f"    {'现金':<8}{dn['cash']:>+9.2%}")
    print(f"    {'成本':<8}{dn['cost']:>+9.2%}")
    print(f"    {'合计':<8}{dn['total']:>+9.2%}")
    print(f"    股指信号 {dn['signal_min']:.1f}~{dn['signal_max']:.1f}，"
          f"仓位 {dn['index_w_min']:.1%}~{dn['index_w_max']:.1%}")

    print(f"\n[谷底至今] {tr.date()} → {r.index[-1].date()}（{post['n']} 交易日，累加 {post['total']:+.2%}）")
    for k in CLASS_ORDER:
        if k in post['cls']:
            print(f"    {k:<8}{post['cls'][k]:>+9.2%}")
    print(f"    {'现金':<8}{post['cash']:>+9.2%}")
    print(f"    {'成本':<8}{post['cost']:>+9.2%}")
    print(f"    {'合计':<8}{post['total']:>+9.2%}")
    cur_dd = float(nav.iloc[-1] / nav.loc[pk] - 1)
    print(f"\n    距峰值仍差 {cur_dd:.2%}；若按近期日均速度，需再涨 "
          f"{np.log(nav.loc[pk] / nav.iloc[-1]) / max(post['total'] / post['n'], 1e-9):.0f} 交易日回本"
          f"（参考值，仅线性外推）")

# ---------- 导出 ----------
p1 = fresh(f'{OUT_METRICS}/回撤归因_逐次明细_2018-2026-{d["as_of"].date()}.csv')
attrib_df.to_csv(p1, index=False, encoding='utf-8-sig')
print(f'\n已导出：{p1}')

daily_out = pd.DataFrame({
    '策略日收益': d['total'],
    '净值': nav,
    '回撤': nav / nav.cummax() - 1,
    '现金贡献': d['cash'],
    '交易成本': d['cost'],
    '保证金占用': d['margin'],
    '股指信号': d['signal'],
})
for a in d['assets']:
    daily_out[f'贡献_{a}'] = d['C'][a]
    daily_out[f'权重_{a}'] = d['W'][a]
cls_daily = pd.DataFrame({k: d['C'][[a for a in d['assets'] if classify(a) == k]].sum(axis=1)
                          for k in CLASS_ORDER if any(classify(a) == k for a in d['assets'])})
for k in cls_daily.columns:
    daily_out[f'贡献_{k}'] = cls_daily[k]
daily_out.index.name = 'date'
p2 = fresh(f'{OUT_METRICS}/回撤归因_日度贡献_2018-2026-{d["as_of"].date()}.csv')
daily_out.to_csv(p2, encoding='utf-8-sig')
print(f'已导出：{p2}')
print('\n完成。')
