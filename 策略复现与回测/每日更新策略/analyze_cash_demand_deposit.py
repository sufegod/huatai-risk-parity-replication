# -*- coding: utf-8 -*-
"""生产 v0.19 现金口径情景：活期存款利率 对照 GC001 / 不含现金（2026-09-29）

【用户需求】现金部分的收益 = (净资产 − 保证金) × 当年银行活期存款利率
  · 净资产归一到 1，故 现金余额 = 1 − 保证金占用；现金日收益 = 现金余额 × 活期日利率
  · 活期日利率 = 当年活期年利率 / 365 × 自然日（无逆回购手续费）

【实现】非破坏性 what-if：导入生产策略模块（资产风险平价策略0.19）的求解器与常量，
  复刻 daily_update_strategy.build_backtest_result 的逐日重放，仅将现金利率由 GC001 逆回购
  替换为「六大行活期存款挂牌利率」历史分段（DEMAND_SCHEDULE）。不改生产脚本、不写回任何数据。

【校验】GC001 口径应复现生产报告（2026-09-28：净值 2.5766 / 年化 11.91% / 夏普 1.75）。
"""
import sys
import numpy as np
import pandas as pd

PROJ = 'C:/Users/aa/WorkBuddy/2026-05-28-14-41-45/huatai-risk-parity-replication'
SCRIPT_DIR = f'{PROJ}/策略复现与回测/每日更新策略'
sys.path.insert(0, SCRIPT_DIR)
import daily_update_strategy as D

strategy = D.load_strategy_module()

# 六大行活期存款挂牌利率（年率）历史分段 —— 百度百科/新浪财经 2024-10-18、2025-05-20 调整公告
# 2016~2022-09 大行活期长期 0.30%；其后历次下调至 0.05%（2025-05-20）
DEMAND_SCHEDULE = [
    ('2016-01-01', 0.0030),
    ('2022-09-15', 0.0025),
    ('2023-06-08', 0.0020),
    ('2024-07-25', 0.0015),
    ('2024-10-18', 0.0010),
    ('2025-05-20', 0.0005),
]

TRADING_DAYS = 252


def replay():
    """复刻 build_backtest_result，返回 资产端 / GC001口径 / 活期口径 三套日收益序列 + 现金余额序列。"""
    df_weight_raw, df_trade_raw, index_signal, active_assets = D.load_strategy_inputs(strategy)

    df_weight_all = df_weight_raw / 100.0
    df_trade_all_raw = df_trade_raw / 100.0
    df_trade_all = df_trade_all_raw.fillna(0)

    as_of_date = min(df_weight_all.index.max(), df_trade_all.index.max()).normalize()
    df_weight_all = df_weight_all.loc[:as_of_date]
    df_trade_all_raw = df_trade_all_raw.loc[:as_of_date]
    df_trade_all = df_trade_all.loc[:as_of_date]

    assets = [a for a in active_assets if a in df_weight_all.columns and a in df_trade_all.columns]
    risk_parity_assets = [a for a in assets if a not in strategy.INDEX_FUTURES]

    df_weight = df_weight_all[assets].fillna(0)
    df_trade = df_trade_all[assets]
    listing_dates = {a: df_trade_all_raw[a].first_valid_index() for a in assets}

    signal_on_trade_dates = index_signal.reindex(df_trade.index, method='ffill')
    first_signal_date = index_signal.first_valid_index()

    calendar_days = df_trade_all.index.to_series().diff().dt.days.fillna(1)

    # GC001 逆回购净日利率（生产口径）
    repo_rate_ann = df_trade_all.get('一天期国债逆回购', pd.Series(0.0, index=df_trade_all.index))
    repo_shifted = repo_rate_ann.shift(1).fillna(0)
    repo_net = np.maximum((repo_shifted / 365.0) * calendar_days - strategy.REPO_FEE_RATE, 0.0)

    # 活期存款日利率（无逆回购手续费）
    demand_ann = pd.Series(0.0, index=df_trade_all.index)
    for s, r in DEMAND_SCHEDULE:
        demand_ann.loc[df_trade_all.index >= pd.Timestamp(s)] = r
    demand_net = (demand_ann.shift(1).fillna(0) / 365.0) * calendar_days

    m_ratios = pd.Series({a: strategy.MARGIN_RATIOS.get(a, 1.0) for a in assets})
    observation_dates = D.get_observation_dates(df_trade.index)

    ret_asset = pd.Series(0.0, index=df_trade.index)
    ret_gc = pd.Series(0.0, index=df_trade.index)
    ret_dm = pd.Series(0.0, index=df_trade.index)
    idle_series = pd.Series(0.0, index=df_trade.index)

    curr_w = pd.Series(0.0, index=assets)
    curr_margin = 0.0
    first_date = None

    for i in range(len(observation_dates) - 1):
        rb = observation_dates[i]
        if rb < first_signal_date:
            continue
        raw_signal = signal_on_trade_dates.loc[rb]
        if pd.isna(raw_signal):
            continue
        elig = [a for a in risk_parity_assets if listing_dates.get(a) is not None and listing_dates[a] <= rb]
        if len(elig) == 0:
            continue
        lookback = df_weight.loc[rb - pd.DateOffset(months=12):rb, elig]
        if len(lookback) < 150:
            continue
        idx_target = strategy.allocate_index_futures(raw_signal, assets, listing_dates, rb)
        idx_w = float(idx_target.sum())
        rem = max(0.0, 1.0 - idx_w)
        rp = strategy.get_risk_parity_weights(strategy.calculate_ewma_semi_cov(lookback, strategy.EWMA_DECAY))
        target = pd.Series(0.0, index=assets)
        target.loc[idx_target.index] = idx_target
        target.loc[elig] = rp * rem

        holding = df_trade.loc[rb + pd.Timedelta(days=1):observation_dates[i + 1]]
        if len(holding) == 0:
            continue
        if first_date is None:
            first_date = holding.index[0]

        for date, dr in holding.iterrows():
            idle = max(0.0, 1.0 - (target * m_ratios).sum() if date == holding.index[0] else 1.0 - curr_margin)
            g = float((target * dr).sum()) if date == holding.index[0] else float((curr_w * dr).sum())
            cost = float((target - curr_w).abs().sum() * strategy.FEE_RATE) if date == holding.index[0] else 0.0
            asset = g - cost
            ret_asset.loc[date] = asset
            ret_gc.loc[date] = asset + idle * float(repo_net.loc[date])
            ret_dm.loc[date] = asset + idle * float(demand_net.loc[date])
            idle_series.loc[date] = idle
            if date == holding.index[0]:
                curr_w = target.copy()
            else:
                gw = float((curr_w * (1 + dr)).sum())
                curr_w = (curr_w * (1 + dr)) / (gw if gw else 1.0)
            curr_margin = float((curr_w * m_ratios).sum())

    return (pd.Timestamp(first_date), ret_asset.loc[first_date:], ret_gc.loc[first_date:],
            ret_dm.loc[first_date:], idle_series.loc[first_date:], as_of_date)


def metrics(r):
    r = r.fillna(0.0)
    nav = (1 + r).cumprod()
    y = len(r) / TRADING_DAYS
    ann = nav.iloc[-1] ** (1 / y) - 1 if nav.iloc[-1] > 0 else -1.0
    vol = r.std() * np.sqrt(TRADING_DAYS)
    mdd = ((nav / nav.cummax()) - 1).min()
    sharpe = (r.mean() * TRADING_DAYS) / vol if vol > 0 else np.nan
    return dict(累计=nav.iloc[-1] - 1, 净值=nav.iloc[-1], 年化=ann, 波动=vol, 夏普=sharpe,
                最大回撤=mdd, 日胜率=(r > 0).mean(), 交易日数=len(r), nav=nav)


print('=' * 100)
print('生产 v0.19 现金口径情景：活期存款利率 vs GC001 vs 不含现金')
print('=' * 100)
first_date, ra, rg, rd, idle, as_of = replay()
print(f'\n样本起点(持仓首日)：{first_date.date()}   数据日期：{as_of.date()}   '
      f'共 {len(rg)} 交易日 / {len(rg)/TRADING_DAYS:.2f} 年（252 折算）')

mg = metrics(rg)
mdm = metrics(rd)
ma = metrics(ra)

print('\n' + '-' * 100)
print(f"{'口径':<22}{'期末净值':>10}{'年化':>10}{'波动':>9}{'夏普':>8}{'最大回撤':>11}{'日胜率':>9}")
for lab, m in [('① 不含现金(资产端)', ma), ('② 含现金·GC001逆回购(生产)', mg),
               ('③ 含现金·活期存款利率(本次)', mdm)]:
    print(f"{lab:<22}{m['净值']:>10.4f}{m['年化']:>9.2%}{m['波动']:>9.2%}{m['夏普']:>8.2f}"
          f"{m['最大回撤']:>11.2%}{m['日胜率']:>9.2%}")

print('\n--- 校验：GC001 口径应≈生产报告(2026-09-28：净值2.5766/年化11.91%/夏普1.75) ---')
print(f"    实际 GC001：净值 {mg['净值']:.4f} / 年化 {mg['年化']:.2%} / 夏普 {mg['夏普']:.2f} / 回撤 {mg['最大回撤']:.2%}")

yn = len(rg) / TRADING_DAYS
cash_gc = (idle * (rg - ra)).sum() / yn
cash_dm = (idle * (rd - ra)).sum() / yn
print('\n--- 现金收益年均贡献（百分点）---')
print(f"    资产端年化      ：{ma['年化']:+.2%}")
print(f"    + GC001现金      ：+{cash_gc:.2%}  →  GC001口径年化 {mg['年化']:+.2%}")
print(f"    + 活期现金        ：+{cash_dm:.2%}  →  活期口径年化 {mdm['年化']:+.2%}")
print(f"    活期相对GC001现金少赚：{cash_gc - cash_dm:+.2%}/年")

# 逐年
print('\n' + '-' * 100)
print('逐年表现（年化区间收益）')
print('-' * 100)
yy = [y for y in sorted(set(rg.index.year)) if (rg.index.year == y).sum() > 20]
print(f"{'年份':<7}{'资产端':>10}{'GC001口径':>11}{'活期口径':>11}{'活期少赚':>10}")
for y in yy:
    m = rg.index.year == y
    na = (1 + ra[m]).prod() - 1
    ng = (1 + rg[m]).prod() - 1
    nd = (1 + rd[m]).prod() - 1
    print(f"{y:<7}{na:>9.2%}{ng:>10.2%}{nd:>10.2%}{ng - nd:>9.2%}")

# 导出
out = f'{SCRIPT_DIR}/输出/对比分析/现金口径_活期vsGC001_生产v0.19_{as_of.date()}.csv'
daily = pd.DataFrame({
    '资产端日收益': ra, 'GC001口径日收益': rg, '活期口径日收益': rd,
    '现金余额比例': idle,
    'GC001净值': metrics(rg)['nav'], '活期净值': metrics(rd)['nav'],
})
daily.index.name = 'date'
daily.to_csv(out, encoding='utf-8-sig')
print(f'\n已导出：{out}')
print('完成。')
