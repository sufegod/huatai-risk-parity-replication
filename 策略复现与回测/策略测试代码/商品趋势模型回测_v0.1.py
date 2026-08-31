# -*- coding: utf-8 -*-
"""
商品趋势模型回测 v0.1 —— 第一步：仅用价格验证商品趋势（多空双向）是否存在 alpha

背景：
  此前在风险平价组合内做"趋势过滤"是 0/正（趋势向下只能剔除、空仓），赚不到下跌的钱，
  实证结果是破坏分散、恶化表现。本脚本改为商品期货【多空双向】(+1/-1)：
  趋势向上做多、向下做空 —— 期货天然支持做空，这才是趋势模型的完整形态。

品种（不含股指期货与红利ETF）：沪金/豆粕/沪铜/沪铝/PTA/原油
模型：
  - 唐奇安通道突破 Donchian(N)：t日收盘突破过去N日最高做多(+1)，跌破N日最低做空(-1)，否则沿用前信号
  - 时序动量 TSMOM(M月)：过去M月累计收益 >0 做多，<0 做空（Moskowitz-Ooi-Pedersen 2012）
数据：前复权收盘价（已处理主力换月跳空），区间自原油上市日 2018-03-26 起（6品种齐全）
防前视：position = signal.shift(1)，用 t-1 日信号承担 t 日收益
成本：单边手续费 FEE（保守估计，未单独计滑点）
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parents[1]
PRICE_FILE = PROJECT / '数据' / '日度收益数据更新' / '期货主力前复权收盘价.csv'
OUT_DIR = PROJECT / '策略复现与回测' / '策略测试结果' / '商品趋势模型_v0.1'

ASSETS = ['沪金主连', '豆粕主连', '沪铜主连', '沪铝主连', 'PTA主连', '原油主连']
START = '2018-03-26'      # 原油上市日，此后6品种齐全
FEE = 0.0005              # 单边手续费
ANN = 252                 # 年化交易日
WARMUP = 253              # 最长模型(TSMOM12月=252日)预热期，评估区间从此后开始以保证公平


def load_prices():
    """读取前复权收盘价（前复权已处理主力合约换月跳空，是趋势回测的正确数据）"""
    df = pd.read_csv(PRICE_FILE, encoding='utf-8-sig')
    c0 = df.columns[0]
    df[c0] = pd.to_datetime(df[c0])
    df = df.set_index(c0).sort_index()
    df = df[ASSETS].apply(pd.to_numeric, errors='coerce')
    return df.loc[START:].dropna(how='all')


def donchian_signal(price, n):
    """唐奇安通道突破：突破过去N日(至t-1)最高做多，跌破最低做空，区间内沿用前信号"""
    hh = price.shift(1).rolling(n).max()
    ll = price.shift(1).rolling(n).min()
    sig = pd.Series(np.nan, index=price.index)
    sig[price > hh] = 1.0
    sig[price < ll] = -1.0
    return sig.ffill().fillna(0.0)


def tsmom_signal(price, months):
    """时序动量：过去M月累计收益为正做多、为负做空"""
    n = int(round(months * 21))
    ret = price / price.shift(n) - 1.0
    sig = pd.Series(np.nan, index=price.index)
    sig[ret > 0] = 1.0
    sig[ret < 0] = -1.0
    return sig.fillna(0.0)


def run_backtest(ret_df, sig_df, fee=FEE):
    """等权多空组合：每品种权重=信号/品种数，扣换手手续费"""
    pos = sig_df.shift(1).fillna(0.0)
    w = pos / len(sig_df.columns)
    gross = (w * ret_df).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    return gross - turnover * fee


def single_asset_backtest(ret, sig, fee=FEE):
    """单品种：仓位=信号(+1/-1/0)，扣换手手续费"""
    pos = sig.shift(1).fillna(0.0)
    gross = pos * ret
    turnover = pos.diff().abs().fillna(0.0)
    return gross - turnover * fee


def calc_metrics(r):
    r = r.dropna()
    if len(r) == 0:
        return {}
    nav = (1 + r).cumprod()
    n = len(r)
    total = float(nav.iloc[-1] - 1)
    ann = float(nav.iloc[-1] ** (ANN / n) - 1)
    vol = float(r.std() * np.sqrt(ANN))
    sharpe = float(ann / vol) if vol > 0 else np.nan
    dd = float((nav / nav.cummax() - 1).min())
    win = float((r > 0).mean())
    calmar = float(ann / abs(dd)) if dd < 0 else np.nan
    return {
        '累计收益': total, '年化收益': ann, '年化波动': vol,
        '夏普': sharpe, '最大回撤': dd, '日胜率': win, 'Calmar': calmar,
        '起': r.index[0].date(), '止': r.index[-1].date(), '天数': n,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    ret = prices.pct_change().fillna(0.0)

    print('=' * 78)
    print('商品趋势模型回测 v0.1（多空双向 +1/-1）')
    print('=' * 78)
    print(f"品种：{'、'.join(ASSETS)}")
    print(f"数据：{prices.index[0].date()} → {prices.index[-1].date()}  共{len(prices)}个交易日")

    eval_idx = ret.index[WARMUP:]
    print(f"评估区间（剔除{WARMUP}日预热，保证各模型信号完整可比）："
          f"{eval_idx[0].date()} → {eval_idx[-1].date()}  共{len(eval_idx)}日\n")

    # ---------- 构建模型 ----------
    models = {}
    for n in [20, 55, 90]:
        models[f'唐奇安{n}日'] = {a: donchian_signal(prices[a], n) for a in ASSETS}
    for m in [1, 3, 6, 12]:
        models[f'TSMOM{m}月'] = {a: tsmom_signal(prices[a], m) for a in ASSETS}

    # 补充诊断：两个较优短期模型的『仅做多』变体（信号为负时空仓、不做空）
    # 目的：区分『趋势逻辑本身无效』与『亏损主要来自做空端』
    for base in ['唐奇安20日', 'TSMOM1月']:
        models[f'{base}·仅做多'] = {a: models[base][a].where(models[base][a] > 0, 0.0) for a in ASSETS}

    # ---------- 组合层面回测 ----------
    rows, navs = [], {}
    for name, sigdic in models.items():
        sig_df = pd.DataFrame(sigdic)
        r = run_backtest(ret, sig_df).loc[eval_idx]
        m = calc_metrics(r)
        m['模型'] = name
        rows.append(m)
        navs[name] = (1 + r).cumprod()

    bh = ret.mean(axis=1).loc[eval_idx]           # 基准：等权买入持有（纯多头，不择时）
    m = calc_metrics(bh)
    m['模型'] = '基准·等权买入持有'
    rows.append(m)
    navs['基准·等权买入持有'] = (1 + bh).cumprod()

    res = pd.DataFrame(rows).set_index('模型')
    order = ['累计收益', '年化收益', '年化波动', '夏普', '最大回撤', '日胜率', 'Calmar']
    disp = res[order].copy()
    for c in ['累计收益', '年化收益', '年化波动', '最大回撤', '日胜率']:
        disp[c] = (disp[c] * 100).round(2).astype(str) + '%'
    for c in ['夏普', 'Calmar']:
        disp[c] = disp[c].round(2)

    print('【组合层面：商品趋势策略 vs 买入持有】')
    print(disp.to_string())
    print()

    # ---------- 分品种：趋势在各品种上的表现 ----------
    print('【分品种夏普：趋势模型 vs 买入持有】')
    per_rows = []
    for a in ASSETS:
        row = {'品种': a}
        bh_a = calc_metrics(ret[a].loc[eval_idx])
        row['买入持有'] = round(bh_a['夏普'], 2)
        for name, sigdic in models.items():
            r_a = single_asset_backtest(ret[a], sigdic[a]).loc[eval_idx]
            row[name] = round(calc_metrics(r_a)['夏普'], 2)
        per_rows.append(row)
    per = pd.DataFrame(per_rows).set_index('品种')
    print(per.to_string())
    print()

    # ---------- 输出文件 ----------
    res[order].to_csv(OUT_DIR / '商品趋势_组合指标对比.csv', encoding='utf-8-sig')
    per.to_csv(OUT_DIR / '商品趋势_分品种夏普.csv', encoding='utf-8-sig')
    pd.DataFrame(navs).to_csv(OUT_DIR / '商品趋势_净值曲线.csv', encoding='utf-8-sig')

    # ---------- 净值曲线图 ----------
    fig, ax = plt.subplots(figsize=(13, 6.5))
    nav_df = pd.DataFrame(navs)
    colors = plt.cm.tab10(np.linspace(0, 1, len(nav_df.columns)))
    for col, c in zip(nav_df.columns, colors):
        lw = 2.6 if '基准' in col else 1.4
        ax.plot(nav_df.index, nav_df[col], label=col, linewidth=lw, color=c, alpha=0.9)
    ax.axhline(1.0, color='#888', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_title('商品趋势模型净值曲线（多空双向 vs 等权买入持有，2019-03 起）', fontsize=14, pad=12)
    ax.set_xlabel('日期')
    ax.set_ylabel('净值')
    ax.legend(fontsize=8.5, ncol=4, loc='upper left')
    ax.grid(alpha=0.3, linestyle=':')
    fig.tight_layout()
    fig.savefig(OUT_DIR / '商品趋势_净值曲线.png', dpi=130)
    plt.close(fig)

    print(f"输出目录：{OUT_DIR}")
    print('  - 商品趋势_组合指标对比.csv')
    print('  - 商品趋势_分品种夏普.csv')
    print('  - 商品趋势_净值曲线.csv')
    print('  - 商品趋势_净值曲线.png')


if __name__ == '__main__':
    main()
