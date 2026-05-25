import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
BACKTEST_DIR = BASE_DIR.parent
PROJECT_DIR = BACKTEST_DIR.parent
MPLCONFIG_DIR = BASE_DIR / '.matplotlib'
MPLCONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(MPLCONFIG_DIR))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

# ================= 配置参数 =================
VERSION = '0.15'
FILE_PATH_WEIGHT_RETURNS = PROJECT_DIR / '数据' / 'JYDB期货数据替换' / 'JYDB主力日涨跌幅_填充.csv'
FILE_PATH_TRADE_RETURNS = PROJECT_DIR / '数据' / 'JYDB期货数据替换' / 'JYDB主力日涨跌幅_未填充.csv'
FILE_PATH_MOM = PROJECT_DIR / '买方宏观预期指标合成' / '预期动量' / '增长预期动量与通胀预期动量数据.csv'
METRICS_DIR = BACKTEST_DIR / '回测指标'
CHART_DIR = BACKTEST_DIR / '回测图表'
MONTH_END_FREQ = 'M'
FEE_RATE = 0.0005
REPO_FEE_RATE = 0.000001
EWMA_DECAY = 0.97

MARGIN_RATIOS = {
    '沪深300主连': 0.15, '中证1000主连': 0.15, '红利低波ETF': 1.0,
    '10年国债主连': 0.03, '30年国债主连': 0.03,
    '沪铜主连': 0.10, '沪铝主连': 0.10, 'PTA主连': 0.10, '原油主连': 0.10, '豆粕主连': 0.10,
    '沪金主连': 0.10
}

ASSET_CLASSES = {
    '股票': ['沪深300主连', '中证1000主连', '红利低波ETF'],
    '债券': ['10年国债主连', '30年国债主连'],
    '商品': ['沪铜主连', '沪铝主连', 'PTA主连', '原油主连', '豆粕主连'],
    '黄金': ['沪金主连']
}

MACRO_QUADRANTS = {
    '增长超预期': ['沪深300主连', '中证1000主连', '沪铜主连', '沪铝主连', 'PTA主连', '原油主连', '豆粕主连'],
    '增长不及预期': ['10年国债主连', '30年国债主连', '沪金主连'],
    '通胀超预期': ['沪金主连', '豆粕主连', '沪铜主连', '沪铝主连', 'PTA主连', '原油主连'],
    '通胀不及预期': ['10年国债主连', '30年国债主连', '沪金主连', '红利低波ETF']
}

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ================= 核心模型函数 =================
def calculate_ewma_semi_cov(returns_df, decay=0.97):
    downside = np.minimum(returns_df.values, 0.0)
    T, N = downside.shape
    w = (decay ** np.arange(T - 1, -1, -1))
    w /= np.sum(w)
    weighted = downside * np.sqrt(w[:, np.newaxis])
    return np.dot(weighted.T, weighted) * 252 + np.eye(N) * 1e-8


def risk_parity_convex_objective(x, cov_matrix):
    n = len(x)
    return 0.5 * np.dot(x.T, np.dot(cov_matrix, x)) - np.sum(np.log(x)) / n


def risk_parity_convex_jacobian(x, cov_matrix):
    n = len(x)
    return np.dot(cov_matrix, x) - 1.0 / (n * x)


def get_risk_parity_weights(cov_matrix):
    n = cov_matrix.shape[0]
    res = minimize(risk_parity_convex_objective, np.ones(n), args=(cov_matrix,),
                   method='L-BFGS-B', jac=risk_parity_convex_jacobian,
                   bounds=[(1e-8, None)] * n, options={'ftol': 1e-12})
    return res.x / np.sum(res.x)


def calculate_metrics(ret_series, margin_series=None):
    if len(ret_series) < 5:
        return {k: "0.00%" for k in ['年化收益', '年化波动', '夏普比率', '最大回撤', '月度胜率']}

    ret_series = ret_series.fillna(0)
    nav = (1 + ret_series).cumprod()
    y = len(ret_series) / 252.0

    ann_ret = nav.iloc[-1] ** (1 / y) - 1 if y > 0 else 0.0
    ann_vol = ret_series.std() * np.sqrt(252)

    sharpe = (ret_series.mean() * 252) / ann_vol if ann_vol > 0 else 0.0

    max_dd = ((nav / nav.cummax()) - 1).min()
    monthly_ret = ret_series.resample(MONTH_END_FREQ).apply(lambda x: (1 + x).prod() - 1)
    win_rate = (monthly_ret > 0).sum() / len(monthly_ret) if len(monthly_ret) > 0 else 0.0

    res = {
        '年化收益': f"{ann_ret:.2%}",
        '年化波动': f"{ann_vol:.2%}",
        '夏普比率': f"{sharpe:.2f}",
        '最大回撤': f"{max_dd:.2%}",
        '月度胜率': f"{win_rate:.2%}"
    }
    if margin_series is not None:
        res['平均资金占用'] = f"{margin_series.mean():.2%}"
    return res


def load_returns_csv(file_path):
    with file_path.open('r', encoding='utf-8-sig') as returns_file:
        df = pd.read_csv(returns_file, index_col=0, parse_dates=True)
    return df.dropna(how='all')


# ================= 主流程 =================
def main():
    print(f"正在执行回测框架 v{VERSION}...")
    METRICS_DIR.mkdir(exist_ok=True)
    CHART_DIR.mkdir(exist_ok=True)

    df_weight_raw = load_returns_csv(FILE_PATH_WEIGHT_RETURNS)
    df_trade_raw = load_returns_csv(FILE_PATH_TRADE_RETURNS)

    # 原油主连不再使用布油连续补缺；布油连续仅作为原始保留列，后续不参与仓位或交易。
    for df in (df_weight_raw, df_trade_raw):
        if '布油连续' in df.columns:
            df.drop(columns=['布油连续'], inplace=True)

    df_weight_all = df_weight_raw / 100.0
    df_trade_all_raw = df_trade_raw / 100.0
    df_trade_all = df_trade_all_raw.fillna(0)

    repo_rate_ann = df_trade_all.get('一天期国债逆回购', pd.Series(0.0, index=df_trade_all.index))

    # ✅ 修复：通过 ASSET_CLASSES 字典生成严格的“白名单”资产池
    active_assets = []
    for class_assets in ASSET_CLASSES.values():
        for asset in class_assets:
            if asset not in active_assets:
                active_assets.append(asset)
    # 按 ASSET_CLASSES 固定顺序保留资产，避免仓位明细列顺序随机变化。
    assets = [a for a in active_assets if a in df_weight_all.columns and a in df_trade_all.columns]

    # 仓位优化使用填充数据；交易收益使用未填充数据。
    df_weight = df_weight_all[assets].fillna(0)
    df_trade = df_trade_all[assets]
    listing_dates = {
        asset: df_trade_all_raw[asset].first_valid_index()
        for asset in assets
    }

    calendar_days = df_trade_all.index.to_series().diff().dt.days.fillna(1)
    repo_shifted = repo_rate_ann.shift(1).fillna(0)
    repo_net_yield = np.maximum((repo_shifted / 365.0) * calendar_days - REPO_FEE_RATE, 0.0)

    m_ratios = pd.Series({a: MARGIN_RATIOS.get(a, 1.0) for a in assets})
    with FILE_PATH_MOM.open('r', encoding='utf-8-sig') as mom_file:
        df_mom = pd.read_csv(mom_file, index_col=0, parse_dates=True).resample(MONTH_END_FREQ).last().ffill()
    month_ends = df_trade.resample(MONTH_END_FREQ).last().index

    strats = ['风险平价策略', '全天候策略', '全天候增强策略 (动量择时)']
    ret_dfs = {s: pd.Series(0.0, index=df_trade.index) for s in strats}
    margin_dfs = {s: pd.Series(0.0, index=df_trade.index) for s in strats}
    weight_recs = {s: [] for s in strats}

    curr_ws = {s: pd.Series(0.0, index=assets) for s in strats}
    curr_margin = {s: 0.0 for s in strats}
    first_date = None

    for i in range(len(month_ends) - 1):
        reb = month_ends[i]
        if reb < pd.to_datetime('2013-12-01') or reb not in df_mom.index or pd.isna(df_mom.loc[reb, '增长预期动量']):
            continue

        eligible_assets = [
            asset for asset in assets
            if listing_dates.get(asset) is not None and listing_dates[asset] <= reb
        ]
        if len(eligible_assets) == 0:
            continue

        look = df_weight.loc[reb - pd.DateOffset(months=12):reb, eligible_assets]
        if len(look) < 150: continue

        active_quadrants = {
            n: [a for a in al if a in eligible_assets]
            for n, al in MACRO_QUADRANTS.items()
        }
        active_quadrants = {n: al for n, al in active_quadrants.items() if len(al) > 0}
        if len(active_quadrants) == 0:
            continue

        look_q = pd.DataFrame({n: look[al].mean(axis=1) for n, al in active_quadrants.items()})

        t_w_rp_active = get_risk_parity_weights(calculate_ewma_semi_cov(look, EWMA_DECAY))
        t_w_rp = pd.Series(0.0, index=assets)
        t_w_rp.loc[eligible_assets] = t_w_rp_active

        t_q_aw = get_risk_parity_weights(calculate_ewma_semi_cov(look_q, EWMA_DECAY))
        t_w_aw = pd.Series(0.0, index=assets)
        for idx, qn in enumerate(list(look_q.columns)):
            qa = active_quadrants[qn]
            t_w_aw[qa] += t_q_aw[idx] / len(qa)

        mq = [('增长超预期' if df_mom.loc[reb, '增长预期动量'] > 0 else '增长不及预期'),
              ('通胀超预期' if df_mom.loc[reb, '通胀预期动量'] > 0 else '通胀不及预期')]
        mq = [qn for qn in mq if qn in look_q.columns]
        if len(mq) == 0:
            continue
        t_q_enh = get_risk_parity_weights(calculate_ewma_semi_cov(look_q[mq], EWMA_DECAY))
        t_w_enh = pd.Series(0.0, index=assets)
        for idx, qn in enumerate(mq):
            qa = active_quadrants[qn]
            t_w_enh[qa] += t_q_enh[idx] / len(qa)

        targets = {'风险平价策略': t_w_rp, '全天候策略': t_w_aw, '全天候增强策略 (动量择时)': t_w_enh}

        next_m = df_trade.loc[reb + pd.Timedelta(days=1):month_ends[i + 1]]
        if first_date is None and len(next_m) > 0: first_date = next_m.index[0]

        for date, dr in next_m.iterrows():
            daily_repo = repo_net_yield.loc[date]

            for s in strats:
                if date == next_m.index[0]:
                    new_margin = (targets[s] * m_ratios).sum()
                    idle_cash = max(0.0, 1.0 - new_margin)
                    idle_return = idle_cash * daily_repo

                    cost = (targets[s] - curr_ws[s]).abs().sum() * FEE_RATE
                    ret_dfs[s].loc[date] = (targets[s] * dr).sum() - cost + idle_return

                    curr_ws[s] = targets[s].copy()
                    weight_recs[s].append({'date': reb, **{a: targets[s].loc[a] for a in assets}})
                else:
                    idle_cash = max(0.0, 1.0 - curr_margin[s])
                    idle_return = idle_cash * daily_repo
                    ret_dfs[s].loc[date] = (curr_ws[s] * dr).sum() + idle_return

                gross_weight = (curr_ws[s] * (1 + dr)).sum()
                curr_ws[s] = (curr_ws[s] * (1 + dr)) / (gross_weight or 1)
                curr_margin[s] = (curr_ws[s] * m_ratios).sum()
                margin_dfs[s].loc[date] = curr_margin[s]

    if first_date is None: raise ValueError("日期或数据不满足条件")

    print(f"正在生成每日净值数据...")
    df_navs = pd.DataFrame(index=df_trade.loc[first_date:].index)
    for s in strats:
        df_navs[s] = (1 + ret_dfs[s].loc[first_date:]).cumprod()
    navs_filename = METRICS_DIR / f'策略每日净值走势_v{VERSION}.csv'
    df_navs.to_csv(str(navs_filename), encoding='utf-8-sig')

    print(f"正在计算年度与全局指标...")
    all_metrics = []

    def append_metrics(period_label, start_d, end_d):
        for asset in assets:
            m = calculate_metrics(df_trade.loc[start_d:end_d, asset])
            m['回测区间'] = period_label
            m['组合/资产'] = asset
            all_metrics.append(m)
        for s in strats:
            m = calculate_metrics(ret_dfs[s].loc[start_d:end_d], margin_dfs[s].loc[start_d:end_d])
            m['回测区间'] = period_label
            m['组合/资产'] = s
            all_metrics.append(m)

    append_metrics('全局 (Total)', first_date, df_trade.index[-1])

    years = sorted(set(df_trade.loc[first_date:].index.year))
    for y in years:
        year_mask = (df_trade.index.year == y) & (df_trade.index >= first_date)
        if year_mask.sum() > 20:
            y_start = df_trade.index[year_mask][0]
            y_end = df_trade.index[year_mask][-1]
            append_metrics(f"{y}年", y_start, y_end)

    df_m_all = pd.DataFrame(all_metrics)
    cols_order = ['回测区间', '组合/资产', '年化收益', '年化波动', '夏普比率', '最大回撤', '月度胜率', '平均资金占用']
    cols_order = [c for c in cols_order if c in df_m_all.columns]
    df_m_all = df_m_all[cols_order]

    metrics_filename = METRICS_DIR / f'年度及全局回测指标_v{VERSION}.csv'
    df_m_all.to_csv(str(metrics_filename), index=False, encoding='utf-8-sig')

    print("\n[全局回测总览]")
    print(df_m_all[(df_m_all['回测区间'] == '全局 (Total)') & (df_m_all['组合/资产'].isin(strats))].set_index(
        '组合/资产').to_string())

    print(f"\n正在生成月度仓位明细...")
    all_weight_dfs = []
    for s in strats:
        df_w_temp = pd.DataFrame(weight_recs[s])
        df_w_temp.insert(1, '策略名称', s)
        all_weight_dfs.append(df_w_temp)

    df_weights_all = pd.concat(all_weight_dfs, ignore_index=True)
    df_weights_all = df_weights_all[['date', '策略名称'] + assets]
    weights_filename = METRICS_DIR / f'策略月度仓位明细_v{VERSION}.csv'
    df_weights_all.to_csv(str(weights_filename), index=False, encoding='utf-8-sig')

    print(f"\n数据文件已生成：\n 1. {navs_filename}\n 2. {metrics_filename}\n 3. {weights_filename}")

    fig, axes = plt.subplots(4, 1, figsize=(16, 20))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    axes[0].plot((1 + ret_dfs[strats[2]].loc[first_date:]).cumprod(), label=strats[2], color='red', lw=2)
    axes[0].plot((1 + ret_dfs[strats[1]].loc[first_date:]).cumprod(), label=strats[1], color='orange', lw=2)
    axes[0].plot((1 + ret_dfs[strats[0]].loc[first_date:]).cumprod(), label=strats[0], color='purple', lw=2)

    if '沪深300主连' in df_trade.columns:
        axes[0].plot((1 + df_trade.loc[first_date:, '沪深300主连']).cumprod(), label='沪深300主连', color='blue',
                     alpha=0.3)
    if '10年国债主连' in df_trade.columns:
        axes[0].plot((1 + df_trade.loc[first_date:, '10年国债主连']).cumprod(), label='10年国债主连', color='green',
                     alpha=0.3)

    axes[0].set_title('策略累计净值走势', fontsize=14);
    axes[0].legend(loc='upper left');
    axes[0].grid(True, ls='--', alpha=0.5)

    def plot_w(ax, s_name, rec):
        df_w = pd.DataFrame(rec).set_index('date')
        df_c = pd.DataFrame({cn: df_w[[a for a in al if a in assets]].sum(axis=1) for cn, al in ASSET_CLASSES.items()})
        ax.stackplot(df_c.index, df_c.T, labels=df_c.columns, alpha=0.8, colors=colors)
        ax.set_title(f'{s_name} - 大类资产权重', fontsize=14);
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0));
        ax.legend(loc='upper left')

    for i, s in enumerate(strats): plot_w(axes[i + 1], s, weight_recs[s])

    plt.tight_layout()
    plt.savefig(str(CHART_DIR / f'回测图表_v{VERSION}.png'), dpi=300)


if __name__ == '__main__':
    main()
