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
VERSION = '0.17'
STRATS = ['风险平价策略', '全天候策略', '全天候增强策略 (动量择时)']

FILE_PATH_WEIGHT_RETURNS = PROJECT_DIR / '数据' / 'JYDB数据替换' / '日涨跌幅_填充.csv'
FILE_PATH_TRADE_RETURNS = PROJECT_DIR / '数据' / 'JYDB数据替换' / '日涨跌幅_未填充.csv'
FILE_PATH_INDEX_SIGNAL = PROJECT_DIR / '数据' / '原始数据' / '股指期货信号.xlsx'
FILE_PATH_MOM = PROJECT_DIR / '买方宏观预期指标合成' / '预期动量' / '增长预期动量与通胀预期动量数据.csv'
METRICS_DIR = BACKTEST_DIR / '回测指标'
CHART_DIR = BACKTEST_DIR / '回测图表'

MONTH_END_FREQ = 'M'
WEEKLY_REBALANCE_FREQ = 'W-FRI'
FEE_RATE = 0.0005
REPO_FEE_RATE = 0.000001
EWMA_DECAY = 0.97
INDEX_BASE_WEIGHT = 0.30
INDEX_FUTURES = ['沪深300主连', '中证1000主连']

MARGIN_RATIOS = {
    '沪深300主连': 0.15, '中证1000主连': 0.15, '红利低波ETF': 1.0,
    '10年国债主连': 0.03, '30年国债主连': 0.03,
    '沪铜主连': 0.10, '沪铝主连': 0.10, 'PTA主连': 0.10, '原油主连': 0.10, '豆粕主连': 0.10,
    '沪金主连': 0.10
}

RISK_PARITY_ASSET_CLASSES = {
    '股票': ['红利低波ETF'],
    '债券': ['10年国债主连', '30年国债主连'],
    '商品': ['沪铜主连', '沪铝主连', 'PTA主连', '原油主连', '豆粕主连'],
    '黄金': ['沪金主连']
}

MACRO_QUADRANTS = {
    '增长超预期': ['沪铜主连', '沪铝主连', 'PTA主连', '原油主连', '豆粕主连'],
    '增长不及预期': ['10年国债主连', '30年国债主连', '沪金主连'],
    '通胀超预期': ['沪金主连', '豆粕主连', '沪铜主连', '沪铝主连', 'PTA主连', '原油主连'],
    '通胀不及预期': ['10年国债主连', '30年国债主连', '沪金主连', '红利低波ETF']
}

PLOT_ASSET_CLASSES = {
    '股指期货': INDEX_FUTURES,
    **RISK_PARITY_ASSET_CLASSES
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
    res = minimize(
        risk_parity_convex_objective,
        np.ones(n),
        args=(cov_matrix,),
        method='L-BFGS-B',
        jac=risk_parity_convex_jacobian,
        bounds=[(1e-8, None)] * n,
        options={'ftol': 1e-12}
    )
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


def load_index_signal(file_path):
    raw = pd.read_excel(file_path, sheet_name=0, header=None)
    signal_col = None
    for col in raw.columns:
        values = raw[col].astype(str).str.strip()
        if (values == '股指期货').any():
            signal_col = col
            break
    if signal_col is None:
        signal_col = 1

    df = raw[[0, signal_col]].copy()
    df.columns = ['date', 'signal']
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['signal'] = pd.to_numeric(df['signal'], errors='coerce')
    df = df.dropna(subset=['date']).set_index('date').sort_index()
    df = df[~df.index.duplicated(keep='last')]

    first_valid = df['signal'].first_valid_index()
    if first_valid is None:
        raise ValueError("股指期货信号文件没有有效信号")
    return df.loc[first_valid:, 'signal'].ffill()


def load_macro_momentum(file_path):
    with file_path.open('r', encoding='utf-8-sig') as mom_file:
        df = pd.read_csv(mom_file, index_col=0, parse_dates=True).sort_index()
    df = df[['增长预期动量', '通胀预期动量']].apply(pd.to_numeric, errors='coerce')
    df = df[~df.index.duplicated(keep='last')]
    return df.ffill()


def get_weekly_observation_dates(index):
    observations = []
    date_series = pd.Series(index=index, data=index)
    for _, group in date_series.groupby(pd.Grouper(freq=WEEKLY_REBALANCE_FREQ)):
        if len(group) > 0:
            observations.append(group.index[-1])
    return pd.DatetimeIndex(observations)


def normalize_index_signal(signal):
    if pd.isna(signal) or signal <= 0:
        return 0.0
    return min(float(signal), 1.0)


def allocate_index_futures(signal, assets, listing_dates, rebalance_date):
    target = pd.Series(0.0, index=assets)
    total_weight = INDEX_BASE_WEIGHT * normalize_index_signal(signal)
    if total_weight <= 0:
        return target

    listed_index_futures = [
        asset for asset in INDEX_FUTURES
        if asset in assets
        and listing_dates.get(asset) is not None
        and listing_dates[asset] <= rebalance_date
    ]
    if not listed_index_futures:
        return target

    target.loc[listed_index_futures] = total_weight / len(listed_index_futures)
    return target


def build_all_weather_target(look_q, active_quadrants, assets):
    quadrant_weights = get_risk_parity_weights(calculate_ewma_semi_cov(look_q, EWMA_DECAY))
    target = pd.Series(0.0, index=assets)
    for idx, quadrant_name in enumerate(list(look_q.columns)):
        quadrant_assets = active_quadrants[quadrant_name]
        target.loc[quadrant_assets] += quadrant_weights[idx] / len(quadrant_assets)
    return target


def build_enhanced_target(look_q, active_quadrants, macro_row, assets):
    selected_quadrants = [
        '增长超预期' if macro_row['增长预期动量'] > 0 else '增长不及预期',
        '通胀超预期' if macro_row['通胀预期动量'] > 0 else '通胀不及预期'
    ]
    selected_quadrants = [q for q in selected_quadrants if q in look_q.columns]
    if len(selected_quadrants) == 0:
        return None

    quadrant_weights = get_risk_parity_weights(calculate_ewma_semi_cov(look_q[selected_quadrants], EWMA_DECAY))
    target = pd.Series(0.0, index=assets)
    for idx, quadrant_name in enumerate(selected_quadrants):
        quadrant_assets = active_quadrants[quadrant_name]
        target.loc[quadrant_assets] += quadrant_weights[idx] / len(quadrant_assets)
    return target


def combine_with_index_target(index_target, non_index_target, remaining_weight, assets):
    target = pd.Series(0.0, index=assets)
    target.loc[index_target.index] = index_target
    target.loc[non_index_target.index] += non_index_target * remaining_weight
    return target


# ================= 主流程 =================
def main():
    print(f"正在执行回测框架 v{VERSION}...")
    METRICS_DIR.mkdir(exist_ok=True)
    CHART_DIR.mkdir(exist_ok=True)

    df_weight_raw = load_returns_csv(FILE_PATH_WEIGHT_RETURNS)
    df_trade_raw = load_returns_csv(FILE_PATH_TRADE_RETURNS)
    index_signal = load_index_signal(FILE_PATH_INDEX_SIGNAL)
    df_mom = load_macro_momentum(FILE_PATH_MOM)

    # 原油主连不再使用布油连续补缺；布油连续仅作为原始保留列，后续不参与仓位或交易。
    for df in (df_weight_raw, df_trade_raw):
        if '布油连续' in df.columns:
            df.drop(columns=['布油连续'], inplace=True)

    df_weight_all = df_weight_raw / 100.0
    df_trade_all_raw = df_trade_raw / 100.0
    df_trade_all = df_trade_all_raw.fillna(0)

    repo_rate_ann = df_trade_all.get('一天期国债逆回购', pd.Series(0.0, index=df_trade_all.index))

    active_assets = []
    for asset in INDEX_FUTURES:
        if asset not in active_assets:
            active_assets.append(asset)
    for class_assets in RISK_PARITY_ASSET_CLASSES.values():
        for asset in class_assets:
            if asset not in active_assets:
                active_assets.append(asset)

    assets = [a for a in active_assets if a in df_weight_all.columns and a in df_trade_all.columns]
    non_index_assets = [a for a in assets if a not in INDEX_FUTURES]

    df_weight = df_weight_all[assets].fillna(0)
    df_trade = df_trade_all[assets]
    listing_dates = {
        asset: df_trade_all_raw[asset].first_valid_index()
        for asset in assets
    }

    signal_on_trade_dates = index_signal.reindex(df_trade.index, method='ffill')
    mom_on_trade_dates = df_mom.reindex(df_trade.index, method='ffill')
    first_signal_date = index_signal.first_valid_index()
    first_mom_date = df_mom.dropna(how='any').index.min()

    calendar_days = df_trade_all.index.to_series().diff().dt.days.fillna(1)
    repo_shifted = repo_rate_ann.shift(1).fillna(0)
    repo_net_yield = np.maximum((repo_shifted / 365.0) * calendar_days - REPO_FEE_RATE, 0.0)

    m_ratios = pd.Series({a: MARGIN_RATIOS.get(a, 1.0) for a in assets})
    week_ends = get_weekly_observation_dates(df_trade.index)

    ret_dfs = {s: pd.Series(0.0, index=df_trade.index) for s in STRATS}
    margin_dfs = {s: pd.Series(0.0, index=df_trade.index) for s in STRATS}
    weight_recs = {s: [] for s in STRATS}

    curr_ws = {s: pd.Series(0.0, index=assets) for s in STRATS}
    curr_margin = {s: 0.0 for s in STRATS}
    first_date = None

    for i in range(len(week_ends) - 1):
        reb = week_ends[i]
        if reb < first_signal_date or reb < first_mom_date:
            continue

        raw_signal = signal_on_trade_dates.loc[reb]
        macro_row = mom_on_trade_dates.loc[reb]
        if pd.isna(raw_signal) or macro_row.isna().any():
            continue

        eligible_non_index_assets = [
            asset for asset in non_index_assets
            if listing_dates.get(asset) is not None and listing_dates[asset] <= reb
        ]
        if len(eligible_non_index_assets) == 0:
            continue

        look = df_weight.loc[reb - pd.DateOffset(months=12):reb, eligible_non_index_assets]
        if len(look) < 150:
            continue

        active_quadrants = {
            name: [asset for asset in quadrant_assets if asset in eligible_non_index_assets]
            for name, quadrant_assets in MACRO_QUADRANTS.items()
        }
        active_quadrants = {name: al for name, al in active_quadrants.items() if len(al) > 0}
        if len(active_quadrants) == 0:
            continue

        look_q = pd.DataFrame({name: look[al].mean(axis=1) for name, al in active_quadrants.items()})

        index_target = allocate_index_futures(raw_signal, assets, listing_dates, reb)
        index_weight = float(index_target.sum())
        remaining_weight = max(0.0, 1.0 - index_weight)

        rp_active = get_risk_parity_weights(calculate_ewma_semi_cov(look, EWMA_DECAY))
        rp_non_index_target = pd.Series(0.0, index=assets)
        rp_non_index_target.loc[eligible_non_index_assets] = rp_active

        aw_non_index_target = build_all_weather_target(look_q, active_quadrants, assets)
        enh_non_index_target = build_enhanced_target(look_q, active_quadrants, macro_row, assets)
        if enh_non_index_target is None:
            continue

        targets = {
            '风险平价策略': combine_with_index_target(index_target, rp_non_index_target, remaining_weight, assets),
            '全天候策略': combine_with_index_target(index_target, aw_non_index_target, remaining_weight, assets),
            '全天候增强策略 (动量择时)': combine_with_index_target(
                index_target,
                enh_non_index_target,
                remaining_weight,
                assets
            )
        }

        next_week = df_trade.loc[reb + pd.Timedelta(days=1):week_ends[i + 1]]
        if len(next_week) == 0:
            continue
        if first_date is None:
            first_date = next_week.index[0]

        for date, dr in next_week.iterrows():
            daily_repo = repo_net_yield.loc[date]

            for strategy in STRATS:
                target = targets[strategy]
                if date == next_week.index[0]:
                    new_margin = (target * m_ratios).sum()
                    idle_cash = max(0.0, 1.0 - new_margin)
                    idle_return = idle_cash * daily_repo

                    cost = (target - curr_ws[strategy]).abs().sum() * FEE_RATE
                    ret_dfs[strategy].loc[date] = (target * dr).sum() - cost + idle_return

                    curr_ws[strategy] = target.copy()
                    weight_recs[strategy].append({
                        'date': reb,
                        '策略名称': strategy,
                        '股指期货信号': float(raw_signal),
                        '股指期货仓位': index_weight,
                        **{a: target.loc[a] for a in assets}
                    })
                else:
                    idle_cash = max(0.0, 1.0 - curr_margin[strategy])
                    idle_return = idle_cash * daily_repo
                    ret_dfs[strategy].loc[date] = (curr_ws[strategy] * dr).sum() + idle_return

                gross_weight = (curr_ws[strategy] * (1 + dr)).sum()
                curr_ws[strategy] = (curr_ws[strategy] * (1 + dr)) / (gross_weight or 1)
                curr_margin[strategy] = (curr_ws[strategy] * m_ratios).sum()
                margin_dfs[strategy].loc[date] = curr_margin[strategy]

    if first_date is None:
        raise ValueError("日期或数据不满足条件")

    print("正在生成每日净值数据...")
    df_navs = pd.DataFrame(index=df_trade.loc[first_date:].index)
    for strategy in STRATS:
        df_navs[strategy] = (1 + ret_dfs[strategy].loc[first_date:]).cumprod()
    navs_filename = METRICS_DIR / f'策略每日净值走势_v{VERSION}.csv'
    df_navs.to_csv(str(navs_filename), encoding='utf-8-sig')

    print("正在计算年度与全局指标...")
    all_metrics = []

    def append_metrics(period_label, start_d, end_d):
        for asset in assets:
            m = calculate_metrics(df_trade.loc[start_d:end_d, asset])
            m['回测区间'] = period_label
            m['组合/资产'] = asset
            all_metrics.append(m)
        for strategy in STRATS:
            m = calculate_metrics(ret_dfs[strategy].loc[start_d:end_d], margin_dfs[strategy].loc[start_d:end_d])
            m['回测区间'] = period_label
            m['组合/资产'] = strategy
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
    print(df_m_all[(df_m_all['回测区间'] == '全局 (Total)') & (df_m_all['组合/资产'].isin(STRATS))].set_index(
        '组合/资产').to_string())

    print("\n正在生成周度仓位明细...")
    all_weight_dfs = []
    for strategy in STRATS:
        all_weight_dfs.append(pd.DataFrame(weight_recs[strategy]))

    df_weights_all = pd.concat(all_weight_dfs, ignore_index=True)
    weight_cols = ['date', '策略名称', '股指期货信号', '股指期货仓位'] + assets
    df_weights_all = df_weights_all[weight_cols]
    weights_filename = METRICS_DIR / f'策略周度仓位明细_v{VERSION}.csv'
    df_weights_all.to_csv(str(weights_filename), index=False, encoding='utf-8-sig')

    print(f"\n数据文件已生成：\n 1. {navs_filename}\n 2. {metrics_filename}\n 3. {weights_filename}")

    fig, axes = plt.subplots(5, 1, figsize=(16, 26), sharex=False)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    axes[0].plot(df_navs.index, df_navs[STRATS[2]], label=STRATS[2], color='red', lw=2)
    axes[0].plot(df_navs.index, df_navs[STRATS[1]], label=STRATS[1], color='orange', lw=2)
    axes[0].plot(df_navs.index, df_navs[STRATS[0]], label=STRATS[0], color='purple', lw=2)
    if '沪深300主连' in df_trade.columns:
        axes[0].plot((1 + df_trade.loc[first_date:, '沪深300主连']).cumprod(), label='沪深300主连', color='blue',
                     alpha=0.3)
    if '10年国债主连' in df_trade.columns:
        axes[0].plot((1 + df_trade.loc[first_date:, '10年国债主连']).cumprod(), label='10年国债主连', color='green',
                     alpha=0.3)
    axes[0].set_title('策略累计净值走势', fontsize=14)
    axes[0].legend(loc='upper left')
    axes[0].grid(True, ls='--', alpha=0.5)

    def plot_weights(ax, strategy):
        df_w = df_weights_all[df_weights_all['策略名称'] == strategy].set_index('date')
        df_c = pd.DataFrame({
            cn: df_w[[a for a in al if a in assets]].sum(axis=1)
            for cn, al in PLOT_ASSET_CLASSES.items()
        })
        ax.stackplot(df_c.index, df_c.T, labels=df_c.columns, alpha=0.8, colors=colors)
        ax.set_title(f'{strategy} - 周度大类资产权重', fontsize=14)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
        ax.legend(loc='upper left')
        ax.grid(True, ls='--', alpha=0.4)

    for idx, strategy in enumerate(STRATS):
        plot_weights(axes[idx + 1], strategy)

    df_signal = df_weights_all[df_weights_all['策略名称'] == STRATS[0]].set_index('date')
    axes[4].plot(df_signal.index, df_signal['股指期货信号'], label='股指期货信号', color='black', lw=1.5)
    axes[4].plot(df_signal.index, df_signal['股指期货仓位'], label='股指期货仓位', color='blue', lw=1.5)
    axes[4].set_title('股指期货信号与仓位', fontsize=14)
    axes[4].set_ylim(-1.1, 1.1)
    axes[4].yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    axes[4].legend(loc='upper left')
    axes[4].grid(True, ls='--', alpha=0.4)

    plt.tight_layout()
    chart_filename = CHART_DIR / f'回测图表_v{VERSION}.png'
    plt.savefig(str(chart_filename), dpi=300)
    print(f" 4. {chart_filename}")


if __name__ == '__main__':
    main()

