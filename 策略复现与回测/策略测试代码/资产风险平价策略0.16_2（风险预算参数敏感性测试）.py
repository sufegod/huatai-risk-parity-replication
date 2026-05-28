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
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

VERSION = '0.16_2'
STRATEGY_NAME = '风险平价策略'

FILE_PATH_WEIGHT_RETURNS = PROJECT_DIR / '数据' / 'JYDB期货数据替换' / 'JYDB主力日涨跌幅_填充.csv'
FILE_PATH_TRADE_RETURNS = PROJECT_DIR / '数据' / 'JYDB期货数据替换' / 'JYDB主力日涨跌幅_未填充.csv'
FILE_PATH_INDEX_SIGNAL = PROJECT_DIR / '数据' / '原始数据' / '股指期货信号.xlsx'
METRICS_DIR = BACKTEST_DIR / '回测指标'
FILE_PATH_V016_WEIGHTS = METRICS_DIR / '策略周度仓位明细_v0.16.csv'

try:
    pd.Series([1], index=pd.to_datetime(['2000-01-01'])).resample('ME').sum()
    MONTH_END_FREQ = 'ME'
except ValueError:
    MONTH_END_FREQ = 'M'
WEEKLY_REBALANCE_FREQ = 'W-FRI'
FEE_RATE = 0.0005
REPO_FEE_RATE = 0.000001
EWMA_DECAY = 0.97
TRAIN_END_DATE = pd.Timestamp('2020-12-31')
VALIDATION_START_DATE = pd.Timestamp('2021-01-01')

INDEX_FUTURES = ['沪深300主连', '中证1000主连']
LOW_SIGNAL_MULTIPLIERS = [0.25, 0.50, 0.75]
POSITIVE_SIGNAL_MAX_MULTIPLIERS = [1.50, 2.00, 2.50, 3.00, 4.00, 5.00, 6.00, 7.00]

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


def calculate_ewma_semi_cov(returns_df, decay=0.97):
    downside = np.minimum(returns_df.values, 0.0)
    T, N = downside.shape
    w = decay ** np.arange(T - 1, -1, -1)
    w /= np.sum(w)
    weighted = downside * np.sqrt(w[:, np.newaxis])
    return np.dot(weighted.T, weighted) * 252 + np.eye(N) * 1e-8


def risk_budget_convex_objective(x, cov_matrix, risk_budget):
    return 0.5 * np.dot(x.T, np.dot(cov_matrix, x)) - np.dot(risk_budget, np.log(x))


def risk_budget_convex_jacobian(x, cov_matrix, risk_budget):
    return np.dot(cov_matrix, x) - risk_budget / x


def get_risk_budget_parity_weights(cov_matrix, risk_budget):
    risk_budget = np.asarray(risk_budget, dtype=float)
    risk_budget = risk_budget / risk_budget.sum()
    n = cov_matrix.shape[0]
    res = minimize(
        risk_budget_convex_objective,
        np.ones(n),
        args=(cov_matrix, risk_budget),
        method='L-BFGS-B',
        jac=risk_budget_convex_jacobian,
        bounds=[(1e-8, None)] * n,
        options={'ftol': 1e-12}
    )
    return res.x / np.sum(res.x)


def load_returns_csv(file_path):
    with file_path.open('r', encoding='utf-8-sig') as returns_file:
        df = pd.read_csv(returns_file, index_col=0, parse_dates=True)
    return df.dropna(how='all')


def load_index_signal(file_path):
    with file_path.open('rb') as signal_file:
        raw = pd.read_excel(signal_file, sheet_name=0, header=None)
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


def get_weekly_observation_dates(index):
    observations = []
    date_series = pd.Series(index=index, data=index)
    for _, group in date_series.groupby(pd.Grouper(freq=WEEKLY_REBALANCE_FREQ)):
        if len(group) > 0:
            observations.append(group.index[-1])
    return pd.DatetimeIndex(observations)


def get_index_budget_multiplier(signal, low_signal_multiplier, positive_signal_max_multiplier):
    if signal <= 0:
        return low_signal_multiplier
    if signal <= 1:
        return 1.0 + float(signal) * (positive_signal_max_multiplier - 1.0)
    return positive_signal_max_multiplier


def build_risk_budget(eligible_assets, signal, low_signal_multiplier, positive_signal_max_multiplier):
    raw_budget = pd.Series(1.0, index=eligible_assets, dtype=float)
    multiplier = get_index_budget_multiplier(signal, low_signal_multiplier, positive_signal_max_multiplier)
    listed_index_futures = [asset for asset in INDEX_FUTURES if asset in raw_budget.index]
    raw_budget.loc[listed_index_futures] *= multiplier
    return raw_budget / raw_budget.sum(), multiplier


def calculate_risk_contributions(weights, cov_matrix):
    w = weights.values
    marginal_risk = np.dot(cov_matrix, w)
    portfolio_var = np.dot(w.T, marginal_risk)
    if portfolio_var <= 0:
        return pd.Series(0.0, index=weights.index)
    return pd.Series(w * marginal_risk / portfolio_var, index=weights.index)


def calculate_metrics_numeric(ret_series, margin_series=None):
    ret_series = ret_series.fillna(0.0)
    if len(ret_series) < 5:
        res = {'期末净值': 1.0, '年化收益': 0.0, '年化波动': 0.0, '夏普比率': 0.0, '最大回撤': 0.0, '月度胜率': 0.0}
        if margin_series is not None:
            res['平均资金占用'] = 0.0
        return res

    nav = (1.0 + ret_series).cumprod()
    years = len(ret_series) / 252.0
    ann_ret = nav.iloc[-1] ** (1.0 / years) - 1.0
    ann_vol = ret_series.std() * np.sqrt(252)
    sharpe = ret_series.mean() * 252 / ann_vol if ann_vol > 0 else 0.0
    max_dd = (nav / nav.cummax() - 1.0).min()
    monthly_ret = ret_series.resample(MONTH_END_FREQ).apply(lambda x: (1.0 + x).prod() - 1.0)
    win_rate = (monthly_ret > 0).sum() / len(monthly_ret) if len(monthly_ret) > 0 else 0.0
    res = {'期末净值': nav.iloc[-1], '年化收益': ann_ret, '年化波动': ann_vol, '夏普比率': sharpe, '最大回撤': max_dd, '月度胜率': win_rate}
    if margin_series is not None:
        res['平均资金占用'] = margin_series.mean()
    return res


def format_percent(value):
    return f"{value:.2%}"


def load_reference_v016_weights():
    if not FILE_PATH_V016_WEIGHTS.exists():
        return None
    with FILE_PATH_V016_WEIGHTS.open('r', encoding='utf-8-sig') as weights_file:
        df = pd.read_csv(weights_file, parse_dates=['date'])
    if '股指期货仓位' not in df.columns:
        return None
    return df


def calculate_reference_index_weight(reference_weights, period_label, year=None):
    if reference_weights is None:
        return np.nan
    if period_label == '全局 (Total)':
        subset = reference_weights
    elif period_label == '训练期 (2018-2020)':
        subset = reference_weights[reference_weights['date'] <= TRAIN_END_DATE]
    elif period_label == '验证期 (2021-2026)':
        subset = reference_weights[reference_weights['date'] >= VALIDATION_START_DATE]
    elif year is not None:
        subset = reference_weights[reference_weights['date'].dt.year == year]
    else:
        subset = reference_weights.iloc[0:0]
    if len(subset) == 0:
        return np.nan
    return subset['股指期货仓位'].mean()


def load_data():
    df_weight_raw = load_returns_csv(FILE_PATH_WEIGHT_RETURNS)
    df_trade_raw = load_returns_csv(FILE_PATH_TRADE_RETURNS)
    index_signal = load_index_signal(FILE_PATH_INDEX_SIGNAL)

    for df in (df_weight_raw, df_trade_raw):
        if '布油连续' in df.columns:
            df.drop(columns=['布油连续'], inplace=True)

    active_assets = []
    for asset in INDEX_FUTURES:
        if asset not in active_assets:
            active_assets.append(asset)
    for class_assets in RISK_PARITY_ASSET_CLASSES.values():
        for asset in class_assets:
            if asset not in active_assets:
                active_assets.append(asset)

    df_weight_all = df_weight_raw / 100.0
    df_trade_all_raw = df_trade_raw / 100.0
    df_trade_all = df_trade_all_raw.fillna(0)
    assets = [a for a in active_assets if a in df_weight_all.columns and a in df_trade_all.columns]
    df_weight = df_weight_all[assets].fillna(0)
    df_trade = df_trade_all[assets]
    listing_dates = {asset: df_trade_all_raw[asset].first_valid_index() for asset in assets}

    repo_rate_ann = df_trade_all.get('一天期国债逆回购', pd.Series(0.0, index=df_trade_all.index))
    calendar_days = df_trade_all.index.to_series().diff().dt.days.fillna(1)
    repo_net_yield = np.maximum((repo_rate_ann.shift(1).fillna(0) / 365.0) * calendar_days - REPO_FEE_RATE, 0.0)

    return {
        'assets': assets,
        'df_weight': df_weight,
        'df_trade': df_trade,
        'listing_dates': listing_dates,
        'signal_on_trade_dates': index_signal.reindex(df_trade.index, method='ffill'),
        'first_signal_date': index_signal.first_valid_index(),
        'repo_net_yield': repo_net_yield,
        'm_ratios': pd.Series({a: MARGIN_RATIOS.get(a, 1.0) for a in assets}),
        'week_ends': get_weekly_observation_dates(df_trade.index)
    }


def run_backtest(data, low_signal_multiplier, positive_signal_max_multiplier):
    assets = data['assets']
    df_weight = data['df_weight']
    df_trade = data['df_trade']
    listing_dates = data['listing_dates']
    signal_on_trade_dates = data['signal_on_trade_dates']
    first_signal_date = data['first_signal_date']
    repo_net_yield = data['repo_net_yield']
    m_ratios = data['m_ratios']
    week_ends = data['week_ends']

    ret_series = pd.Series(0.0, index=df_trade.index)
    margin_series = pd.Series(0.0, index=df_trade.index)
    weight_records = []
    curr_w = pd.Series(0.0, index=assets)
    curr_margin = 0.0
    first_date = None

    for i in range(len(week_ends) - 1):
        reb = week_ends[i]
        if reb < first_signal_date:
            continue
        raw_signal = signal_on_trade_dates.loc[reb]
        if pd.isna(raw_signal):
            continue
        eligible_assets = [asset for asset in assets if listing_dates.get(asset) is not None and listing_dates[asset] <= reb]
        if len(eligible_assets) == 0:
            continue
        look = df_weight.loc[reb - pd.DateOffset(months=12):reb, eligible_assets]
        if len(look) < 150:
            continue

        cov_matrix = calculate_ewma_semi_cov(look, EWMA_DECAY)
        risk_budget, multiplier = build_risk_budget(
            eligible_assets,
            raw_signal,
            low_signal_multiplier,
            positive_signal_max_multiplier
        )
        active_weights = pd.Series(
            get_risk_budget_parity_weights(cov_matrix, risk_budget.values),
            index=eligible_assets
        )
        target = pd.Series(0.0, index=assets)
        target.loc[eligible_assets] = active_weights

        risk_contrib = calculate_risk_contributions(active_weights, cov_matrix)
        listed_index_futures = [asset for asset in INDEX_FUTURES if asset in eligible_assets]
        index_weight = float(target.loc[listed_index_futures].sum()) if listed_index_futures else 0.0
        index_budget = float(risk_budget.loc[listed_index_futures].sum()) if listed_index_futures else 0.0
        index_contrib = float(risk_contrib.loc[listed_index_futures].sum()) if listed_index_futures else 0.0

        next_week = df_trade.loc[reb + pd.Timedelta(days=1):week_ends[i + 1]]
        if len(next_week) == 0:
            continue
        if first_date is None:
            first_date = next_week.index[0]

        for date, dr in next_week.iterrows():
            daily_repo = repo_net_yield.loc[date]
            if date == next_week.index[0]:
                new_margin = (target * m_ratios).sum()
                idle_return = max(0.0, 1.0 - new_margin) * daily_repo
                cost = (target - curr_w).abs().sum() * FEE_RATE
                ret_series.loc[date] = (target * dr).sum() - cost + idle_return
                curr_w = target.copy()
                weight_records.append({
                    'date': reb,
                    '股指期货信号': float(raw_signal),
                    '股指风险预算倍率': multiplier,
                    '股指目标风险预算': index_budget,
                    '股指实际风险贡献': index_contrib,
                    '股指期货仓位': index_weight,
                    '权重合计': target.sum()
                })
            else:
                idle_return = max(0.0, 1.0 - curr_margin) * daily_repo
                ret_series.loc[date] = (curr_w * dr).sum() + idle_return

            gross_weight = (curr_w * (1.0 + dr)).sum()
            curr_w = (curr_w * (1.0 + dr)) / (gross_weight or 1.0)
            curr_margin = (curr_w * m_ratios).sum()
            margin_series.loc[date] = curr_margin

    if first_date is None:
        raise ValueError("日期或数据不满足条件")
    weights = pd.DataFrame(weight_records)
    return ret_series.loc[first_date:], margin_series.loc[first_date:], weights, first_date


def append_period_result(
    results,
    label,
    ret_series,
    margin_series,
    weights,
    low_multiplier,
    positive_multiplier,
    reference_index_weight
):
    metrics = calculate_metrics_numeric(ret_series, margin_series)
    strategy_index_weight = weights['股指期货仓位'].mean() if len(weights) else 0.0
    index_weight_diff = strategy_index_weight - reference_index_weight if not pd.isna(reference_index_weight) else np.nan
    results.append({
        '低信号倍率': f"{low_multiplier:.2f}",
        '正信号满值倍率': f"{positive_multiplier:.2f}",
        '参数组合': f"低信号{low_multiplier:.2f}_满值{positive_multiplier:.2f}",
        '回测区间': label,
        '期末净值': f"{metrics['期末净值']:.4f}",
        '年化收益': format_percent(metrics['年化收益']),
        '年化波动': format_percent(metrics['年化波动']),
        '夏普比率': f"{metrics['夏普比率']:.2f}",
        '最大回撤': format_percent(metrics['最大回撤']),
        '月度胜率': format_percent(metrics['月度胜率']),
        '平均资金占用': format_percent(metrics.get('平均资金占用', 0.0)),
        '股指平均仓位': format_percent(strategy_index_weight),
        'v0.16股指平均仓位': format_percent(reference_index_weight) if not pd.isna(reference_index_weight) else '',
        '股指仓位差异': format_percent(index_weight_diff) if not pd.isna(index_weight_diff) else '',
        '股指目标风险预算均值': format_percent(weights['股指目标风险预算'].mean() if len(weights) else 0.0),
        '股指实际风险贡献均值': format_percent(weights['股指实际风险贡献'].mean() if len(weights) else 0.0),
        '周度记录数': len(weights)
    })


def main():
    print(f"正在执行 v{VERSION} 风险预算参数敏感性测试...")
    METRICS_DIR.mkdir(exist_ok=True)
    data = load_data()
    reference_weights = load_reference_v016_weights()
    results = []

    for low_multiplier in LOW_SIGNAL_MULTIPLIERS:
        for positive_multiplier in POSITIVE_SIGNAL_MAX_MULTIPLIERS:
            ret_series, margin_series, weights, first_date = run_backtest(data, low_multiplier, positive_multiplier)
            append_period_result(
                results,
                '全局 (Total)',
                ret_series,
                margin_series,
                weights,
                low_multiplier,
                positive_multiplier,
                calculate_reference_index_weight(reference_weights, '全局 (Total)')
            )
            train_mask = ret_series.index <= TRAIN_END_DATE
            validation_mask = ret_series.index >= VALIDATION_START_DATE
            append_period_result(
                results,
                '训练期 (2018-2020)',
                ret_series.loc[train_mask],
                margin_series.loc[train_mask],
                weights[weights['date'] <= TRAIN_END_DATE],
                low_multiplier,
                positive_multiplier,
                calculate_reference_index_weight(reference_weights, '训练期 (2018-2020)')
            )
            append_period_result(
                results,
                '验证期 (2021-2026)',
                ret_series.loc[validation_mask],
                margin_series.loc[validation_mask],
                weights[weights['date'] >= VALIDATION_START_DATE],
                low_multiplier,
                positive_multiplier,
                calculate_reference_index_weight(reference_weights, '验证期 (2021-2026)')
            )
            for year in sorted(set(ret_series.index.year)):
                year_mask = ret_series.index.year == year
                if year_mask.sum() > 20:
                    append_period_result(
                        results,
                        f"{year}年",
                        ret_series.loc[year_mask],
                        margin_series.loc[year_mask],
                        weights[weights['date'].dt.year == year],
                        low_multiplier,
                        positive_multiplier,
                        calculate_reference_index_weight(reference_weights, f"{year}年", year)
                    )

    df_results = pd.DataFrame(results)
    output_file = METRICS_DIR / f'风险预算参数敏感性测试_v{VERSION}.csv'
    df_results.to_csv(str(output_file), index=False, encoding='utf-8-sig')
    print(f"敏感性测试结果已生成：{output_file}")


if __name__ == '__main__':
    main()
