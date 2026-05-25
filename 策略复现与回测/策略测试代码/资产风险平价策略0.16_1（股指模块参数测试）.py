import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
VERSION = '0.16_1'
STRATEGY_NAME = '风险平价策略'
BASELINE_SCENARIO = '固定仓位_30%'

FILE_PATH_WEIGHT_RETURNS = PROJECT_DIR / '数据' / 'JYDB期货数据替换' / 'JYDB主力日涨跌幅_填充.csv'
FILE_PATH_TRADE_RETURNS = PROJECT_DIR / '数据' / 'JYDB期货数据替换' / 'JYDB主力日涨跌幅_未填充.csv'
FILE_PATH_INDEX_SIGNAL = PROJECT_DIR / '数据' / '原始数据' / '股指期货信号.xlsx'
METRICS_DIR = BACKTEST_DIR / '回测指标'
TEST_OUTPUT_DIR = METRICS_DIR / f'股指模块参数测试_v{VERSION}'

MONTH_END_FREQ = 'M'
WEEKLY_REBALANCE_FREQ = 'W-FRI'
FEE_RATE = 0.0005
REPO_FEE_RATE = 0.000001
EWMA_DECAY = 0.97
VOL_WINDOW = 60
TRAIN_PERIOD_LABEL = '训练期 (2018-2020)'
VALIDATION_PERIOD_LABEL = '验证期 (2021-2026)'
TRAIN_END_DATE = pd.Timestamp('2020-12-31')
VALIDATION_START_DATE = pd.Timestamp('2021-01-01')

INDEX_FUTURES = ['沪深300主连', '中证1000主连']
FIXED_BASE_WEIGHTS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
VOL_TARGETS = [0.12, 0.15, 0.18]
RISK_BUDGETS = [0.15, 0.20, 0.25]
INDEX_CHANGE_LIMITS = [0.10, 0.15, 0.20]

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

PLOT_ASSET_CLASSES = {
    '股指期货': INDEX_FUTURES,
    **RISK_PARITY_ASSET_CLASSES
}

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


@dataclass(frozen=True)
class IndexModuleConfig:
    scenario_name: str
    experiment_group: str
    mode: str
    base_weight: float
    vol_target: Optional[float] = None
    risk_budget: Optional[float] = None
    max_weekly_index_change: Optional[float] = None


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


def normalize_index_signal(signal):
    if pd.isna(signal) or signal <= 0:
        return 0.0
    return min(float(signal), 1.0)


def annualized_vol(ret_series):
    clean = ret_series.dropna()
    if len(clean) < 20:
        return np.nan
    return clean.std() * np.sqrt(252)


def get_listed_index_futures(assets, listing_dates, rebalance_date):
    return [
        asset for asset in INDEX_FUTURES
        if asset in assets
        and listing_dates.get(asset) is not None
        and listing_dates[asset] <= rebalance_date
    ]


def calculate_index_equal_return(df_weight, listed_index_futures, end_date):
    if not listed_index_futures:
        return pd.Series(dtype=float)
    return df_weight.loc[:end_date, listed_index_futures].tail(VOL_WINDOW).mean(axis=1)


def calculate_non_index_portfolio_return(df_weight, rp_weights, end_date):
    look = df_weight.loc[:end_date, rp_weights.index].tail(VOL_WINDOW)
    if len(look) < 20:
        return pd.Series(dtype=float)
    return look.mul(rp_weights, axis=1).sum(axis=1)


def calculate_desired_index_weight(
    config,
    signal,
    df_weight,
    listed_index_futures,
    rp_weights,
    rebalance_date
):
    signal_scale = normalize_index_signal(signal)
    if signal_scale <= 0:
        return 0.0

    base_target = config.base_weight * signal_scale
    if config.mode == 'fixed':
        return base_target

    index_ret = calculate_index_equal_return(df_weight, listed_index_futures, rebalance_date)
    index_vol = annualized_vol(index_ret)
    if pd.isna(index_vol) or index_vol <= 0:
        return base_target

    if config.mode == 'vol_scaled':
        scale = min(1.0, config.vol_target / index_vol)
        return base_target * scale

    if config.mode == 'risk_budget':
        non_index_ret = calculate_non_index_portfolio_return(df_weight, rp_weights, rebalance_date)
        non_index_vol = annualized_vol(non_index_ret)
        if pd.isna(non_index_vol) or non_index_vol <= 0:
            return base_target

        budget = config.risk_budget
        risk_weight = (budget * non_index_vol) / ((1.0 - budget) * index_vol + budget * non_index_vol)
        return min(config.base_weight, risk_weight) * signal_scale

    raise ValueError(f"未知股指模块模式: {config.mode}")


def apply_index_change_limit(desired_weight, previous_weight, max_weekly_index_change):
    if max_weekly_index_change is None:
        return desired_weight
    delta = desired_weight - previous_weight
    if abs(delta) <= max_weekly_index_change:
        return desired_weight
    return previous_weight + np.sign(delta) * max_weekly_index_change


def allocate_index_futures(total_weight, assets, listing_dates, rebalance_date):
    target = pd.Series(0.0, index=assets)
    if total_weight <= 0:
        return target

    listed_index_futures = get_listed_index_futures(assets, listing_dates, rebalance_date)
    if not listed_index_futures:
        return target

    target.loc[listed_index_futures] = total_weight / len(listed_index_futures)
    return target


def calculate_metrics_numeric(ret_series, margin_series=None):
    if len(ret_series) < 5:
        res = {
            '年化收益': 0.0,
            '年化波动': 0.0,
            '夏普比率': 0.0,
            '最大回撤': 0.0,
            '月度胜率': 0.0,
            '期末净值': 1.0
        }
        if margin_series is not None:
            res['平均资金占用'] = 0.0
        return res

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
        '年化收益': ann_ret,
        '年化波动': ann_vol,
        '夏普比率': sharpe,
        '最大回撤': max_dd,
        '月度胜率': win_rate,
        '期末净值': nav.iloc[-1]
    }
    if margin_series is not None:
        res['平均资金占用'] = margin_series.mean()
    return res


def format_percent(value):
    return f"{value:.2%}"


def format_metric_rows(df_rows):
    df = pd.DataFrame(df_rows)
    percent_cols = [
        '基础股指仓位', '目标波动', '风险预算', '单周股指变化上限',
        '年化收益', '年化波动', '最大回撤', '月度胜率', '平均资金占用',
        '平均周换手', '最大单周换手', '股指平均仓位', '股指最大仓位',
        '2020回撤窗口收益', '2020回撤窗口最大回撤'
    ]
    for col in percent_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: '' if pd.isna(x) else format_percent(float(x)))
    if '夏普比率' in df.columns:
        df['夏普比率'] = df['夏普比率'].apply(lambda x: f"{float(x):.2f}" if not pd.isna(x) else '')
    if '期末净值' in df.columns:
        df['期末净值'] = df['期末净值'].apply(lambda x: f"{float(x):.4f}" if not pd.isna(x) else '')
    return df


def get_period_bounds(label, first_date, end_date):
    if label == '全局 (Total)':
        return first_date, end_date
    if label == TRAIN_PERIOD_LABEL:
        return first_date, min(TRAIN_END_DATE, end_date)
    if label == VALIDATION_PERIOD_LABEL:
        return max(VALIDATION_START_DATE, first_date), end_date
    year = int(label.replace('年', ''))
    start = max(pd.Timestamp(year=year, month=1, day=1), first_date)
    end = min(pd.Timestamp(year=year, month=12, day=31), end_date)
    return start, end


def build_scenario_rows(result):
    config = result['config']
    rows = []
    for label in result['period_labels']:
        start_d, end_d = get_period_bounds(label, result['first_date'], result['end_date'])
        if start_d > end_d:
            continue
        ret_slice = result['returns'].loc[start_d:end_d]
        margin_slice = result['margin'].loc[start_d:end_d]
        if len(ret_slice) < 5:
            continue
        metrics = calculate_metrics_numeric(ret_slice, margin_slice)
        rows.append({
            '方案名称': config.scenario_name,
            '实验分组': config.experiment_group,
            '股指模式': config.mode,
            '基础股指仓位': config.base_weight,
            '目标波动': config.vol_target,
            '风险预算': config.risk_budget,
            '单周股指变化上限': config.max_weekly_index_change,
            '回测区间': label,
            '期末净值': metrics['期末净值'],
            '年化收益': metrics['年化收益'],
            '年化波动': metrics['年化波动'],
            '夏普比率': metrics['夏普比率'],
            '最大回撤': metrics['最大回撤'],
            '月度胜率': metrics['月度胜率'],
            '平均资金占用': metrics['平均资金占用'],
            **result['diagnostics']
        })
    return rows


def run_backtest(config, data):
    df_weight_all = data['df_weight_all']
    df_trade_all_raw = data['df_trade_all_raw']
    df_trade_all = data['df_trade_all']
    index_signal = data['index_signal']
    repo_rate_ann = data['repo_rate_ann']
    assets = data['assets']
    risk_parity_assets = data['risk_parity_assets']
    listing_dates = data['listing_dates']

    df_weight = df_weight_all[assets].fillna(0)
    df_trade = df_trade_all[assets]
    signal_on_trade_dates = index_signal.reindex(df_trade.index, method='ffill')
    first_signal_date = index_signal.first_valid_index()

    calendar_days = df_trade_all.index.to_series().diff().dt.days.fillna(1)
    repo_shifted = repo_rate_ann.shift(1).fillna(0)
    repo_net_yield = np.maximum((repo_shifted / 365.0) * calendar_days - REPO_FEE_RATE, 0.0)

    m_ratios = pd.Series({a: MARGIN_RATIOS.get(a, 1.0) for a in assets})
    week_ends = get_weekly_observation_dates(df_trade.index)

    ret_series = pd.Series(0.0, index=df_trade.index)
    margin_series = pd.Series(0.0, index=df_trade.index)
    weight_recs = []
    curr_w = pd.Series(0.0, index=assets)
    curr_margin = 0.0
    prev_index_weight = 0.0
    first_date = None

    for i in range(len(week_ends) - 1):
        reb = week_ends[i]
        if reb < first_signal_date:
            continue

        raw_signal = signal_on_trade_dates.loc[reb]
        if pd.isna(raw_signal):
            continue

        eligible_rp_assets = [
            asset for asset in risk_parity_assets
            if listing_dates.get(asset) is not None and listing_dates[asset] <= reb
        ]
        if len(eligible_rp_assets) == 0:
            continue

        look = df_weight.loc[reb - pd.DateOffset(months=12):reb, eligible_rp_assets]
        if len(look) < 150:
            continue

        rp_active = pd.Series(
            get_risk_parity_weights(calculate_ewma_semi_cov(look, EWMA_DECAY)),
            index=eligible_rp_assets
        )

        listed_index_futures = get_listed_index_futures(assets, listing_dates, reb)
        desired_index_weight = calculate_desired_index_weight(
            config,
            raw_signal,
            df_weight,
            listed_index_futures,
            rp_active,
            reb
        )
        index_weight = apply_index_change_limit(
            desired_index_weight,
            prev_index_weight,
            config.max_weekly_index_change
        )
        index_weight = min(max(index_weight, 0.0), config.base_weight)
        prev_index_weight = index_weight
        remaining_weight = max(0.0, 1.0 - index_weight)

        index_target = allocate_index_futures(index_weight, assets, listing_dates, reb)
        target = pd.Series(0.0, index=assets)
        target.loc[index_target.index] = index_target
        target.loc[eligible_rp_assets] = rp_active * remaining_weight

        next_week = df_trade.loc[reb + pd.Timedelta(days=1):week_ends[i + 1]]
        if len(next_week) == 0:
            continue
        if first_date is None:
            first_date = next_week.index[0]

        for date, dr in next_week.iterrows():
            daily_repo = repo_net_yield.loc[date]

            if date == next_week.index[0]:
                new_margin = (target * m_ratios).sum()
                idle_cash = max(0.0, 1.0 - new_margin)
                idle_return = idle_cash * daily_repo
                cost = (target - curr_w).abs().sum() * FEE_RATE
                ret_series.loc[date] = (target * dr).sum() - cost + idle_return
                curr_w = target.copy()
                weight_recs.append({
                    'date': reb,
                    '方案名称': config.scenario_name,
                    '策略名称': STRATEGY_NAME,
                    '股指期货信号': float(raw_signal),
                    '股指期货目标仓位': desired_index_weight,
                    '股指期货仓位': index_weight,
                    **{a: target.loc[a] for a in assets}
                })
            else:
                idle_cash = max(0.0, 1.0 - curr_margin)
                idle_return = idle_cash * daily_repo
                ret_series.loc[date] = (curr_w * dr).sum() + idle_return

            gross_weight = (curr_w * (1 + dr)).sum()
            curr_w = (curr_w * (1 + dr)) / (gross_weight or 1)
            curr_margin = (curr_w * m_ratios).sum()
            margin_series.loc[date] = curr_margin

    if first_date is None:
        raise ValueError(f"{config.scenario_name} 日期或数据不满足条件")

    df_weights = pd.DataFrame(weight_recs)
    asset_cols = [a for a in assets if a in df_weights.columns]
    turnover = df_weights[asset_cols].diff().abs().sum(axis=1).iloc[1:]
    index_weights = df_weights['股指期货仓位']
    signal_changes = int((df_weights['股指期货信号'].diff() != 0).sum() - 1)

    dd_window_returns = ret_series.loc[pd.Timestamp('2020-01-20'):pd.Timestamp('2020-03-19')]
    dd_window_nav = (1 + dd_window_returns).cumprod()
    if len(dd_window_nav) > 0:
        dd_window_ret = dd_window_nav.iloc[-1] - 1
        dd_window_mdd = (dd_window_nav / dd_window_nav.cummax() - 1).min()
    else:
        dd_window_ret = np.nan
        dd_window_mdd = np.nan

    diagnostics = {
        '平均周换手': turnover.mean() if len(turnover) > 0 else 0.0,
        '最大单周换手': turnover.max() if len(turnover) > 0 else 0.0,
        '股指平均仓位': index_weights.mean() if len(index_weights) > 0 else 0.0,
        '股指最大仓位': index_weights.max() if len(index_weights) > 0 else 0.0,
        '信号切换次数': signal_changes,
        '2020回撤窗口收益': dd_window_ret,
        '2020回撤窗口最大回撤': dd_window_mdd
    }

    years = sorted(set(df_trade.loc[first_date:].index.year))
    year_labels = []
    for y in years:
        year_mask = (df_trade.index.year == y) & (df_trade.index >= first_date)
        if year_mask.sum() > 20:
            year_labels.append(f"{y}年")

    return {
        'config': config,
        'returns': ret_series.loc[first_date:],
        'nav': (1 + ret_series.loc[first_date:]).cumprod(),
        'margin': margin_series.loc[first_date:],
        'weights': df_weights,
        'first_date': first_date,
        'end_date': df_trade.index[-1],
        'period_labels': ['全局 (Total)', TRAIN_PERIOD_LABEL, VALIDATION_PERIOD_LABEL] + year_labels,
        'diagnostics': diagnostics
    }


def load_data():
    df_weight_raw = load_returns_csv(FILE_PATH_WEIGHT_RETURNS)
    df_trade_raw = load_returns_csv(FILE_PATH_TRADE_RETURNS)
    index_signal = load_index_signal(FILE_PATH_INDEX_SIGNAL)

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
    risk_parity_assets = [a for a in assets if a not in INDEX_FUTURES]
    listing_dates = {
        asset: df_trade_all_raw[asset].first_valid_index()
        for asset in assets
    }

    return {
        'df_weight_all': df_weight_all,
        'df_trade_all_raw': df_trade_all_raw,
        'df_trade_all': df_trade_all,
        'index_signal': index_signal,
        'repo_rate_ann': repo_rate_ann,
        'assets': assets,
        'risk_parity_assets': risk_parity_assets,
        'listing_dates': listing_dates
    }


def build_fixed_configs():
    return [
        IndexModuleConfig(
            scenario_name=f"固定仓位_{int(weight * 100)}%",
            experiment_group='固定仓位比例测试',
            mode='fixed',
            base_weight=weight
        )
        for weight in FIXED_BASE_WEIGHTS
    ]


def numeric_metric(df_results, scenario_name, period, column):
    row = df_results[
        (df_results['方案名称'] == scenario_name)
        & (df_results['回测区间'] == period)
    ]
    if row.empty:
        return np.nan
    return float(row.iloc[0][column])


def choose_dynamic_base_weights(fixed_rows):
    df_fixed = pd.DataFrame(fixed_rows)
    validation = df_fixed[df_fixed['回测区间'] == VALIDATION_PERIOD_LABEL].copy()
    validation = validation.sort_values(
        ['夏普比率', '最大回撤', '年化收益'],
        ascending=[False, False, False]
    )
    selected = [0.30]
    for _, row in validation.head(2).iterrows():
        selected.append(float(row['基础股指仓位']))
    return sorted(set(selected))


def build_dynamic_configs(base_weights):
    configs = []
    for base_weight in base_weights:
        base_label = int(base_weight * 100)
        for vol_target in VOL_TARGETS:
            vol_label = int(vol_target * 100)
            configs.append(IndexModuleConfig(
                scenario_name=f"波动率调整_{base_label}%_目标{vol_label}%",
                experiment_group='波动率调整测试',
                mode='vol_scaled',
                base_weight=base_weight,
                vol_target=vol_target
            ))
        for risk_budget in RISK_BUDGETS:
            budget_label = int(risk_budget * 100)
            configs.append(IndexModuleConfig(
                scenario_name=f"风险预算_{base_label}%_预算{budget_label}%",
                experiment_group='风险预算测试',
                mode='risk_budget',
                base_weight=base_weight,
                risk_budget=risk_budget
            ))
        for limit in INDEX_CHANGE_LIMITS:
            limit_label = int(limit * 100)
            configs.append(IndexModuleConfig(
                scenario_name=f"固定仓位限速_{base_label}%_周变{limit_label}pct",
                experiment_group='信号限速测试',
                mode='fixed',
                base_weight=base_weight,
                max_weekly_index_change=limit
            ))
            configs.append(IndexModuleConfig(
                scenario_name=f"波动率调整限速_{base_label}%_目标15%_周变{limit_label}pct",
                experiment_group='信号限速测试',
                mode='vol_scaled',
                base_weight=base_weight,
                vol_target=0.15,
                max_weekly_index_change=limit
            ))
    return configs


def select_candidate_scenarios(raw_rows):
    df = pd.DataFrame(raw_rows)
    baseline = df[
        (df['方案名称'] == BASELINE_SCENARIO)
        & (df['回测区间'] == '全局 (Total)')
    ].iloc[0]
    validation = df[df['回测区间'] == VALIDATION_PERIOD_LABEL].copy()
    global_rows = df[df['回测区间'] == '全局 (Total)'].copy()
    global_lookup = global_rows.set_index('方案名称')

    candidate_names = []
    for _, row in validation.iterrows():
        name = row['方案名称']
        global_row = global_lookup.loc[name]
        if (
            global_row['夏普比率'] > baseline['夏普比率']
            and global_row['最大回撤'] >= baseline['最大回撤']
            and global_row['平均周换手'] <= baseline['平均周换手']
            and global_row['年化收益'] >= 0.108
        ):
            candidate_names.append(name)

    if len(candidate_names) == 0:
        ranked = validation.sort_values(
            ['夏普比率', '最大回撤', '年化收益', '平均周换手'],
            ascending=[False, False, False, True]
        )
        candidate_names = ranked['方案名称'].head(3).tolist()

    if BASELINE_SCENARIO not in candidate_names:
        candidate_names = [BASELINE_SCENARIO] + candidate_names
    return candidate_names


def build_candidate_annual_rows(results, candidate_names):
    rows = []
    for result in results:
        config = result['config']
        if config.scenario_name not in candidate_names:
            continue
        for label in result['period_labels']:
            if not label.endswith('年'):
                continue
            start_d, end_d = get_period_bounds(label, result['first_date'], result['end_date'])
            ret_slice = result['returns'].loc[start_d:end_d]
            margin_slice = result['margin'].loc[start_d:end_d]
            if len(ret_slice) < 5:
                continue
            metrics = calculate_metrics_numeric(ret_slice, margin_slice)
            rows.append({
                '方案名称': config.scenario_name,
                '实验分组': config.experiment_group,
                '回测区间': label,
                '期末净值': metrics['期末净值'],
                '年化收益': metrics['年化收益'],
                '年化波动': metrics['年化波动'],
                '夏普比率': metrics['夏普比率'],
                '最大回撤': metrics['最大回撤'],
                '月度胜率': metrics['月度胜率'],
                '平均资金占用': metrics['平均资金占用']
            })
    return rows


def build_candidate_weight_details(results, candidate_names):
    frames = []
    for result in results:
        config = result['config']
        if config.scenario_name not in candidate_names:
            continue
        frames.append(result['weights'])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def save_comparison_chart(results, candidate_names):
    chart_results = [
        result for result in results
        if result['config'].scenario_name in candidate_names
    ]
    if not chart_results:
        return

    fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=False)
    for result in chart_results:
        axes[0].plot(
            result['nav'].index,
            result['nav'].values,
            label=result['config'].scenario_name,
            lw=2 if result['config'].scenario_name == BASELINE_SCENARIO else 1.4
        )
    axes[0].set_title('股指模块候选方案累计净值', fontsize=14)
    axes[0].legend(loc='upper left')
    axes[0].grid(True, ls='--', alpha=0.4)

    val_rows = []
    for result in chart_results:
        start_d, end_d = get_period_bounds(VALIDATION_PERIOD_LABEL, result['first_date'], result['end_date'])
        metrics = calculate_metrics_numeric(result['returns'].loc[start_d:end_d], result['margin'].loc[start_d:end_d])
        val_rows.append((result['config'].scenario_name, metrics['夏普比率']))
    val_df = pd.DataFrame(val_rows, columns=['方案名称', '验证期夏普']).sort_values('验证期夏普')
    axes[1].barh(val_df['方案名称'], val_df['验证期夏普'], color='#4c78a8')
    axes[1].set_title('候选方案验证期夏普比率', fontsize=14)
    axes[1].xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    axes[1].grid(True, axis='x', ls='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(str(TEST_OUTPUT_DIR / f'股指模块参数测试对比_v{VERSION}.png'), dpi=300)
    plt.close(fig)


def main():
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"正在执行 v{VERSION} 股指模块参数测试...")
    data = load_data()

    results = []
    raw_rows = []
    fixed_rows = []
    fixed_configs = build_fixed_configs()

    print("\n[第一部分] 固定仓位比例测试")
    for config in fixed_configs:
        print(f"  - {config.scenario_name}")
        result = run_backtest(config, data)
        rows = build_scenario_rows(result)
        results.append(result)
        raw_rows.extend(rows)
        fixed_rows.extend(rows)

    dynamic_base_weights = choose_dynamic_base_weights(fixed_rows)
    print(f"\n动态模块测试使用基础仓位: {', '.join(format_percent(v) for v in dynamic_base_weights)}")

    print("\n[第二部分] 动态股指模块测试")
    for config in build_dynamic_configs(dynamic_base_weights):
        print(f"  - {config.scenario_name}")
        result = run_backtest(config, data)
        results.append(result)
        raw_rows.extend(build_scenario_rows(result))

    candidate_names = select_candidate_scenarios(raw_rows)
    print(f"\n候选方案: {', '.join(candidate_names)}")

    summary_filename = TEST_OUTPUT_DIR / f'股指模块参数测试结果_v{VERSION}.csv'
    annual_filename = TEST_OUTPUT_DIR / f'股指模块候选方案年度指标_v{VERSION}.csv'
    weights_filename = TEST_OUTPUT_DIR / f'股指模块候选方案周度仓位明细_v{VERSION}.csv'

    df_summary = format_metric_rows(raw_rows)
    df_summary.to_csv(str(summary_filename), index=False, encoding='utf-8-sig')

    df_annual = format_metric_rows(build_candidate_annual_rows(results, candidate_names))
    df_annual.to_csv(str(annual_filename), index=False, encoding='utf-8-sig')

    df_weights = build_candidate_weight_details(results, candidate_names)
    df_weights.to_csv(str(weights_filename), index=False, encoding='utf-8-sig')

    save_comparison_chart(results, candidate_names)

    print("\n股指模块参数测试输出已生成：")
    print(f"  1. {summary_filename}")
    print(f"  2. {annual_filename}")
    print(f"  3. {weights_filename}")
    print(f"  4. {TEST_OUTPUT_DIR / f'股指模块参数测试对比_v{VERSION}.png'}")

    baseline_row = pd.DataFrame(raw_rows)
    baseline_row = baseline_row[
        (baseline_row['方案名称'] == BASELINE_SCENARIO)
        & (baseline_row['回测区间'] == '全局 (Total)')
    ]
    if len(baseline_row) == 1:
        row = baseline_row.iloc[0]
        print(
            f"\n基准回归 v0.16: 年化收益={row['年化收益']:.2%}, "
            f"年化波动={row['年化波动']:.2%}, 夏普={row['夏普比率']:.2f}, "
            f"最大回撤={row['最大回撤']:.2%}"
        )


if __name__ == '__main__':
    main()
