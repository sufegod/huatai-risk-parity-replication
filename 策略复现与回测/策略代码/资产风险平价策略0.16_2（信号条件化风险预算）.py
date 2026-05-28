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
VERSION = '0.16_2'
STRATEGY_NAME = '风险平价策略'

FILE_PATH_WEIGHT_RETURNS = PROJECT_DIR / '数据' / 'JYDB期货数据替换' / 'JYDB主力日涨跌幅_填充.csv'
FILE_PATH_TRADE_RETURNS = PROJECT_DIR / '数据' / 'JYDB期货数据替换' / 'JYDB主力日涨跌幅_未填充.csv'
FILE_PATH_INDEX_SIGNAL = PROJECT_DIR / '数据' / '原始数据' / '股指期货信号.xlsx'
METRICS_DIR = BACKTEST_DIR / '回测指标'
CHART_DIR = BACKTEST_DIR / '回测图表'

try:
    pd.Series([1], index=pd.to_datetime(['2000-01-01'])).resample('ME').sum()
    MONTH_END_FREQ = 'ME'
except ValueError:
    MONTH_END_FREQ = 'M'
WEEKLY_REBALANCE_FREQ = 'W-FRI'
FEE_RATE = 0.0005
REPO_FEE_RATE = 0.000001
EWMA_DECAY = 0.97
INDEX_FUTURES = ['沪深300主连', '中证1000主连']
INDEX_NEGATIVE_SIGNAL_MULTIPLIER = 0.50
INDEX_POSITIVE_SIGNAL_MAX_MULTIPLIER = 2.00

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


# ================= 核心模型函数 =================
def calculate_ewma_semi_cov(returns_df, decay=0.97):
    downside = np.minimum(returns_df.values, 0.0)
    T, N = downside.shape
    w = (decay ** np.arange(T - 1, -1, -1))
    w /= np.sum(w)
    weighted = downside * np.sqrt(w[:, np.newaxis])
    return np.dot(weighted.T, weighted) * 252 + np.eye(N) * 1e-8


def risk_budget_convex_objective(x, cov_matrix, risk_budget):
    return 0.5 * np.dot(x.T, np.dot(cov_matrix, x)) - np.dot(risk_budget, np.log(x))


def risk_budget_convex_jacobian(x, cov_matrix, risk_budget):
    return np.dot(cov_matrix, x) - risk_budget / x


def get_risk_parity_weights(cov_matrix, risk_budget=None):
    n = cov_matrix.shape[0]
    if risk_budget is None:
        risk_budget = np.ones(n) / n
    else:
        risk_budget = np.asarray(risk_budget, dtype=float)
        if len(risk_budget) != n:
            raise ValueError("风险预算长度必须与协方差矩阵维度一致")
        if np.any(risk_budget <= 0):
            raise ValueError("风险预算必须为正数")
        risk_budget = risk_budget / risk_budget.sum()

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


def get_weekly_observation_dates(index):
    observations = []
    date_series = pd.Series(index=index, data=index)
    for _, group in date_series.groupby(pd.Grouper(freq=WEEKLY_REBALANCE_FREQ)):
        if len(group) > 0:
            observations.append(group.index[-1])
    return pd.DatetimeIndex(observations)


def get_index_budget_multiplier(signal):
    if pd.isna(signal):
        raise ValueError("股指期货信号缺失，无法生成风险预算")
    if signal <= 0:
        return INDEX_NEGATIVE_SIGNAL_MULTIPLIER
    if signal <= 1:
        return 1.0 + float(signal)
    return INDEX_POSITIVE_SIGNAL_MAX_MULTIPLIER


def build_signal_conditioned_risk_budget(eligible_assets, signal):
    raw_budget = pd.Series(1.0, index=eligible_assets, dtype=float)
    multiplier = get_index_budget_multiplier(signal)
    listed_index_futures = [asset for asset in INDEX_FUTURES if asset in raw_budget.index]
    raw_budget.loc[listed_index_futures] *= multiplier
    risk_budget = raw_budget / raw_budget.sum()
    return risk_budget, multiplier


def calculate_risk_contributions(weights, cov_matrix):
    w = weights.values
    marginal_risk = np.dot(cov_matrix, w)
    portfolio_var = np.dot(w.T, marginal_risk)
    if portfolio_var <= 0:
        return pd.Series(0.0, index=weights.index)
    return pd.Series(w * marginal_risk / portfolio_var, index=weights.index)


# ================= 主流程 =================
def main():
    print(f"正在执行回测框架 v{VERSION}...")
    METRICS_DIR.mkdir(exist_ok=True)
    CHART_DIR.mkdir(exist_ok=True)

    df_weight_raw = load_returns_csv(FILE_PATH_WEIGHT_RETURNS)
    df_trade_raw = load_returns_csv(FILE_PATH_TRADE_RETURNS)
    index_signal = load_index_signal(FILE_PATH_INDEX_SIGNAL)

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
    risk_parity_assets = assets

    df_weight = df_weight_all[assets].fillna(0)
    df_trade = df_trade_all[assets]
    listing_dates = {
        asset: df_trade_all_raw[asset].first_valid_index()
        for asset in assets
    }

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

        cov_matrix = calculate_ewma_semi_cov(look, EWMA_DECAY)
        risk_budget, index_budget_multiplier = build_signal_conditioned_risk_budget(eligible_rp_assets, raw_signal)
        rp_active = get_risk_parity_weights(cov_matrix, risk_budget.values)
        target = pd.Series(0.0, index=assets)
        target.loc[eligible_rp_assets] = rp_active

        active_weights = pd.Series(rp_active, index=eligible_rp_assets)
        risk_contrib = calculate_risk_contributions(active_weights, cov_matrix)
        listed_index_futures = [asset for asset in INDEX_FUTURES if asset in eligible_rp_assets]
        index_weight = float(target.loc[listed_index_futures].sum()) if listed_index_futures else 0.0
        index_target_budget = float(risk_budget.loc[listed_index_futures].sum()) if listed_index_futures else 0.0
        index_actual_contribution = float(risk_contrib.loc[listed_index_futures].sum()) if listed_index_futures else 0.0

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
                    '策略名称': STRATEGY_NAME,
                    '股指期货信号': float(raw_signal),
                    '股指风险预算倍率': index_budget_multiplier,
                    '股指目标风险预算': index_target_budget,
                    '股指实际风险贡献': index_actual_contribution,
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
        raise ValueError("日期或数据不满足条件")

    print("正在生成每日净值数据...")
    df_navs = pd.DataFrame(index=df_trade.loc[first_date:].index)
    df_navs[STRATEGY_NAME] = (1 + ret_series.loc[first_date:]).cumprod()
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

        m = calculate_metrics(ret_series.loc[start_d:end_d], margin_series.loc[start_d:end_d])
        m['回测区间'] = period_label
        m['组合/资产'] = STRATEGY_NAME
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
    print(df_m_all[(df_m_all['回测区间'] == '全局 (Total)') & (df_m_all['组合/资产'] == STRATEGY_NAME)].set_index(
        '组合/资产').to_string())

    print("\n正在生成周度仓位明细...")
    df_weights_all = pd.DataFrame(weight_recs)
    weight_cols = [
        'date', '策略名称', '股指期货信号', '股指风险预算倍率',
        '股指目标风险预算', '股指实际风险贡献', '股指期货仓位'
    ] + assets
    df_weights_all = df_weights_all[weight_cols]
    weights_filename = METRICS_DIR / f'策略周度仓位明细_v{VERSION}.csv'
    df_weights_all.to_csv(str(weights_filename), index=False, encoding='utf-8-sig')

    print(f"\n数据文件已生成：\n 1. {navs_filename}\n 2. {metrics_filename}\n 3. {weights_filename}")

    fig, axes = plt.subplots(3, 1, figsize=(16, 16), sharex=False)

    axes[0].plot(df_navs.index, df_navs[STRATEGY_NAME], label=STRATEGY_NAME, color='purple', lw=2)
    if '沪深300主连' in df_trade.columns:
        axes[0].plot((1 + df_trade.loc[first_date:, '沪深300主连']).cumprod(), label='沪深300主连', color='blue',
                     alpha=0.3)
    if '10年国债主连' in df_trade.columns:
        axes[0].plot((1 + df_trade.loc[first_date:, '10年国债主连']).cumprod(), label='10年国债主连', color='green',
                     alpha=0.3)
    axes[0].set_title('策略累计净值走势', fontsize=14)
    axes[0].legend(loc='upper left')
    axes[0].grid(True, ls='--', alpha=0.5)

    df_w = df_weights_all.set_index('date')
    df_c = pd.DataFrame({
        cn: df_w[[a for a in al if a in assets]].sum(axis=1)
        for cn, al in PLOT_ASSET_CLASSES.items()
    })
    axes[1].stackplot(
        df_c.index,
        df_c.T,
        labels=df_c.columns,
        alpha=0.8,
        colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    )
    axes[1].set_title('周度大类资产权重', fontsize=14)
    axes[1].set_ylim(0, 1)
    axes[1].yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    axes[1].legend(loc='upper left')
    axes[1].grid(True, ls='--', alpha=0.4)

    axes[2].plot(df_w.index, df_w['股指期货信号'], label='股指期货信号', color='black', lw=1.5)
    axes[2].plot(df_w.index, df_w['股指期货仓位'], label='股指期货仓位', color='blue', lw=1.5)
    axes[2].set_title('股指期货信号与仓位', fontsize=14)
    axes[2].set_ylim(-1.1, 1.1)
    axes[2].yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    axes[2].legend(loc='upper left')
    axes[2].grid(True, ls='--', alpha=0.4)

    plt.tight_layout()
    chart_filename = CHART_DIR / f'回测图表_v{VERSION}.png'
    plt.savefig(str(chart_filename), dpi=300)
    print(f" 4. {chart_filename}")


if __name__ == '__main__':
    main()
