import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

# ================= 配置参数 =================
VERSION = '0.09'
FILE_PATH_ETF = 'ETF风险平价回测数据.xlsx'
FILE_PATH_MOM = r'C:\Users\tstone1\Documents\trae_projects\风险平价回测\华泰风险平价策略复现\买方宏观预期指标合成\预期动量\增长预期动量与通胀预期动量数据.csv'
SHEET_NAME = '日涨跌幅'
FEE_RATE = 0.0005
RISK_FREE_RATE = 0.0
EWMA_DECAY = 0.97

ASSET_CLASSES = {
    '股票': ['沪深300ETF', '中证1000ETF', '红利低波ETF'],
    '债券': ['10年国债ETF', '30年国债ETF'],
    '商品': ['有色ETF', '能源化工ETF', '豆粕ETF'],
    '黄金': ['黄金ETF']
}

MACRO_QUADRANTS = {
    '增长超预期': ['沪深300ETF', '中证1000ETF', '有色ETF', '能源化工ETF', '豆粕ETF'],
    '增长不及预期': ['10年国债ETF', '30年国债ETF', '黄金ETF'],
    '通胀超预期': ['黄金ETF', '豆粕ETF', '有色ETF', '能源化工ETF'],
    '通胀不及预期': ['10年国债ETF', '30年国债ETF', '黄金ETF', '红利低波ETF']
}

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ================= 核心模型函数 =================
def calculate_ewma_semi_cov(returns_df, decay=0.97):
    downside_returns = np.minimum(returns_df.values, 0.0)
    T, N = downside_returns.shape
    weights = decay ** np.arange(T - 1, -1, -1)
    weights /= np.sum(weights)
    weighted_downside = downside_returns * np.sqrt(weights[:, np.newaxis])
    semi_cov_matrix = np.dot(weighted_downside.T, weighted_downside) * 252
    semi_cov_matrix += np.eye(N) * 1e-8
    return semi_cov_matrix


def risk_parity_convex_objective(x, cov_matrix):
    n = len(x)
    port_var = 0.5 * np.dot(x.T, np.dot(cov_matrix, x))
    log_term = np.sum(np.log(x)) / n
    return port_var - log_term


def risk_parity_convex_jacobian(x, cov_matrix):
    n = len(x)
    return np.dot(cov_matrix, x) - 1.0 / (n * x)


def get_risk_parity_weights(cov_matrix):
    n_assets = cov_matrix.shape[0]
    if n_assets == 0: return np.array([])
    if n_assets == 1: return np.array([1.0])

    x0 = np.ones(n_assets)
    bounds = [(1e-8, None)] * n_assets
    result = minimize(risk_parity_convex_objective, x0, args=(cov_matrix,),
                      method='L-BFGS-B', jac=risk_parity_convex_jacobian,
                      bounds=bounds, options={'ftol': 1e-12, 'maxiter': 1000})
    return result.x / np.sum(result.x)


def calculate_metrics(returns_series):
    returns_series = returns_series.fillna(0)
    cum_nav = (1 + returns_series).cumprod()
    total_years = len(returns_series) / 252.0
    annualized_return = cum_nav.iloc[-1] ** (1 / total_years) - 1 if total_years > 0 else 0.0
    annualized_volatility = returns_series.std() * np.sqrt(252)
    sharpe_ratio = (annualized_return - RISK_FREE_RATE) / annualized_volatility if annualized_volatility > 0 else 0.0
    max_drawdown = ((cum_nav / cum_nav.cummax()) - 1).min()
    monthly_returns = returns_series.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    win_rate = (monthly_returns > 0).sum() / len(monthly_returns) if len(monthly_returns) > 0 else 0

    return {
        '年化收益': f"{annualized_return:.2%}",
        '年化波动': f"{annualized_volatility:.2%}",
        '夏普比率': f"{sharpe_ratio:.2f}",
        '最大回撤': f"{max_drawdown:.2%}",
        '月度胜率': f"{win_rate:.2%}"
    }


# ================= 主流程 =================
def main():
    print(f"正在执行回测框架 (v{VERSION})...")

    df_etf = pd.read_excel(FILE_PATH_ETF, sheet_name=SHEET_NAME, index_col=0, parse_dates=True)
    df_etf = df_etf.dropna(how='all').fillna(0) / 100.0
    assets = df_etf.columns.tolist()

    df_mom = pd.read_csv(FILE_PATH_MOM, index_col=0, parse_dates=True)
    df_mom_monthly = df_mom.resample('ME').last().ffill()
    month_ends = df_etf.resample('ME').last().index

    df_quadrants = pd.DataFrame(index=df_etf.index)
    for q_name, q_assets in MACRO_QUADRANTS.items():
        valid_assets = [a for a in q_assets if a in assets]
        df_quadrants[q_name] = df_etf[valid_assets].mean(axis=1)
    quadrant_names = list(MACRO_QUADRANTS.keys())

    ret_rp = pd.Series(0.0, index=df_etf.index)
    ret_aw = pd.Series(0.0, index=df_etf.index)
    ret_enh = pd.Series(0.0, index=df_etf.index)

    rec_rp = []
    rec_aw = []
    rec_enh = []

    w_rp = np.zeros(len(assets))
    w_aw = np.zeros(len(assets))
    w_enh = np.zeros(len(assets))
    first_trade_date = None

    for i in range(len(month_ends) - 1):
        reb_date = month_ends[i]

        if reb_date < pd.to_datetime('2013-12-01') or reb_date not in df_mom_monthly.index or pd.isna(
                df_mom_monthly.loc[reb_date, '增长预期动量']):
            continue

        lookback = df_etf.loc[reb_date - pd.DateOffset(months=12):reb_date]
        lookback_q = df_quadrants.loc[reb_date - pd.DateOffset(months=12):reb_date]

        if len(lookback) < 150:
            continue

        # 策略1：风险平价
        target_rp = get_risk_parity_weights(calculate_ewma_semi_cov(lookback, EWMA_DECAY))
        rec_rp.append({'date': reb_date, **dict(zip(assets, target_rp))})

        # 策略2：全天候
        target_q_aw = get_risk_parity_weights(calculate_ewma_semi_cov(lookback_q, EWMA_DECAY))
        target_aw_s = pd.Series(0.0, index=assets)
        for idx, q_name in enumerate(quadrant_names):
            q_assets = [a for a in MACRO_QUADRANTS[q_name] if a in assets]
            target_aw_s[q_assets] += target_q_aw[idx] / len(q_assets)
        target_aw = target_aw_s.values
        rec_aw.append({'date': reb_date, **dict(zip(assets, target_aw))})

        # 策略3：全天候增强 (动量择时)
        g_mom = df_mom_monthly.loc[reb_date, '增长预期动量']
        i_mom = df_mom_monthly.loc[reb_date, '通胀预期动量']
        active_quads = [
            '增长超预期' if g_mom > 0 else '增长不及预期',
            '通胀超预期' if i_mom > 0 else '通胀不及预期'
        ]
        target_q_enh = get_risk_parity_weights(calculate_ewma_semi_cov(lookback_q[active_quads], EWMA_DECAY))

        target_enh_s = pd.Series(0.0, index=assets)
        for idx, q_name in enumerate(active_quads):
            q_assets = [a for a in MACRO_QUADRANTS[q_name] if a in assets]
            target_enh_s[q_assets] += target_q_enh[idx] / len(q_assets)
        target_enh = target_enh_s.values
        rec_enh.append({'date': reb_date, **dict(zip(assets, target_enh))})

        # 调仓及收益计算
        next_m_data = df_etf.loc[reb_date + pd.Timedelta(days=1):month_ends[i + 1]]
        if first_trade_date is None and len(next_m_data) > 0:
            first_trade_date = next_m_data.index[0]

        for date, d_ret in next_m_data.iterrows():
            if date == next_m_data.index[0]:
                ret_rp.loc[date] = np.dot(target_rp, d_ret.values) - np.sum(np.abs(target_rp - w_rp)) * FEE_RATE
                ret_aw.loc[date] = np.dot(target_aw, d_ret.values) - np.sum(np.abs(target_aw - w_aw)) * FEE_RATE
                ret_enh.loc[date] = np.dot(target_enh, d_ret.values) - np.sum(np.abs(target_enh - w_enh)) * FEE_RATE
                w_rp, w_aw, w_enh = target_rp, target_aw, target_enh
            else:
                ret_rp.loc[date] = np.dot(w_rp, d_ret.values)
                ret_aw.loc[date] = np.dot(w_aw, d_ret.values)
                ret_enh.loc[date] = np.dot(w_enh, d_ret.values)

            w_rp = (w_rp * (1 + d_ret.values)) / (np.sum(w_rp * (1 + d_ret.values)) or 1)
            w_aw = (w_aw * (1 + d_ret.values)) / (np.sum(w_aw * (1 + d_ret.values)) or 1)
            w_enh = (w_enh * (1 + d_ret.values)) / (np.sum(w_enh * (1 + d_ret.values)) or 1)

    if first_trade_date is None:
        raise ValueError("日期或数据不满足回测条件")

    ret_rp = ret_rp.loc[first_trade_date:]
    ret_aw = ret_aw.loc[first_trade_date:]
    ret_enh = ret_enh.loc[first_trade_date:]
    df_eval = df_etf.loc[first_trade_date:]

    df_w_rp = pd.DataFrame(rec_rp).set_index('date')
    df_w_aw = pd.DataFrame(rec_aw).set_index('date')
    df_w_enh = pd.DataFrame(rec_enh).set_index('date')

    # 指标输出
    metrics = []
    for asset in assets:
        m = calculate_metrics(df_eval[asset])
        m['组合/资产'] = asset
        metrics.append(m)

    strats = {
        '风险平价策略': ret_rp,
        '全天候策略': ret_aw,
        '全天候增强策略 (动量择时)': ret_enh
    }

    for name, ret in strats.items():
        m = calculate_metrics(ret)
        m['组合/资产'] = name
        metrics.append(m)

    df_metrics = pd.DataFrame(metrics).set_index('组合/资产')
    print(df_metrics.tail(3).to_markdown())

    csv_file = f'回测指标_v{VERSION}.csv'
    df_metrics.to_csv(csv_file, encoding='utf-8-sig')
    print(f"数据已保存至 {csv_file}")

    # 绘图模块
    fig, axes = plt.subplots(4, 1, figsize=(16, 20))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    axes[0].plot((1 + ret_enh).cumprod(), label='全天候增强策略 (动量择时)', color='red', linewidth=2)
    axes[0].plot((1 + ret_aw).cumprod(), label='全天候策略', color='darkorange', linewidth=2)
    axes[0].plot((1 + ret_rp).cumprod(), label='风险平价策略', color='purple', linewidth=2)
    axes[0].plot((1 + df_eval['沪深300ETF']).cumprod(), label='沪深300ETF', color='blue', alpha=0.4)
    axes[0].plot((1 + df_eval['10年国债ETF']).cumprod(), label='10年国债ETF', color='green', alpha=0.4)
    axes[0].set_title('策略累计净值走势', fontsize=14)
    axes[0].legend(loc='upper left')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    def plot_class_weights(ax, df_w, title):
        df_class = pd.DataFrame(index=df_w.index)
        for c_name, c_assets in ASSET_CLASSES.items():
            valid = [a for a in c_assets if a in assets]
            if valid: df_class[c_name] = df_w[valid].sum(axis=1)
        ax.stackplot(df_class.index, df_class.T, labels=df_class.columns, alpha=0.8, colors=colors)
        ax.set_title(title, fontsize=14)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.6)

    plot_class_weights(axes[1], df_w_rp, '风险平价策略 - 大类资产权重')
    plot_class_weights(axes[2], df_w_aw, '全天候策略 - 大类资产权重')
    plot_class_weights(axes[3], df_w_enh, '全天候增强策略 (动量择时) - 大类资产权重')

    plt.tight_layout()
    img_file = f'回测图表_v{VERSION}.png'
    plt.savefig(img_file, dpi=300)
    print(f"图表已保存至 {img_file}")


if __name__ == '__main__':
    main()