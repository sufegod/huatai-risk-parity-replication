import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

# ================= 配置参数 =================
VERSION = '0.06'  # 更新为尾部风险管理版本 0.06
FILE_PATH = '../数据/原始数据/ETF风险平价回测数据.xlsx'
SHEET_NAME = '日涨跌幅'
FEE_RATE = 0.0005  # 单边万分之五交易费率
RISK_FREE_RATE = 0.0  # 计算夏普比率时的无风险利率
EWMA_DECAY = 0.97  # EWMA半协方差衰减参数

# 资产分类映射
ASSET_CLASSES = {
    '股票': ['沪深300ETF', '中证1000ETF', '红利低波ETF'],
    '债券': ['10年国债ETF', '30年国债ETF'],
    '商品': ['有色ETF', '能源化工ETF', '豆粕ETF'],
    '黄金': ['黄金ETF']
}

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ================= 风险输入升级：EWMA半协方差计算 =================
def calculate_ewma_semi_cov(returns_df, decay=0.97):
    """
    计算 EWMA 半协方差矩阵 (只考虑下行风险，且赋予近期更高权重)
    """
    # 1. 提取下行收益率 (仅计算收益率 < 0 的部分，涨幅部分全部设为0)
    downside_returns = np.minimum(returns_df.values, 0.0)

    T, N = downside_returns.shape

    # 2. 生成 EWMA 权重序列
    # Pandas 中的数据按照时间顺序排列（最后一行是最新的）
    # 生成的指数序列为 [decay^(T-1), decay^(T-2), ... , decay^0]
    weights = decay ** np.arange(T - 1, -1, -1)
    weights /= np.sum(weights)  # 归一化使得权重之和为1

    # 3. 将权重应用到下行收益率矩阵中
    # 等效于 X^T * W * X (其中 W 是权重的对角矩阵)
    weighted_downside = downside_returns * np.sqrt(weights[:, np.newaxis])

    # 4. 计算加权半协方差并年化
    semi_cov_matrix = np.dot(weighted_downside.T, weighted_downside) * 252

    # 【数学安全锁】半协方差矩阵极易由于部分资产从未下跌而降秩
    # 这里在对角线上加上极小的正数（正则化），确保矩阵严格正定，防止后续的 Spinu 凸优化报错
    semi_cov_matrix += np.eye(N) * 1e-8

    return semi_cov_matrix


# ================= 风险平价优化函数 (Spinu 凸优化) =================
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
    x0 = np.ones(n_assets)
    bounds = [(1e-8, None)] * n_assets

    result = minimize(risk_parity_convex_objective,
                      x0,
                      args=(cov_matrix,),
                      method='L-BFGS-B',
                      jac=risk_parity_convex_jacobian,
                      bounds=bounds,
                      options={'ftol': 1e-12, 'maxiter': 1000})

    x_opt = result.x
    w_opt = x_opt / np.sum(x_opt)
    return w_opt


# ================= 指标计算函数 =================
def calculate_metrics(returns_series):
    returns_series = returns_series.fillna(0)
    cum_nav = (1 + returns_series).cumprod()

    total_years = len(returns_series) / 252.0
    if total_years > 0:
        annualized_return = cum_nav.iloc[-1] ** (1 / total_years) - 1
    else:
        annualized_return = 0.0

    annualized_volatility = returns_series.std() * np.sqrt(252)

    if annualized_volatility > 0:
        sharpe_ratio = (annualized_return - RISK_FREE_RATE) / annualized_volatility
    else:
        sharpe_ratio = 0.0

    running_max = cum_nav.cummax()
    drawdown = (cum_nav / running_max) - 1
    max_drawdown = drawdown.min()

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
    print(f"正在加载数据 (v{VERSION})...")
    df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME, index_col=0, parse_dates=True)
    df = df.dropna(how='all').fillna(0)

    # 还原百分比数据为真实小数
    df = df / 100.0

    assets = df.columns.tolist()
    month_ends = df.resample('ME').last().index
    strategy_returns = pd.Series(0.0, index=df.index, name='策略')
    weight_records = []

    print("执行严密的凸优化计算（搭载 EWMA 半协方差输入）...")
    current_weights = np.zeros(len(assets))
    first_trade_date = None

    for i in range(len(month_ends) - 1):
        rebalance_date = month_ends[i]

        # 确保第一次调仓在2013年12月末及以后
        if rebalance_date < pd.to_datetime('2013-12-01'):
            continue

        lookback_start = rebalance_date - pd.DateOffset(months=12)
        historical_data = df.loc[lookback_start:rebalance_date]

        if len(historical_data) < 150:
            continue

        # 【核心修改点】采用引入近期衰减权重的半下行风险矩阵
        cov_matrix = calculate_ewma_semi_cov(historical_data, decay=EWMA_DECAY)
        target_weights = get_risk_parity_weights(cov_matrix)

        weight_records.append({
            'date': rebalance_date,
            **dict(zip(assets, target_weights))
        })

        next_month_start = rebalance_date + pd.Timedelta(days=1)
        next_month_end = month_ends[i + 1]
        next_month_data = df.loc[next_month_start:next_month_end]

        if first_trade_date is None and len(next_month_data) > 0:
            first_trade_date = next_month_data.index[0]

        for date, daily_ret in next_month_data.iterrows():
            if date == next_month_data.index[0]:
                turnover = np.sum(np.abs(target_weights - current_weights))
                fee = turnover * FEE_RATE
                day_ret = np.dot(target_weights, daily_ret.values) - fee
                current_weights = target_weights
            else:
                day_ret = np.dot(current_weights, daily_ret.values)

            strategy_returns.loc[date] = day_ret
            # 权重随每日资产涨跌幅产生漂移
            current_weights = current_weights * (1 + daily_ret.values)
            weight_sum = np.sum(current_weights)
            if weight_sum > 0:
                current_weights /= weight_sum

    if first_trade_date is None:
        raise ValueError("数据不足以生成调仓记录！")

    strategy_returns = strategy_returns.loc[first_trade_date:]
    df_eval = df.loc[first_trade_date:]
    df_weights = pd.DataFrame(weight_records).set_index('date')

    print("\n[回测结果]")
    metrics_list = []
    for asset in assets:
        metrics = calculate_metrics(df_eval[asset])
        metrics['资产'] = asset
        metrics_list.append(metrics)

    strat_metrics = calculate_metrics(strategy_returns)
    strat_metrics['资产'] = '策略组合'
    metrics_list.append(strat_metrics)

    df_metrics = pd.DataFrame(metrics_list)
    cols = ['资产'] + [c for c in df_metrics.columns if c != '资产']
    df_metrics = df_metrics[cols].set_index('资产')

    print(df_metrics.to_markdown())

    csv_filename = f'回测指标_v{VERSION}.csv'
    df_metrics.to_csv(csv_filename, encoding='utf-8-sig')
    print(f"\n指标已存至: {csv_filename}")

    # ================= 绘图 =================
    fig = plt.figure(figsize=(16, 12))

    ax1 = plt.subplot(2, 1, 1)
    nav_strategy = (1 + strategy_returns).cumprod()
    nav_hs300 = (1 + df_eval['沪深300ETF']).cumprod()
    nav_10ybond = (1 + df_eval['10年国债ETF']).cumprod()

    ax1.plot(nav_strategy.index, nav_strategy, label='策略组合', color='red', linewidth=2)
    ax1.plot(nav_hs300.index, nav_hs300, label='沪深300ETF', color='blue', alpha=0.7)
    ax1.plot(nav_10ybond.index, nav_10ybond, label='10年国债ETF', color='green', alpha=0.7)
    ax1.set_title('累计净值走势', fontsize=15)
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = plt.subplot(2, 1, 2)
    class_weights = pd.DataFrame(index=df_weights.index)
    for class_name, class_assets in ASSET_CLASSES.items():
        valid_assets = [a for a in class_assets if a in assets]
        if valid_assets:
            class_weights[class_name] = df_weights[valid_assets].sum(axis=1)

    ax2.stackplot(class_weights.index, class_weights.T,
                  labels=class_weights.columns, alpha=0.8,
                  colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

    ax2.set_title('大类资产动态权重', fontsize=15)
    ax2.set_ylim(0, 1)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    img_filename = f'回测图表_v{VERSION}.png'
    plt.savefig(img_filename, dpi=300)
    print(f"图表已存至: {img_filename}")


if __name__ == '__main__':
    main()