import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

# ================= 配置参数 =================
VERSION = '0.10'
# 【修复1】修改为新的文件名
FILE_PATH_ETF = '../数据/原始数据/风险平价回测数据.xlsx'
FILE_PATH_MOM = '../买方宏观预期指标合成/预期动量/增长预期动量与通胀预期动量数据.csv'
SHEET_NAME = '日涨跌幅'
FEE_RATE = 0.0005
RISK_FREE_RATE = 0.0
EWMA_DECAY = 0.97

# 保证金比例设置 (期货按交易所标准，ETF为1.0)
MARGIN_RATIOS = {
    '沪深300主连': 0.15, '中证1000主连': 0.15, '红利低波ETF': 1.0,
    '10年国债主连': 0.03, '30年国债主连': 0.03,
    '有色ETF': 1.0, '能源化工ETF': 1.0, '豆粕主连': 0.10, '沪金主连': 0.10
}

ASSET_CLASSES = {
    '股票': ['沪深300主连', '中证1000主连', '红利低波ETF'],
    '债券': ['10年国债主连', '30年国债主连'],
    '商品': ['有色ETF', '能源化工ETF', '豆粕主连'],
    '黄金': ['沪金主连']
}

MACRO_QUADRANTS = {
    '增长超预期': ['沪深300主连', '中证1000主连', '有色ETF', '能源化工ETF', '豆粕主连'],
    '增长不及预期': ['10年国债主连', '30年国债主连', '沪金主连'],
    '通胀超预期': ['沪金主连', '豆粕主连', '有色ETF', '能源化工ETF'],
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
    ret_series = ret_series.fillna(0)
    nav = (1 + ret_series).cumprod()
    y = len(ret_series) / 252.0

    res = {
        '年化收益': f"{nav.iloc[-1] ** (1 / y) - 1:.2%}" if y > 0 else "0%",
        '年化波动': f"{ret_series.std() * np.sqrt(252):.2%}",
        '夏普比率': f"{(nav.iloc[-1] ** (1 / y) - 1) / (ret_series.std() * np.sqrt(252)):.2f}" if ret_series.std() > 0 else "0",
        '最大回撤': f"{((nav / nav.cummax()) - 1).min():.2%}",
        '月度胜率': f"{(ret_series.resample('ME').apply(lambda x: (1 + x).prod() - 1) > 0).mean():.2%}"
    }
    if margin_series is not None:
        res['平均资金占用'] = f"{margin_series.mean():.2%}"
    return res


# ================= 主流程 =================
def main():
    print(f"正在执行回测框架 v{VERSION} (期货+现货混合模式)...")

    df_raw = pd.read_excel(FILE_PATH_ETF, sheet_name=SHEET_NAME, index_col=0, parse_dates=True)
    df_etf = df_raw.dropna(how='all').fillna(0) / 100.0
    assets = df_etf.columns.tolist()

    # 【修复2】增加 .get(a, 1.0) 回退保护，防止字典中没有对应键值导致 KeyError
    m_ratios = np.array([MARGIN_RATIOS.get(a, 1.0) for a in assets])

    df_mom = pd.read_csv(FILE_PATH_MOM, index_col=0, parse_dates=True).resample('ME').last().ffill()
    month_ends = df_etf.resample('ME').last().index

    df_q = pd.DataFrame({n: df_etf[[a for a in al if a in assets]].mean(axis=1) for n, al in MACRO_QUADRANTS.items()})

    strats = ['风险平价策略', '全天候策略', '全天候增强策略 (动量择时)']
    ret_dfs = {s: pd.Series(0.0, index=df_etf.index) for s in strats}
    margin_dfs = {s: pd.Series(0.0, index=df_etf.index) for s in strats}
    weight_recs = {s: [] for s in strats}

    curr_ws = {s: np.zeros(len(assets)) for s in strats}
    first_date = None

    for i in range(len(month_ends) - 1):
        reb = month_ends[i]
        if reb < pd.to_datetime('2013-12-01') or reb not in df_mom.index or pd.isna(df_mom.loc[reb, '增长预期动量']):
            continue

        look = df_etf.loc[reb - pd.DateOffset(months=12):reb]
        look_q = df_q.loc[reb - pd.DateOffset(months=12):reb]
        if len(look) < 150: continue

        # 1. RP
        t_w_rp = get_risk_parity_weights(calculate_ewma_semi_cov(look, EWMA_DECAY))

        # 2. AW
        t_q_aw = get_risk_parity_weights(calculate_ewma_semi_cov(look_q, EWMA_DECAY))
        t_w_aw = pd.Series(0.0, index=assets)
        for idx, qn in enumerate(list(MACRO_QUADRANTS.keys())):
            qa = [a for a in MACRO_QUADRANTS[qn] if a in assets]
            t_w_aw[qa] += t_q_aw[idx] / len(qa)
        t_w_aw = t_w_aw.values

        # 3. AWE
        mq = [('增长超预期' if df_mom.loc[reb, '增长预期动量'] > 0 else '增长不及预期'),
              ('通胀超预期' if df_mom.loc[reb, '通胀预期动量'] > 0 else '通胀不及预期')]
        t_q_enh = get_risk_parity_weights(calculate_ewma_semi_cov(look_q[mq], EWMA_DECAY))
        t_w_enh = pd.Series(0.0, index=assets)
        for idx, qn in enumerate(mq):
            qa = [a for a in MACRO_QUADRANTS[qn] if a in assets]
            t_w_enh[qa] += t_q_enh[idx] / len(qa)
        t_w_enh = t_w_enh.values

        targets = {'风险平价策略': t_w_rp, '全天候策略': t_w_aw, '全天候增强策略 (动量择时)': t_w_enh}

        next_m = df_etf.loc[reb + pd.Timedelta(days=1):month_ends[i + 1]]
        if first_date is None and len(next_m) > 0: first_date = next_m.index[0]

        for date, dr in next_m.iterrows():
            for s in strats:
                if date == next_m.index[0]:
                    cost = np.sum(np.abs(targets[s] - curr_ws[s])) * FEE_RATE
                    ret_dfs[s].loc[date] = np.dot(targets[s], dr.values) - cost
                    curr_ws[s] = targets[s]
                    weight_recs[s].append({'date': reb, **dict(zip(assets, targets[s]))})
                else:
                    ret_dfs[s].loc[date] = np.dot(curr_ws[s], dr.values)

                # 计算资金占用: 权重 * 保证金比例
                margin_dfs[s].loc[date] = np.sum(curr_ws[s] * m_ratios)
                curr_ws[s] = (curr_ws[s] * (1 + dr.values)) / (np.sum(curr_ws[s] * (1 + dr.values)) or 1)

    if first_date is None: raise ValueError("日期或数据不满足条件")

    metrics = []
    for asset in assets:
        m = calculate_metrics(df_etf.loc[first_date:, asset])
        m['组合/资产'] = asset
        metrics.append(m)

    for s in strats:
        m = calculate_metrics(ret_dfs[s].loc[first_date:], margin_dfs[s].loc[first_date:])
        m['组合/资产'] = s
        metrics.append(m)

    df_m = pd.DataFrame(metrics).set_index('组合/资产')
    print(df_m.tail(3).to_markdown())
    df_m.to_csv(f'回测指标_v{VERSION}.csv', encoding='utf-8-sig')

    # 绘图
    fig, axes = plt.subplots(4, 1, figsize=(16, 20))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    axes[0].plot((1 + ret_dfs[strats[2]].loc[first_date:]).cumprod(), label=strats[2], color='red', lw=2)
    axes[0].plot((1 + ret_dfs[strats[1]].loc[first_date:]).cumprod(), label=strats[1], color='orange', lw=2)
    axes[0].plot((1 + ret_dfs[strats[0]].loc[first_date:]).cumprod(), label=strats[0], color='purple', lw=2)

    # 兼容处理底图参考线（如果表里没有这些列就不会画，不会报错）
    if '沪深300主连' in df_etf.columns:
        axes[0].plot((1 + df_etf.loc[first_date:, '沪深300主连']).cumprod(), label='沪深300主连', color='blue',
                     alpha=0.3)
    if '10年国债主连' in df_etf.columns:
        axes[0].plot((1 + df_etf.loc[first_date:, '10年国债主连']).cumprod(), label='10年国债主连', color='green',
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
    plt.savefig(f'回测图表_v{VERSION}.png', dpi=300)


if __name__ == '__main__':
    main()
