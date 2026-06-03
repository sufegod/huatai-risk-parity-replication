import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parents[1]
BACKTEST_DIR = PROJECT_DIR / "策略复现与回测"
BASE_DIR = BACKTEST_DIR / "策略测试代码"
MPLCONFIG_DIR = BASE_DIR / ".matplotlib"
MPLCONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings


warnings.filterwarnings("ignore")

# ================= 配置参数 =================
STRATEGY_NAME = "风险平价策略"

FILE_PATH_WEIGHT_RETURNS = PROJECT_DIR / "数据" / "日度收益数据更新" / "日涨跌幅_填充.csv"
FILE_PATH_TRADE_RETURNS = PROJECT_DIR / "数据" / "日度收益数据更新" / "日涨跌幅_未填充.csv"
FILE_PATH_INDEX_SIGNAL = PROJECT_DIR / "数据" / "原始数据" / "股指期货信号.xlsx"
METRICS_DIR = BACKTEST_DIR / "回测指标"
NAV_DIR = METRICS_DIR / "净值"
PERFORMANCE_DIR = METRICS_DIR / "指标"
WEIGHTS_DIR = METRICS_DIR / "仓位明细"
COMPARISON_DIR = METRICS_DIR / "对比结果"
CHART_DIR = BACKTEST_DIR / "回测图表"
COMPARE_DOC_DIR = BACKTEST_DIR / "回测结果对比说明"

MONTH_END_FREQ = "M"
WEEKLY_REBALANCE_FREQ = "W-FRI"
FEE_RATE = 0.0005
REPO_FEE_RATE = 0.000001
EWMA_DECAY = 0.97
INDEX_BASE_WEIGHT = 0.30
INDEX_FUTURES = ["沪深300主连", "中证1000主连"]

MARGIN_RATIOS = {
    "沪深300主连": 0.15,
    "中证1000主连": 0.15,
    "红利低波ETF": 1.0,
    "10年国债主连": 0.03,
    "30年国债主连": 0.03,
    "沪铜主连": 0.10,
    "沪铝主连": 0.10,
    "PTA主连": 0.10,
    "原油主连": 0.10,
    "豆粕主连": 0.10,
    "沪金主连": 0.10,
}

RISK_PARITY_ASSET_CLASSES = {
    "股票": ["红利低波ETF"],
    "债券": ["10年国债主连", "30年国债主连"],
    "商品": ["沪铜主连", "沪铝主连", "PTA主连", "原油主连", "豆粕主连"],
    "黄金": ["沪金主连"],
}

PLOT_ASSET_CLASSES = {
    "股指期货": INDEX_FUTURES,
    **RISK_PARITY_ASSET_CLASSES,
}

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def calculate_ewma_semi_cov(returns_df, decay=EWMA_DECAY):
    downside = np.minimum(returns_df.values, 0.0)
    t_count, asset_count = downside.shape
    weights = decay ** np.arange(t_count - 1, -1, -1)
    weights /= np.sum(weights)
    weighted = downside * np.sqrt(weights[:, np.newaxis])
    return np.dot(weighted.T, weighted) * 252 + np.eye(asset_count) * 1e-8


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
        method="L-BFGS-B",
        jac=risk_parity_convex_jacobian,
        bounds=[(1e-8, None)] * n,
        options={"ftol": 1e-12},
    )
    return res.x / np.sum(res.x)


def calculate_metrics(ret_series, margin_series=None):
    if len(ret_series) < 5:
        result = {k: "0.00%" for k in ["年化收益", "年化波动", "夏普比率", "最大回撤", "月度胜率"]}
        if margin_series is not None:
            result["平均资金占用"] = "0.00%"
        return result

    ret_series = ret_series.fillna(0)
    nav = (1 + ret_series).cumprod()
    years = len(ret_series) / 252.0

    ann_ret = nav.iloc[-1] ** (1 / years) - 1 if years > 0 else 0.0
    ann_vol = ret_series.std() * np.sqrt(252)
    sharpe = (ret_series.mean() * 252) / ann_vol if ann_vol > 0 else 0.0
    max_dd = ((nav / nav.cummax()) - 1).min()
    monthly_ret = ret_series.resample(MONTH_END_FREQ).apply(lambda x: (1 + x).prod() - 1)
    win_rate = (monthly_ret > 0).sum() / len(monthly_ret) if len(monthly_ret) > 0 else 0.0

    result = {
        "年化收益": f"{ann_ret:.2%}",
        "年化波动": f"{ann_vol:.2%}",
        "夏普比率": f"{sharpe:.2f}",
        "最大回撤": f"{max_dd:.2%}",
        "月度胜率": f"{win_rate:.2%}",
    }
    if margin_series is not None:
        result["平均资金占用"] = f"{margin_series.mean():.2%}"
    return result


def load_returns_csv(file_path):
    with file_path.open("r", encoding="utf-8-sig") as returns_file:
        df = pd.read_csv(returns_file, index_col=0, parse_dates=True)
    return df.dropna(how="all")


def load_index_signal(file_path):
    raw = pd.read_excel(file_path, sheet_name=0, header=None)
    signal_col = None
    for col in raw.columns:
        values = raw[col].astype(str).str.strip()
        if (values == "股指期货").any():
            signal_col = col
            break
    if signal_col is None:
        signal_col = 1

    df = raw[[0, signal_col]].copy()
    df.columns = ["date", "signal"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["signal"] = pd.to_numeric(df["signal"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    first_valid = df["signal"].first_valid_index()
    if first_valid is None:
        raise ValueError("股指期货信号文件没有有效信号")
    return df.loc[first_valid:, "signal"].ffill()


def get_observation_dates(index, rebalance_mode):
    if rebalance_mode == "daily":
        return pd.DatetimeIndex(index)
    if rebalance_mode != "weekly":
        raise ValueError("rebalance_mode must be 'weekly' or 'daily'")

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
        asset
        for asset in INDEX_FUTURES
        if asset in assets and listing_dates.get(asset) is not None and listing_dates[asset] <= rebalance_date
    ]
    if not listed_index_futures:
        return target

    target.loc[listed_index_futures] = total_weight / len(listed_index_futures)
    return target


def parse_percent(value):
    if isinstance(value, str):
        return float(value.replace("%", "")) / 100.0
    return float(value)


def parse_float(value):
    if isinstance(value, str):
        return float(value.replace("%", "")) / 100.0 if value.endswith("%") else float(value)
    return float(value)


def evaluate_optimization(weekly_row, daily_row):
    weekly_nav = parse_float(weekly_row["期末净值"])
    daily_nav = parse_float(daily_row["期末净值"])
    weekly_sharpe = parse_float(weekly_row["夏普比率"])
    daily_sharpe = parse_float(daily_row["夏普比率"])
    weekly_drawdown = parse_percent(weekly_row["最大回撤"])
    daily_drawdown = parse_percent(daily_row["最大回撤"])
    weekly_return = parse_percent(weekly_row["年化收益"])
    daily_return = parse_percent(daily_row["年化收益"])
    weekly_cost = parse_float(weekly_row.get("交易成本合计", 0.0) or 0.0)
    daily_cost = parse_float(daily_row.get("交易成本合计", 0.0) or 0.0)

    sharpe_improved = daily_sharpe > weekly_sharpe
    drawdown_not_worse = daily_drawdown >= weekly_drawdown
    return_or_nav_improved = daily_return > weekly_return or daily_nav > weekly_nav
    cost_significantly_higher = weekly_cost > 0 and daily_cost > weekly_cost * 1.3

    if sharpe_improved and drawdown_not_worse and return_or_nav_improved and not cost_significantly_higher:
        return "日频调仓夏普提升、最大回撤未恶化，且收益或期末净值改善，判定为优化。"

    issues = []
    if daily_sharpe < weekly_sharpe:
        issues.append("夏普下降")
    if daily_drawdown < weekly_drawdown:
        issues.append("最大回撤扩大")
    if cost_significantly_higher:
        issues.append("交易成本显著增加")

    if daily_return > weekly_return and issues:
        return f"日频调仓年化收益更高，但{ '、'.join(issues) }，不判定为明确优化。"
    if sharpe_improved and return_or_nav_improved and issues:
        return f"日频调仓收益和夏普比率提升，但{ '、'.join(issues) }，不判定为明确优化。"
    return "日频调仓核心指标方向混杂，未形成稳健优化。"


def dataframe_to_markdown(df):
    headers = [str(col) for col in df.columns]
    rows = []
    for _, row in df.iterrows():
        rows.append([str(row[col]) if not pd.isna(row[col]) else "" for col in df.columns])

    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_lines = ["| " + " | ".join(values) + " |" for values in rows]
    return "\n".join([header_line, separator_line, *data_lines])


def build_input_data():
    df_weight_raw = load_returns_csv(FILE_PATH_WEIGHT_RETURNS)
    df_trade_raw = load_returns_csv(FILE_PATH_TRADE_RETURNS)
    index_signal = load_index_signal(FILE_PATH_INDEX_SIGNAL)

    for df in (df_weight_raw, df_trade_raw):
        if "布油连续" in df.columns:
            df.drop(columns=["布油连续"], inplace=True)

    df_weight_all = df_weight_raw / 100.0
    df_trade_all_raw = df_trade_raw / 100.0
    df_trade_all = df_trade_all_raw.fillna(0)

    active_assets = []
    for asset in INDEX_FUTURES:
        if asset not in active_assets:
            active_assets.append(asset)
    for class_assets in RISK_PARITY_ASSET_CLASSES.values():
        for asset in class_assets:
            if asset not in active_assets:
                active_assets.append(asset)

    assets = [a for a in active_assets if a in df_weight_all.columns and a in df_trade_all.columns]
    return df_weight_all, df_trade_all_raw, df_trade_all, index_signal, assets


def run_backtest(rebalance_mode, version):
    if rebalance_mode not in {"weekly", "daily"}:
        raise ValueError("rebalance_mode must be 'weekly' or 'daily'")

    print(f"正在执行 v0.16 {rebalance_mode} 调仓回测，输出版本 v{version}...")
    METRICS_DIR.mkdir(exist_ok=True)
    NAV_DIR.mkdir(exist_ok=True)
    PERFORMANCE_DIR.mkdir(exist_ok=True)
    WEIGHTS_DIR.mkdir(exist_ok=True)
    CHART_DIR.mkdir(exist_ok=True)

    df_weight_all, df_trade_all_raw, df_trade_all, index_signal, assets = build_input_data()
    risk_parity_assets = [a for a in assets if a not in INDEX_FUTURES]

    df_weight = df_weight_all[assets].fillna(0)
    df_trade = df_trade_all[assets]
    listing_dates = {asset: df_trade_all_raw[asset].first_valid_index() for asset in assets}

    signal_on_trade_dates = index_signal.reindex(df_trade.index, method="ffill")
    first_signal_date = index_signal.first_valid_index()

    repo_rate_ann = df_trade_all.get("一天期国债逆回购", pd.Series(0.0, index=df_trade_all.index))
    calendar_days = df_trade_all.index.to_series().diff().dt.days.fillna(1)
    repo_shifted = repo_rate_ann.shift(1).fillna(0)
    repo_net_yield = np.maximum((repo_shifted / 365.0) * calendar_days - REPO_FEE_RATE, 0.0)

    margin_ratios = pd.Series({asset: MARGIN_RATIOS.get(asset, 1.0) for asset in assets})
    observation_dates = get_observation_dates(df_trade.index, rebalance_mode)

    ret_series = pd.Series(0.0, index=df_trade.index)
    margin_series = pd.Series(0.0, index=df_trade.index)
    weight_recs = []

    curr_w = pd.Series(0.0, index=assets)
    curr_margin = 0.0
    first_date = None
    rebalance_count = 0
    total_turnover = 0.0
    total_cost = 0.0

    for i in range(len(observation_dates) - 1):
        rebalance_date = observation_dates[i]
        if rebalance_date < first_signal_date:
            continue

        raw_signal = signal_on_trade_dates.loc[rebalance_date]
        if pd.isna(raw_signal):
            continue

        eligible_rp_assets = [
            asset
            for asset in risk_parity_assets
            if listing_dates.get(asset) is not None and listing_dates[asset] <= rebalance_date
        ]
        if len(eligible_rp_assets) == 0:
            continue

        lookback = df_weight.loc[rebalance_date - pd.DateOffset(months=12):rebalance_date, eligible_rp_assets]
        if len(lookback) < 150:
            continue

        index_target = allocate_index_futures(raw_signal, assets, listing_dates, rebalance_date)
        index_weight = float(index_target.sum())
        remaining_weight = max(0.0, 1.0 - index_weight)

        rp_active = get_risk_parity_weights(calculate_ewma_semi_cov(lookback, EWMA_DECAY))
        target = pd.Series(0.0, index=assets)
        target.loc[index_target.index] = index_target
        target.loc[eligible_rp_assets] = rp_active * remaining_weight

        hold_period = df_trade.loc[rebalance_date + pd.Timedelta(days=1):observation_dates[i + 1]]
        if len(hold_period) == 0:
            continue
        if first_date is None:
            first_date = hold_period.index[0]

        for date, daily_returns in hold_period.iterrows():
            daily_repo = repo_net_yield.loc[date]

            if date == hold_period.index[0]:
                new_margin = (target * margin_ratios).sum()
                idle_cash = max(0.0, 1.0 - new_margin)
                idle_return = idle_cash * daily_repo

                turnover = float((target - curr_w).abs().sum())
                cost = turnover * FEE_RATE
                ret_series.loc[date] = (target * daily_returns).sum() - cost + idle_return

                curr_w = target.copy()
                rebalance_count += 1
                total_turnover += turnover
                total_cost += cost

                weight_recs.append(
                    {
                        "date": rebalance_date,
                        "执行日期": date,
                        "调仓模式": rebalance_mode,
                        "策略名称": STRATEGY_NAME,
                        "股指期货信号": float(raw_signal),
                        "股指期货仓位": index_weight,
                        "换手率": turnover,
                        "交易成本": cost,
                        **{asset: target.loc[asset] for asset in assets},
                    }
                )
            else:
                idle_cash = max(0.0, 1.0 - curr_margin)
                idle_return = idle_cash * daily_repo
                ret_series.loc[date] = (curr_w * daily_returns).sum() + idle_return

            gross_weight = (curr_w * (1 + daily_returns)).sum()
            curr_w = (curr_w * (1 + daily_returns)) / (gross_weight or 1)
            curr_margin = (curr_w * margin_ratios).sum()
            margin_series.loc[date] = curr_margin

    if first_date is None:
        raise ValueError("日期或数据不满足条件")

    df_navs = pd.DataFrame(index=df_trade.loc[first_date:].index)
    df_navs[STRATEGY_NAME] = (1 + ret_series.loc[first_date:]).cumprod()
    navs_filename = NAV_DIR / f"策略每日净值走势_v{version}.csv"
    df_navs.to_csv(str(navs_filename), encoding="utf-8-sig")

    all_metrics = []

    def append_metrics(period_label, start_date, end_date):
        for asset in assets:
            metrics = calculate_metrics(df_trade.loc[start_date:end_date, asset])
            metrics["回测区间"] = period_label
            metrics["组合/资产"] = asset
            all_metrics.append(metrics)

        metrics = calculate_metrics(ret_series.loc[start_date:end_date], margin_series.loc[start_date:end_date])
        metrics["回测区间"] = period_label
        metrics["组合/资产"] = STRATEGY_NAME
        metrics["期末净值"] = f"{df_navs.loc[:end_date, STRATEGY_NAME].iloc[-1]:.4f}"
        metrics["调仓次数"] = rebalance_count if period_label == "全局 (Total)" else ""
        metrics["换手率合计"] = f"{total_turnover:.4f}" if period_label == "全局 (Total)" else ""
        metrics["交易成本合计"] = f"{total_cost:.4f}" if period_label == "全局 (Total)" else ""
        all_metrics.append(metrics)

    append_metrics("全局 (Total)", first_date, df_trade.index[-1])

    years = sorted(set(df_trade.loc[first_date:].index.year))
    for year in years:
        year_mask = (df_trade.index.year == year) & (df_trade.index >= first_date)
        if year_mask.sum() > 20:
            year_start = df_trade.index[year_mask][0]
            year_end = df_trade.index[year_mask][-1]
            append_metrics(f"{year}年", year_start, year_end)

    df_metrics = pd.DataFrame(all_metrics)
    cols_order = [
        "回测区间",
        "组合/资产",
        "期末净值",
        "年化收益",
        "年化波动",
        "夏普比率",
        "最大回撤",
        "月度胜率",
        "平均资金占用",
        "调仓次数",
        "换手率合计",
        "交易成本合计",
    ]
    cols_order = [col for col in cols_order if col in df_metrics.columns]
    df_metrics = df_metrics[cols_order]

    metrics_filename = PERFORMANCE_DIR / f"年度及全局回测指标_v{version}.csv"
    df_metrics.to_csv(str(metrics_filename), index=False, encoding="utf-8-sig")

    df_weights = pd.DataFrame(weight_recs)
    weight_cols = [
        "date",
        "执行日期",
        "调仓模式",
        "策略名称",
        "股指期货信号",
        "股指期货仓位",
        "换手率",
        "交易成本",
    ] + assets
    df_weights = df_weights[weight_cols]
    weights_filename = WEIGHTS_DIR / f"策略调仓仓位明细_v{version}.csv"
    df_weights.to_csv(str(weights_filename), index=False, encoding="utf-8-sig")

    chart_filename = CHART_DIR / f"回测图表_v{version}.png"
    plot_backtest_chart(df_navs, df_trade, df_weights, first_date, assets, rebalance_mode, chart_filename)

    global_metrics = df_metrics[
        (df_metrics["回测区间"] == "全局 (Total)") & (df_metrics["组合/资产"] == STRATEGY_NAME)
    ].iloc[0].to_dict()

    print(
        df_metrics[
            (df_metrics["回测区间"] == "全局 (Total)") & (df_metrics["组合/资产"] == STRATEGY_NAME)
        ].set_index("组合/资产").to_string()
    )
    print(f"数据文件已生成：{navs_filename} | {metrics_filename} | {weights_filename} | {chart_filename}")

    return {
        "mode": rebalance_mode,
        "version": version,
        "first_date": first_date,
        "end_date": df_trade.index[-1],
        "assets": assets,
        "nav_path": navs_filename,
        "metrics_path": metrics_filename,
        "weights_path": weights_filename,
        "chart_path": chart_filename,
        "df_navs": df_navs,
        "df_metrics": df_metrics,
        "df_weights": df_weights,
        "global_metrics": global_metrics,
    }


def plot_backtest_chart(df_navs, df_trade, df_weights, first_date, assets, rebalance_mode, chart_filename):
    fig, axes = plt.subplots(3, 1, figsize=(16, 16), sharex=False)

    axes[0].plot(df_navs.index, df_navs[STRATEGY_NAME], label=STRATEGY_NAME, color="purple", lw=2)
    if "沪深300主连" in df_trade.columns:
        axes[0].plot((1 + df_trade.loc[first_date:, "沪深300主连"]).cumprod(), label="沪深300主连", color="blue", alpha=0.3)
    if "10年国债主连" in df_trade.columns:
        axes[0].plot((1 + df_trade.loc[first_date:, "10年国债主连"]).cumprod(), label="10年国债主连", color="green", alpha=0.3)
    axes[0].set_title("策略累计净值走势", fontsize=14)
    axes[0].legend(loc="upper left")
    axes[0].grid(True, ls="--", alpha=0.5)

    df_w = df_weights.set_index("date")
    df_classes = pd.DataFrame(
        {class_name: df_w[[a for a in class_assets if a in assets]].sum(axis=1) for class_name, class_assets in PLOT_ASSET_CLASSES.items()}
    )
    axes[1].stackplot(
        df_classes.index,
        df_classes.T,
        labels=df_classes.columns,
        alpha=0.8,
        colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
    )
    axes[1].set_title(f"{'日频' if rebalance_mode == 'daily' else '周频'}大类资产权重", fontsize=14)
    axes[1].set_ylim(0, 1)
    axes[1].yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    axes[1].legend(loc="upper left")
    axes[1].grid(True, ls="--", alpha=0.4)

    axes[2].plot(df_w.index, df_w["股指期货信号"], label="股指期货信号", color="black", lw=1.5)
    axes[2].plot(df_w.index, df_w["股指期货仓位"], label="股指期货仓位", color="blue", lw=1.5)
    axes[2].set_title("股指期货信号与仓位", fontsize=14)
    axes[2].set_ylim(-1.1, 1.1)
    axes[2].yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    axes[2].legend(loc="upper left")
    axes[2].grid(True, ls="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(str(chart_filename), dpi=300)
    plt.close(fig)


def strategy_global_row(result, label):
    row = result["global_metrics"].copy()
    row["方案"] = label
    return row


def build_comparison(weekly_result, daily_result):
    weekly_row = strategy_global_row(weekly_result, "周频基准")
    daily_row = strategy_global_row(daily_result, "日频调仓")
    ordered_cols = [
        "方案",
        "回测区间",
        "组合/资产",
        "期末净值",
        "年化收益",
        "年化波动",
        "夏普比率",
        "最大回撤",
        "月度胜率",
        "平均资金占用",
        "调仓次数",
        "换手率合计",
        "交易成本合计",
    ]
    df_compare = pd.DataFrame([weekly_row, daily_row])
    return df_compare[[col for col in ordered_cols if col in df_compare.columns]]


def write_comparison_report(weekly_result, daily_result, comparison_path, report_path):
    df_compare = build_comparison(weekly_result, daily_result)
    df_compare.to_csv(str(comparison_path), index=False, encoding="utf-8-sig")

    weekly_row = df_compare[df_compare["方案"] == "周频基准"].iloc[0].to_dict()
    daily_row = df_compare[df_compare["方案"] == "日频调仓"].iloc[0].to_dict()
    conclusion = evaluate_optimization(weekly_row, daily_row)

    annual = pd.concat(
        [
            weekly_result["df_metrics"].assign(方案="周频基准"),
            daily_result["df_metrics"].assign(方案="日频调仓"),
        ],
        ignore_index=True,
    )
    annual_strategy = annual[
        (annual["组合/资产"] == STRATEGY_NAME) & (annual["回测区间"] != "全局 (Total)")
    ][["方案", "回测区间", "期末净值", "年化收益", "年化波动", "夏普比率", "最大回撤", "月度胜率", "平均资金占用"]]

    report = f"""# v0.16 周频与日频调仓对比说明

## 实验设置

- 周频基准：沿用 v0.16，每周最后一个实际交易日观察，下一交易日执行。
- 日频实验：每个实际交易日观察，下一交易日执行。
- 两组均使用当前工作区数据、相同资产池、相同股指期货信号、相同手续费率 `FEE_RATE = {FEE_RATE}`。
- 回测区间：{weekly_result["first_date"].date()} 至 {weekly_result["end_date"].date()}。

## 全局核心指标

{dataframe_to_markdown(df_compare)}

## 年度策略指标

{dataframe_to_markdown(annual_strategy)}

## 结论

{conclusion}

## 输出文件

- 周频净值：`{weekly_result["nav_path"]}`
- 周频指标：`{weekly_result["metrics_path"]}`
- 周频仓位：`{weekly_result["weights_path"]}`
- 日频净值：`{daily_result["nav_path"]}`
- 日频指标：`{daily_result["metrics_path"]}`
- 日频仓位：`{daily_result["weights_path"]}`
- 对比表：`{comparison_path}`
"""
    report_path.write_text(report, encoding="utf-8")
    return df_compare, conclusion


def main():
    COMPARE_DOC_DIR.mkdir(exist_ok=True)
    COMPARISON_DIR.mkdir(exist_ok=True)
    weekly_result = run_backtest("weekly", "0.16_weekly_rerun")
    daily_result = run_backtest("daily", "0.16_daily_rebalance_test")

    comparison_path = COMPARISON_DIR / "v0.16周频_vs_日频调仓对比.csv"
    report_path = COMPARE_DOC_DIR / "v0.16周频与日频调仓对比说明.md"
    df_compare, conclusion = write_comparison_report(weekly_result, daily_result, comparison_path, report_path)

    print("\n[周频 vs 日频全局对比]")
    print(df_compare.to_string(index=False))
    print(f"\n结论：{conclusion}")
    print(f"对比文件已生成：{comparison_path}")
    print(f"说明文档已生成：{report_path}")


if __name__ == "__main__":
    main()
