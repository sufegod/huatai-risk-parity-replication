import os
import warnings
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
MPLCONFIG_DIR = PACKAGE_DIR / ".matplotlib"
MPLCONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.optimize import minimize


warnings.filterwarnings("ignore")

# ================= 配置参数 =================
VERSION = "monetary_credit_cycle"
STRATEGY_NAME = "货币信用周期策略"

FILE_PATH_WEIGHT_RETURNS = PROJECT_DIR / "数据" / "日度收益数据更新" / "日涨跌幅_填充.csv"
FILE_PATH_TRADE_RETURNS = PROJECT_DIR / "数据" / "日度收益数据更新" / "日涨跌幅_未填充.csv"
FILE_PATH_CYCLE = PACKAGE_DIR / "国信证券-货币信用划分.xlsx"
FILE_PATH_ASSET_POOL = PACKAGE_DIR / "分析结果" / "三周期优选资产池.csv"

RESULT_DIR = PACKAGE_DIR / "回测结果"
NAV_DIR = RESULT_DIR / "净值"
PERFORMANCE_DIR = RESULT_DIR / "指标"
WEIGHTS_DIR = RESULT_DIR / "仓位明细"
CHART_DIR = RESULT_DIR / "图表"

MONTH_END_FREQ = "M"
FEE_RATE = 0.0005
REPO_FEE_RATE = 0.000001
EWMA_DECAY = 0.97
MIN_LOOKBACK_DAYS = 150
REPO_COLUMN = "一天期国债逆回购"

MAIN_CYCLES = ["宽货币紧信用", "宽货币宽信用", "紧货币紧信用"]
ALL_CYCLES = [*MAIN_CYCLES, "紧货币宽信用"]

INDEX_FUTURES = ["沪深300主连", "中证1000主连", "中证500主连"]

MARGIN_RATIOS_EXCHANGE_MIN = {
    "沪深300主连": 0.08,
    "中证1000主连": 0.08,
    "中证500主连": 0.08,
    "红利低波ETF": 1.00,
    "10年国债主连": 0.02,
    "30年国债主连": 0.035,
    "沪铜主连": 0.05,
    "沪铝主连": 0.05,
    "PTA主连": 0.05,
    "原油主连": 0.05,
    "豆粕主连": 0.05,
    "沪金主连": 0.04,
}

MARGIN_RATIOS_BROKER = {
    "沪深300主连": 0.14,
    "中证1000主连": 0.14,
    "中证500主连": 0.14,
    "红利低波ETF": 1.00,
    "10年国债主连": 0.025,
    "30年国债主连": 0.05,
    "沪铜主连": 0.16,
    "沪铝主连": 0.16,
    "PTA主连": 0.17,
    "原油主连": 0.32,
    "豆粕主连": 0.13,
    "沪金主连": 0.28,
}

# 默认使用期货公司实际保证金比例。
MARGIN_RATIOS = MARGIN_RATIOS_BROKER

PLOT_ASSET_CLASSES = {
    "股指期货": INDEX_FUTURES,
    "股票": ["红利低波ETF"],
    "债券": ["10年国债主连", "30年国债主连"],
    "商品": ["沪铜主连", "沪铝主连", "PTA主连", "原油主连", "豆粕主连"],
    "黄金": ["沪金主连"],
}

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# ================= 核心模型函数 =================
def calculate_ewma_semi_cov(returns_df, decay=0.97):
    downside = np.minimum(returns_df.values, 0.0)
    t_count, n_count = downside.shape
    weights = decay ** np.arange(t_count - 1, -1, -1)
    weights /= np.sum(weights)
    weighted = downside * np.sqrt(weights[:, np.newaxis])
    return np.dot(weighted.T, weighted) * 252 + np.eye(n_count) * 1e-8


def risk_parity_convex_objective(x, cov_matrix):
    n_count = len(x)
    return 0.5 * np.dot(x.T, np.dot(cov_matrix, x)) - np.sum(np.log(x)) / n_count


def risk_parity_convex_jacobian(x, cov_matrix):
    n_count = len(x)
    return np.dot(cov_matrix, x) - 1.0 / (n_count * x)


def get_risk_parity_weights(cov_matrix):
    n_count = cov_matrix.shape[0]
    res = minimize(
        risk_parity_convex_objective,
        np.ones(n_count),
        args=(cov_matrix,),
        method="L-BFGS-B",
        jac=risk_parity_convex_jacobian,
        bounds=[(1e-8, None)] * n_count,
        options={"ftol": 1e-12},
    )
    return res.x / np.sum(res.x)


def calculate_metrics(ret_series, margin_series=None):
    if len(ret_series) < 5:
        return {k: "0.00%" for k in ["年化收益", "年化波动", "夏普比率", "最大回撤", "月度胜率"]}

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


def get_asset_margin_series(asset, index):
    margin_ratio = MARGIN_RATIOS.get(asset, 1.0)
    return pd.Series(margin_ratio, index=index)


def calculate_position_margin_usage(weights):
    return float(sum(weight * MARGIN_RATIOS.get(asset, 1.0) for asset, weight in weights.items()))


def load_returns_csv(file_path):
    with file_path.open("r", encoding="utf-8-sig") as returns_file:
        df = pd.read_csv(returns_file, index_col=0, parse_dates=True)
    return df.dropna(how="all")


def load_monetary_credit_cycle_map(file_path):
    raw = pd.read_excel(file_path)
    raw = raw.rename(columns={raw.columns[0]: "日期"})
    if "周期划分" not in raw.columns:
        raise ValueError("货币信用划分文件缺少 `周期划分` 列")
    raw["日期"] = pd.to_datetime(raw["日期"], errors="coerce")
    raw = raw.dropna(subset=["日期", "周期划分"]).copy()
    raw["月份"] = raw["日期"].dt.to_period("M")
    raw["周期划分"] = raw["周期划分"].astype(str).str.strip()
    raw = raw[~raw["月份"].duplicated(keep="last")].sort_values("月份")
    return pd.Series(raw["周期划分"].values, index=pd.PeriodIndex(raw["月份"], freq="M"), name="周期划分")


def split_asset_pool_text(value):
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split("、") if item.strip()]


def parse_asset_pool_table(pool_df, full_asset_pool):
    full_asset_pool = list(full_asset_pool)
    pool_map = {}
    for _, row in pool_df.iterrows():
        cycle = str(row["周期划分"]).strip()
        parsed_assets = split_asset_pool_text(row["资产池"])
        pool_map[cycle] = [asset for asset in parsed_assets if asset in full_asset_pool]

    for cycle in MAIN_CYCLES:
        if cycle not in pool_map:
            raise ValueError(f"资产池文件缺少 `{cycle}`")
    pool_map["紧货币宽信用"] = full_asset_pool.copy()
    return pool_map


def load_asset_pool_map(file_path, full_asset_pool):
    pool_df = pd.read_csv(file_path, encoding="utf-8-sig")
    required_cols = {"周期划分", "资产池"}
    missing = required_cols - set(pool_df.columns)
    if missing:
        raise ValueError(f"资产池文件缺少必要列: {sorted(missing)}")
    return parse_asset_pool_table(pool_df, full_asset_pool)


def get_month_start_rebalance_dates(trade_index, cycle_months):
    trade_index = pd.DatetimeIndex(trade_index).sort_values()
    cycle_months = pd.PeriodIndex(cycle_months, freq="M")
    dates = []
    for month in cycle_months:
        month_dates = trade_index[trade_index.to_period("M") == month]
        if len(month_dates) > 0:
            dates.append(month_dates[0])
    return pd.DatetimeIndex(dates)


def select_eligible_assets(pool_assets, listing_dates, rebalance_date, available_assets):
    available_assets = set(available_assets)
    eligible = []
    for asset in pool_assets:
        listing_date = listing_dates.get(asset)
        if asset in available_assets and listing_date is not None and listing_date <= rebalance_date:
            eligible.append(asset)
    return eligible


def get_previous_trade_date(index, date):
    previous_dates = index[index < date]
    if len(previous_dates) == 0:
        return None
    return previous_dates[-1]


def get_holding_period_end_date(trade_index, rebalance_dates, position):
    trade_index = pd.DatetimeIndex(trade_index).sort_values()
    rebalance_dates = pd.DatetimeIndex(rebalance_dates).sort_values()
    rebalance_date = rebalance_dates[position]
    if position + 1 < len(rebalance_dates):
        next_rebalance_date = rebalance_dates[position + 1]
        holding_dates = trade_index[(trade_index >= rebalance_date) & (trade_index < next_rebalance_date)]
    else:
        holding_month = rebalance_date.to_period("M")
        holding_dates = trade_index[(trade_index >= rebalance_date) & (trade_index.to_period("M") == holding_month)]
    if len(holding_dates) == 0:
        return None
    return holding_dates[-1]


def build_nav_frame(ret_series, first_date, last_date):
    active_returns = ret_series.loc[first_date:last_date]
    df_navs = pd.DataFrame(index=active_returns.index)
    df_navs[STRATEGY_NAME] = (1 + active_returns).cumprod()
    df_navs.index.name = "date"
    return df_navs


def ensure_output_dirs():
    for directory in (RESULT_DIR, NAV_DIR, PERFORMANCE_DIR, WEIGHTS_DIR, CHART_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def write_strategy_note(asset_pool_map, output_files):
    lines = [
        "# 货币信用周期策略说明",
        "",
        "## 策略口径",
        "",
        "- 基于 v0.18 风险平价框架，取消股指期货估值信号。",
        "- 每月首个交易日调仓，使用同月货币信用周期象限选择资产池。",
        "- 三类有效周期读取 `分析结果/三周期优选资产池.csv`；`紧货币宽信用` 使用全资产池。",
        "- 风险估计窗口为调仓日前一交易日起向前 12 个月，避免使用调仓当日收益。",
        "- 收益单位输入为百分比点；策略内部转换为小数收益。",
        "",
        "## 周期资产池",
        "",
    ]
    for cycle in ALL_CYCLES:
        lines.append(f"- `{cycle}`：{'、'.join(asset_pool_map[cycle])}")
    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            *[f"- `{path}`" for path in output_files],
            "",
        ]
    )
    note_path = RESULT_DIR / "货币信用周期策略说明.md"
    note_path.write_text("\n".join(lines), encoding="utf-8")
    return note_path


# ================= 主流程 =================
def main():
    print(f"正在执行 {STRATEGY_NAME}（月初调仓）...")
    ensure_output_dirs()

    df_weight_raw = load_returns_csv(FILE_PATH_WEIGHT_RETURNS)
    df_trade_raw = load_returns_csv(FILE_PATH_TRADE_RETURNS)
    cycle_map = load_monetary_credit_cycle_map(FILE_PATH_CYCLE)

    for df in (df_weight_raw, df_trade_raw):
        if "布油连续" in df.columns:
            df.drop(columns=["布油连续"], inplace=True)

    df_weight_all = df_weight_raw / 100.0
    df_trade_all_raw = df_trade_raw / 100.0
    df_trade_all = df_trade_all_raw.fillna(0)

    full_asset_pool = [
        asset
        for asset in df_weight_all.columns
        if asset != REPO_COLUMN and asset in df_trade_all.columns
    ]
    asset_pool_map = load_asset_pool_map(FILE_PATH_ASSET_POOL, full_asset_pool)

    df_weight = df_weight_all[full_asset_pool].fillna(0)
    df_trade = df_trade_all[full_asset_pool]
    listing_dates = {
        asset: df_trade_all_raw[asset].first_valid_index()
        for asset in full_asset_pool
    }

    repo_rate_ann = df_trade_all.get(REPO_COLUMN, pd.Series(0.0, index=df_trade_all.index))
    calendar_days = df_trade_all.index.to_series().diff().dt.days.fillna(1)
    repo_shifted = repo_rate_ann.shift(1).fillna(0)
    repo_net_yield = np.maximum((repo_shifted / 365.0) * calendar_days - REPO_FEE_RATE, 0.0)

    m_ratios = pd.Series({asset: MARGIN_RATIOS.get(asset, 1.0) for asset in full_asset_pool})
    rebalance_dates = get_month_start_rebalance_dates(df_trade.index, cycle_map.index)

    ret_series = pd.Series(0.0, index=df_trade.index)
    margin_series = pd.Series(0.0, index=df_trade.index)
    weight_recs = []
    skipped_recs = []

    curr_w = pd.Series(0.0, index=full_asset_pool)
    curr_margin = 0.0
    first_date = None
    last_date = None

    for i in range(len(rebalance_dates)):
        rebalance_date = rebalance_dates[i]
        cycle_month = rebalance_date.to_period("M")
        cycle = cycle_map.get(cycle_month)
        if cycle is None:
            continue

        previous_trade_date = get_previous_trade_date(df_weight.index, rebalance_date)
        if previous_trade_date is None:
            skipped_recs.append({"date": rebalance_date, "周期划分": cycle, "跳过原因": "缺少调仓日前一交易日"})
            continue

        pool_assets = asset_pool_map.get(cycle, full_asset_pool)
        eligible_assets = select_eligible_assets(pool_assets, listing_dates, rebalance_date, full_asset_pool)
        if len(eligible_assets) == 0:
            skipped_recs.append({"date": rebalance_date, "周期划分": cycle, "跳过原因": "无已上市入选资产"})
            continue

        look = df_weight.loc[
            previous_trade_date - pd.DateOffset(months=12):previous_trade_date,
            eligible_assets,
        ].dropna(how="all")
        valid_assets = [
            asset
            for asset in eligible_assets
            if look[asset].dropna().shape[0] >= MIN_LOOKBACK_DAYS
        ]
        look = look[valid_assets].dropna(how="any")
        if len(valid_assets) == 0 or len(look) < MIN_LOOKBACK_DAYS:
            skipped_recs.append({"date": rebalance_date, "周期划分": cycle, "跳过原因": "估计窗口有效数据不足"})
            continue

        rp_active = get_risk_parity_weights(calculate_ewma_semi_cov(look, EWMA_DECAY))
        target = pd.Series(0.0, index=full_asset_pool)
        target.loc[valid_assets] = rp_active

        holding_end_date = get_holding_period_end_date(df_trade.index, rebalance_dates, i)
        if holding_end_date is None:
            continue
        holding_period = df_trade.loc[rebalance_date:holding_end_date]
        if len(holding_period) == 0:
            continue
        if first_date is None:
            first_date = holding_period.index[0]
        last_date = holding_period.index[-1]

        for date, daily_ret in holding_period.iterrows():
            daily_repo = repo_net_yield.loc[date]

            if date == holding_period.index[0]:
                new_margin = (target * m_ratios).sum()
                idle_cash = max(0.0, 1.0 - new_margin)
                idle_return = idle_cash * daily_repo
                cost = (target - curr_w).abs().sum() * FEE_RATE
                ret_series.loc[date] = (target * daily_ret).sum() - cost + idle_return

                curr_w = target.copy()
                curr_margin = float(new_margin)
                weight_recs.append(
                    {
                        "date": rebalance_date,
                        "策略名称": STRATEGY_NAME,
                        "周期划分": cycle,
                        "资产池": "、".join(pool_assets),
                        "入选资产数": len(valid_assets),
                        "资金占用比例": calculate_position_margin_usage(target),
                        **{asset: target.loc[asset] for asset in full_asset_pool},
                    }
                )
            else:
                idle_cash = max(0.0, 1.0 - curr_margin)
                idle_return = idle_cash * daily_repo
                ret_series.loc[date] = (curr_w * daily_ret).sum() + idle_return

            gross_weight = (curr_w * (1 + daily_ret)).sum()
            curr_w = (curr_w * (1 + daily_ret)) / (gross_weight or 1)
            curr_margin = (curr_w * m_ratios).sum()
            margin_series.loc[date] = curr_margin

    if first_date is None:
        raise ValueError("日期或数据不满足条件")

    print("正在生成每日净值数据...")
    df_navs = build_nav_frame(ret_series, first_date, last_date)
    navs_filename = NAV_DIR / f"策略每日净值走势_{VERSION}.csv"
    df_navs.to_csv(str(navs_filename), encoding="utf-8-sig")

    print("正在计算年度与全局指标...")
    all_metrics = []

    def append_metrics(period_label, start_date, end_date):
        for asset in full_asset_pool:
            asset_returns = df_trade.loc[start_date:end_date, asset]
            asset_margin = get_asset_margin_series(asset, asset_returns.index)
            metrics = calculate_metrics(asset_returns, asset_margin)
            metrics["回测区间"] = period_label
            metrics["组合/资产"] = asset
            all_metrics.append(metrics)

        metrics = calculate_metrics(ret_series.loc[start_date:end_date], margin_series.loc[start_date:end_date])
        metrics["回测区间"] = period_label
        metrics["组合/资产"] = STRATEGY_NAME
        all_metrics.append(metrics)

    append_metrics("全局 (Total)", first_date, last_date)
    years = sorted(set(df_trade.loc[first_date:last_date].index.year))
    for year in years:
        year_mask = (df_trade.index.year == year) & (df_trade.index >= first_date) & (df_trade.index <= last_date)
        if year_mask.sum() > 20:
            year_start = df_trade.index[year_mask][0]
            year_end = df_trade.index[year_mask][-1]
            append_metrics(f"{year}年", year_start, year_end)

    df_metrics = pd.DataFrame(all_metrics)
    cols_order = ["回测区间", "组合/资产", "年化收益", "年化波动", "夏普比率", "最大回撤", "月度胜率", "平均资金占用"]
    df_metrics = df_metrics[[col for col in cols_order if col in df_metrics.columns]]
    metrics_filename = PERFORMANCE_DIR / f"年度及全局回测指标_{VERSION}.csv"
    df_metrics.to_csv(str(metrics_filename), index=False, encoding="utf-8-sig")

    print("\n[全局回测总览]")
    print(
        df_metrics[
            (df_metrics["回测区间"] == "全局 (Total)")
            & (df_metrics["组合/资产"] == STRATEGY_NAME)
        ].set_index("组合/资产").to_string()
    )

    print("\n正在生成月度调仓仓位明细...")
    df_weights = pd.DataFrame(weight_recs)
    weight_cols = ["date", "策略名称", "周期划分", "资产池", "入选资产数", *full_asset_pool, "资金占用比例"]
    df_weights = df_weights[weight_cols]
    weights_filename = WEIGHTS_DIR / f"策略月度调仓仓位明细_{VERSION}.csv"
    df_weights.to_csv(str(weights_filename), index=False, encoding="utf-8-sig")

    skipped_filename = None
    if skipped_recs:
        skipped_filename = WEIGHTS_DIR / f"策略跳过调仓记录_{VERSION}.csv"
        pd.DataFrame(skipped_recs).to_csv(str(skipped_filename), index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(3, 1, figsize=(16, 16), sharex=False)
    axes[0].plot(df_navs.index, df_navs[STRATEGY_NAME], label=STRATEGY_NAME, color="purple", lw=2)
    if "沪深300主连" in df_trade.columns:
        axes[0].plot((1 + df_trade.loc[first_date:, "沪深300主连"]).cumprod(), label="沪深300主连", color="blue", alpha=0.3)
    if "10年国债主连" in df_trade.columns:
        axes[0].plot((1 + df_trade.loc[first_date:, "10年国债主连"]).cumprod(), label="10年国债主连", color="green", alpha=0.3)
    axes[0].set_title("策略累计净值走势", fontsize=14)
    axes[0].legend(loc="upper left")
    axes[0].grid(True, ls="--", alpha=0.5)

    df_weight_plot = df_weights.set_index("date")
    df_class = pd.DataFrame(
        {
            class_name: df_weight_plot[[asset for asset in class_assets if asset in full_asset_pool]].sum(axis=1)
            for class_name, class_assets in PLOT_ASSET_CLASSES.items()
        }
    )
    axes[1].stackplot(
        df_class.index,
        df_class.T,
        labels=df_class.columns,
        alpha=0.8,
        colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
    )
    axes[1].set_title("月度调仓大类资产权重", fontsize=14)
    axes[1].set_ylim(0, 1)
    axes[1].yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    axes[1].legend(loc="upper left")
    axes[1].grid(True, ls="--", alpha=0.4)

    cycle_codes = pd.Categorical(df_weight_plot["周期划分"], categories=ALL_CYCLES, ordered=True).codes
    axes[2].step(df_weight_plot.index, cycle_codes, where="post", label="周期划分", color="black", lw=1.5)
    axes[2].plot(df_weight_plot.index, df_weight_plot["入选资产数"] / max(len(full_asset_pool), 1), label="入选资产数/全资产数", color="blue", lw=1.2)
    axes[2].plot(df_weight_plot.index, df_weight_plot["资金占用比例"], label="资金占用比例", color="orange", lw=1.2)
    axes[2].set_title("周期划分、入选资产数与资金占用", fontsize=14)
    axes[2].set_yticks(range(len(ALL_CYCLES)), ALL_CYCLES)
    axes[2].legend(loc="upper left")
    axes[2].grid(True, ls="--", alpha=0.4)

    plt.tight_layout()
    chart_filename = CHART_DIR / f"回测图表_{VERSION}.png"
    plt.savefig(str(chart_filename), dpi=300)

    output_files = [navs_filename, metrics_filename, weights_filename, chart_filename]
    if skipped_filename is not None:
        output_files.append(skipped_filename)
    note_path = write_strategy_note(asset_pool_map, output_files)
    output_files.append(note_path)

    print("\n数据文件已生成：")
    for idx, file_path in enumerate(output_files, start=1):
        print(f" {idx}. {file_path}")


if __name__ == "__main__":
    main()
