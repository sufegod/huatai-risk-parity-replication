import importlib.util
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


warnings.filterwarnings("ignore")


def _load_base_strategy_module():
    base_path = PACKAGE_DIR / "货币信用周期策略.py"
    spec = importlib.util.spec_from_file_location("monetary_credit_cycle_base_strategy", base_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load base strategy module: {base_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base_strategy_module()

# ================= 配置参数 =================
VERSION = "monetary_credit_cycle_v0_2_index_signal"
STRATEGY_NAME = "货币信用周期策略0.2（股指信号）"

FILE_PATH_WEIGHT_RETURNS = base.FILE_PATH_WEIGHT_RETURNS
FILE_PATH_TRADE_RETURNS = base.FILE_PATH_TRADE_RETURNS
FILE_PATH_CYCLE = base.FILE_PATH_CYCLE
FILE_PATH_ASSET_POOL = base.FILE_PATH_ASSET_POOL
FILE_PATH_INDEX_SIGNAL = PROJECT_DIR / "数据" / "原始数据" / "股指期货信号.xlsx"

RESULT_DIR = PACKAGE_DIR / "回测结果_0.2_股指信号"
NAV_DIR = RESULT_DIR / "净值"
PERFORMANCE_DIR = RESULT_DIR / "指标"
WEIGHTS_DIR = RESULT_DIR / "仓位明细"
CHART_DIR = RESULT_DIR / "图表"

MONTH_END_FREQ = base.MONTH_END_FREQ
FEE_RATE = base.FEE_RATE
REPO_FEE_RATE = base.REPO_FEE_RATE
EWMA_DECAY = base.EWMA_DECAY
MIN_LOOKBACK_DAYS = base.MIN_LOOKBACK_DAYS
REPO_COLUMN = base.REPO_COLUMN

MAIN_CYCLES = base.MAIN_CYCLES
ALL_CYCLES = base.ALL_CYCLES

INDEX_BASE_WEIGHT = 0.30
INDEX_FUTURES = ["沪深300主连", "中证1000主连", "中证500主连"]

MARGIN_RATIOS = base.MARGIN_RATIOS

PLOT_ASSET_CLASSES = {
    "股指期货": INDEX_FUTURES,
    "股票": ["红利低波ETF"],
    "债券": ["10年国债主连", "30年国债主连"],
    "商品": ["沪铜主连", "沪铝主连", "PTA主连", "原油主连", "豆粕主连"],
    "黄金": ["沪金主连"],
}

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# ================= 复用无信号版本的通用函数 =================
calculate_ewma_semi_cov = base.calculate_ewma_semi_cov
get_risk_parity_weights = base.get_risk_parity_weights
calculate_metrics = base.calculate_metrics
get_asset_margin_series = base.get_asset_margin_series
load_returns_csv = base.load_returns_csv
load_monetary_credit_cycle_map = base.load_monetary_credit_cycle_map
load_asset_pool_map = base.load_asset_pool_map
get_month_start_rebalance_dates = base.get_month_start_rebalance_dates
select_eligible_assets = base.select_eligible_assets
get_previous_trade_date = base.get_previous_trade_date
get_holding_period_end_date = base.get_holding_period_end_date


def calculate_position_margin_usage(weights):
    return float(sum(weight * MARGIN_RATIOS.get(asset, 1.0) for asset, weight in weights.items()))


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
        if asset in assets
        and listing_dates.get(asset) is not None
        and listing_dates[asset] <= rebalance_date
    ]
    if not listed_index_futures:
        return target

    target.loc[listed_index_futures] = total_weight / len(listed_index_futures)
    return target


def get_risk_parity_pool_assets(pool_assets):
    return [asset for asset in pool_assets if asset not in INDEX_FUTURES]


def combine_index_and_risk_parity_weights(index_target, risk_parity_assets, risk_parity_weights, full_asset_pool):
    target = pd.Series(0.0, index=full_asset_pool)
    target.loc[index_target.index] = index_target.reindex(target.index).fillna(0)
    index_weight = float(target.loc[[asset for asset in INDEX_FUTURES if asset in target.index]].sum())
    remaining_weight = max(0.0, 1.0 - index_weight)
    target.loc[risk_parity_assets] = np.asarray(risk_parity_weights) * remaining_weight
    return target


def build_nav_frame(ret_series, first_date, last_date):
    active_returns = ret_series.loc[first_date:last_date]
    df_navs = pd.DataFrame(index=active_returns.index)
    df_navs[STRATEGY_NAME] = (1 + active_returns).cumprod()
    df_navs.index.name = "date"
    return df_navs


def build_weight_record(
    rebalance_date,
    cycle,
    pool_assets,
    risk_parity_assets,
    raw_signal,
    index_weight,
    target,
    full_asset_pool,
):
    return {
        "date": rebalance_date,
        "策略名称": STRATEGY_NAME,
        "周期划分": cycle,
        "资产池": "、".join(pool_assets),
        "风险平价资产池": "、".join(risk_parity_assets),
        "风险平价入选资产数": len(risk_parity_assets),
        "股指期货信号": float(raw_signal),
        "股指期货仓位": float(index_weight),
        "资金占用比例": calculate_position_margin_usage(target),
        **{asset: target.loc[asset] for asset in full_asset_pool},
    }


def ensure_output_dirs():
    for directory in (RESULT_DIR, NAV_DIR, PERFORMANCE_DIR, WEIGHTS_DIR, CHART_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def write_strategy_note(asset_pool_map, output_files, first_signal_date):
    relative_outputs = [Path(path).resolve().relative_to(PACKAGE_DIR) for path in output_files]
    lines = [
        "# 货币信用周期策略0.2（股指信号）说明",
        "",
        "## 策略定位",
        "",
        "- 本策略是在无信号版 `货币信用周期策略` 基础上新增股指期货信号的 0.2 版本。",
        "- 股指期货信号采用独立覆盖口径，股指仓位由信号单独决定，剩余权重给当前周期下的非股指资产做风险平价。",
        "- 股指信号覆盖：沪深300主连、中证1000主连、中证500主连。",
        "",
        "## 数据输入",
        "",
        f"- 权重估计收益：`{FILE_PATH_WEIGHT_RETURNS.relative_to(PROJECT_DIR)}`，收益单位为百分比点，策略内部转换为小数收益。",
        f"- 交易收益：`{FILE_PATH_TRADE_RETURNS.relative_to(PROJECT_DIR)}`，用于实际组合收益计算。",
        f"- 货币信用周期：`{FILE_PATH_CYCLE.relative_to(PROJECT_DIR)}`，按月份读取 `周期划分`。",
        f"- 周期资产池：`{FILE_PATH_ASSET_POOL.relative_to(PROJECT_DIR)}`，三类有效周期读取优选资产池。",
        f"- 股指期货信号：`{FILE_PATH_INDEX_SIGNAL.relative_to(PROJECT_DIR)}`，首个有效日期为 `{first_signal_date.date()}`。",
        "",
        "## 调仓规则",
        "",
        "- 回测从股指信号有效后的首个共同交易日开始。",
        "- 每月首个交易日调仓，持有到下月首个交易日前一交易日；最后一期持有到当月最后一个交易日。",
        "- 月初调仓使用同月货币信用周期象限，这是历史划分复现口径，不声明为无前视实盘信号。",
        "- 非股指风险平价资产使用调仓日前一交易日起向前 12 个月估计窗口，避免使用调仓当日收益。",
        "- 入选非股指资产需已上市，且估计窗口有效数据不少于 150 个交易日。",
        "",
        "## 资产池与周期映射",
        "",
    ]
    for cycle in ALL_CYCLES:
        lines.append(f"- `{cycle}`：{'、'.join(asset_pool_map[cycle])}")
    lines.extend(
        [
            "",
            "## 风险平价与资金占用口径",
            "",
            f"- 股指目标仓位 = `{INDEX_BASE_WEIGHT:.0%} * min(max(signal, 0), 1)`，在已上市股指期货中等权分配。",
            "- 负信号、空信号和 0 信号均按 0 仓位处理，信号大于 1 时按 1 截断。",
            "- 当前周期资产池中的股指期货会从风险平价资产池中剔除，避免股指仓位重复配置。",
            "- 非股指资产使用 EWMA 半协方差风险平价模型，权重缩放到 `1 - 股指期货仓位`。",
            "- 资金占用比例按各资产目标权重乘以期货公司保证金比例计算；ETF 保证金比例按 100% 处理。",
            "",
            "## 闲置资金收益、手续费、保证金说明",
            "",
            "- 未被保证金占用的闲置资金按 `一天期国债逆回购` 前一交易日收益计息。",
            "- 调仓换手成本使用 `0.0005`，逆回购日成本使用 `0.000001`。",
            "- 保证金比例默认沿用无信号版策略中的 `MARGIN_RATIOS_BROKER`。",
            "",
            "## 输出目录与结果文件",
            "",
            "- 输出目录：`回测结果_0.2_股指信号/`。",
            *[f"- `{path}`" for path in relative_outputs],
            "",
            "## 注意事项",
            "",
            "- 股指信号是独立覆盖，不受周期优选资产池是否包含股指期货影响。",
            "- 若非股指风险平价有效资产不足，策略跳过该月调仓，不单独开股指裸仓。",
            "- 若未来更新 `分析结果/三周期优选资产池.csv`，策略会自动使用新的周期资产池。",
            "",
        ]
    )
    note_path = PACKAGE_DIR / "货币信用周期策略0.2（股指信号）说明.md"
    note_path.write_text("\n".join(lines), encoding="utf-8")
    return note_path


# ================= 主流程 =================
def main():
    print(f"正在执行 {STRATEGY_NAME}（月初调仓 + 股指信号）...")
    ensure_output_dirs()

    df_weight_raw = load_returns_csv(FILE_PATH_WEIGHT_RETURNS)
    df_trade_raw = load_returns_csv(FILE_PATH_TRADE_RETURNS)
    cycle_map = load_monetary_credit_cycle_map(FILE_PATH_CYCLE)
    index_signal = load_index_signal(FILE_PATH_INDEX_SIGNAL)

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

    signal_on_trade_dates = index_signal.reindex(df_trade.index, method="ffill")
    first_signal_date = index_signal.first_valid_index()

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
        if rebalance_date < first_signal_date:
            continue

        cycle_month = rebalance_date.to_period("M")
        cycle = cycle_map.get(cycle_month)
        if cycle is None:
            continue

        raw_signal = signal_on_trade_dates.loc[rebalance_date]
        if pd.isna(raw_signal):
            skipped_recs.append({"date": rebalance_date, "周期划分": cycle, "跳过原因": "缺少股指期货信号"})
            continue

        previous_trade_date = get_previous_trade_date(df_weight.index, rebalance_date)
        if previous_trade_date is None:
            skipped_recs.append({"date": rebalance_date, "周期划分": cycle, "跳过原因": "缺少调仓日前一交易日"})
            continue

        pool_assets = asset_pool_map.get(cycle, full_asset_pool)
        risk_parity_pool = get_risk_parity_pool_assets(pool_assets)
        eligible_rp_assets = select_eligible_assets(risk_parity_pool, listing_dates, rebalance_date, full_asset_pool)
        if len(eligible_rp_assets) == 0:
            skipped_recs.append({"date": rebalance_date, "周期划分": cycle, "跳过原因": "无已上市非股指入选资产"})
            continue

        look = df_weight.loc[
            previous_trade_date - pd.DateOffset(months=12):previous_trade_date,
            eligible_rp_assets,
        ].dropna(how="all")
        valid_rp_assets = [
            asset
            for asset in eligible_rp_assets
            if look[asset].dropna().shape[0] >= MIN_LOOKBACK_DAYS
        ]
        look = look[valid_rp_assets].dropna(how="any")
        if len(valid_rp_assets) == 0 or len(look) < MIN_LOOKBACK_DAYS:
            skipped_recs.append({"date": rebalance_date, "周期划分": cycle, "跳过原因": "非股指估计窗口有效数据不足"})
            continue

        index_target = allocate_index_futures(raw_signal, full_asset_pool, listing_dates, rebalance_date)
        index_weight = float(index_target.loc[[asset for asset in INDEX_FUTURES if asset in index_target.index]].sum())
        rp_active = get_risk_parity_weights(calculate_ewma_semi_cov(look, EWMA_DECAY))
        target = combine_index_and_risk_parity_weights(index_target, valid_rp_assets, rp_active, full_asset_pool)

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
                    build_weight_record(
                        rebalance_date=rebalance_date,
                        cycle=cycle,
                        pool_assets=pool_assets,
                        risk_parity_assets=valid_rp_assets,
                        raw_signal=raw_signal,
                        index_weight=index_weight,
                        target=target,
                        full_asset_pool=full_asset_pool,
                    )
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
    weight_cols = [
        "date",
        "策略名称",
        "周期划分",
        "资产池",
        "风险平价资产池",
        "风险平价入选资产数",
        "股指期货信号",
        "股指期货仓位",
        *full_asset_pool,
        "资金占用比例",
    ]
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
        axes[0].plot((1 + df_trade.loc[first_date:last_date, "沪深300主连"]).cumprod(), label="沪深300主连", color="blue", alpha=0.3)
    if "10年国债主连" in df_trade.columns:
        axes[0].plot((1 + df_trade.loc[first_date:last_date, "10年国债主连"]).cumprod(), label="10年国债主连", color="green", alpha=0.3)
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
    axes[2].set_yticks(range(len(ALL_CYCLES)), ALL_CYCLES)
    axes[2].set_title("周期划分、股指期货信号、股指仓位与资金占用", fontsize=14)
    axes[2].grid(True, ls="--", alpha=0.4)
    signal_axis = axes[2].twinx()
    signal_axis.plot(df_weight_plot.index, df_weight_plot["股指期货信号"], label="股指期货信号", color="blue", lw=1.2)
    signal_axis.plot(df_weight_plot.index, df_weight_plot["股指期货仓位"], label="股指期货仓位", color="red", lw=1.2)
    signal_axis.plot(df_weight_plot.index, df_weight_plot["资金占用比例"], label="资金占用比例", color="orange", lw=1.2)
    signal_axis.set_ylim(-1.05, 1.05)
    signal_axis.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    lines, labels = axes[2].get_legend_handles_labels()
    lines2, labels2 = signal_axis.get_legend_handles_labels()
    axes[2].legend(lines + lines2, labels + labels2, loc="upper left")

    plt.tight_layout()
    chart_filename = CHART_DIR / f"回测图表_{VERSION}.png"
    plt.savefig(str(chart_filename), dpi=300)

    output_files = [navs_filename, metrics_filename, weights_filename, chart_filename]
    if skipped_filename is not None:
        output_files.append(skipped_filename)
    note_path = write_strategy_note(asset_pool_map, output_files, first_signal_date)
    output_files.append(note_path)

    print("\n数据文件已生成：")
    for idx, file_path in enumerate(output_files, start=1):
        print(f" {idx}. {file_path}")


if __name__ == "__main__":
    main()
