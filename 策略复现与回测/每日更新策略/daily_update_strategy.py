import argparse
import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
BACKTEST_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = BACKTEST_DIR.parent

DATA_UPDATE_SCRIPT = PROJECT_ROOT / "数据" / "JYDB数据替换" / "update_daily_returns.py"
V016_SCRIPT = BACKTEST_DIR / "策略代码" / "资产风险平价策略0.16（周频调仓+股指信号）.py"
OUTPUT_DIR = SCRIPT_DIR / "输出"

WEIGHT_RETURNS_PATH = PROJECT_ROOT / "数据" / "JYDB数据替换" / "日涨跌幅_填充.csv"
TRADE_RETURNS_PATH = PROJECT_ROOT / "数据" / "JYDB数据替换" / "日涨跌幅_未填充.csv"
INDEX_SIGNAL_PATH = PROJECT_ROOT / "数据" / "原始数据" / "股指期货信号.xlsx"

WEEKLY_REBALANCE_FREQ = "W-FRI"


@dataclass(frozen=True)
class ObservationSelection:
    observation_date: pd.Timestamp
    is_new_observation: bool


@dataclass(frozen=True)
class TargetReport:
    as_of_date: pd.Timestamp
    observation_date: pd.Timestamp
    is_new_observation: bool
    raw_signal: float
    index_weight: float
    target_weights: pd.Series
    margin_ratios: pd.Series


@dataclass(frozen=True)
class BacktestResult:
    as_of_date: pd.Timestamp
    first_date: pd.Timestamp
    assets: list[str]
    df_navs: pd.DataFrame
    df_metrics: pd.DataFrame
    df_weekly_weights: pd.DataFrame
    df_trade: pd.DataFrame | None = None


Runner = Callable[..., subprocess.CompletedProcess]


def load_v016_module(script_path: Path = V016_SCRIPT):
    spec = importlib.util.spec_from_file_location("risk_parity_v016", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载v0.16策略脚本: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="先更新日度数据，再生成v0.16每日策略仓位报告")
    parser.add_argument("--data-end-date", help="传递给每日数据更新脚本的结束日期")
    parser.add_argument("--skip-data-update", action="store_true", help="跳过每日数据更新，仅基于现有CSV生成报告")
    parser.add_argument("--force-observation", action="store_true", help="强制将策略数据日期作为周度观察日")
    return parser.parse_args(argv)


def run_data_update(
    data_end_date: str | None = None,
    runner: Runner = subprocess.run,
    python_executable: str = sys.executable,
) -> subprocess.CompletedProcess:
    cmd = [python_executable, str(DATA_UPDATE_SCRIPT)]
    if data_end_date:
        cmd.extend(["--end-date", data_end_date])
    return runner(cmd, cwd=PROJECT_ROOT, check=True)


def maybe_run_data_update(args: argparse.Namespace, runner: Runner = subprocess.run) -> str:
    if args.skip_data_update:
        return "skipped"
    run_data_update(args.data_end_date, runner=runner)
    return "updated"


def output_paths(output_dir: Path, as_of_date: pd.Timestamp) -> dict[str, Path]:
    suffix = pd.Timestamp(as_of_date).strftime("%Y-%m-%d")
    return {
        "positions": output_dir / "仓位" / f"仓位_{suffix}.csv",
        "nav": output_dir / "净值" / f"策略每日净值走势_{suffix}.csv",
        "metrics": output_dir / "指标" / f"年度及全局回测指标_{suffix}.csv",
        "weekly_weights": output_dir / "仓位明细" / f"策略周度仓位明细_{suffix}.csv",
        "chart": output_dir / "图表" / f"回测图表_{suffix}.png",
        "report": output_dir / "报告" / f"回测报告_{suffix}.md",
    }


def get_weekly_observation_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    observations = []
    date_series = pd.Series(index=index, data=index)
    for _, group in date_series.groupby(pd.Grouper(freq=WEEKLY_REBALANCE_FREQ)):
        if len(group) > 0:
            observations.append(group.index[-1])
    return pd.DatetimeIndex(observations)


def select_observation_date(
    trade_index: pd.DatetimeIndex,
    as_of_date: pd.Timestamp,
    force_observation: bool,
) -> ObservationSelection:
    if len(trade_index) == 0:
        raise ValueError("交易收益数据为空，无法选择观察日")

    as_of_date = pd.Timestamp(as_of_date).normalize()
    available_index = pd.DatetimeIndex(trade_index).sort_values()
    available_index = available_index[available_index <= as_of_date]
    if len(available_index) == 0:
        raise ValueError(f"没有不晚于 {as_of_date:%Y-%m-%d} 的交易收益数据")

    actual_as_of = pd.Timestamp(available_index[-1]).normalize()
    if force_observation or actual_as_of.weekday() == 4:
        return ObservationSelection(actual_as_of, True)

    week_ends = get_weekly_observation_dates(available_index)
    if len(week_ends) == 0:
        raise ValueError("没有可用周度观察日")
    if week_ends[-1].normalize() == actual_as_of:
        week_ends = week_ends[:-1]
    if len(week_ends) == 0:
        raise ValueError("当前日期之前没有已完成的周度观察日")
    return ObservationSelection(pd.Timestamp(week_ends[-1]).normalize(), False)


def validate_returns_frame(df: pd.DataFrame, required_columns: list[str], label: str) -> None:
    if df.empty:
        raise ValueError(f"{label}为空")
    if df.index.has_duplicates:
        raise ValueError(f"{label}存在重复日期")
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{label}日期未递增")
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{label}缺少必要列: {', '.join(missing)}")


def load_strategy_inputs(v016):
    df_weight_raw = v016.load_returns_csv(WEIGHT_RETURNS_PATH)
    df_trade_raw = v016.load_returns_csv(TRADE_RETURNS_PATH)
    index_signal = v016.load_index_signal(INDEX_SIGNAL_PATH)

    for df in (df_weight_raw, df_trade_raw):
        if "布油连续" in df.columns:
            df.drop(columns=["布油连续"], inplace=True)

    active_assets = []
    for asset in v016.INDEX_FUTURES:
        if asset not in active_assets:
            active_assets.append(asset)
    for class_assets in v016.RISK_PARITY_ASSET_CLASSES.values():
        for asset in class_assets:
            if asset not in active_assets:
                active_assets.append(asset)

    validate_returns_frame(df_weight_raw, active_assets, "权重估计收益")
    validate_returns_frame(df_trade_raw, active_assets, "交易收益")
    return df_weight_raw, df_trade_raw, index_signal, active_assets


def compute_target_for_observation(
    v016,
    observation_date: pd.Timestamp,
    assets: list[str],
    risk_parity_assets: list[str],
    df_weight: pd.DataFrame,
    signal_on_trade_dates: pd.Series,
    first_signal_date: pd.Timestamp,
    listing_dates: dict[str, pd.Timestamp | None],
) -> tuple[pd.Series, float, float]:
    observation_date = pd.Timestamp(observation_date).normalize()
    if observation_date < first_signal_date:
        raise ValueError(f"{observation_date:%Y-%m-%d} 早于首个有效股指期货信号日")

    raw_signal = signal_on_trade_dates.loc[observation_date]
    if pd.isna(raw_signal):
        raise ValueError(f"{observation_date:%Y-%m-%d} 没有可用股指期货信号")

    eligible_rp_assets = [
        asset
        for asset in risk_parity_assets
        if listing_dates.get(asset) is not None and listing_dates[asset] <= observation_date
    ]
    if not eligible_rp_assets:
        raise ValueError(f"{observation_date:%Y-%m-%d} 没有可配置风险平价资产")

    lookback = df_weight.loc[
        observation_date - pd.DateOffset(months=12) : observation_date,
        eligible_rp_assets,
    ]
    if len(lookback) < 150:
        raise ValueError(f"{observation_date:%Y-%m-%d} 的回看窗口不足150个交易日")

    index_target = v016.allocate_index_futures(raw_signal, assets, listing_dates, observation_date)
    index_weight = float(index_target.sum())
    remaining_weight = max(0.0, 1.0 - index_weight)

    rp_weights = v016.get_risk_parity_weights(v016.calculate_ewma_semi_cov(lookback, v016.EWMA_DECAY))
    target = pd.Series(0.0, index=assets)
    target.loc[index_target.index] = index_target
    target.loc[eligible_rp_assets] = rp_weights * remaining_weight
    return target, float(raw_signal), index_weight


def find_valid_target(
    v016,
    selection: ObservationSelection,
    week_ends: pd.DatetimeIndex,
    assets: list[str],
    risk_parity_assets: list[str],
    df_weight: pd.DataFrame,
    signal_on_trade_dates: pd.Series,
    first_signal_date: pd.Timestamp,
    listing_dates: dict[str, pd.Timestamp | None],
) -> tuple[pd.Timestamp, pd.Series, float, float]:
    if selection.is_new_observation:
        target, raw_signal, index_weight = compute_target_for_observation(
            v016,
            selection.observation_date,
            assets,
            risk_parity_assets,
            df_weight,
            signal_on_trade_dates,
            first_signal_date,
            listing_dates,
        )
        return selection.observation_date, target, raw_signal, index_weight

    candidates = [pd.Timestamp(date).normalize() for date in week_ends if date <= selection.observation_date]
    for candidate in reversed(candidates):
        try:
            target, raw_signal, index_weight = compute_target_for_observation(
                v016,
                candidate,
                assets,
                risk_parity_assets,
                df_weight,
                signal_on_trade_dates,
                first_signal_date,
                listing_dates,
            )
            return candidate, target, raw_signal, index_weight
        except ValueError:
            continue
    raise ValueError("没有可用的历史周度目标仓位")


def build_target_report(force_observation: bool = False) -> TargetReport:
    v016 = load_v016_module()
    df_weight_raw, df_trade_raw, index_signal, active_assets = load_strategy_inputs(v016)

    df_weight_all = df_weight_raw / 100.0
    df_trade_all_raw = df_trade_raw / 100.0
    df_trade_all = df_trade_all_raw.fillna(0)

    as_of_date = min(df_weight_all.index.max(), df_trade_all.index.max()).normalize()
    df_weight_all = df_weight_all.loc[:as_of_date]
    df_trade_all_raw = df_trade_all_raw.loc[:as_of_date]
    df_trade_all = df_trade_all.loc[:as_of_date]

    assets = [asset for asset in active_assets if asset in df_weight_all.columns and asset in df_trade_all.columns]
    risk_parity_assets = [asset for asset in assets if asset not in v016.INDEX_FUTURES]

    df_weight = df_weight_all[assets].fillna(0)
    df_trade = df_trade_all[assets]
    listing_dates = {asset: df_trade_all_raw[asset].first_valid_index() for asset in assets}

    signal_on_trade_dates = index_signal.reindex(df_trade.index, method="ffill")
    first_signal_date = index_signal.first_valid_index()
    if first_signal_date is None:
        raise ValueError("股指期货信号文件没有有效信号")

    selection = select_observation_date(df_trade.index, as_of_date, force_observation)
    week_ends = get_weekly_observation_dates(df_trade.index[df_trade.index <= selection.observation_date])
    observation_date, target, raw_signal, index_weight = find_valid_target(
        v016,
        selection,
        week_ends,
        assets,
        risk_parity_assets,
        df_weight,
        signal_on_trade_dates,
        first_signal_date,
        listing_dates,
    )
    margin_ratios = pd.Series({asset: v016.MARGIN_RATIOS.get(asset, 1.0) for asset in assets})

    return TargetReport(
        as_of_date=as_of_date,
        observation_date=observation_date,
        is_new_observation=selection.is_new_observation,
        raw_signal=raw_signal,
        index_weight=index_weight,
        target_weights=target,
        margin_ratios=margin_ratios,
    )


def build_backtest_result() -> BacktestResult:
    v016 = load_v016_module()
    df_weight_raw, df_trade_raw, index_signal, active_assets = load_strategy_inputs(v016)

    df_weight_all = df_weight_raw / 100.0
    df_trade_all_raw = df_trade_raw / 100.0
    df_trade_all = df_trade_all_raw.fillna(0)
    as_of_date = min(df_weight_all.index.max(), df_trade_all.index.max()).normalize()

    repo_rate_ann = df_trade_all.get("一天期国债逆回购", pd.Series(0.0, index=df_trade_all.index))
    assets = [asset for asset in active_assets if asset in df_weight_all.columns and asset in df_trade_all.columns]
    risk_parity_assets = [asset for asset in assets if asset not in v016.INDEX_FUTURES]

    df_weight = df_weight_all[assets].fillna(0)
    df_trade = df_trade_all[assets]
    listing_dates = {asset: df_trade_all_raw[asset].first_valid_index() for asset in assets}

    signal_on_trade_dates = index_signal.reindex(df_trade.index, method="ffill")
    first_signal_date = index_signal.first_valid_index()
    if first_signal_date is None:
        raise ValueError("股指期货信号文件没有有效信号")

    calendar_days = df_trade_all.index.to_series().diff().dt.days.fillna(1)
    repo_shifted = repo_rate_ann.shift(1).fillna(0)
    repo_net_yield = np.maximum((repo_shifted / 365.0) * calendar_days - v016.REPO_FEE_RATE, 0.0)

    m_ratios = pd.Series({asset: v016.MARGIN_RATIOS.get(asset, 1.0) for asset in assets})
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
            asset
            for asset in risk_parity_assets
            if listing_dates.get(asset) is not None and listing_dates[asset] <= reb
        ]
        if len(eligible_rp_assets) == 0:
            continue

        lookback = df_weight.loc[reb - pd.DateOffset(months=12) : reb, eligible_rp_assets]
        if len(lookback) < 150:
            continue

        index_target = v016.allocate_index_futures(raw_signal, assets, listing_dates, reb)
        index_weight = float(index_target.sum())
        remaining_weight = max(0.0, 1.0 - index_weight)

        rp_active = v016.get_risk_parity_weights(v016.calculate_ewma_semi_cov(lookback, v016.EWMA_DECAY))
        target = pd.Series(0.0, index=assets)
        target.loc[index_target.index] = index_target
        target.loc[eligible_rp_assets] = rp_active * remaining_weight

        next_week = df_trade.loc[reb + pd.Timedelta(days=1) : week_ends[i + 1]]
        if len(next_week) == 0:
            continue
        if first_date is None:
            first_date = next_week.index[0]

        for date, daily_return in next_week.iterrows():
            daily_repo = repo_net_yield.loc[date]

            if date == next_week.index[0]:
                new_margin = (target * m_ratios).sum()
                idle_cash = max(0.0, 1.0 - new_margin)
                idle_return = idle_cash * daily_repo

                cost = (target - curr_w).abs().sum() * v016.FEE_RATE
                ret_series.loc[date] = (target * daily_return).sum() - cost + idle_return

                curr_w = target.copy()
                weight_recs.append(
                    {
                        "date": reb,
                        "策略名称": v016.STRATEGY_NAME,
                        "股指期货信号": float(raw_signal),
                        "股指期货仓位": index_weight,
                        **{asset: target.loc[asset] for asset in assets},
                    }
                )
            else:
                idle_cash = max(0.0, 1.0 - curr_margin)
                idle_return = idle_cash * daily_repo
                ret_series.loc[date] = (curr_w * daily_return).sum() + idle_return

            gross_weight = (curr_w * (1 + daily_return)).sum()
            curr_w = (curr_w * (1 + daily_return)) / (gross_weight or 1)
            curr_margin = (curr_w * m_ratios).sum()
            margin_series.loc[date] = curr_margin

    if first_date is None:
        raise ValueError("日期或数据不满足条件")

    df_navs = pd.DataFrame(index=df_trade.loc[first_date:].index)
    df_navs[v016.STRATEGY_NAME] = (1 + ret_series.loc[first_date:]).cumprod()

    all_metrics = []

    def append_metrics(period_label: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> None:
        for asset in assets:
            metrics = v016.calculate_metrics(df_trade.loc[start_date:end_date, asset])
            metrics["回测区间"] = period_label
            metrics["组合/资产"] = asset
            all_metrics.append(metrics)

        metrics = v016.calculate_metrics(ret_series.loc[start_date:end_date], margin_series.loc[start_date:end_date])
        metrics["回测区间"] = period_label
        metrics["组合/资产"] = v016.STRATEGY_NAME
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
    cols_order = ["回测区间", "组合/资产", "年化收益", "年化波动", "夏普比率", "最大回撤", "月度胜率", "平均资金占用"]
    cols_order = [col for col in cols_order if col in df_metrics.columns]
    df_metrics = df_metrics[cols_order]

    df_weekly_weights = pd.DataFrame(weight_recs)
    weight_cols = ["date", "策略名称", "股指期货信号", "股指期货仓位"] + assets
    df_weekly_weights = df_weekly_weights[weight_cols]

    return BacktestResult(
        as_of_date=as_of_date,
        first_date=pd.Timestamp(first_date),
        assets=assets,
        df_navs=df_navs,
        df_metrics=df_metrics,
        df_weekly_weights=df_weekly_weights,
        df_trade=df_trade,
    )


def build_position_dataframe(report: TargetReport) -> pd.DataFrame:
    margin_usage = report.target_weights * report.margin_ratios
    report_type = "新调仓目标" if report.is_new_observation else "当前有效仓位"
    rows = []
    for asset in report.target_weights.index:
        rows.append(
            {
                "日期": report.as_of_date.strftime("%Y-%m-%d"),
                "观察日": report.observation_date.strftime("%Y-%m-%d"),
                "报告类型": report_type,
                "是否新调仓": report.is_new_observation,
                "股指期货信号": report.raw_signal,
                "股指期货仓位": report.index_weight,
                "资产": asset,
                "目标权重": report.target_weights.loc[asset],
                "保证金比例": report.margin_ratios.loc[asset],
                "资金占用": margin_usage.loc[asset],
            }
        )
    return pd.DataFrame(rows)


def _period_return(df_navs: pd.DataFrame, start_date: pd.Timestamp, strategy_name: str) -> str:
    nav = df_navs[strategy_name].dropna()
    if nav.empty:
        return "0.00%"
    subset = nav.loc[nav.index >= start_date]
    if len(subset) < 2:
        return "0.00%"
    return f"{subset.iloc[-1] / subset.iloc[0] - 1:.2%}"


def build_summary_markdown(
    report: TargetReport,
    positions: pd.DataFrame,
    backtest_result: BacktestResult,
    paths: dict[str, Path],
) -> str:
    report_type = "新调仓目标" if report.is_new_observation else "当前有效仓位"
    weight_sum = positions["目标权重"].sum()
    margin_sum = positions["资金占用"].sum()
    top_positions = positions.sort_values("目标权重", ascending=False).head(8)
    strategy_name = "风险平价策略"
    strategy_metrics = backtest_result.df_metrics[
        (backtest_result.df_metrics["回测区间"] == "全局 (Total)")
        & (backtest_result.df_metrics["组合/资产"] == strategy_name)
    ]
    global_metrics = strategy_metrics.iloc[0].to_dict() if not strategy_metrics.empty else {}
    nav = backtest_result.df_navs[strategy_name].dropna()
    ending_nav = nav.iloc[-1] if len(nav) else 0.0
    latest_date = backtest_result.df_navs.index[-1]
    one_month = _period_return(backtest_result.df_navs, latest_date - pd.DateOffset(months=1), strategy_name)
    three_month = _period_return(backtest_result.df_navs, latest_date - pd.DateOffset(months=3), strategy_name)
    ytd = _period_return(backtest_result.df_navs, pd.Timestamp(year=latest_date.year, month=1, day=1), strategy_name)

    lines = [
        "# 每日更新策略回测报告",
        "",
        f"- 策略数据日期：`{report.as_of_date:%Y-%m-%d}`",
        f"- 回测区间：`{backtest_result.first_date:%Y-%m-%d}` 至 `{latest_date:%Y-%m-%d}`",
        f"- 观察日：`{report.observation_date:%Y-%m-%d}`",
        f"- 报告类型：`{report_type}`",
        f"- 股指期货信号：`{report.raw_signal:.4f}`",
        f"- 股指期货仓位：`{report.index_weight:.2%}`",
        f"- 权重合计：`{weight_sum:.6f}`",
        f"- 资金占用合计：`{margin_sum:.2%}`",
        "",
        "## 全局核心指标",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| 期末净值 | {ending_nav:.4f} |",
        f"| 年化收益 | {global_metrics.get('年化收益', '')} |",
        f"| 年化波动 | {global_metrics.get('年化波动', '')} |",
        f"| 夏普比率 | {global_metrics.get('夏普比率', '')} |",
        f"| 最大回撤 | {global_metrics.get('最大回撤', '')} |",
        f"| 月度胜率 | {global_metrics.get('月度胜率', '')} |",
        f"| 平均资金占用 | {global_metrics.get('平均资金占用', '')} |",
        "",
        "## 近期表现",
        "",
        "| 区间 | 收益 |",
        "| --- | ---: |",
        f"| 近1月 | {one_month} |",
        f"| 近3月 | {three_month} |",
        f"| 年初至今 | {ytd} |",
        "",
        "## 主要仓位",
        "",
        "| 资产 | 目标权重 | 资金占用 |",
        "| --- | ---: | ---: |",
    ]
    for _, row in top_positions.iterrows():
        lines.append(f"| {row['资产']} | {row['目标权重']:.2%} | {row['资金占用']:.2%} |")
    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            "| 类型 | 路径 |",
            "| --- | --- |",
            f"| 仓位 | `{paths['positions']}` |",
            f"| 净值 | `{paths['nav']}` |",
            f"| 指标 | `{paths['metrics']}` |",
            f"| 周度仓位明细 | `{paths['weekly_weights']}` |",
            f"| 图表 | `{paths['chart']}` |",
            f"| 报告 | `{paths['report']}` |",
        ]
    )
    return "\n".join(lines) + "\n"


def render_backtest_chart(backtest_result: BacktestResult, chart_path: Path) -> None:
    v016 = load_v016_module()
    df_navs = backtest_result.df_navs
    df_weekly_weights = backtest_result.df_weekly_weights
    df_trade = backtest_result.df_trade
    first_date = backtest_result.first_date
    assets = backtest_result.assets

    fig, axes = plt.subplots(3, 1, figsize=(16, 16), sharex=False)

    axes[0].plot(df_navs.index, df_navs[v016.STRATEGY_NAME], label=v016.STRATEGY_NAME, color="purple", lw=2)
    if df_trade is not None and "沪深300主连" in df_trade.columns:
        axes[0].plot((1 + df_trade.loc[first_date:, "沪深300主连"]).cumprod(), label="沪深300主连", color="blue", alpha=0.3)
    if df_trade is not None and "10年国债主连" in df_trade.columns:
        axes[0].plot((1 + df_trade.loc[first_date:, "10年国债主连"]).cumprod(), label="10年国债主连", color="green", alpha=0.3)
    axes[0].set_title("策略累计净值走势", fontsize=14)
    axes[0].legend(loc="upper left")
    axes[0].grid(True, ls="--", alpha=0.5)

    df_weights = df_weekly_weights.set_index("date")
    df_classes = pd.DataFrame(
        {
            class_name: df_weights[[asset for asset in class_assets if asset in assets]].sum(axis=1)
            for class_name, class_assets in v016.PLOT_ASSET_CLASSES.items()
        }
    )
    axes[1].stackplot(
        df_classes.index,
        df_classes.T,
        labels=df_classes.columns,
        alpha=0.8,
        colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
    )
    axes[1].set_title("周度大类资产权重", fontsize=14)
    axes[1].set_ylim(0, 1)
    axes[1].yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    axes[1].legend(loc="upper left")
    axes[1].grid(True, ls="--", alpha=0.4)

    axes[2].plot(df_weights.index, df_weights["股指期货信号"], label="股指期货信号", color="black", lw=1.5)
    axes[2].plot(df_weights.index, df_weights["股指期货仓位"], label="股指期货仓位", color="blue", lw=1.5)
    axes[2].set_title("股指期货信号与仓位", fontsize=14)
    axes[2].set_ylim(-1.1, 1.1)
    axes[2].yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    axes[2].legend(loc="upper left")
    axes[2].grid(True, ls="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(str(chart_path), dpi=300)
    plt.close(fig)


def write_report(
    report: TargetReport,
    backtest_result: BacktestResult,
    output_dir: Path = OUTPUT_DIR,
    render_chart: bool = True,
) -> dict[str, Path]:
    paths = output_paths(output_dir, report.as_of_date)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    positions = build_position_dataframe(report)
    positions.to_csv(paths["positions"], index=False, encoding="utf-8-sig", float_format="%.10f")
    backtest_result.df_navs.to_csv(paths["nav"], encoding="utf-8-sig")
    backtest_result.df_metrics.to_csv(paths["metrics"], index=False, encoding="utf-8-sig")
    backtest_result.df_weekly_weights.to_csv(paths["weekly_weights"], index=False, encoding="utf-8-sig")
    if render_chart:
        render_backtest_chart(backtest_result, paths["chart"])
    else:
        paths["chart"].write_bytes(b"")
    paths["report"].write_text(build_summary_markdown(report, positions, backtest_result, paths), encoding="utf-8-sig")
    return paths


def run(argv: list[str] | None = None, runner: Runner = subprocess.run) -> int:
    args = parse_args(argv)
    update_status = maybe_run_data_update(args, runner=runner)
    backtest_result = build_backtest_result()
    report = build_target_report(force_observation=args.force_observation)
    paths = write_report(report, backtest_result)
    print(f"data_update={update_status}")
    for name, path in paths.items():
        print(f"{name}_file={path}")
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
