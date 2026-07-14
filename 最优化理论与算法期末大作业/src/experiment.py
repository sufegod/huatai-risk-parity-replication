from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from .backtest import BacktestConfig, calculate_performance_metrics, run_backtest
from .optimizer_evolution import evaluate_optimizer_evolution
from .report import build_html_and_pdf
from .risk_parity import estimate_covariance, risk_contributions, solve_erc


STRATEGY_LABELS = {
    "equal_weight": "等权组合",
    "inverse_downside_vol": "逆下行波动率",
    "erc": "风险平价",
}
SOLVER_LABELS = {"newton": "阻尼牛顿法", "lbfgsb": "L-BFGS-B", "slsqp": "SLSQP"}
ESTIMATOR_LABELS = {
    "sample": "样本协方差",
    "ewma_full": "EWMA全协方差",
    "ewma_semi": "EWMA半协方差",
}
PALETTE = {
    "blue": "#1F5A94",
    "blue_light": "#A9C4DE",
    "gold": "#C28B2C",
    "orange": "#D36C32",
    "olive": "#7A8445",
    "pink": "#B45A75",
    "ink": "#25313C",
    "gray": "#8B949E",
    "grid": "#D9DEE3",
}
SOURCE_NOTE = "数据来源：项目内 ETF风险平价回测数据.xlsx；作者计算"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    return value


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _load_config(config_path: Path) -> dict[str, Any]:
    return json.loads(config_path.read_text(encoding="utf-8"))


def _course_paths(course_dir: Path) -> dict[str, Path]:
    output = course_dir / "output"
    return {
        "course": course_dir,
        "repo": course_dir.parent,
        "data": course_dir / "data",
        "tables": output / "tables",
        "figures": output / "figures",
        "html": output / "html",
        "pdf": output / "pdf",
        "tmp_pdf": course_dir / "tmp" / "pdfs",
    }


def _ensure_output_dirs(paths: dict[str, Path]) -> None:
    for key in ("data", "tables", "figures", "html", "pdf", "tmp_pdf"):
        paths[key].mkdir(parents=True, exist_ok=True)


def load_and_clean_returns(paths: dict[str, Path], config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    source = paths["repo"] / Path(config["source_excel"])
    raw = pd.read_excel(source, sheet_name=int(config["source_sheet_index"]), index_col=0, parse_dates=True)
    raw.index = pd.to_datetime(raw.index, errors="coerce")
    raw = raw.loc[raw.index.notna()].sort_index()
    raw = raw.apply(pd.to_numeric, errors="coerce")

    duplicate_dates = int(raw.index.duplicated().sum())
    exact_duplicates = int(raw.reset_index().duplicated().sum())
    missing_before = raw.isna().sum()
    if duplicate_dates or exact_duplicates:
        raise ValueError("source data contains duplicate dates or rows")
    if not raw.index.is_monotonic_increasing:
        raise ValueError("source dates are not ordered")
    if raw.shape[1] != 9:
        raise ValueError(f"expected 9 assets, found {raw.shape[1]}")

    cleaned = raw.fillna(0.0) / 100.0
    if not np.isfinite(cleaned.to_numpy()).all():
        raise ValueError("cleaned returns contains non-finite values")
    cleaned.index.name = "date"
    cleaned.to_csv(paths["data"] / "etf_returns.csv", encoding="utf-8-sig", float_format="%.10f")

    profile = pd.DataFrame(
        {
            "asset": cleaned.columns,
            "missing_before_fill": missing_before.reindex(cleaned.columns).to_numpy(dtype=int),
            "min_daily_return": cleaned.min().to_numpy(),
            "max_daily_return": cleaned.max().to_numpy(),
            "mean_daily_return": cleaned.mean().to_numpy(),
            "daily_volatility": cleaned.std(ddof=1).to_numpy(),
            "first_date": str(cleaned.index.min().date()),
            "last_date": str(cleaned.index.max().date()),
            "observations": len(cleaned),
        }
    )
    profile.to_csv(paths["tables"] / "data_quality_profile.csv", index=False, encoding="utf-8-sig")
    summary = {
        "source": config["source_excel"],
        "sheet_index": int(config["source_sheet_index"]),
        "rows": int(cleaned.shape[0]),
        "assets": int(cleaned.shape[1]),
        "start_date": str(cleaned.index.min().date()),
        "end_date": str(cleaned.index.max().date()),
        "duplicate_dates": duplicate_dates,
        "exact_duplicate_rows": exact_duplicates,
        "missing_cells_before_fill": int(missing_before.sum()),
        "missing_cells_after_fill": int(cleaned.isna().sum().sum()),
        "min_return": float(cleaned.min().min()),
        "max_return": float(cleaned.max().max()),
        "date_monotonic": bool(cleaned.index.is_monotonic_increasing),
        "fill_rule": "three isolated missing asset-day returns filled with zero",
        "known_limitation": "pre-ETF-history uses project-provided index proxies",
    }
    _json_dump(paths["tables"] / "data_quality_summary.json", summary)
    return cleaned, profile, summary


def _month_end_positions(index: pd.DatetimeIndex) -> list[int]:
    positions = pd.Series(np.arange(len(index)), index=index)
    return positions.groupby(index.to_period("M")).last().astype(int).tolist()


def run_optimizer_evolution(
    returns: pd.DataFrame,
    paths: dict[str, Path],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    details, summary = evaluate_optimizer_evolution(
        returns,
        window=int(config["window"]),
        decay=float(config["decay"]),
        ridge=float(config["ridge"]),
        solver_tol=float(config["solver_tol"]),
        solver_max_iter=int(config["solver_max_iter"]),
    )
    details.to_csv(paths["tables"] / "optimizer_evolution_details.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(paths["tables"] / "optimizer_evolution_summary.csv", index=False, encoding="utf-8-sig")
    return details, summary


def run_solver_comparison(
    returns: pd.DataFrame,
    paths: dict[str, Path],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    window = int(config["window"])
    methods = ["newton", "lbfgsb", "slsqp"]
    records: list[dict[str, Any]] = []
    representative_covariance: np.ndarray | None = None
    representative_date: pd.Timestamp | None = None
    representative_cutoff = pd.Timestamp("2025-12-31")

    for position in _month_end_positions(returns.index):
        if position < window - 1:
            continue
        observation_date = returns.index[position]
        history = returns.iloc[position - window + 1 : position + 1]
        covariance = estimate_covariance(
            history,
            method="ewma_semi",
            decay=float(config["decay"]),
            ridge=float(config["ridge"]),
        )
        if observation_date <= representative_cutoff:
            representative_covariance = covariance
            representative_date = observation_date
        for method in methods:
            result = solve_erc(
                covariance,
                method=method,
                tol=float(config["solver_tol"]),
                max_iter=int(config["solver_max_iter"]),
            )
            records.append(
                {
                    "observation_date": observation_date,
                    "method": method,
                    "success": result.success,
                    "iterations": result.iterations,
                    "runtime_ms": result.runtime_ms,
                    "objective": result.objective,
                    "gradient_norm": result.gradient_norm,
                    "rc_max_error": result.rc_max_error,
                    "weight_sum_error": result.weight_sum_error,
                    "min_weight": result.min_weight,
                    "status": result.status,
                }
            )

    details = pd.DataFrame(records)
    details.to_csv(paths["tables"] / "algorithm_monthly_details.csv", index=False, encoding="utf-8-sig")
    summary = (
        details.groupby("method", as_index=False)
        .agg(
            observations=("observation_date", "count"),
            success_rate=("success", "mean"),
            median_iterations=("iterations", "median"),
            median_runtime_ms=("runtime_ms", "median"),
            p95_runtime_ms=("runtime_ms", lambda s: s.quantile(0.95)),
            median_rc_error=("rc_max_error", "median"),
            max_rc_error=("rc_max_error", "max"),
            max_weight_sum_error=("weight_sum_error", "max"),
        )
        .sort_values("method")
    )
    summary.to_csv(paths["tables"] / "algorithm_summary.csv", index=False, encoding="utf-8-sig")

    if representative_covariance is None or representative_date is None:
        raise RuntimeError("representative covariance was not created")
    convergence_records: list[dict[str, Any]] = []
    representative_results = {}
    for method in methods:
        result = solve_erc(
            representative_covariance,
            method=method,
            tol=float(config["solver_tol"]),
            max_iter=int(config["solver_max_iter"]),
        )
        representative_results[method] = result
        for item in result.history:
            convergence_records.append({"method": method, "date": representative_date, **item})
    convergence = pd.DataFrame(convergence_records)
    convergence.to_csv(paths["tables"] / "representative_convergence.csv", index=False, encoding="utf-8-sig")

    primary_result = representative_results["newton"]
    rc = risk_contributions(primary_result.weights, representative_covariance)
    optimal = pd.DataFrame(
        {
            "asset": returns.columns,
            "weight": primary_result.weights,
            "risk_contribution": rc,
            "target_risk_contribution": np.full(len(returns.columns), 1.0 / len(returns.columns)),
            "representative_date": representative_date,
        }
    )
    optimal.to_csv(paths["tables"] / "representative_optimal_solution.csv", index=False, encoding="utf-8-sig")
    return details, summary, convergence, optimal


def run_risk_budget_extension(
    returns: pd.DataFrame,
    paths: dict[str, Path],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    window = int(config["window"])
    representative_cutoff = pd.Timestamp("2025-12-31")
    positions = [
        position
        for position in _month_end_positions(returns.index)
        if position >= window - 1 and returns.index[position] <= representative_cutoff
    ]
    if not positions:
        raise RuntimeError("no representative window is available for the risk-budget extension")
    position = positions[-1]
    representative_date = returns.index[position]
    history = returns.iloc[position - window + 1 : position + 1]
    covariance = estimate_covariance(
        history,
        method="ewma_semi",
        decay=float(config["decay"]),
        ridge=float(config["ridge"]),
    )

    tilted_assets = ["沪深300ETF", "中证1000ETF"]
    missing_assets = [asset for asset in tilted_assets if asset not in returns.columns]
    if missing_assets:
        raise ValueError(f"risk-budget extension is missing assets: {missing_assets}")
    raw_budget = pd.Series(1.0, index=returns.columns, dtype=float)
    raw_budget.loc[tilted_assets] = 2.0
    target_budget = raw_budget / raw_budget.sum()
    result = solve_erc(
        covariance,
        method="newton",
        risk_budget=target_budget.to_numpy(),
        tol=float(config["solver_tol"]),
        max_iter=int(config["solver_max_iter"]),
    )
    realized = risk_contributions(result.weights, covariance)
    extension = pd.DataFrame(
        {
            "asset": returns.columns,
            "raw_budget_multiplier": raw_budget.to_numpy(),
            "target_risk_budget": target_budget.to_numpy(),
            "actual_risk_contribution": realized,
            "weight": result.weights,
            "absolute_rc_error": np.abs(realized - target_budget.to_numpy()),
            "representative_date": representative_date,
        }
    )
    extension.to_csv(paths["tables"] / "risk_budget_extension.csv", index=False, encoding="utf-8-sig")
    summary = {
        "representative_date": str(representative_date.date()),
        "tilted_assets": tilted_assets,
        "tilted_multiplier": 2.0,
        "other_multiplier": 1.0,
        "solver_success": bool(result.success),
        "iterations": int(result.iterations),
        "rc_max_error": float(extension["absolute_rc_error"].max()),
        "weight_sum_error": float(abs(extension["weight"].sum() - 1.0)),
    }
    _json_dump(paths["tables"] / "risk_budget_extension_summary.json", summary)
    return extension, summary


def run_stress_tests(paths: dict[str, Path], config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(config["random_seed"]))
    methods = ["newton", "lbfgsb", "slsqp"]
    conditions = [1e2, 1e4, 1e6, 1e8]
    records: list[dict[str, Any]] = []
    n_assets = 9
    for condition in conditions:
        for replication in range(20):
            q, _ = np.linalg.qr(rng.normal(size=(n_assets, n_assets)))
            eigenvalues = np.geomspace(0.04, 0.04 / condition, n_assets)
            covariance = q @ np.diag(eigenvalues) @ q.T + np.eye(n_assets) * 1e-12
            for method in methods:
                result = solve_erc(
                    covariance,
                    method=method,
                    tol=float(config["solver_tol"]),
                    max_iter=int(config["solver_max_iter"]),
                )
                records.append(
                    {
                        "condition_number": condition,
                        "replication": replication,
                        "method": method,
                        "success": result.success,
                        "iterations": result.iterations,
                        "runtime_ms": result.runtime_ms,
                        "rc_max_error": result.rc_max_error,
                        "weight_sum_error": result.weight_sum_error,
                    }
                )
    details = pd.DataFrame(records)
    details.to_csv(paths["tables"] / "stress_test_details.csv", index=False, encoding="utf-8-sig")
    summary = (
        details.groupby(["condition_number", "method"], as_index=False)
        .agg(
            success_rate=("success", "mean"),
            median_iterations=("iterations", "median"),
            median_runtime_ms=("runtime_ms", "median"),
            median_rc_error=("rc_max_error", "median"),
            max_rc_error=("rc_max_error", "max"),
        )
        .sort_values(["condition_number", "method"])
    )
    summary.to_csv(paths["tables"] / "stress_test_summary.csv", index=False, encoding="utf-8-sig")
    return details, summary


def _base_backtest_config(config: dict[str, Any]) -> BacktestConfig:
    return BacktestConfig(
        covariance_method="ewma_semi",
        window=int(config["window"]),
        decay=float(config["decay"]),
        ridge=float(config["ridge"]),
        fee_rate=float(config["fee_rate"]),
        solver="newton",
        solver_tol=float(config["solver_tol"]),
        solver_max_iter=int(config["solver_max_iter"]),
        train_start=config["train_start"],
        train_end=config["train_end"],
        validation_start=config["validation_start"],
        validation_end=config["validation_end"],
    )


def run_strategy_backtests(
    returns: pd.DataFrame,
    paths: dict[str, Path],
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = _base_backtest_config(config)
    results = {
        strategy: run_backtest(returns, replace(base, strategy=strategy))
        for strategy in ("equal_weight", "inverse_downside_vol", "erc")
    }
    nav = pd.concat({key: value.nav for key, value in results.items()}, axis=1)
    nav.to_csv(paths["tables"] / "strategy_nav.csv", encoding="utf-8-sig")
    metrics = pd.concat(
        [value.metrics.assign(strategy=key) for key, value in results.items()], ignore_index=True
    )
    metrics = metrics[["strategy", "period"] + [c for c in metrics.columns if c not in {"strategy", "period"}]]
    metrics.to_csv(paths["tables"] / "strategy_metrics.csv", index=False, encoding="utf-8-sig")

    erc = results["erc"]
    erc.target_weights.to_csv(paths["tables"] / "erc_target_weights.csv", encoding="utf-8-sig")
    erc.solver_diagnostics.to_csv(paths["tables"] / "erc_solver_diagnostics.csv", encoding="utf-8-sig")

    yearly_records: list[dict[str, Any]] = []
    for strategy, result in results.items():
        for year, group in result.returns.groupby(result.returns.index.year):
            metrics_year = calculate_performance_metrics(group)
            yearly_records.append({"strategy": strategy, "year": int(year), **metrics_year})
    yearly = pd.DataFrame(yearly_records)
    yearly.to_csv(paths["tables"] / "yearly_metrics.csv", index=False, encoding="utf-8-sig")
    return results, nav, metrics, yearly


def run_estimator_comparison(
    returns: pd.DataFrame,
    paths: dict[str, Path],
    config: dict[str, Any],
) -> pd.DataFrame:
    base = _base_backtest_config(config)
    records: list[pd.DataFrame] = []
    for estimator in ("sample", "ewma_full", "ewma_semi"):
        result = run_backtest(returns, replace(base, strategy="erc", covariance_method=estimator))
        records.append(result.metrics.assign(estimator=estimator))
    comparison = pd.concat(records, ignore_index=True)
    comparison = comparison[["estimator", "period"] + [c for c in comparison.columns if c not in {"estimator", "period"}]]
    comparison.to_csv(paths["tables"] / "estimator_comparison.csv", index=False, encoding="utf-8-sig")
    return comparison


def run_parameter_sensitivity(
    returns: pd.DataFrame,
    paths: dict[str, Path],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    base = _base_backtest_config(config)
    records: list[pd.DataFrame] = []
    for decay in (0.90, 0.94, 0.97, 0.99):
        for window in (126, 252, 504):
            result = run_backtest(
                returns,
                replace(
                    base,
                    strategy="erc",
                    covariance_method="ewma_semi",
                    decay=decay,
                    window=window,
                ),
            )
            records.append(result.metrics.assign(decay=decay, window=window))
    sensitivity = pd.concat(records, ignore_index=True)
    sensitivity = sensitivity[["decay", "window", "period"] + [c for c in sensitivity.columns if c not in {"decay", "window", "period"}]]
    sensitivity.to_csv(paths["tables"] / "parameter_sensitivity.csv", index=False, encoding="utf-8-sig")
    train = sensitivity.loc[sensitivity["period"] == "train"].sort_values(
        ["sharpe", "max_drawdown"], ascending=[False, False]
    )
    selected_row = train.iloc[0]
    selected = {
        "selection_period": "train",
        "selection_rule": "highest Sharpe, then less severe maximum drawdown",
        "decay": float(selected_row["decay"]),
        "window": int(selected_row["window"]),
        "train_sharpe": float(selected_row["sharpe"]),
        "train_max_drawdown": float(selected_row["max_drawdown"]),
    }
    validation = sensitivity.loc[
        (sensitivity["period"] == "validation")
        & (sensitivity["decay"] == selected["decay"])
        & (sensitivity["window"] == selected["window"])
    ].iloc[0]
    selected["validation_sharpe"] = float(validation["sharpe"])
    selected["validation_max_drawdown"] = float(validation["max_drawdown"])
    _json_dump(paths["tables"] / "selected_parameter.json", selected)
    return sensitivity, selected


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": PALETTE["ink"],
            "axes.labelcolor": PALETTE["ink"],
            "text.color": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.7,
            "grid.alpha": 0.65,
            "savefig.dpi": 180,
        }
    )


def _finish_figure(fig: plt.Figure, path: Path, subtitle: str) -> None:
    fig.text(0.01, 0.965, subtitle, fontsize=9, color=PALETTE["gray"], va="top")
    fig.text(0.01, 0.008, SOURCE_NOTE, fontsize=7.5, color=PALETTE["gray"], va="bottom")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_static_figures(
    paths: dict[str, Path],
    nav: pd.DataFrame,
    metrics: pd.DataFrame,
    yearly: pd.DataFrame,
    optimizer_evolution: pd.DataFrame,
    algorithm_summary: pd.DataFrame,
    convergence: pd.DataFrame,
    optimal: pd.DataFrame,
    risk_budget_extension: pd.DataFrame,
    stress_summary: pd.DataFrame,
    estimator: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> list[dict[str, str]]:
    _configure_matplotlib()
    figures = paths["figures"]
    chart_map: list[dict[str, str]] = []

    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    colors = [PALETTE["gray"], PALETTE["gold"], PALETTE["blue"]]
    styles = ["--", "-.", "-"]
    for column, color, style in zip(nav.columns, colors, styles):
        ax.plot(nav.index, nav[column], label=STRATEGY_LABELS[column], color=color, linestyle=style, linewidth=2.0)
    ax.set_title("三类资产配置策略累计净值")
    ax.set_ylabel("累计净值")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    _finish_figure(fig, figures / "strategy_nav.png", "2014-01 至 2026-04；月末观察、下一交易日调仓，含单边 5bp 成本")
    chart_map.append({"section": "实证结果", "question": "风险平价是否改善风险调整后收益", "family": "Trend", "type": "multi-series line", "path": "output/figures/strategy_nav.png"})

    evolution = optimizer_evolution.sort_values("stage_order")
    evolution_labels = [label.replace(" ", "\n", 1) for label in evolution["label"]]
    evolution_colors = [PALETTE["gray"], PALETTE["blue_light"], PALETTE["gold"], PALETTE["orange"], PALETTE["blue"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    bars = axes[0].bar(
        evolution_labels,
        evolution["rc_pass_rate"],
        color=evolution_colors,
        edgecolor=PALETTE["ink"],
    )
    axes[0].set_ylim(0, 1.08)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].set_title("RC误差验收率")
    axes[0].set_ylabel("最大RC误差不超过1e-6")
    axes[0].bar_label(bars, labels=[f"{value:.0%}" for value in evolution["rc_pass_rate"]], padding=3, fontsize=8)
    axes[1].bar(
        evolution_labels,
        evolution["median_rc_error"].clip(lower=1e-16),
        color=evolution_colors,
        edgecolor=PALETTE["ink"],
    )
    axes[1].axhline(1e-6, color=PALETTE["ink"], linestyle=":", linewidth=1.2, label="验收阈值1e-6")
    axes[1].set_yscale("log")
    axes[1].set_title("中位风险贡献误差")
    axes[1].set_ylabel("绝对误差（对数轴）")
    axes[1].legend(frameon=False, loc="upper right")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", labelsize=8)
    _finish_figure(fig, figures / "optimizer_evolution.png", "148个相同滚动EWMA半协方差矩阵；仅改变目标函数与求解设置")
    chart_map.append({"section": "优化器演进", "question": "历史数值修补与凸重构分别解决了什么问题", "family": "Comparison", "type": "paired bars", "path": "output/figures/optimizer_evolution.png"})

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    order = ["newton", "lbfgsb", "slsqp"]
    view = algorithm_summary.set_index("method").loc[order]
    axes[0].bar([SOLVER_LABELS[x] for x in order], view["median_iterations"], color=[PALETTE["blue"], PALETTE["gold"], PALETTE["orange"]], edgecolor=PALETTE["ink"])
    axes[0].set_title("滚动窗口中位迭代次数")
    axes[0].set_ylabel("次")
    axes[1].bar([SOLVER_LABELS[x] for x in order], view["max_rc_error"].clip(lower=1e-16), color=[PALETTE["blue"], PALETTE["gold"], PALETTE["orange"]], edgecolor=PALETTE["ink"])
    axes[1].set_yscale("log")
    axes[1].set_title("最大风险贡献误差")
    axes[1].set_ylabel("绝对误差（对数轴）")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", visible=False)
    _finish_figure(fig, figures / "solver_summary.png", "全部月度 EWMA 半协方差矩阵；成功阈值为最大风险贡献误差不超过 1e-6")
    chart_map.append({"section": "算法比较", "question": "三类算法的效率和精度如何", "family": "Comparison", "type": "paired bars", "path": "output/figures/solver_summary.png"})

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for method, color, marker in [("newton", PALETTE["blue"], "o"), ("lbfgsb", PALETTE["gold"], "s"), ("slsqp", PALETTE["orange"], "^")]:
        subset = convergence.loc[convergence["method"] == method]
        ax.semilogy(subset["iteration"], subset["rc_max_error"].clip(lower=1e-16), label=SOLVER_LABELS[method], color=color, marker=marker, linewidth=1.8, markersize=4)
    ax.axhline(1e-6, color=PALETTE["ink"], linestyle=":", linewidth=1.2, label="验收阈值 1e-6")
    ax.set_title("代表窗口的风险贡献误差收敛")
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("最大风险贡献误差（对数轴）")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _finish_figure(fig, figures / "solver_convergence.png", "代表日期为不晚于 2025-12-31 的最后一个月末观察日")
    chart_map.append({"section": "算法比较", "question": "代表窗口中误差如何随迭代下降", "family": "Trend", "type": "semilog convergence", "path": "output/figures/solver_convergence.png"})

    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    x = np.arange(len(optimal))
    width = 0.38
    ax.bar(x - width / 2, optimal["weight"], width, label="组合权重", color=PALETTE["blue"], edgecolor=PALETTE["ink"])
    ax.bar(x + width / 2, optimal["risk_contribution"], width, label="风险贡献", color=PALETTE["gold"], edgecolor=PALETTE["ink"])
    ax.axhline(1.0 / len(optimal), color=PALETTE["ink"], linestyle=":", linewidth=1.2, label="目标风险贡献 11.11%")
    ax.set_xticks(x, optimal["asset"], rotation=28, ha="right")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("代表窗口的最优权重与风险贡献")
    ax.set_ylabel("占比")
    ax.legend(frameon=False, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", visible=False)
    fig.subplots_adjust(bottom=0.24)
    _finish_figure(fig, figures / "weights_risk_contributions.png", "权重不等于风险贡献；低波动债券获得较高名义权重")
    chart_map.append({"section": "最优解分析", "question": "不等权如何实现等风险贡献", "family": "Comparison", "type": "grouped bar", "path": "output/figures/weights_risk_contributions.png"})

    extension = risk_budget_extension.copy()
    x = np.arange(len(extension))
    width = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    axes[0].bar(x - width / 2, extension["target_risk_budget"], width, label="目标风险预算", color=PALETTE["blue_light"], edgecolor=PALETTE["ink"])
    axes[0].bar(x + width / 2, extension["actual_risk_contribution"], width, label="实际风险贡献", color=PALETTE["gold"], edgecolor=PALETTE["ink"])
    axes[0].set_title("一般风险预算跟踪")
    axes[0].set_ylabel("风险贡献占比")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(x, extension["weight"], color=PALETTE["blue"], edgecolor=PALETTE["ink"])
    axes[1].set_title("对应资金权重")
    axes[1].set_ylabel("组合权重")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    for ax in axes:
        ax.set_xticks(x, extension["asset"], rotation=32, ha="right")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", labelsize=7.5)
    fig.subplots_adjust(bottom=0.28)
    _finish_figure(fig, figures / "risk_budget_extension.png", "代表窗口截至2025-12-31；两只宽基股票ETF的原始风险预算倍率设为2，其余为1")
    chart_map.append({"section": "模型扩展", "question": "一般风险预算能否准确映射为实际风险贡献", "family": "Comparison", "type": "grouped bars", "path": "output/figures/risk_budget_extension.png"})

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for method, color, marker in [("newton", PALETTE["blue"], "o"), ("lbfgsb", PALETTE["gold"], "s"), ("slsqp", PALETTE["orange"], "^")]:
        subset = stress_summary.loc[stress_summary["method"] == method]
        ax.loglog(subset["condition_number"], subset["max_rc_error"].clip(lower=1e-16), label=SOLVER_LABELS[method], color=color, marker=marker, linewidth=1.8)
    ax.axhline(1e-6, color=PALETTE["ink"], linestyle=":", linewidth=1.2)
    ax.set_title("病态协方差矩阵压力测试")
    ax.set_xlabel("条件数（对数轴）")
    ax.set_ylabel("20 次实验最大风险贡献误差（对数轴）")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _finish_figure(fig, figures / "stress_test.png", "9维随机正定矩阵；条件数 1e2 至 1e8，每档固定种子重复 20 次")
    chart_map.append({"section": "稳健性", "question": "病态矩阵是否破坏求解精度", "family": "Uncertainty", "type": "log-log line", "path": "output/figures/stress_test.png"})

    validation_estimator = estimator.loc[estimator["period"] == "validation"].copy()
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar([ESTIMATOR_LABELS[x] for x in validation_estimator["estimator"]], validation_estimator["sharpe"], color=[PALETTE["gray"], PALETTE["gold"], PALETTE["blue"]], edgecolor=PALETTE["ink"])
    ax.set_title("不同风险估计方法的样本外夏普比率")
    ax.set_ylabel("夏普比率")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", visible=False)
    _finish_figure(fig, figures / "estimator_comparison.png", "验证期 2021-01 至 2026-04；其他回测规则保持一致")
    chart_map.append({"section": "风险估计", "question": "半协方差相对全协方差是否稳健", "family": "Comparison", "type": "bar", "path": "output/figures/estimator_comparison.png"})

    validation_sensitivity = sensitivity.loc[sensitivity["period"] == "validation"]
    pivot = validation_sensitivity.pivot(index="decay", columns="window", values="sharpe").sort_index()
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    image = ax.imshow(pivot.to_numpy(), cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)), [str(x) for x in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), [f"{x:.2f}" for x in pivot.index])
    ax.set_xlabel("回看窗口（交易日）")
    ax.set_ylabel("EWMA 衰减系数")
    ax.set_title("参数组合的样本外夏普比率")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value > np.nanmedian(pivot.to_numpy()) else PALETTE["ink"], fontsize=9)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.85)
    colorbar.set_label("夏普比率")
    ax.grid(False)
    _finish_figure(fig, figures / "sensitivity_heatmap.png", "验证期 2021-01 至 2026-04；参数选择只使用 2014-2020 训练期")
    chart_map.append({"section": "敏感性", "question": "结论是否依赖单点参数", "family": "Matrix", "type": "annotated heatmap", "path": "output/figures/sensitivity_heatmap.png"})

    validation_yearly = yearly.loc[yearly["year"] >= 2021].copy()
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    years = sorted(validation_yearly["year"].unique())
    width = 0.24
    for offset, strategy, color in [(-1, "equal_weight", PALETTE["gray"]), (0, "inverse_downside_vol", PALETTE["gold"]), (1, "erc", PALETTE["blue"])]:
        subset = validation_yearly.loc[validation_yearly["strategy"] == strategy].set_index("year").reindex(years)
        ax.bar(np.arange(len(years)) + offset * width, subset["cumulative_return"], width, label=STRATEGY_LABELS[strategy], color=color, edgecolor=PALETTE["ink"])
    ax.axhline(0, color=PALETTE["ink"], linewidth=0.9)
    ax.set_xticks(np.arange(len(years)), [str(y) for y in years])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("验证期分年度收益")
    ax.set_ylabel("年度累计收益")
    ax.legend(frameon=False, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", visible=False)
    _finish_figure(fig, figures / "yearly_returns.png", "2026 年仅统计至 4 月 3 日，不与完整年度直接比较")
    chart_map.append({"section": "情景分析", "question": "不同年份下策略表现是否一致", "family": "Comparison", "type": "grouped bars", "path": "output/figures/yearly_returns.png"})

    _json_dump(paths["tables"] / "chart_map.json", chart_map)
    return chart_map


def _headline_summary(
    metrics: pd.DataFrame,
    optimizer_evolution: pd.DataFrame,
    algorithm_summary: pd.DataFrame,
    stress_summary: pd.DataFrame,
    estimator: pd.DataFrame,
    selected: dict[str, Any],
    data_quality: dict[str, Any],
    risk_budget_extension: dict[str, Any],
) -> dict[str, Any]:
    validation = metrics.loc[(metrics["strategy"] == "erc") & (metrics["period"] == "validation")].iloc[0]
    equal_validation = metrics.loc[(metrics["strategy"] == "equal_weight") & (metrics["period"] == "validation")].iloc[0]
    newton = algorithm_summary.loc[algorithm_summary["method"] == "newton"].iloc[0]
    lbfgsb = algorithm_summary.loc[algorithm_summary["method"] == "lbfgsb"].iloc[0]
    slsqp = algorithm_summary.loc[algorithm_summary["method"] == "slsqp"].iloc[0]
    estimator_validation = estimator.loc[estimator["period"] == "validation"].set_index("estimator")
    worst_condition = stress_summary["condition_number"].max()
    newton_stress = stress_summary.loc[
        (stress_summary["method"] == "newton") & (stress_summary["condition_number"] == worst_condition)
    ].iloc[0]
    evolution_baseline = optimizer_evolution.loc[
        optimizer_evolution["variant"] == "v0_02_raw_slsqp"
    ].iloc[0]
    evolution_current = optimizer_evolution.loc[
        optimizer_evolution["variant"] == "course_newton"
    ].iloc[0]
    return {
        "validation_erc": validation.to_dict(),
        "validation_equal_weight": equal_validation.to_dict(),
        "newton_summary": newton.to_dict(),
        "lbfgsb_summary": lbfgsb.to_dict(),
        "slsqp_summary": slsqp.to_dict(),
        "stress_newton_1e8": newton_stress.to_dict(),
        "optimizer_evolution": {
            "baseline": evolution_baseline.to_dict(),
            "current": evolution_current.to_dict(),
        },
        "risk_budget_extension": risk_budget_extension,
        "estimator_validation": estimator_validation[["annual_return", "annual_volatility", "sharpe", "max_drawdown"]].to_dict(orient="index"),
        "selected_parameter": selected,
        "data_quality": data_quality,
    }


def generate_all_outputs(config_path: Path) -> None:
    config_path = Path(config_path).resolve()
    course_dir = config_path.parent
    paths = _course_paths(course_dir)
    _ensure_output_dirs(paths)
    config = _load_config(config_path)

    print("[1/9] 清洗数据并执行数据质量检查...")
    returns, _, data_quality = load_and_clean_returns(paths, config)
    print("[2/9] 复现历史优化器演进并统一执行RC误差验收...")
    _, optimizer_evolution = run_optimizer_evolution(returns, paths, config)
    print("[3/9] 比较当前阻尼牛顿法、L-BFGS-B 与 SLSQP...")
    _, algorithm_summary, convergence, optimal = run_solver_comparison(returns, paths, config)
    print("[4/9] 演示一般风险预算并执行病态矩阵压力测试...")
    risk_budget_extension, risk_budget_summary = run_risk_budget_extension(returns, paths, config)
    _, stress_summary = run_stress_tests(paths, config)
    print("[5/9] 回测等权、逆下行波动率与风险平价策略...")
    _, nav, metrics, yearly = run_strategy_backtests(returns, paths, config)
    print("[6/9] 比较风险估计方法并运行参数敏感性实验...")
    estimator = run_estimator_comparison(returns, paths, config)
    sensitivity, selected = run_parameter_sensitivity(returns, paths, config)
    print("[7/9] 生成静态图表...")
    chart_map = generate_static_figures(
        paths,
        nav,
        metrics,
        yearly,
        optimizer_evolution,
        algorithm_summary,
        convergence,
        optimal,
        risk_budget_extension,
        stress_summary,
        estimator,
        sensitivity,
    )
    summary = _headline_summary(
        metrics,
        optimizer_evolution,
        algorithm_summary,
        stress_summary,
        estimator,
        selected,
        data_quality,
        risk_budget_summary,
    )
    summary["chart_map"] = chart_map
    summary["config"] = config
    _json_dump(paths["tables"] / "analysis_summary.json", summary)
    print("[8/9] 构建自包含技术报告 HTML 并导出 PDF...")
    build_html_and_pdf(course_dir, config, summary)
    print("[9/9] 全部输出已生成。")
