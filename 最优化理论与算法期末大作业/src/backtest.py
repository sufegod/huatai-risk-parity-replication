from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .risk_parity import estimate_covariance, risk_contributions, solve_erc


ANNUALIZATION = 252


@dataclass(frozen=True)
class BacktestConfig:
    strategy: str = "erc"
    covariance_method: str = "ewma_semi"
    window: int = 252
    decay: float = 0.97
    ridge: float = 1e-8
    fee_rate: float = 0.0005
    solver: str = "newton"
    solver_tol: float = 1e-10
    solver_max_iter: int = 1000
    train_start: str = "2014-01-01"
    train_end: str = "2020-12-31"
    validation_start: str = "2021-01-01"
    validation_end: str = "2026-04-03"


@dataclass
class BacktestResult:
    config: BacktestConfig
    returns: pd.Series
    nav: pd.Series
    target_weights: pd.DataFrame
    turnover: pd.Series
    risk_contribution_error: pd.Series
    solver_diagnostics: pd.DataFrame
    metrics: pd.DataFrame


def _validate_returns(returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("returns index must be a DatetimeIndex")
    if returns.shape[1] < 2:
        raise ValueError("returns must contain at least two assets")
    if returns.index.duplicated().any() or not returns.index.is_monotonic_increasing:
        raise ValueError("returns dates must be unique and sorted")
    numeric = returns.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("returns contains missing or non-numeric values")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("returns contains non-finite values")
    return numeric.astype(float)


def _observation_positions(index: pd.DatetimeIndex) -> list[int]:
    positions = pd.Series(np.arange(len(index)), index=index)
    return positions.groupby(index.to_period("M")).last().astype(int).tolist()


def _target_weights(history: pd.DataFrame, config: BacktestConfig) -> tuple[np.ndarray, dict[str, float]]:
    n_assets = history.shape[1]
    if config.strategy == "equal_weight":
        return np.full(n_assets, 1.0 / n_assets), {
            "iterations": 0.0,
            "runtime_ms": 0.0,
            "success": 1.0,
            "rc_max_error": float("nan"),
        }

    covariance = estimate_covariance(
        history,
        method=config.covariance_method,
        decay=config.decay,
        ridge=config.ridge,
    )
    if config.strategy == "inverse_downside_vol":
        vol = np.sqrt(np.diag(covariance))
        inverse = 1.0 / np.maximum(vol, 1e-12)
        weights = inverse / inverse.sum()
        rc_error = float(
            np.max(np.abs(risk_contributions(weights, covariance) - np.full(n_assets, 1.0 / n_assets)))
        )
        return weights, {
            "iterations": 0.0,
            "runtime_ms": 0.0,
            "success": 1.0,
            "rc_max_error": rc_error,
        }
    if config.strategy != "erc":
        raise ValueError(f"unsupported strategy: {config.strategy}")

    result = solve_erc(
        covariance,
        method=config.solver,
        tol=config.solver_tol,
        max_iter=config.solver_max_iter,
    )
    return result.weights, {
        "iterations": float(result.iterations),
        "runtime_ms": result.runtime_ms,
        "success": float(result.success),
        "rc_max_error": result.rc_max_error,
    }


def calculate_performance_metrics(
    returns: pd.Series,
    turnover: pd.Series | None = None,
    target_weights: pd.DataFrame | None = None,
    rc_error: pd.Series | None = None,
) -> dict[str, float]:
    series = returns.dropna().astype(float)
    if series.empty:
        return {
            "observations": 0,
            "cumulative_return": float("nan"),
            "annual_return": float("nan"),
            "annual_volatility": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "calmar": float("nan"),
            "monthly_win_rate": float("nan"),
            "annual_turnover": float("nan"),
            "average_max_weight": float("nan"),
            "average_rc_error": float("nan"),
        }
    nav = (1.0 + series).cumprod()
    years = len(series) / ANNUALIZATION
    annual_return = float(nav.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    annual_volatility = float(series.std(ddof=1) * np.sqrt(ANNUALIZATION))
    sharpe = float(series.mean() * ANNUALIZATION / annual_volatility) if annual_volatility > 0 else float("nan")
    drawdown = nav / nav.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    calmar = float(annual_return / abs(max_drawdown)) if max_drawdown < 0 else float("nan")
    monthly = (1.0 + series).resample("ME").prod() - 1.0
    monthly_win_rate = float((monthly > 0).mean()) if len(monthly) else float("nan")
    annual_turnover = float(turnover.reindex(series.index, fill_value=0.0).sum() / years) if turnover is not None else float("nan")
    average_max_weight = (
        float(target_weights.loc[series.index.min() : series.index.max()].max(axis=1).mean())
        if target_weights is not None and not target_weights.empty
        else float("nan")
    )
    average_rc_error = (
        float(rc_error.loc[series.index.min() : series.index.max()].dropna().mean())
        if rc_error is not None and not rc_error.dropna().empty
        else float("nan")
    )
    return {
        "observations": int(len(series)),
        "cumulative_return": float(nav.iloc[-1] - 1.0),
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "monthly_win_rate": monthly_win_rate,
        "annual_turnover": annual_turnover,
        "average_max_weight": average_max_weight,
        "average_rc_error": average_rc_error,
    }


def _period_metrics(result: BacktestResult) -> pd.DataFrame:
    periods = {
        "total": (result.returns.index.min(), result.returns.index.max()),
        "train": (pd.Timestamp(result.config.train_start), pd.Timestamp(result.config.train_end)),
        "validation": (
            pd.Timestamp(result.config.validation_start),
            pd.Timestamp(result.config.validation_end),
        ),
    }
    records: list[dict[str, float | str]] = []
    for period, (start, end) in periods.items():
        sliced = result.returns.loc[start:end]
        metrics = calculate_performance_metrics(
            sliced,
            result.turnover,
            result.target_weights,
            result.risk_contribution_error,
        )
        records.append({"period": period, **metrics})
    return pd.DataFrame(records)


def run_backtest(returns: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    returns = _validate_returns(returns)
    if config.window < 20 or config.window >= len(returns):
        raise ValueError("window must be at least 20 and shorter than the dataset")
    observation_positions = _observation_positions(returns.index)
    execution_targets: dict[pd.Timestamp, np.ndarray] = {}
    target_records: list[dict[str, float | str | pd.Timestamp]] = []
    solver_records: list[dict[str, float | str | pd.Timestamp]] = []
    rc_records: dict[pd.Timestamp, float] = {}

    for position in observation_positions:
        if position < config.window - 1 or position + 1 >= len(returns):
            continue
        observation_date = returns.index[position]
        execution_date = returns.index[position + 1]
        history = returns.iloc[position - config.window + 1 : position + 1]
        weights, diagnostics = _target_weights(history, config)
        execution_targets[execution_date] = weights

        if config.strategy == "equal_weight":
            rc_error = float("nan")
        else:
            covariance = estimate_covariance(
                history,
                method=config.covariance_method,
                decay=config.decay,
                ridge=config.ridge,
            )
            rc_error = float(
                np.max(
                    np.abs(
                        risk_contributions(weights, covariance)
                        - np.full(returns.shape[1], 1.0 / returns.shape[1])
                    )
                )
            )
        rc_records[execution_date] = rc_error
        target_records.append(
            {
                "observation_date": observation_date,
                "execution_date": execution_date,
                **{asset: float(weight) for asset, weight in zip(returns.columns, weights)},
            }
        )
        solver_records.append(
            {
                "observation_date": observation_date,
                "execution_date": execution_date,
                **diagnostics,
            }
        )

    if not execution_targets:
        raise ValueError("no rebalance date has enough history")

    first_execution = min(execution_targets)
    portfolio_returns = pd.Series(0.0, index=returns.loc[first_execution:].index, name=config.strategy)
    turnover_series = pd.Series(0.0, index=portfolio_returns.index, name="turnover")
    current_weights = np.zeros(returns.shape[1], dtype=float)

    for date in portfolio_returns.index:
        transaction_cost = 0.0
        if date in execution_targets:
            target = execution_targets[date]
            turnover = float(np.abs(target - current_weights).sum())
            transaction_cost = config.fee_rate * turnover
            current_weights = target.copy()
            turnover_series.loc[date] = turnover

        daily_returns = returns.loc[date].to_numpy(dtype=float)
        portfolio_returns.loc[date] = float(current_weights @ daily_returns - transaction_cost)
        holdings = current_weights * (1.0 + daily_returns)
        gross = float(holdings.sum())
        if gross > 0:
            current_weights = holdings / gross

    target_weights = pd.DataFrame(target_records).set_index("execution_date")
    target_weights.index = pd.DatetimeIndex(target_weights.index)
    risk_error = pd.Series(rc_records, name="rc_max_error", dtype=float).sort_index()
    solver_diagnostics = pd.DataFrame(solver_records).set_index("execution_date")
    solver_diagnostics.index = pd.DatetimeIndex(solver_diagnostics.index)
    nav = (1.0 + portfolio_returns).cumprod().rename(config.strategy)

    result = BacktestResult(
        config=config,
        returns=portfolio_returns,
        nav=nav,
        target_weights=target_weights.drop(columns="observation_date"),
        turnover=turnover_series,
        risk_contribution_error=risk_error,
        solver_diagnostics=solver_diagnostics,
        metrics=pd.DataFrame(),
    )
    result.metrics = _period_metrics(result)
    return result

