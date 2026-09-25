from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .risk_parity import estimate_covariance, original_erc_objective, risk_contributions, solve_erc


RC_ACCEPTANCE_TOLERANCE = 1e-6

VARIANT_SPECS = (
    ("v0_02_raw_slsqp", "v0.02 原始SLSQP", "原始RC平方差", 1),
    ("v0_03_scaled_slsqp", "v0.03 目标放大", "目标乘1e9", 2),
    ("v0_04_relative_slsqp", "v0.04 相对误差", "无量纲相对RC误差", 3),
    ("v0_05_convex_lbfgsb", "v0.05 凸重构", "对数障碍凸目标", 4),
    ("course_newton", "当前阻尼牛顿法", "解析Hessian与线搜索", 5),
)
VARIANT_METADATA = {
    variant: {"label": label, "objective_family": objective_family, "stage_order": stage_order}
    for variant, label, objective_family, stage_order in VARIANT_SPECS
}
OPTIMIZER_VARIANTS = tuple(variant for variant, _, _, _ in VARIANT_SPECS)


@dataclass
class EvolutionSolverResult:
    variant: str
    label: str
    objective_family: str
    stage_order: int
    weights: np.ndarray
    solver_success: bool
    status: str
    iterations: int
    runtime_ms: float
    rc_max_error: float
    rc_pass: bool
    weight_sum_error: float
    min_weight: float
    distance_from_equal: float


def scaled_erc_objective(weights: np.ndarray, covariance: np.ndarray) -> float:
    """Reproduce v0.03: enlarge the original ERC objective by 1e9."""
    return original_erc_objective(weights, covariance, np.full(len(weights), 1.0 / len(weights))) * 1e9


def relative_erc_objective(weights: np.ndarray, covariance: np.ndarray) -> float:
    """Reproduce v0.04: minimize dimensionless relative RC errors."""
    portfolio_variance = float(weights @ covariance @ weights)
    if portfolio_variance < 1e-12:
        return 1e9
    contributions = risk_contributions(weights, covariance, normalize=False)
    target = portfolio_variance / len(weights)
    return float(np.sum(np.square(contributions / target - 1.0)))


def historical_convex_objective(x: np.ndarray, covariance: np.ndarray) -> float:
    """Reproduce the equal-budget log-barrier objective introduced in v0.05."""
    if np.any(x <= 0):
        return float("inf")
    return float(0.5 * x @ covariance @ x - np.log(x).mean())


def historical_convex_gradient(x: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    return covariance @ x - 1.0 / (len(x) * x)


def _validate_covariance(covariance: np.ndarray) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix")
    if covariance.shape[0] < 2 or not np.isfinite(covariance).all():
        raise ValueError("covariance must contain finite values for at least two assets")
    return (covariance + covariance.T) / 2.0


def _package_result(
    variant: str,
    weights: np.ndarray,
    solver_success: bool,
    status: str,
    iterations: int,
    runtime_ms: float,
    covariance: np.ndarray,
) -> EvolutionSolverResult:
    weights = np.asarray(weights, dtype=float)
    budget = np.full(len(weights), 1.0 / len(weights))
    weight_sum_error = float(abs(weights.sum() - 1.0))
    min_weight = float(weights.min())
    if np.isfinite(weights).all() and weights.sum() > 0:
        normalized = weights / weights.sum()
        rc_error = float(np.max(np.abs(risk_contributions(normalized, covariance) - budget)))
    else:
        rc_error = float("inf")
    metadata = VARIANT_METADATA[variant]
    return EvolutionSolverResult(
        variant=variant,
        label=str(metadata["label"]),
        objective_family=str(metadata["objective_family"]),
        stage_order=int(metadata["stage_order"]),
        weights=weights,
        solver_success=bool(solver_success),
        status=str(status),
        iterations=int(iterations),
        runtime_ms=float(runtime_ms),
        rc_max_error=rc_error,
        rc_pass=bool(
            rc_error <= RC_ACCEPTANCE_TOLERANCE
            and weight_sum_error <= 1e-10
            and min_weight >= -1e-12
        ),
        weight_sum_error=weight_sum_error,
        min_weight=min_weight,
        distance_from_equal=float(np.max(np.abs(weights - budget))),
    )


def solve_optimizer_variant(
    covariance: np.ndarray,
    variant: str,
    *,
    solver_tol: float = 1e-10,
    solver_max_iter: int = 1000,
) -> EvolutionSolverResult:
    """Solve one controlled historical variant without importing archival scripts."""
    covariance = _validate_covariance(covariance)
    if variant not in VARIANT_METADATA:
        raise ValueError(f"unsupported optimizer evolution variant: {variant}")

    n_assets = covariance.shape[0]
    equal_weights = np.full(n_assets, 1.0 / n_assets)
    bounds = [(0.0, 1.0)] * n_assets
    constraint = {"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)}
    started = perf_counter()

    if variant == "v0_02_raw_slsqp":
        result = minimize(
            original_erc_objective,
            equal_weights,
            args=(covariance, equal_weights),
            method="SLSQP",
            bounds=bounds,
            constraints=constraint,
        )
        weights = np.asarray(result.x, dtype=float)
    elif variant == "v0_03_scaled_slsqp":
        result = minimize(
            scaled_erc_objective,
            equal_weights,
            args=(covariance,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraint,
            options={"ftol": 1e-10, "maxiter": 1000},
        )
        weights = np.asarray(result.x, dtype=float)
    elif variant == "v0_04_relative_slsqp":
        result = minimize(
            relative_erc_objective,
            equal_weights,
            args=(covariance,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraint,
            options={"ftol": 1e-9, "maxiter": 1000},
        )
        weights = np.asarray(result.x, dtype=float)
    elif variant == "v0_05_convex_lbfgsb":
        result = minimize(
            historical_convex_objective,
            np.ones(n_assets, dtype=float),
            args=(covariance,),
            jac=historical_convex_gradient,
            method="L-BFGS-B",
            bounds=[(1e-8, None)] * n_assets,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        x = np.asarray(result.x, dtype=float)
        weights = x / x.sum()
    else:
        newton = solve_erc(
            covariance,
            method="newton",
            tol=solver_tol,
            max_iter=solver_max_iter,
        )
        return _package_result(
            variant,
            newton.weights,
            newton.success,
            newton.status,
            newton.iterations,
            newton.runtime_ms,
            covariance,
        )

    runtime_ms = (perf_counter() - started) * 1000.0
    return _package_result(
        variant,
        weights,
        bool(result.success),
        str(result.message),
        int(result.nit),
        runtime_ms,
        covariance,
    )


def evaluate_optimizer_evolution(
    returns: pd.DataFrame,
    *,
    window: int = 252,
    decay: float = 0.97,
    ridge: float = 1e-8,
    solver_tol: float = 1e-10,
    solver_max_iter: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all controlled variants on the same monthly EWMA semi-covariance matrices."""
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("returns index must be a DatetimeIndex")
    if window < 20 or window > len(returns):
        raise ValueError("window must be at least 20 and no longer than the dataset")

    positions = pd.Series(np.arange(len(returns)), index=returns.index)
    month_ends = positions.groupby(returns.index.to_period("M")).last().astype(int)
    records: list[dict[str, object]] = []
    for position in month_ends:
        if position < window - 1:
            continue
        observation_date = returns.index[position]
        history = returns.iloc[position - window + 1 : position + 1]
        covariance = estimate_covariance(history, method="ewma_semi", decay=decay, ridge=ridge)
        for variant in OPTIMIZER_VARIANTS:
            result = solve_optimizer_variant(
                covariance,
                variant,
                solver_tol=solver_tol,
                solver_max_iter=solver_max_iter,
            )
            records.append(
                {
                    "observation_date": observation_date,
                    "variant": result.variant,
                    "label": result.label,
                    "objective_family": result.objective_family,
                    "stage_order": result.stage_order,
                    "solver_success": result.solver_success,
                    "status": result.status,
                    "iterations": result.iterations,
                    "runtime_ms": result.runtime_ms,
                    "rc_max_error": result.rc_max_error,
                    "rc_pass": result.rc_pass,
                    "weight_sum_error": result.weight_sum_error,
                    "min_weight": result.min_weight,
                    "distance_from_equal": result.distance_from_equal,
                }
            )

    details = pd.DataFrame(records)
    if details.empty:
        raise ValueError("no monthly observation has enough history for optimizer evolution")
    summaries: list[dict[str, object]] = []
    for variant, label, objective_family, stage_order in VARIANT_SPECS:
        subset = details.loc[details["variant"] == variant]
        summaries.append(
            {
                "variant": variant,
                "label": label,
                "objective_family": objective_family,
                "stage_order": stage_order,
                "observations": int(len(subset)),
                "solver_success_rate": float(subset["solver_success"].mean()),
                "rc_pass_rate": float(subset["rc_pass"].mean()),
                "median_iterations": float(subset["iterations"].median()),
                "median_runtime_ms": float(subset["runtime_ms"].median()),
                "median_rc_error": float(subset["rc_max_error"].median()),
                "max_rc_error": float(subset["rc_max_error"].max()),
                "max_weight_sum_error": float(subset["weight_sum_error"].max()),
                "min_weight": float(subset["min_weight"].min()),
                "median_distance_from_equal": float(subset["distance_from_equal"].median()),
            }
        )
    return details, pd.DataFrame(summaries)
