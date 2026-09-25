from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ANNUALIZATION = 252


@dataclass
class SolverResult:
    method: str
    weights: np.ndarray
    success: bool
    status: str
    iterations: int
    runtime_ms: float
    objective: float
    gradient_norm: float
    rc_max_error: float
    weight_sum_error: float
    min_weight: float
    history: list[dict[str, float]] = field(default_factory=list)


def _as_2d_array(returns: pd.DataFrame | np.ndarray) -> np.ndarray:
    array = np.asarray(returns, dtype=float)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 2:
        raise ValueError("returns must be a two-dimensional array with at least 2 rows and 2 columns")
    if not np.isfinite(array).all():
        raise ValueError("returns contains non-finite values")
    return array


def _normalized_budget(risk_budget: Iterable[float] | None, n_assets: int) -> np.ndarray:
    if risk_budget is None:
        budget = np.full(n_assets, 1.0 / n_assets)
    else:
        budget = np.asarray(list(risk_budget), dtype=float)
        if budget.shape != (n_assets,):
            raise ValueError("risk_budget length must equal the number of assets")
        if not np.isfinite(budget).all() or np.any(budget <= 0):
            raise ValueError("risk_budget must contain finite positive values")
        budget = budget / budget.sum()
    return budget


def estimate_covariance(
    returns: pd.DataFrame | np.ndarray,
    method: str = "ewma_semi",
    decay: float = 0.97,
    ridge: float = 1e-8,
) -> np.ndarray:
    """Estimate an annualized covariance or downside semi-covariance matrix."""
    array = _as_2d_array(returns)
    n_obs, n_assets = array.shape
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    if method not in {"sample", "ewma_full", "ewma_semi"}:
        raise ValueError(f"unsupported covariance method: {method}")
    if method != "sample" and not 0 < decay < 1:
        raise ValueError("decay must be in (0, 1)")

    if method == "sample":
        covariance = np.cov(array, rowvar=False, ddof=1) * ANNUALIZATION
    else:
        weights = decay ** np.arange(n_obs - 1, -1, -1, dtype=float)
        weights /= weights.sum()
        if method == "ewma_semi":
            transformed = np.minimum(array, 0.0)
        else:
            mean = np.sum(array * weights[:, None], axis=0)
            transformed = array - mean
        weighted = transformed * np.sqrt(weights[:, None])
        covariance = weighted.T @ weighted * ANNUALIZATION

    covariance = (covariance + covariance.T) / 2.0
    covariance += np.eye(n_assets) * ridge
    if not np.isfinite(covariance).all():
        raise ValueError("estimated covariance contains non-finite values")
    return covariance


def convex_objective(x: np.ndarray, covariance: np.ndarray, budget: np.ndarray) -> float:
    if np.any(x <= 0):
        return float("inf")
    return float(0.5 * x @ covariance @ x - budget @ np.log(x))


def convex_gradient(x: np.ndarray, covariance: np.ndarray, budget: np.ndarray) -> np.ndarray:
    return covariance @ x - budget / x


def convex_hessian(x: np.ndarray, covariance: np.ndarray, budget: np.ndarray) -> np.ndarray:
    return covariance + np.diag(budget / np.square(x))


def risk_contributions(weights: np.ndarray, covariance: np.ndarray, normalize: bool = True) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    contributions = weights * (covariance @ weights)
    if normalize:
        total = contributions.sum()
        if total <= 0 or not np.isfinite(total):
            raise ValueError("portfolio variance must be positive")
        return contributions / total
    return contributions


def original_erc_objective(weights: np.ndarray, covariance: np.ndarray, budget: np.ndarray) -> float:
    contributions = risk_contributions(weights, covariance, normalize=False)
    target = budget * float(weights @ covariance @ weights)
    return float(np.sum(np.square(contributions - target)))


def original_erc_gradient(weights: np.ndarray, covariance: np.ndarray, budget: np.ndarray) -> np.ndarray:
    marginal = covariance @ weights
    variance = float(weights @ marginal)
    residual = weights * marginal - budget * variance
    residual_jacobian = np.diag(marginal) + np.diag(weights) @ covariance - np.outer(budget, 2.0 * marginal)
    return 2.0 * residual_jacobian.T @ residual


def _finite_difference_gradient(function, x: np.ndarray, epsilon: float = 1e-7) -> np.ndarray:
    gradient = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        step = epsilon * max(1.0, abs(float(x[i])))
        upper = x.copy()
        lower = x.copy()
        upper[i] += step
        lower[i] -= step
        gradient[i] = (function(upper) - function(lower)) / (2.0 * step)
    return gradient


def _result_from_solution(
    method: str,
    x: np.ndarray,
    covariance: np.ndarray,
    budget: np.ndarray,
    success: bool,
    status: str,
    iterations: int,
    runtime_ms: float,
    objective: float,
    gradient_norm: float,
    history: list[dict[str, float]],
) -> SolverResult:
    weights = np.asarray(x, dtype=float)
    weights = np.maximum(weights, 0.0)
    weights /= weights.sum()
    rc_error = float(np.max(np.abs(risk_contributions(weights, covariance) - budget)))
    weight_sum_error = float(abs(weights.sum() - 1.0))
    feasible = bool(weight_sum_error <= 1e-10 and weights.min() >= -1e-12)
    return SolverResult(
        method=method,
        weights=weights,
        success=bool(success and feasible and rc_error <= 1e-6),
        status=status,
        iterations=int(iterations),
        runtime_ms=float(runtime_ms),
        objective=float(objective),
        gradient_norm=float(gradient_norm),
        rc_max_error=rc_error,
        weight_sum_error=weight_sum_error,
        min_weight=float(weights.min()),
        history=history,
    )


def _solve_newton(
    covariance: np.ndarray,
    budget: np.ndarray,
    tol: float,
    max_iter: int,
) -> SolverResult:
    x = np.ones(len(budget), dtype=float)
    history: list[dict[str, float]] = []
    started = perf_counter()
    success = False
    status = "maximum iterations reached"
    iteration = 0

    for iteration in range(max_iter + 1):
        gradient = convex_gradient(x, covariance, budget)
        gradient_norm = float(np.linalg.norm(gradient, ord=np.inf))
        objective = convex_objective(x, covariance, budget)
        rc_error = float(np.max(np.abs(risk_contributions(x / x.sum(), covariance) - budget)))
        history.append(
            {
                "iteration": float(iteration),
                "objective": objective,
                "gradient_norm": gradient_norm,
                "rc_max_error": rc_error,
                "step": 0.0,
            }
        )
        if gradient_norm <= tol:
            success = True
            status = "gradient tolerance satisfied"
            break
        if iteration == max_iter:
            break

        hessian = convex_hessian(x, covariance, budget)
        try:
            direction = -np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            status = "Hessian solve failed"
            break
        directional_derivative = float(gradient @ direction)
        if directional_derivative >= 0:
            status = "Newton direction is not a descent direction"
            break

        negative = direction < 0
        boundary_step = 1.0
        if np.any(negative):
            boundary_step = min(1.0, float(0.99 * np.min(-x[negative] / direction[negative])))
        step = boundary_step
        armijo = 1e-4
        while step > 1e-16:
            candidate = x + step * direction
            if np.all(candidate > 0) and convex_objective(candidate, covariance, budget) <= (
                objective + armijo * step * directional_derivative
            ):
                x = candidate
                history[-1]["step"] = float(step)
                break
            step *= 0.5
        else:
            status = "Armijo line search failed"
            break

    runtime_ms = (perf_counter() - started) * 1000.0
    final_gradient = convex_gradient(x, covariance, budget)
    return _result_from_solution(
        "newton",
        x,
        covariance,
        budget,
        success,
        status,
        iteration,
        runtime_ms,
        convex_objective(x, covariance, budget),
        float(np.linalg.norm(final_gradient, ord=np.inf)),
        history,
    )


def _solve_lbfgsb(
    covariance: np.ndarray,
    budget: np.ndarray,
    tol: float,
    max_iter: int,
) -> SolverResult:
    x0 = np.ones(len(budget), dtype=float)
    history: list[dict[str, float]] = []

    def record(x: np.ndarray) -> None:
        weights = x / x.sum()
        history.append(
            {
                "iteration": float(len(history) + 1),
                "objective": convex_objective(x, covariance, budget),
                "gradient_norm": float(np.linalg.norm(convex_gradient(x, covariance, budget), ord=np.inf)),
                "rc_max_error": float(np.max(np.abs(risk_contributions(weights, covariance) - budget))),
                "step": float("nan"),
            }
        )

    started = perf_counter()
    result = minimize(
        convex_objective,
        x0,
        args=(covariance, budget),
        jac=convex_gradient,
        method="L-BFGS-B",
        bounds=[(1e-12, None)] * len(budget),
        callback=record,
        options={"ftol": np.finfo(float).eps, "gtol": tol, "maxiter": max_iter, "maxls": 50},
    )
    runtime_ms = (perf_counter() - started) * 1000.0
    x = np.asarray(result.x, dtype=float)
    if not history:
        record(x)
    return _result_from_solution(
        "lbfgsb",
        x,
        covariance,
        budget,
        bool(result.success),
        str(result.message),
        int(result.nit),
        runtime_ms,
        float(result.fun),
        float(np.linalg.norm(convex_gradient(x, covariance, budget), ord=np.inf)),
        history,
    )


def _solve_slsqp(
    covariance: np.ndarray,
    budget: np.ndarray,
    tol: float,
    max_iter: int,
) -> SolverResult:
    n_assets = len(budget)
    x0 = np.full(n_assets, 1.0 / n_assets)
    history: list[dict[str, float]] = []
    natural_scale = max(float(np.square(np.trace(covariance) / n_assets)), 1e-16)

    def scaled_objective(weights: np.ndarray) -> float:
        return original_erc_objective(weights, covariance, budget) / natural_scale

    def scaled_gradient(weights: np.ndarray) -> np.ndarray:
        return original_erc_gradient(weights, covariance, budget) / natural_scale

    def record(weights: np.ndarray) -> None:
        history.append(
            {
                "iteration": float(len(history) + 1),
                "objective": original_erc_objective(weights, covariance, budget),
                "gradient_norm": float(
                    np.linalg.norm(scaled_gradient(weights), ord=np.inf)
                ),
                "rc_max_error": float(np.max(np.abs(risk_contributions(weights, covariance) - budget))),
                "step": float("nan"),
            }
        )

    started = perf_counter()
    result = minimize(
        scaled_objective,
        x0,
        method="SLSQP",
        jac=scaled_gradient,
        bounds=[(1e-12, 1.0)] * n_assets,
        constraints={"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
        callback=record,
        options={"ftol": min(tol * 1e-4, 1e-14), "maxiter": max_iter, "disp": False},
    )
    runtime_ms = (perf_counter() - started) * 1000.0
    weights = np.asarray(result.x, dtype=float)
    if not history:
        record(weights)
    projected_gradient = scaled_gradient(weights)
    projected_gradient -= projected_gradient.mean()
    return _result_from_solution(
        "slsqp",
        weights,
        covariance,
        budget,
        bool(result.success),
        str(result.message),
        int(result.nit),
        runtime_ms,
        original_erc_objective(weights, covariance, budget),
        float(np.linalg.norm(projected_gradient, ord=np.inf)),
        history,
    )


def solve_erc(
    covariance: np.ndarray,
    method: str = "newton",
    risk_budget: Iterable[float] | None = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> SolverResult:
    """Solve the equal/general risk-budgeting portfolio problem."""
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix")
    if not np.isfinite(covariance).all():
        raise ValueError("covariance contains non-finite values")
    covariance = (covariance + covariance.T) / 2.0
    min_eigenvalue = float(np.linalg.eigvalsh(covariance).min())
    if min_eigenvalue <= 0:
        raise ValueError("covariance must be positive definite")
    budget = _normalized_budget(risk_budget, covariance.shape[0])
    if tol <= 0 or max_iter <= 0:
        raise ValueError("tol and max_iter must be positive")

    normalized = method.lower().replace("-", "")
    if normalized == "newton":
        return _solve_newton(covariance, budget, tol, max_iter)
    if normalized in {"lbfgsb", "lbgfsb"}:
        return _solve_lbfgsb(covariance, budget, tol, max_iter)
    if normalized == "slsqp":
        return _solve_slsqp(covariance, budget, tol, max_iter)
    raise ValueError(f"unsupported solver method: {method}")
