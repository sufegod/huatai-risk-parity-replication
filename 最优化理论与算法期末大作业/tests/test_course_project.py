from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


COURSE_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = COURSE_DIR.parent
if str(COURSE_DIR) not in sys.path:
    sys.path.insert(0, str(COURSE_DIR))

from src.backtest import BacktestConfig, run_backtest
from src.risk_parity import (
    convex_gradient,
    convex_hessian,
    convex_objective,
    estimate_covariance,
    risk_contributions,
    solve_erc,
)


class RiskParityModelTests(unittest.TestCase):
    def setUp(self):
        self.covariance = np.array(
            [
                [0.040, 0.006, 0.002],
                [0.006, 0.090, 0.004],
                [0.002, 0.004, 0.160],
            ],
            dtype=float,
        )
        self.budget = np.full(3, 1.0 / 3.0)

    def test_analytic_gradient_matches_finite_difference(self):
        x = np.array([0.8, 1.1, 1.4])
        step = 1e-6
        numerical = np.empty_like(x)
        for i in range(len(x)):
            upper = x.copy()
            lower = x.copy()
            upper[i] += step
            lower[i] -= step
            numerical[i] = (
                convex_objective(upper, self.covariance, self.budget)
                - convex_objective(lower, self.covariance, self.budget)
            ) / (2 * step)
        np.testing.assert_allclose(convex_gradient(x, self.covariance, self.budget), numerical, rtol=1e-6, atol=1e-8)

    def test_analytic_hessian_matches_gradient_difference_and_is_spd(self):
        x = np.array([0.8, 1.1, 1.4])
        step = 1e-6
        numerical = np.empty((3, 3))
        for i in range(len(x)):
            upper = x.copy()
            lower = x.copy()
            upper[i] += step
            lower[i] -= step
            numerical[:, i] = (
                convex_gradient(upper, self.covariance, self.budget)
                - convex_gradient(lower, self.covariance, self.budget)
            ) / (2 * step)
        analytic = convex_hessian(x, self.covariance, self.budget)
        np.testing.assert_allclose(analytic, numerical, rtol=1e-6, atol=1e-8)
        self.assertGreater(np.linalg.eigvalsh(analytic).min(), 0.0)

    def test_diagonal_covariance_matches_inverse_volatility_closed_form(self):
        diagonal = np.diag(np.square([0.10, 0.20, 0.40, 0.80]))
        expected = 1.0 / np.sqrt(np.diag(diagonal))
        expected /= expected.sum()
        for method in ("newton", "lbfgsb", "slsqp"):
            result = solve_erc(diagonal, method=method, tol=1e-10, max_iter=1000)
            np.testing.assert_allclose(result.weights, expected, atol=2e-6)
            self.assertLess(result.rc_max_error, 1e-6)

    def test_solvers_agree_and_obey_constraints(self):
        results = {method: solve_erc(self.covariance, method=method) for method in ("newton", "lbfgsb", "slsqp")}
        for result in results.values():
            self.assertAlmostEqual(float(result.weights.sum()), 1.0, places=12)
            self.assertGreater(result.weights.min(), 0.0)
            self.assertLess(result.rc_max_error, 1e-6)
        np.testing.assert_allclose(results["newton"].weights, results["lbfgsb"].weights, atol=2e-6)
        np.testing.assert_allclose(results["newton"].weights, results["slsqp"].weights, atol=2e-6)

    def test_newton_is_deterministic(self):
        first = solve_erc(self.covariance, method="newton")
        second = solve_erc(self.covariance, method="newton")
        np.testing.assert_array_equal(first.weights, second.weights)
        self.assertEqual(first.iterations, second.iterations)

    def test_ewma_semi_covariance_is_positive_definite(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.01, size=(252, 9))
        covariance = estimate_covariance(returns, method="ewma_semi", decay=0.97, ridge=1e-8)
        self.assertGreater(np.linalg.eigvalsh(covariance).min(), 0.0)


class BacktestTests(unittest.TestCase):
    @staticmethod
    def synthetic_returns() -> pd.DataFrame:
        rng = np.random.default_rng(7)
        dates = pd.bdate_range("2018-01-01", periods=700)
        values = rng.normal(0.0002, [0.008, 0.012, 0.004], size=(700, 3))
        return pd.DataFrame(values, index=dates, columns=["A", "B", "C"])

    def config(self, **changes) -> BacktestConfig:
        base = BacktestConfig(
            strategy="erc",
            window=126,
            decay=0.97,
            fee_rate=0.0005,
            train_start="2018-01-01",
            train_end="2019-12-31",
            validation_start="2020-01-01",
            validation_end="2021-12-31",
        )
        return replace(base, **changes)

    def test_future_changes_do_not_affect_first_target(self):
        returns = self.synthetic_returns()
        original = run_backtest(returns, self.config())
        changed = returns.copy()
        first_execution = original.target_weights.index[0]
        changed.loc[changed.index > first_execution] *= 5.0
        rerun = run_backtest(changed, self.config())
        np.testing.assert_allclose(
            original.target_weights.iloc[0].to_numpy(),
            rerun.target_weights.iloc[0].to_numpy(),
            atol=1e-12,
        )

    def test_transaction_cost_reduces_nav(self):
        returns = self.synthetic_returns() * 0.0
        without_cost = run_backtest(returns, self.config(strategy="equal_weight", fee_rate=0.0))
        with_cost = run_backtest(returns, self.config(strategy="equal_weight", fee_rate=0.0005))
        self.assertLess(with_cost.nav.iloc[-1], without_cost.nav.iloc[-1])
        self.assertAlmostEqual(without_cost.nav.iloc[-1], 1.0, places=12)

    def test_rebalance_executes_after_observation(self):
        result = run_backtest(self.synthetic_returns(), self.config())
        observations = pd.to_datetime(result.solver_diagnostics["observation_date"])
        self.assertTrue((result.solver_diagnostics.index > observations).all())


class SourceDataTests(unittest.TestCase):
    def test_source_data_shape_and_quality(self):
        source = REPO_DIR / "数据" / "原始数据" / "ETF风险平价回测数据.xlsx"
        frame = pd.read_excel(source, sheet_name=0, index_col=0, parse_dates=True)
        self.assertEqual(frame.shape, (3216, 9))
        self.assertEqual(int(frame.index.duplicated().sum()), 0)
        self.assertEqual(int(frame.isna().sum().sum()), 3)
        self.assertEqual(str(frame.index.min().date()), "2013-01-04")
        self.assertEqual(str(frame.index.max().date()), "2026-04-03")

    def test_newton_converges_quickly_on_representative_real_window(self):
        source = REPO_DIR / "数据" / "原始数据" / "ETF风险平价回测数据.xlsx"
        frame = pd.read_excel(source, sheet_name=0, index_col=0, parse_dates=True).fillna(0.0) / 100.0
        covariance = estimate_covariance(frame.iloc[-252:], method="ewma_semi", decay=0.97, ridge=1e-8)
        result = solve_erc(covariance, method="newton", tol=1e-10)
        self.assertLessEqual(result.iterations, 10)
        self.assertLess(result.rc_max_error, 1e-6)
        np.testing.assert_allclose(risk_contributions(result.weights, covariance), np.full(9, 1 / 9), atol=1e-6)


if __name__ == "__main__":
    unittest.main()

