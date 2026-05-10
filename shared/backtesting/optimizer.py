"""
Strategy Optimizer
====================

Provides grid search, Bayesian optimization (via Optuna),
walk-forward analysis, and Monte Carlo simulation for
trading strategy parameter tuning.

Usage:
    optimizer = StrategyOptimizer(engine)
    best = optimizer.bayesian_optimize(factory, param_space, n_trials=50)
    wf = optimizer.walk_forward(factory, best.params, n_splits=5)
    mc = optimizer.monte_carlo(result, n_simulations=1000)
"""

from __future__ import annotations

import itertools
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import optuna  # type: ignore[import-untyped]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False
    logger.debug("optuna not installed — Bayesian optimization unavailable")


@dataclass
class OptimizationResult:
    """Result of a single optimization trial."""

    params: Dict[str, Any]
    metric_value: float
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Result of walk-forward analysis."""

    in_sample_metrics: List[Dict[str, float]]
    out_of_sample_metrics: List[Dict[str, float]]
    avg_in_sample_sharpe: float = 0.0
    avg_oos_sharpe: float = 0.0
    avg_in_sample_return: float = 0.0
    avg_oos_return: float = 0.0
    robustness_ratio: float = 0.0  # OOS Sharpe / IS Sharpe
    n_splits: int = 0


@dataclass
class MonteCarloResult:
    """Result of Monte Carlo simulation."""

    median_return: float = 0.0
    mean_return: float = 0.0
    worst_return: float = 0.0
    best_return: float = 0.0
    median_drawdown: float = 0.0
    worst_drawdown: float = 0.0
    p5_return: float = 0.0
    p95_return: float = 0.0
    p5_drawdown: float = 0.0
    p95_drawdown: float = 0.0
    median_sharpe: float = 0.0
    return_distribution: list[float] = field(default_factory=list)
    drawdown_distribution: list[float] = field(default_factory=list)
    n_simulations: int = 0


class StrategyOptimizer:
    """Optimizer for trading strategy parameters.

    Uses BacktestEngineV2 to evaluate strategies across different
    parameter combinations.

    Args:
        engine: A BacktestEngineV2 instance with data already loaded.
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def grid_search(
        self,
        strategy_factory: Callable[[Dict[str, Any]], Callable],
        param_grid: Dict[str, List[Any]],
        metric: str = "sharpe_ratio",
    ) -> List[OptimizationResult]:
        """Exhaustive grid search over parameter combinations.

        Args:
            strategy_factory: Callable that takes params dict and returns
                a strategy function compatible with BacktestEngineV2.run().
            param_grid: Dict mapping param names to lists of values.
                Example: {"rsi_length": [10, 14, 20], "threshold": [30, 35]}
            metric: Metric to optimize (attribute of BacktestResultV2).

        Returns:
            List of OptimizationResult sorted by metric (descending).
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))

        logger.info(
            "Grid search: %d combinations over %d parameters",
            len(combinations), len(keys),
        )

        results: List[OptimizationResult] = []

        for i, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            strategy_fn = strategy_factory(params)

            try:
                bt_result = self.engine.run(strategy_fn)
                metric_val = getattr(bt_result, metric, 0.0)

                results.append(OptimizationResult(
                    params=params,
                    metric_value=metric_val,
                    metrics={
                        "sharpe_ratio": bt_result.sharpe_ratio,
                        "total_return": bt_result.total_return,
                        "max_drawdown": bt_result.max_drawdown,
                        "win_rate": bt_result.win_rate,
                        "profit_factor": bt_result.profit_factor,
                        "total_trades": bt_result.total_trades,
                    },
                ))

                if (i + 1) % 10 == 0:
                    logger.info("Grid search progress: %d/%d", i + 1, len(combinations))

            except Exception as e:
                logger.warning("Grid search trial %d failed: %s", i, e)
                results.append(OptimizationResult(
                    params=params,
                    metric_value=float("-inf"),
                ))

        results.sort(key=lambda r: r.metric_value, reverse=True)

        if results:
            best = results[0]
            logger.info(
                "Grid search complete. Best %s=%.4f with params=%s",
                metric, best.metric_value, best.params,
            )

        return results

    def bayesian_optimize(
        self,
        strategy_factory: Callable[[Dict[str, Any]], Callable],
        param_space: Dict[str, Dict[str, Any]],
        n_trials: int = 100,
        metric: str = "sharpe_ratio",
    ) -> OptimizationResult:
        """Bayesian optimization using Optuna.

        Args:
            strategy_factory: Callable that takes params dict and returns
                a strategy function.
            param_space: Dict mapping param names to space definitions.
                Each entry: {"type": "int"|"float"|"categorical", "low": ..., "high": ..., "choices": [...]}
            n_trials: Number of optimization trials.
            metric: Metric to maximize.

        Returns:
            Best OptimizationResult.

        Raises:
            ImportError: If optuna is not installed.
        """
        if not _HAS_OPTUNA:
            raise ImportError(
                "optuna is required for Bayesian optimization. "
                "Install with: pip install optuna"
            )

        def objective(trial: optuna.Trial) -> float:
            params: Dict[str, Any] = {}
            for name, spec in param_space.items():
                param_type = spec.get("type", "float")
                if param_type == "int":
                    params[name] = trial.suggest_int(name, spec["low"], spec["high"])
                elif param_type == "float":
                    params[name] = trial.suggest_float(
                        name, spec["low"], spec["high"],
                        log=spec.get("log", False),
                    )
                elif param_type == "categorical":
                    params[name] = trial.suggest_categorical(name, spec["choices"])

            strategy_fn = strategy_factory(params)
            try:
                result = self.engine.run(strategy_fn)
                return getattr(result, metric, 0.0)
            except Exception as e:
                logger.warning("Trial failed: %s", e)
                return float("-inf")

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_value = study.best_value

        # Re-run best to get full metrics
        strategy_fn = strategy_factory(best_params)
        bt_result = self.engine.run(strategy_fn)

        logger.info(
            "Bayesian optimization complete. Best %s=%.4f after %d trials. Params=%s",
            metric, best_value, n_trials, best_params,
        )

        return OptimizationResult(
            params=best_params,
            metric_value=best_value,
            metrics={
                "sharpe_ratio": bt_result.sharpe_ratio,
                "total_return": bt_result.total_return,
                "max_drawdown": bt_result.max_drawdown,
                "win_rate": bt_result.win_rate,
                "profit_factor": bt_result.profit_factor,
                "total_trades": bt_result.total_trades,
                "sortino_ratio": bt_result.sortino_ratio,
            },
        )

    def walk_forward(
        self,
        strategy_factory: Callable[[Dict[str, Any]], Callable],
        params: Dict[str, Any],
        n_splits: int = 5,
        train_pct: float = 0.7,
    ) -> WalkForwardResult:
        """Walk-forward analysis with rolling train/test windows.

        Splits data into rolling windows. For each window, trains on
        in-sample period and validates on out-of-sample period.

        Args:
            strategy_factory: Callable that takes params and returns strategy fn.
            params: Strategy parameters to use.
            n_splits: Number of rolling windows.
            train_pct: Fraction of each window for training.

        Returns:
            WalkForwardResult with IS/OOS metrics per split.
        """
        # Use the first symbol's data for splitting
        first_symbol = list(self.engine._data.keys())[0]
        total_bars = len(self.engine._data[first_symbol])
        window_size = total_bars // n_splits

        if window_size < 20:
            raise ValueError(
                f"Insufficient data for {n_splits} splits. "
                f"Total bars: {total_bars}, window size: {window_size}"
            )

        in_sample_metrics: List[Dict[str, float]] = []
        oos_metrics: List[Dict[str, float]] = []

        from shared.backtesting.backtest_engine_v2 import BacktestEngineV2

        for i in range(n_splits):
            start = i * window_size
            end = min(start + window_size, total_bars)
            split_point = start + int((end - start) * train_pct)

            # Build IS and OOS datasets
            is_data = {}
            oos_data = {}
            for sym, df in self.engine._data.items():
                if end <= len(df):
                    is_data[sym] = df.iloc[start:split_point].reset_index(drop=True)
                    oos_data[sym] = df.iloc[split_point:end].reset_index(drop=True)
                else:
                    max_end = min(end, len(df))
                    max_split = min(split_point, len(df))
                    is_data[sym] = df.iloc[start:max_split].reset_index(drop=True)
                    oos_data[sym] = df.iloc[max_split:max_end].reset_index(drop=True)

            strategy_fn = strategy_factory(params)

            # In-sample
            is_engine = BacktestEngineV2(
                initial_capital=self.engine.initial_capital,
                commission=self.engine.commission,
                slippage=self.engine.slippage,
            )
            is_engine.load_data(is_data)
            try:
                is_result = is_engine.run(strategy_fn)
                is_metrics = {
                    "sharpe_ratio": is_result.sharpe_ratio,
                    "total_return": is_result.total_return,
                    "max_drawdown": is_result.max_drawdown,
                    "total_trades": is_result.total_trades,
                }
            except Exception:
                is_metrics = {"sharpe_ratio": 0, "total_return": 0, "max_drawdown": 0, "total_trades": 0}
            in_sample_metrics.append(is_metrics)

            # Out-of-sample
            oos_engine = BacktestEngineV2(
                initial_capital=self.engine.initial_capital,
                commission=self.engine.commission,
                slippage=self.engine.slippage,
            )
            oos_engine.load_data(oos_data)
            try:
                strategy_fn = strategy_factory(params)
                oos_result = oos_engine.run(strategy_fn)
                oos_m = {
                    "sharpe_ratio": oos_result.sharpe_ratio,
                    "total_return": oos_result.total_return,
                    "max_drawdown": oos_result.max_drawdown,
                    "total_trades": oos_result.total_trades,
                }
            except Exception:
                oos_m = {"sharpe_ratio": 0, "total_return": 0, "max_drawdown": 0, "total_trades": 0}
            oos_metrics.append(oos_m)

            logger.info(
                "Walk-forward split %d/%d: IS Sharpe=%.4f, OOS Sharpe=%.4f",
                i + 1, n_splits,
                is_metrics["sharpe_ratio"], oos_m["sharpe_ratio"],
            )

        avg_is_sharpe = np.mean([m["sharpe_ratio"] for m in in_sample_metrics])
        avg_oos_sharpe = np.mean([m["sharpe_ratio"] for m in oos_metrics])
        avg_is_return = np.mean([m["total_return"] for m in in_sample_metrics])
        avg_oos_return = np.mean([m["total_return"] for m in oos_metrics])
        robustness = avg_oos_sharpe / avg_is_sharpe if avg_is_sharpe != 0 else 0.0

        logger.info(
            "Walk-forward complete: Avg IS Sharpe=%.4f, Avg OOS Sharpe=%.4f, Robustness=%.2f",
            avg_is_sharpe, avg_oos_sharpe, robustness,
        )

        return WalkForwardResult(
            in_sample_metrics=in_sample_metrics,
            out_of_sample_metrics=oos_metrics,
            avg_in_sample_sharpe=round(float(avg_is_sharpe), 4),
            avg_oos_sharpe=round(float(avg_oos_sharpe), 4),
            avg_in_sample_return=round(float(avg_is_return), 6),
            avg_oos_return=round(float(avg_oos_return), 6),
            robustness_ratio=round(robustness, 4),
            n_splits=n_splits,
        )

    @staticmethod
    def monte_carlo(
        backtest_result: Any,
        n_simulations: int = 1000,
        seed: Optional[int] = None,
    ) -> MonteCarloResult:
        """Monte Carlo simulation by shuffling trade sequences.

        Randomly reorders completed trades and recomputes equity
        curves to generate distributions of returns and drawdowns.

        Args:
            backtest_result: BacktestResultV2 with trades list.
            n_simulations: Number of simulated equity curves.
            seed: Random seed for reproducibility.

        Returns:
            MonteCarloResult with return and drawdown distributions.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        trades = getattr(backtest_result, "trades", [])
        if not trades:
            # Fall back to trade_log PnL
            pnls = [
                t.get("pnl", 0) for t in backtest_result.trade_log
                if "pnl" in t
            ]
        else:
            pnls = [t.pnl for t in trades]

        if not pnls:
            logger.warning("No trades for Monte Carlo simulation")
            return MonteCarloResult(n_simulations=0)

        initial = backtest_result.equity_curve[0] if backtest_result.equity_curve else 100000

        sim_returns: list[float] = []
        sim_drawdowns: list[float] = []
        sim_sharpes: list[float] = []

        for _ in range(n_simulations):
            shuffled = pnls.copy()
            random.shuffle(shuffled)

            equity = initial
            peak = equity
            max_dd = 0.0
            daily_rets = []

            for pnl in shuffled:
                prev = equity
                equity += pnl
                if prev > 0:
                    daily_rets.append((equity - prev) / prev)
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd

            total_ret = (equity - initial) / initial
            sim_returns.append(total_ret)
            sim_drawdowns.append(max_dd)

            if daily_rets:
                arr = np.array(daily_rets)
                std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0
                sharpe = float(np.mean(arr)) / std * np.sqrt(252) if std > 0 else 0
            else:
                sharpe = 0.0
            sim_sharpes.append(sharpe)

        sim_returns.sort()
        sim_drawdowns.sort()

        result = MonteCarloResult(
            median_return=round(float(np.median(sim_returns)), 6),
            mean_return=round(float(np.mean(sim_returns)), 6),
            worst_return=round(min(sim_returns), 6),
            best_return=round(max(sim_returns), 6),
            median_drawdown=round(float(np.median(sim_drawdowns)), 6),
            worst_drawdown=round(max(sim_drawdowns), 6),
            p5_return=round(float(np.percentile(sim_returns, 5)), 6),
            p95_return=round(float(np.percentile(sim_returns, 95)), 6),
            p5_drawdown=round(float(np.percentile(sim_drawdowns, 5)), 6),
            p95_drawdown=round(float(np.percentile(sim_drawdowns, 95)), 6),
            median_sharpe=round(float(np.median(sim_sharpes)), 4),
            return_distribution=sim_returns,
            drawdown_distribution=sim_drawdowns,
            n_simulations=n_simulations,
        )

        logger.info(
            "Monte Carlo (%d sims): median return=%.2f%%, median DD=%.2f%%, "
            "worst DD=%.2f%%",
            n_simulations,
            result.median_return * 100,
            result.median_drawdown * 100,
            result.worst_drawdown * 100,
        )
        return result
