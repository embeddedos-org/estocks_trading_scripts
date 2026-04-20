"""
LEAN CLI Runner
=================

Interface to the LEAN CLI for running backtests locally.

Usage:
    from lean_adapter.lean_cli import LEANCLIRunner, LEANCLIConfig
    runner = LEANCLIRunner(LEANCLIConfig())
    result = runner.generate_and_run(strategy_config, output_dir="./lean_projects")
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from shared.backtesting.backtest_engine_v2 import BacktestResultV2
    _HAS_ENGINE = True
except ImportError:
    _HAS_ENGINE = False


@dataclass
class LEANCLIConfig:
    """Configuration for LEAN CLI integration."""
    lean_cli_path: str = "lean"
    data_dir: str = "~/.lean/data"
    results_dir: str = "~/.lean/results"
    docker_image: str = "quantconnect/lean:latest"


class LEANCLIRunner:
    """Run LEAN backtests via the LEAN CLI."""

    def __init__(self, config: Optional[LEANCLIConfig] = None):
        self.config = config or LEANCLIConfig()

    def generate_and_run(
        self,
        strategy_config: Dict[str, Any],
        output_dir: str,
    ) -> Any:
        """Generate a LEAN project and run it via CLI.

        Args:
            strategy_config: Dict with name, symbols, indicators, etc.
            output_dir: Directory to generate project in

        Returns:
            BacktestResultV2 if available, else raw results dict
        """
        from lean_adapter.lean_bridge import LEANProjectGenerator

        gen = LEANProjectGenerator()
        project_dir = gen.generate_project(
            name=strategy_config.get("name", "Strategy"),
            symbols=strategy_config.get("symbols", ["SPY"]),
            output_dir=output_dir,
            indicators=strategy_config.get("indicators"),
            start_date=strategy_config.get("start_date", "2020-01-01"),
            end_date=strategy_config.get("end_date", "2024-01-01"),
            initial_capital=strategy_config.get("initial_capital", 100000),
        )

        # Run via LEAN CLI
        try:
            cmd = [
                self.config.lean_cli_path, "backtest", project_dir,
                "--output", self.config.results_dir,
            ]
            logger.info("Running: %s", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                logger.error("LEAN CLI failed: %s", result.stderr)
                raise RuntimeError(f"LEAN CLI error: {result.stderr}")

            return self._parse_lean_results(self.config.results_dir)
        except FileNotFoundError:
            logger.error(
                "LEAN CLI not found at '%s'. Install: pip install lean",
                self.config.lean_cli_path,
            )
            raise

    def _parse_lean_results(self, results_dir: str) -> Any:
        """Parse LEAN backtest results JSON into BacktestResultV2."""
        expanded_dir = os.path.expanduser(results_dir)
        if not os.path.isdir(expanded_dir):
            return {}
        results_file = None
        for f in os.listdir(expanded_dir):
            if f.endswith(".json") and "backtest" in f.lower():
                results_file = os.path.join(expanded_dir, f)
                break

        if results_file is None:
            raise FileNotFoundError(f"No results JSON found in {results_dir}")

        with open(results_file) as f:
            data = json.load(f)

        def _safe_float(val: str, scale: float = 1.0) -> float:
            """Parse a string to float, stripping '%' and scaling. Returns 0.0 on failure."""
            try:
                return float(val.replace("%", "")) * scale
            except (ValueError, AttributeError):
                return 0.0

        def _safe_int(val: str) -> int:
            try:
                return int(val)
            except (ValueError, TypeError):
                return 0

        stats = data.get("Statistics", {})
        total_return = _safe_float(stats.get("Total Net Profit", "0"), 0.01)
        sharpe = _safe_float(stats.get("Sharpe Ratio", "0"))
        max_dd = _safe_float(stats.get("Drawdown", "0"), -0.01)

        if _HAS_ENGINE:
            return BacktestResultV2(
                total_return=total_return,
                cagr=_safe_float(stats.get("Compounding Annual Return", "0"), 0.01),
                sharpe_ratio=sharpe,
                sortino_ratio=_safe_float(stats.get("Sortino Ratio", "0")),
                max_drawdown=max_dd,
                calmar_ratio=0.0,
                win_rate=_safe_float(stats.get("Win Rate", "0"), 0.01),
                profit_factor=_safe_float(stats.get("Profit-Loss Ratio", "0")),
                total_trades=_safe_int(stats.get("Total Trades", "0")),
                expectancy=_safe_float(stats.get("Expectation", "0")),
                avg_trade_duration=0.0,
                max_consecutive_wins=0,
                max_consecutive_losses=0,
                monthly_returns=pd.Series(dtype=float),
                alpha=_safe_float(stats.get("Alpha", "0")),
                beta=_safe_float(stats.get("Beta", "0")),
                information_ratio=_safe_float(stats.get("Information Ratio", "0")),
                tracking_error=_safe_float(stats.get("Tracking Error", "0")),
                equity_curve=pd.Series(dtype=float),
                trade_log=[],
                trades=[],
                long_trades=0,
                short_trades=0,
                avg_win=0.0,
                avg_loss=0.0,
            )
        return data
