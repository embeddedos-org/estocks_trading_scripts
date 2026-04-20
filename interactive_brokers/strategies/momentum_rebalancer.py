"""
Momentum Rebalancer Bot for Interactive Brokers
=================================================

Ranks a universe of symbols by composite momentum score, filters
by 200-day SMA, selects top N%, and generates rebalance orders to
achieve equal-weight target allocation.

Usage:
    rebalancer = MomentumRebalancer(
        connection, order_manager, fetcher,
        universe=["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"],
    )
    orders = rebalancer.rebalance()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RebalanceConfig:
    """Configuration for the momentum rebalancer."""

    # Momentum scoring weights
    weight_1m: float = 0.40
    weight_3m: float = 0.35
    weight_6m: float = 0.25

    # Selection
    top_pct: float = 0.20  # top 20% of universe
    min_holdings: int = 3
    max_holdings: int = 10

    # Filter
    require_above_200sma: bool = True

    # Allocation
    equal_weight: bool = True
    rebalance_threshold_pct: float = 5.0  # only rebalance if drift > 5%

    # Data
    lookback_duration: str = "1 Y"
    bar_size: str = "1 day"

    # Capital
    total_capital: float = 100000.0


@dataclass
class MomentumScore:
    """Momentum ranking for a single symbol."""
    symbol: str
    roc_1m: float
    roc_3m: float
    roc_6m: float
    composite_score: float
    above_200sma: bool
    current_price: float
    rank: int = 0


@dataclass
class RebalanceOrder:
    """A single rebalance order to execute."""
    symbol: str
    action: str  # "BUY" or "SELL"
    quantity: int
    current_shares: int
    target_shares: int
    current_weight: float
    target_weight: float


class MomentumRebalancer:
    """Monthly/weekly momentum-based portfolio rebalancer.

    Fetches historical data, ranks by composite momentum, filters
    by trend (200-SMA), and generates rebalance orders.

    Args:
        connection: An IBInsyncConnection instance.
        order_manager: OrderManager for trade execution.
        fetcher: HistoricalDataFetcher for bar data.
        risk_manager: Optional RiskManager for risk controls.
        notifier: Optional AlertDispatcher for notifications.
        universe: List of symbols to rank.
        config: RebalanceConfig with strategy parameters.
    """

    def __init__(
        self,
        connection: Any,
        order_manager: Any,
        fetcher: Any,
        risk_manager: Any = None,
        notifier: Any = None,
        universe: Optional[List[str]] = None,
        config: Optional[RebalanceConfig] = None,
    ) -> None:
        self.connection = connection
        self.order_manager = order_manager
        self.fetcher = fetcher
        self.risk_manager = risk_manager
        self.notifier = notifier
        self.config = config or RebalanceConfig()

        self.universe = universe or [
            "XLK", "XLF", "XLE", "XLV", "XLI",
            "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
        ]

        self._current_holdings: Dict[str, int] = {}
        self._rebalance_history: List[dict] = []

    # ─── Momentum Scoring ───

    def score_universe(self) -> List[MomentumScore]:
        """Fetch data and score all symbols in the universe.

        Returns:
            Sorted list of MomentumScore (highest first).
        """
        cfg = self.config
        scores: List[MomentumScore] = []

        if hasattr(self.fetcher, 'fetch_multiple'):
            data = self.fetcher.fetch_multiple(
                self.universe,
                duration=cfg.lookback_duration,
                bar_size=cfg.bar_size,
            )
        else:
            data = {}
            for sym in self.universe:
                try:
                    data[sym] = self.fetcher.fetch(
                        sym,
                        duration=cfg.lookback_duration,
                        bar_size=cfg.bar_size,
                    )
                except Exception as e:
                    logger.warning("Failed to fetch data for %s: %s", sym, e)
                    data[sym] = pd.DataFrame()

        for symbol, df in data.items():
            if df.empty or len(df) < 130:
                logger.warning("Insufficient data for %s (%d bars). Skipping.", symbol, len(df))
                continue

            close = df["close"]
            current_price = close.iloc[-1]

            # Rate of change
            roc_1m = self._calculate_roc(close, 21)
            roc_3m = self._calculate_roc(close, 63)
            roc_6m = self._calculate_roc(close, 126)

            if any(np.isnan(x) for x in [roc_1m, roc_3m, roc_6m]):
                logger.warning("NaN ROC for %s. Skipping.", symbol)
                continue

            composite = (
                roc_1m * cfg.weight_1m
                + roc_3m * cfg.weight_3m
                + roc_6m * cfg.weight_6m
            )

            sma_200 = close.rolling(window=200).mean().iloc[-1]
            above_200 = current_price > sma_200 if not np.isnan(sma_200) else False

            scores.append(
                MomentumScore(
                    symbol=symbol,
                    roc_1m=roc_1m,
                    roc_3m=roc_3m,
                    roc_6m=roc_6m,
                    composite_score=composite,
                    above_200sma=above_200,
                    current_price=current_price,
                )
            )

        # Filter by 200-SMA
        if cfg.require_above_200sma:
            filtered = [s for s in scores if s.above_200sma]
            logger.info(
                "200-SMA filter: %d/%d symbols passed.", len(filtered), len(scores)
            )
            scores = filtered

        # Sort by composite score (descending)
        scores.sort(key=lambda s: s.composite_score, reverse=True)
        for i, s in enumerate(scores):
            s.rank = i + 1

        logger.info("Scored %d symbols. Top: %s", len(scores),
                     [(s.symbol, round(s.composite_score, 4)) for s in scores[:5]])

        return scores

    @staticmethod
    def _calculate_roc(series: pd.Series, periods: int) -> float:
        """Calculate Rate of Change as a percentage."""
        if len(series) < periods + 1:
            return float("nan")
        return (series.iloc[-1] / series.iloc[-periods - 1] - 1) * 100

    # ─── Selection ───

    def select_holdings(self, scores: List[MomentumScore]) -> List[MomentumScore]:
        """Select top N% of scored symbols.

        Args:
            scores: Sorted list of MomentumScore.

        Returns:
            Selected symbols to hold.
        """
        cfg = self.config

        n_select = max(
            cfg.min_holdings,
            min(cfg.max_holdings, int(len(scores) * cfg.top_pct)),
        )

        selected = scores[:n_select]
        logger.info(
            "Selected %d/%d symbols: %s",
            len(selected),
            len(scores),
            [s.symbol for s in selected],
        )
        return selected

    # ─── Rebalance Calculation ───

    def calculate_rebalance(
        self,
        selected: List[MomentumScore],
        current_positions: Optional[Dict[str, int]] = None,
    ) -> List[RebalanceOrder]:
        """Calculate rebalance orders to reach target allocation.

        Args:
            selected: Symbols to hold with their prices.
            current_positions: Current shares held {symbol: qty}.

        Returns:
            List of RebalanceOrder to execute.
        """
        cfg = self.config
        current = current_positions or self._current_holdings

        target_symbols = {s.symbol for s in selected}
        prices = {s.symbol: s.current_price for s in selected}

        # Calculate target weights
        if cfg.equal_weight:
            n = len(selected)
            target_weights = {s.symbol: 1.0 / n for s in selected} if n > 0 else {}
        else:
            total_score = sum(s.composite_score for s in selected)
            if total_score > 0:
                target_weights = {s.symbol: s.composite_score / total_score for s in selected}
            else:
                n = len(selected)
                target_weights = {s.symbol: 1.0 / n for s in selected} if n > 0 else {}

        # Calculate target shares
        target_shares: Dict[str, int] = {}
        for symbol, weight in target_weights.items():
            dollar_amount = cfg.total_capital * weight
            price = prices.get(symbol)
            if price is None:
                logger.warning("No price for %s, skipping", symbol)
                continue
            target_shares[symbol] = int(dollar_amount / price) if price > 0 else 0

        orders: List[RebalanceOrder] = []

        # Sell symbols no longer in target
        for symbol, qty in current.items():
            if symbol not in target_symbols and qty > 0:
                orders.append(
                    RebalanceOrder(
                        symbol=symbol,
                        action="SELL",
                        quantity=qty,
                        current_shares=qty,
                        target_shares=0,
                        current_weight=0.0,
                        target_weight=0.0,
                    )
                )

        # Buy/sell to reach targets
        for symbol, target_qty in target_shares.items():
            current_qty = current.get(symbol, 0)
            diff = target_qty - current_qty

            current_value = current_qty * prices.get(symbol, 0)
            target_value = target_qty * prices.get(symbol, 0)
            current_weight = current_value / cfg.total_capital * 100 if cfg.total_capital > 0 else 0
            target_weight_pct = target_weights.get(symbol, 0) * 100

            # Only rebalance if drift exceeds threshold
            drift = abs(current_weight - target_weight_pct)
            if drift < cfg.rebalance_threshold_pct and symbol in current:
                logger.debug(
                    "Skipping %s: drift %.1f%% < threshold %.1f%%",
                    symbol, drift, cfg.rebalance_threshold_pct,
                )
                continue

            if diff > 0:
                orders.append(
                    RebalanceOrder(
                        symbol=symbol,
                        action="BUY",
                        quantity=diff,
                        current_shares=current_qty,
                        target_shares=target_qty,
                        current_weight=current_weight,
                        target_weight=target_weight_pct,
                    )
                )
            elif diff < 0:
                orders.append(
                    RebalanceOrder(
                        symbol=symbol,
                        action="SELL",
                        quantity=abs(diff),
                        current_shares=current_qty,
                        target_shares=target_qty,
                        current_weight=current_weight,
                        target_weight=target_weight_pct,
                    )
                )

        logger.info(
            "Rebalance plan: %d orders (%d buys, %d sells)",
            len(orders),
            sum(1 for o in orders if o.action == "BUY"),
            sum(1 for o in orders if o.action == "SELL"),
        )

        return orders

    # ─── Execution ───

    def execute_rebalance(self, orders: List[RebalanceOrder]) -> List[Any]:
        """Execute rebalance orders via OrderManager.

        Sells first, then buys.

        Args:
            orders: List of RebalanceOrder to execute.

        Returns:
            List of Trade objects from executed orders.
        """
        if self.risk_manager and not self.risk_manager.can_trade():
            logger.warning("RiskManager blocked rebalance.")
            return []

        trades = []

        # Sells first to free up capital
        sell_orders = [o for o in orders if o.action == "SELL"]
        buy_orders = [o for o in orders if o.action == "BUY"]

        for order in sell_orders + buy_orders:
            try:
                trade = self.order_manager.market_order(
                    symbol=order.symbol,
                    action=order.action,
                    quantity=order.quantity,
                )
                trades.append(trade)

                if order.action == "BUY":
                    self._current_holdings[order.symbol] = (
                        self._current_holdings.get(order.symbol, 0) + order.quantity
                    )
                else:
                    current = self._current_holdings.get(order.symbol, 0)
                    remaining = current - order.quantity
                    if remaining <= 0:
                        self._current_holdings.pop(order.symbol, None)
                    else:
                        self._current_holdings[order.symbol] = remaining

                logger.info(
                    "Executed: %s %d %s (target: %d shares, %.1f%% weight)",
                    order.action,
                    order.quantity,
                    order.symbol,
                    order.target_shares,
                    order.target_weight,
                )

            except Exception as e:
                logger.error("Failed to execute %s %s: %s", order.action, order.symbol, e)

        # Record rebalance
        self._rebalance_history.append({
            "timestamp": datetime.now().isoformat(),
            "orders": len(orders),
            "executed": len(trades),
            "holdings": dict(self._current_holdings),
        })

        # Notify
        if self.notifier and trades:
            summary = f"Rebalanced: {len(trades)} orders executed. Holdings: {list(self._current_holdings.keys())}"
            self.notifier.dispatch("Rebalance Complete", summary)

        return trades

    # ─── Full Rebalance Pipeline ───

    def rebalance(
        self, current_positions: Optional[Dict[str, int]] = None
    ) -> List[Any]:
        """Run the full rebalance pipeline: score → select → calculate → execute.

        Args:
            current_positions: Current shares held. Uses internal tracking if None.

        Returns:
            List of executed Trade objects.
        """
        logger.info("Starting momentum rebalance for %d-symbol universe...", len(self.universe))

        scores = self.score_universe()
        if not scores:
            logger.warning("No valid scores generated. Skipping rebalance.")
            return []

        selected = self.select_holdings(scores)
        orders = self.calculate_rebalance(selected, current_positions)

        if not orders:
            logger.info("No rebalance orders needed (within drift threshold).")
            return []

        return self.execute_rebalance(orders)

    def get_rebalance_history(self) -> List[dict]:
        """Return all rebalance events."""
        return list(self._rebalance_history)
