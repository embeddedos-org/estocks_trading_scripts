"""
Portfolio Tracker for Interactive Brokers
==========================================

Provides real-time portfolio snapshots including equity, cash,
margin, positions, P&L, and Greek exposures.

Usage:
    tracker = PortfolioTracker(connection)
    snapshot = tracker.get_snapshot()
    print(f"Equity: ${snapshot.equity:,.2f}")
    print(f"Daily P&L: ${snapshot.daily_pnl:+,.2f}")
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a single portfolio position."""
    symbol: str
    sec_type: str = "STK"
    quantity: float = 0.0
    avg_cost: float = 0.0
    market_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    sector: str = "Unknown"
    currency: str = "USD"
    account: str = ""

    # Option-specific Greeks
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0

    @property
    def total_cost(self) -> float:
        """Total cost basis of the position."""
        return self.quantity * self.avg_cost

    @property
    def return_pct(self) -> float:
        """Percentage return on the position."""
        if self.total_cost != 0:
            return (self.unrealized_pnl / abs(self.total_cost)) * 100.0
        return 0.0

    @property
    def weight(self) -> float:
        """Position weight as fraction of market value (set externally)."""
        return 0.0


@dataclass
class PortfolioSnapshot:
    """Complete snapshot of portfolio state at a point in time."""
    timestamp: datetime = field(default_factory=datetime.now)
    account_id: str = ""
    equity: float = 0.0
    cash: float = 0.0
    buying_power: float = 0.0
    net_liquidation: float = 0.0
    initial_margin: float = 0.0
    maintenance_margin: float = 0.0
    available_funds: float = 0.0
    excess_liquidity: float = 0.0
    cushion: float = 0.0
    daily_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    positions: List[Position] = field(default_factory=list)

    # Portfolio-level Greeks (aggregate)
    total_delta: float = 0.0
    total_gamma: float = 0.0
    total_theta: float = 0.0
    total_vega: float = 0.0

    @property
    def num_positions(self) -> int:
        return len(self.positions)

    @property
    def margin_utilization(self) -> float:
        """Percentage of available margin being used."""
        if self.net_liquidation > 0:
            return (self.initial_margin / self.net_liquidation) * 100.0
        return 0.0

    @property
    def total_market_value(self) -> float:
        return sum(p.market_value for p in self.positions)


# Standard sector classification
SECTOR_MAP: Dict[str, str] = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "META": "Communication Services", "NFLX": "Communication Services",
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
    "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "PG": "Consumer Staples", "KO": "Consumer Staples", "WMT": "Consumer Staples",
    "CAT": "Industrials", "BA": "Industrials", "HON": "Industrials",
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    "AMT": "Real Estate", "PLD": "Real Estate", "CCI": "Real Estate",
    "LIN": "Materials", "APD": "Materials", "SHW": "Materials",
}


class PortfolioTracker:
    """Tracks portfolio state, positions, and P&L from IB.

    Args:
        connection: An IBInsyncConnection instance.
        sector_map: Optional custom symbol → sector mapping.
    """

    def __init__(
        self,
        connection: Any,
        sector_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self.connection = connection
        self.sector_map = sector_map or SECTOR_MAP
        self._snapshots: List[PortfolioSnapshot] = []

    def _get_account_values(self) -> Dict[str, str]:
        """Fetch account summary values from IB.

        Returns:
            Dictionary of tag → value pairs.
        """
        values: Dict[str, str] = {}
        try:
            summary = self.connection.accountSummary()
            for item in summary:
                values[item.tag] = item.value
        except Exception as e:
            logger.error("Failed to fetch account summary: %s", e)
        return values

    def _get_portfolio_items(self) -> list:
        """Fetch portfolio items from IB."""
        try:
            return self.connection.portfolio()
        except Exception as e:
            logger.error("Failed to fetch portfolio: %s", e)
            return []

    def get_positions(self) -> List[Position]:
        """Get all current positions with market data.

        Returns:
            List of Position objects with current pricing and P&L.
        """
        items = self._get_portfolio_items()
        positions = []

        for item in items:
            symbol = item.contract.symbol
            sec_type = item.contract.secType

            pos = Position(
                symbol=symbol,
                sec_type=sec_type,
                quantity=item.position,
                avg_cost=item.averageCost,
                market_price=item.marketPrice,
                market_value=item.marketValue,
                unrealized_pnl=item.unrealizedPNL,
                realized_pnl=item.realizedPNL,
                sector=self.sector_map.get(symbol, "Unknown"),
                currency=item.contract.currency,
                account=item.account,
            )

            if hasattr(item, 'contract') and sec_type == "OPT":
                try:
                    ticker = self.connection.ib.reqMktData(item.contract)
                    self.connection.ib.sleep(1)
                    if ticker.modelGreeks:
                        multiplier = item.position * 100
                        pos.delta = ticker.modelGreeks.delta * multiplier
                        pos.gamma = ticker.modelGreeks.gamma * multiplier
                        pos.theta = ticker.modelGreeks.theta * multiplier
                        pos.vega = ticker.modelGreeks.vega * multiplier
                    self.connection.ib.cancelMktData(item.contract)
                except Exception as e:
                    logger.debug("Could not fetch Greeks for %s: %s", symbol, e)

            positions.append(pos)

        logger.info("Fetched %d positions", len(positions))
        return positions

    def get_snapshot(self) -> PortfolioSnapshot:
        """Get a complete portfolio snapshot.

        Returns:
            PortfolioSnapshot with equity, cash, margin, positions,
            P&L, and aggregate Greeks.
        """
        values = self._get_account_values()
        positions = self.get_positions()

        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            account_id=values.get("AccountCode", ""),
            equity=float(values.get("EquityWithLoanValue", 0)),
            cash=float(values.get("TotalCashValue", 0)),
            buying_power=float(values.get("BuyingPower", 0)),
            net_liquidation=float(values.get("NetLiquidation", 0)),
            initial_margin=float(values.get("InitMarginReq", 0)),
            maintenance_margin=float(values.get("MaintMarginReq", 0)),
            available_funds=float(values.get("AvailableFunds", 0)),
            excess_liquidity=float(values.get("ExcessLiquidity", 0)),
            cushion=float(values.get("Cushion", 0)),
            unrealized_pnl=sum(p.unrealized_pnl for p in positions),
            realized_pnl=sum(p.realized_pnl for p in positions),
            positions=positions,
            total_delta=sum(p.delta for p in positions),
            total_gamma=sum(p.gamma for p in positions),
            total_theta=sum(p.theta for p in positions),
            total_vega=sum(p.vega for p in positions),
        )

        self._snapshots.append(snapshot)

        logger.info(
            "Portfolio snapshot: equity=$%.2f, cash=$%.2f, "
            "positions=%d, unrealized_pnl=$%.2f",
            snapshot.equity, snapshot.cash,
            snapshot.num_positions, snapshot.unrealized_pnl,
        )
        return snapshot

    def get_pnl_summary(self) -> Dict[str, Any]:
        """Get a P&L summary across all positions.

        Returns:
            Dictionary with total_unrealized, total_realized,
            winners, losers, best_performer, worst_performer.
        """
        positions = self.get_positions()
        if not positions:
            return {"total_positions": 0}

        winners = [p for p in positions if p.unrealized_pnl > 0]
        losers = [p for p in positions if p.unrealized_pnl < 0]

        sorted_by_pnl = sorted(positions, key=lambda p: p.unrealized_pnl)

        return {
            "total_positions": len(positions),
            "total_unrealized_pnl": sum(p.unrealized_pnl for p in positions),
            "total_realized_pnl": sum(p.realized_pnl for p in positions),
            "num_winners": len(winners),
            "num_losers": len(losers),
            "total_market_value": sum(p.market_value for p in positions),
            "best_performer": {
                "symbol": sorted_by_pnl[-1].symbol,
                "pnl": sorted_by_pnl[-1].unrealized_pnl,
                "return_pct": sorted_by_pnl[-1].return_pct,
            } if positions else None,
            "worst_performer": {
                "symbol": sorted_by_pnl[0].symbol,
                "pnl": sorted_by_pnl[0].unrealized_pnl,
                "return_pct": sorted_by_pnl[0].return_pct,
            } if positions else None,
        }

    def sector_exposure(self) -> Dict[str, Dict[str, float]]:
        """Calculate portfolio exposure by sector.

        Returns:
            Dictionary mapping sector → {value, weight, pnl, num_positions}.
        """
        positions = self.get_positions()
        total_value = sum(abs(p.market_value) for p in positions)

        sectors: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"value": 0.0, "weight": 0.0, "pnl": 0.0, "num_positions": 0}
        )

        for pos in positions:
            sector = pos.sector
            sectors[sector]["value"] += pos.market_value
            sectors[sector]["pnl"] += pos.unrealized_pnl
            sectors[sector]["num_positions"] += 1

        if total_value > 0:
            for sector in sectors:
                sectors[sector]["weight"] = (
                    sectors[sector]["value"] / total_value
                ) * 100.0

        logger.info("Sector exposure: %d sectors", len(sectors))
        return dict(sectors)

    def get_snapshot_history(self) -> List[PortfolioSnapshot]:
        """Return all stored portfolio snapshots."""
        return list(self._snapshots)
