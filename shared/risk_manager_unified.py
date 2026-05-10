"""
Unified Portfolio Risk Gate
==============================

Cross-strategy, cross-broker portfolio risk coordination.

Singleton that enforces portfolio-wide limits across ALL strategies and brokers:
- Max total portfolio exposure (% of equity)
- Max single-stock concentration
- Max sector concentration
- Max correlated exposure
- Daily loss limit across all strategies

Usage:
    gate = UnifiedPortfolioRiskGate.get_instance(config)
    ok, reason = gate.can_open_position("AAPL", 15000, sector="Technology")
    if ok:
        gate.register_position("AAPL", 100, 150.0, "momentum", "ib")
    gate.close_position("AAPL", pnl=200.0)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from shared.config.sector_map import SECTOR_MAP as _UNIFIED_SECTOR_MAP
except ImportError:
    _UNIFIED_SECTOR_MAP = None

logger = logging.getLogger(__name__)


@dataclass
class PositionInfo:
    """A single position tracked across all strategies and brokers."""

    symbol: str
    shares: int
    entry_price: float
    notional: float
    strategy: str
    broker: str
    sector: str = "Unknown"
    opened_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class UnifiedRiskConfig:
    """Configuration for the unified portfolio risk gate."""

    max_portfolio_exposure: float = 0.80  # 80% of equity
    max_single_stock_pct: float = 0.15  # 15% of equity per symbol
    max_sector_pct: float = 0.30  # 30% of equity per sector
    max_correlated_exposure: float = 0.40  # 40% correlated group
    max_daily_loss: float = 10000.0
    account_equity: float = 100000.0
    persist_path: str = ""


# ─── Sector classification (lightweight built-in) ───

_SECTOR_MAP: Dict[str, str] = _UNIFIED_SECTOR_MAP if _UNIFIED_SECTOR_MAP is not None else {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "GOOG": "Technology", "META": "Technology", "AMZN": "Consumer Discretionary",
    "NVDA": "Technology", "TSLA": "Consumer Discretionary", "AMD": "Technology",
    "INTC": "Technology", "CRM": "Technology", "ADBE": "Technology",
    "NFLX": "Communication Services", "DIS": "Communication Services",
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
    "MS": "Financials", "WFC": "Financials", "C": "Financials",
    "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare",
    "MRK": "Healthcare", "ABBV": "Healthcare", "LLY": "Healthcare",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "WMT": "Consumer Staples", "PG": "Consumer Staples", "KO": "Consumer Staples",
    "PEP": "Consumer Staples", "COST": "Consumer Staples",
}

# ─── Simple correlation groups ───

_CORRELATION_GROUPS: List[set] = [
    {"AAPL", "MSFT", "GOOGL", "GOOG", "META", "NVDA", "AMD", "INTC", "CRM", "ADBE"},
    {"JPM", "BAC", "GS", "MS", "WFC", "C"},
    {"XOM", "CVX", "COP"},
    {"JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY"},
    {"WMT", "PG", "KO", "PEP", "COST"},
    {"AMZN", "TSLA", "NFLX", "DIS"},
]


class UnifiedPortfolioRiskGate:
    """Cross-strategy, cross-broker portfolio risk coordination.

    Enforces portfolio-wide limits that no single strategy's RiskManager
    can see on its own. Thread-safe singleton.
    """

    _instance: Optional["UnifiedPortfolioRiskGate"] = None
    _init_lock = threading.Lock()

    @classmethod
    def get_instance(
        cls, config: Optional[UnifiedRiskConfig] = None
    ) -> "UnifiedPortfolioRiskGate":
        """Return the singleton instance, creating it if necessary."""
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls(config or UnifiedRiskConfig())
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (useful for tests)."""
        cls._instance = None

    def __init__(self, config: UnifiedRiskConfig) -> None:
        self._positions: Dict[str, PositionInfo] = {}
        self._total_exposure: float = 0.0
        self._max_portfolio_exposure: float = config.max_portfolio_exposure
        self._max_single_stock_pct: float = config.max_single_stock_pct
        self._max_sector_pct: float = config.max_sector_pct
        self._max_correlated_exposure: float = config.max_correlated_exposure
        self._daily_pnl: float = 0.0
        self._daily_pnl_date: date = date.today()
        self._max_daily_loss: float = config.max_daily_loss
        self._account_equity: float = config.account_equity
        self._lock = threading.Lock()

        # Persistence
        self._persist_conn: Optional[sqlite3.Connection] = None
        self._persist_lock = threading.Lock()
        if config.persist_path:
            self._init_persistence(config.persist_path)
            self._load_state()

        logger.info(
            "UnifiedPortfolioRiskGate initialized: equity=$%.0f, "
            "max_exposure=%.0f%%, max_single=%.0f%%, max_sector=%.0f%%",
            self._account_equity,
            self._max_portfolio_exposure * 100,
            self._max_single_stock_pct * 100,
            self._max_sector_pct * 100,
        )

    # ─── Persistence ───

    def _init_persistence(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._persist_conn = sqlite3.connect(db_path, check_same_thread=False)
        # FIX 8: SQLite WAL mode for better concurrency
        self._persist_conn.execute("PRAGMA journal_mode=WAL")
        self._persist_conn.execute("PRAGMA busy_timeout=5000")
        self._persist_conn.execute(
            "CREATE TABLE IF NOT EXISTS unified_risk_state "
            "(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
        )
        self._persist_conn.commit()

    def _save_state(self) -> None:
        if self._persist_conn is None:
            return
        now = datetime.now().isoformat()
        positions_ser = {
            sym: {
                "symbol": p.symbol,
                "shares": p.shares,
                "entry_price": p.entry_price,
                "notional": p.notional,
                "strategy": p.strategy,
                "broker": p.broker,
                "sector": p.sector,
                "opened_at": p.opened_at.isoformat(),
            }
            for sym, p in self._positions.items()
        }
        state = {
            "positions": positions_ser,
            "total_exposure": self._total_exposure,
            "daily_pnl": self._daily_pnl,
            "daily_pnl_date": self._daily_pnl_date.isoformat(),
            "account_equity": self._account_equity,
        }
        with self._persist_lock:
            try:
                for key, value in state.items():
                    self._persist_conn.execute(
                        "INSERT OR REPLACE INTO unified_risk_state (key, value, updated_at) "
                        "VALUES (?, ?, ?)",
                        (key, json.dumps(value), now),
                    )
                self._persist_conn.commit()
            except Exception as e:
                logger.error("Failed to save unified risk state: %s", e)

    def _load_state(self) -> None:
        if self._persist_conn is None:
            return
        with self._persist_lock:
            try:
                rows = self._persist_conn.execute(
                    "SELECT key, value FROM unified_risk_state"
                ).fetchall()
            except Exception as e:
                logger.error("Failed to load unified risk state: %s", e)
                return

        if not rows:
            return

        state = {key: json.loads(value) for key, value in rows}

        saved_date_str = state.get("daily_pnl_date")
        if saved_date_str and date.fromisoformat(saved_date_str) == date.today():
            self._daily_pnl = float(state.get("daily_pnl", 0.0))
        else:
            self._daily_pnl = 0.0

        self._daily_pnl_date = date.today()
        self._account_equity = float(state.get("account_equity", self._account_equity))
        self._total_exposure = float(state.get("total_exposure", 0.0))

        positions_raw = state.get("positions", {})
        if isinstance(positions_raw, dict):
            for sym, p in positions_raw.items():
                self._positions[sym] = PositionInfo(
                    symbol=p["symbol"],
                    shares=int(p["shares"]),
                    entry_price=float(p["entry_price"]),
                    notional=float(p["notional"]),
                    strategy=p["strategy"],
                    broker=p["broker"],
                    sector=p.get("sector", "Unknown"),
                    opened_at=datetime.fromisoformat(p["opened_at"]),
                )

        logger.info(
            "Unified risk state restored: %d positions, exposure=$%.2f, daily_pnl=$%.2f",
            len(self._positions), self._total_exposure, self._daily_pnl,
        )

    # ─── Risk Gates ───

    def _reset_daily_if_needed(self) -> None:
        if self._daily_pnl_date != date.today():
            logger.info("UnifiedRisk: new day — resetting daily P&L (was $%.2f)", self._daily_pnl)
            self._daily_pnl = 0.0
            self._daily_pnl_date = date.today()

    def can_open_position(
        self, symbol: str, notional_value: float, sector: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Check if a new position is allowed across the entire portfolio.

        Args:
            symbol: Ticker symbol.
            notional_value: Dollar value of the proposed position.
            sector: Sector of the symbol (auto-detected if None).

        Returns:
            (allowed, reason) tuple.
        """
        with self._lock:
            self._reset_daily_if_needed()

            if sector is None:
                sector = _SECTOR_MAP.get(symbol, "Unknown")

            # Daily loss limit
            if self._daily_pnl <= -self._max_daily_loss:
                reason = (
                    f"Daily loss limit reached: ${self._daily_pnl:.2f} "
                    f"(max: ${self._max_daily_loss:.2f})"
                )
                logger.warning("UnifiedRisk BLOCKED %s: %s", symbol, reason)
                return False, reason

            # Total portfolio exposure
            new_exposure = self._total_exposure + notional_value
            exposure_pct = new_exposure / self._account_equity if self._account_equity > 0 else 1.0
            if exposure_pct > self._max_portfolio_exposure:
                reason = (
                    f"Portfolio exposure would be {exposure_pct:.1%} "
                    f"(max: {self._max_portfolio_exposure:.1%})"
                )
                logger.warning("UnifiedRisk BLOCKED %s: %s", symbol, reason)
                return False, reason

            # Single stock concentration
            existing_notional = self._positions[symbol].notional if symbol in self._positions else 0.0
            stock_total = existing_notional + notional_value
            stock_pct = stock_total / self._account_equity if self._account_equity > 0 else 1.0
            if stock_pct > self._max_single_stock_pct:
                reason = (
                    f"Single-stock concentration for {symbol} would be {stock_pct:.1%} "
                    f"(max: {self._max_single_stock_pct:.1%})"
                )
                logger.warning("UnifiedRisk BLOCKED %s: %s", symbol, reason)
                return False, reason

            # Sector concentration
            sector_total = notional_value
            for pos in self._positions.values():
                if pos.sector == sector:
                    sector_total += pos.notional
            sector_pct = sector_total / self._account_equity if self._account_equity > 0 else 1.0
            if sector_pct > self._max_sector_pct:
                reason = (
                    f"Sector '{sector}' exposure would be {sector_pct:.1%} "
                    f"(max: {self._max_sector_pct:.1%})"
                )
                logger.warning("UnifiedRisk BLOCKED %s: %s", symbol, reason)
                return False, reason

            # Correlation check
            corr_exposure = self.check_correlation_risk(symbol, set(self._positions.keys()))
            combined_corr = corr_exposure + notional_value
            corr_pct = combined_corr / self._account_equity if self._account_equity > 0 else 1.0
            if corr_pct > self._max_correlated_exposure:
                reason = (
                    f"Correlated exposure would be {corr_pct:.1%} "
                    f"(max: {self._max_correlated_exposure:.1%})"
                )
                logger.warning("UnifiedRisk BLOCKED %s: %s", symbol, reason)
                return False, reason

            return True, "OK"

    def register_position(
        self,
        symbol: str,
        shares: int,
        price: float,
        strategy_name: str,
        broker_name: str,
        sector: Optional[str] = None,
    ) -> None:
        """Register a position from any strategy/broker.

        Args:
            symbol: Ticker symbol.
            shares: Number of shares.
            price: Entry price.
            strategy_name: Name of the strategy.
            broker_name: Name of the broker.
            sector: Sector (auto-detected if None).
        """
        if sector is None:
            sector = _SECTOR_MAP.get(symbol, "Unknown")

        notional = abs(shares * price)

        with self._lock:
            self._positions[symbol] = PositionInfo(
                symbol=symbol,
                shares=shares,
                entry_price=price,
                notional=notional,
                strategy=strategy_name,
                broker=broker_name,
                sector=sector,
            )
            self._total_exposure = sum(p.notional for p in self._positions.values())

        logger.info(
            "UnifiedRisk: registered %s %d shares @ $%.2f ($%.0f) [%s/%s]",
            symbol, shares, price, notional, strategy_name, broker_name,
        )
        self._save_state()

    def close_position(self, symbol: str, pnl: float) -> None:
        """Record a position close and update P&L.

        Args:
            symbol: Ticker symbol.
            pnl: Realized P&L of the closed position.
        """
        with self._lock:
            self._reset_daily_if_needed()
            self._daily_pnl += pnl
            self._account_equity += pnl

            if symbol in self._positions:
                del self._positions[symbol]
                self._total_exposure = sum(p.notional for p in self._positions.values())

        logger.info(
            "UnifiedRisk: closed %s | P&L=$%.2f | Daily P&L=$%.2f | Remaining positions=%d",
            symbol, pnl, self._daily_pnl, len(self._positions),
        )
        self._save_state()

    def get_portfolio_summary(self) -> dict:
        """Return current portfolio exposure, P&L, positions.

        Returns:
            Dict with exposure, P&L, positions, sector breakdown.
        """
        with self._lock:
            self._reset_daily_if_needed()

            sector_exposure: Dict[str, float] = {}
            for pos in self._positions.values():
                sector_exposure[pos.sector] = sector_exposure.get(pos.sector, 0.0) + pos.notional

            exposure_pct = (
                self._total_exposure / self._account_equity
                if self._account_equity > 0 else 0.0
            )

            return {
                "account_equity": round(self._account_equity, 2),
                "total_exposure": round(self._total_exposure, 2),
                "exposure_pct": round(exposure_pct * 100, 2),
                "daily_pnl": round(self._daily_pnl, 2),
                "open_positions": len(self._positions),
                "positions": {
                    sym: {
                        "shares": p.shares,
                        "entry_price": p.entry_price,
                        "notional": p.notional,
                        "strategy": p.strategy,
                        "broker": p.broker,
                        "sector": p.sector,
                    }
                    for sym, p in self._positions.items()
                },
                "sector_exposure": {
                    s: {"notional": round(v, 2), "pct": round(v / self._account_equity * 100, 2)}
                    for s, v in sector_exposure.items()
                },
            }

    def check_correlation_risk(self, symbol: str, existing_symbols: set) -> float:
        """Estimate total notional exposure in the same correlation group.

        Args:
            symbol: New symbol being considered.
            existing_symbols: Set of symbols already in portfolio.

        Returns:
            Total notional value of existing correlated positions.
        """
        correlated_notional = 0.0

        for group in _CORRELATION_GROUPS:
            if symbol in group:
                for sym in existing_symbols:
                    if sym in group and sym in self._positions:
                        correlated_notional += self._positions[sym].notional
                break

        return correlated_notional

    # ─── FIX 7: Mark-to-Market ───

    def update_mark_to_market(self, symbol: str, current_price: float) -> None:
        """Update a position's notional value with the current market price.

        Should be called from broker_bridge on each tick/price update to keep
        exposure calculations accurate.

        Args:
            symbol: Ticker symbol.
            current_price: Current market price.
        """
        with self._lock:
            pos = self._positions.get(symbol)
            if pos is None:
                return
            old_notional = pos.notional
            pos.notional = abs(pos.shares * current_price)
            self._total_exposure = sum(p.notional for p in self._positions.values())

        if abs(old_notional - pos.notional) > 1.0:
            logger.debug(
                "Mark-to-market %s: $%.0f → $%.0f (price=$%.2f)",
                symbol, old_notional, pos.notional, current_price,
            )

    # ─── Split / Dividend Adjustment ───

    def adjust_for_split(self, symbol: str, ratio: float) -> None:
        """Adjust a position for a stock split or reverse split.

        For a 4-for-1 split use ``ratio=4.0``; for a 1-for-10 reverse
        split use ``ratio=0.1``.

        The method multiplies shares by *ratio* and divides entry_price
        by *ratio* so that the notional value stays the same.

        Args:
            symbol: Ticker symbol to adjust.
            ratio: Split ratio (e.g. 4.0 for a 4:1 split).
        """
        if ratio <= 0:
            logger.warning("adjust_for_split: invalid ratio %.4f for %s", ratio, symbol)
            return

        with self._lock:
            pos = self._positions.get(symbol)
            if pos is None:
                logger.warning("adjust_for_split: no position for %s", symbol)
                return

            old_shares = pos.shares
            old_price = pos.entry_price

            pos.shares = int(pos.shares * ratio)
            pos.entry_price = pos.entry_price / ratio
            # Notional stays the same: shares * price is invariant
            pos.notional = abs(pos.shares * pos.entry_price)

            logger.info(
                "Split adjustment for %s (ratio=%.2f): "
                "%d shares @ $%.2f → %d shares @ $%.4f",
                symbol, ratio, old_shares, old_price, pos.shares, pos.entry_price,
            )

        self._save_state()

    def __repr__(self) -> str:
        return (
            f"UnifiedPortfolioRiskGate(positions={len(self._positions)}, "
            f"exposure=${self._total_exposure:,.0f}, daily_pnl=${self._daily_pnl:+,.2f})"
        )
