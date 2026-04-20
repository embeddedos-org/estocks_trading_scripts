"""
Standalone Risk Manager
========================

Provides position sizing, daily loss limits, cooldown after consecutive
losses, portfolio heat limits, trade frequency throttling, and drawdown
circuit breakers. Used by all Python trading strategies.

Usage:
    rm = RiskManager(config=RiskManagerConfig(max_daily_loss=5000))
    size = rm.calculate_position_size("AAPL", 150.0, stop_price=145.0)
    if rm.can_trade():
        # place order
        rm.record_trade(pnl=-200)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class SizingMethod(Enum):
    FIXED_FRACTIONAL = "fixed_fractional"
    KELLY = "kelly"
    FIXED_SHARES = "fixed_shares"
    FIXED_DOLLAR = "fixed_dollar"


@dataclass
class RiskManagerConfig:
    """Configuration for the RiskManager."""

    # Position sizing
    sizing_method: SizingMethod = SizingMethod.FIXED_FRACTIONAL
    risk_per_trade_pct: float = 2.0
    fixed_shares: int = 100
    fixed_dollar_amount: float = 10000.0

    # Kelly criterion parameters
    kelly_win_rate: float = 0.55
    kelly_avg_win: float = 1.5
    kelly_avg_loss: float = 1.0
    kelly_fraction: float = 0.5  # half-Kelly for safety

    # Daily loss limit
    max_daily_loss: float = 5000.0
    auto_flatten_on_daily_loss: bool = True

    # Consecutive loss cooldown
    max_consecutive_losses: int = 3
    cooldown_seconds: int = 1800  # 30 minutes

    # Portfolio heat (total exposure)
    max_portfolio_heat_pct: float = 20.0  # max % of portfolio at risk
    max_open_positions: int = 10

    # Trade frequency
    max_trades_per_hour: int = 10
    min_seconds_between_trades: float = 30.0

    # Drawdown circuit breaker
    max_drawdown_pct: float = 10.0
    circuit_breaker_pause_hours: float = 24.0

    # Capital
    total_capital: float = 100000.0

    # State persistence (crash recovery)
    persist_path: Optional[str] = None


@dataclass
class TradeRecord:
    """Record of a completed trade for risk tracking."""
    symbol: str
    pnl: float
    timestamp: datetime = field(default_factory=datetime.now)
    is_win: bool = field(init=False)

    def __post_init__(self) -> None:
        self.is_win = self.pnl > 0


class RiskManager:
    """Manages risk controls across all trading strategies.

    Tracks daily P&L, consecutive losses, trade frequency, and
    portfolio exposure. All strategies should check can_trade()
    before placing orders and record_trade() after fills.

    Args:
        config: RiskManagerConfig with all risk parameters.
    """

    def __init__(self, config: Optional[RiskManagerConfig] = None) -> None:
        self.config = config or RiskManagerConfig()

        # Daily tracking
        self._daily_pnl: float = 0.0
        self._daily_pnl_date: date = date.today()
        self._daily_trade_count: int = 0

        # Consecutive loss tracking
        self._consecutive_losses: int = 0
        self._cooldown_until: float = 0.0

        # Trade frequency tracking
        self._trade_timestamps: List[float] = []
        self._last_trade_time: float = 0.0

        # Drawdown tracking
        self._peak_equity: float = self.config.total_capital
        self._current_equity: float = self.config.total_capital
        self._circuit_breaker_until: float = 0.0

        # Open positions tracking
        self._open_positions: dict[str, float] = {}  # symbol -> risk_amount

        # Trade history
        self._trade_history: List[TradeRecord] = []

        # State persistence
        self._persist_conn: Optional[sqlite3.Connection] = None
        self._persist_lock = threading.Lock()
        if self.config.persist_path:
            self._init_persistence(self.config.persist_path)
            self._load_state()

    # ─── State Persistence ───

    def _init_persistence(self, db_path: str) -> None:
        """Initialize the SQLite database for state persistence."""
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._persist_conn = sqlite3.connect(db_path, check_same_thread=False)
        # FIX 8: SQLite WAL mode for better concurrency
        self._persist_conn.execute("PRAGMA journal_mode=WAL")
        self._persist_conn.execute("PRAGMA busy_timeout=5000")
        self._persist_conn.execute(
            "CREATE TABLE IF NOT EXISTS risk_state "
            "(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
        )
        self._persist_conn.commit()
        logger.info("RiskManager persistence initialized: %s", db_path)

    def _save_state(self) -> None:
        """Persist current risk state to SQLite."""
        if self._persist_conn is None:
            return

        now = datetime.now().isoformat()
        state = {
            "daily_pnl": self._daily_pnl,
            "daily_pnl_date": self._daily_pnl_date.isoformat(),
            "consecutive_losses": self._consecutive_losses,
            "current_equity": self._current_equity,
            "peak_equity": self._peak_equity,
            "circuit_breaker_until": self._circuit_breaker_until,
            "cooldown_until": self._cooldown_until,
            "daily_trade_count": self._daily_trade_count,
            "open_positions": self._open_positions,
        }

        with self._persist_lock:
            try:
                for key, value in state.items():
                    self._persist_conn.execute(
                        "INSERT OR REPLACE INTO risk_state (key, value, updated_at) "
                        "VALUES (?, ?, ?)",
                        (key, json.dumps(value), now),
                    )
                self._persist_conn.commit()
            except Exception as e:
                logger.error("Failed to save risk state: %s", e)

    def _load_state(self) -> None:
        """Restore risk state from SQLite on startup."""
        if self._persist_conn is None:
            return

        with self._persist_lock:
            try:
                rows = self._persist_conn.execute(
                    "SELECT key, value FROM risk_state"
                ).fetchall()
            except Exception as e:
                logger.error("Failed to load risk state: %s", e)
                return

        if not rows:
            logger.info("No persisted risk state found — starting fresh")
            return

        state = {key: json.loads(value) for key, value in rows}

        saved_date_str = state.get("daily_pnl_date")
        if saved_date_str:
            saved_date = date.fromisoformat(saved_date_str)
            if saved_date == date.today():
                self._daily_pnl = float(state.get("daily_pnl", 0.0))
                self._daily_trade_count = int(state.get("daily_trade_count", 0))
                self._daily_pnl_date = saved_date
            else:
                logger.info("Persisted state is from %s — resetting daily counters", saved_date_str)

        self._consecutive_losses = int(state.get("consecutive_losses", 0))
        self._current_equity = float(state.get("current_equity", self.config.total_capital))
        self._peak_equity = float(state.get("peak_equity", self.config.total_capital))
        self._circuit_breaker_until = float(state.get("circuit_breaker_until", 0.0))
        self._cooldown_until = float(state.get("cooldown_until", 0.0))

        positions = state.get("open_positions")
        if isinstance(positions, dict):
            self._open_positions = {k: float(v) for k, v in positions.items()}

        logger.info(
            "Risk state restored: equity=$%.2f, daily_pnl=$%.2f, "
            "consecutive_losses=%d, positions=%d",
            self._current_equity, self._daily_pnl,
            self._consecutive_losses, len(self._open_positions),
        )

    def _reset_daily_if_needed(self) -> None:
        """Reset daily counters if a new day has started."""
        if self._daily_pnl_date != date.today():
            logger.info(
                "New day — resetting daily counters. Previous day P&L: $%.2f",
                self._daily_pnl,
            )
            self._daily_pnl = 0.0
            self._daily_trade_count = 0
            self._daily_pnl_date = date.today()

    # ─── Position Sizing ───

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_price: Optional[float] = None,
        atr: Optional[float] = None,
    ) -> int:
        """Calculate position size based on configured sizing method.

        Args:
            symbol: Ticker symbol.
            entry_price: Planned entry price.
            stop_price: Stop loss price (required for fixed_fractional).
            atr: Average True Range (alternative to stop_price).

        Returns:
            Number of shares to trade.
        """
        method = self.config.sizing_method

        if method == SizingMethod.FIXED_SHARES:
            return self.config.fixed_shares

        if method == SizingMethod.FIXED_DOLLAR:
            if entry_price <= 0:
                return 0
            return max(1, int(self.config.fixed_dollar_amount / entry_price))

        if method == SizingMethod.KELLY:
            return self._kelly_size(entry_price)

        # FIXED_FRACTIONAL (default)
        return self._fixed_fractional_size(entry_price, stop_price, atr)

    def _fixed_fractional_size(
        self,
        entry_price: float,
        stop_price: Optional[float] = None,
        atr: Optional[float] = None,
    ) -> int:
        """Risk a fixed % of capital per trade.

        risk_amount = capital * risk_pct
        shares = risk_amount / risk_per_share
        """
        risk_amount = self._current_equity * (self.config.risk_per_trade_pct / 100.0)

        if stop_price is not None and stop_price != entry_price:
            risk_per_share = abs(entry_price - stop_price)
        elif atr is not None and atr > 0:
            risk_per_share = atr * 2.0
        else:
            risk_per_share = entry_price * 0.02  # default 2% of price

        if risk_per_share <= 0:
            return 0

        shares = int(risk_amount / risk_per_share)
        return max(1, shares)

    def _kelly_size(self, entry_price: float) -> int:
        """Kelly criterion position sizing (half-Kelly by default).

        f* = (W * R - L) / R
        where W = win rate, R = avg_win / avg_loss, L = loss rate
        """
        w = self.config.kelly_win_rate
        r = (
            self.config.kelly_avg_win / self.config.kelly_avg_loss
            if self.config.kelly_avg_loss > 0
            else 1.0
        )
        loss_rate = 1.0 - w

        kelly_pct = (w * r - loss_rate) / r if r > 0 else 0
        kelly_pct = max(0, kelly_pct) * self.config.kelly_fraction

        dollar_amount = self._current_equity * kelly_pct
        if entry_price <= 0:
            return 0

        return max(1, int(dollar_amount / entry_price))

    # ─── Trade Validation ───

    def can_trade(self) -> bool:
        """Check all risk gates. Returns True if trading is allowed.

        Checks: daily loss, cooldown, circuit breaker, trade frequency.
        """
        self._reset_daily_if_needed()

        # Daily loss limit
        if self._daily_pnl <= -self.config.max_daily_loss:
            logger.warning(
                "Daily loss limit reached: $%.2f (max: $%.2f)",
                self._daily_pnl,
                self.config.max_daily_loss,
            )
            return False

        # Consecutive loss cooldown
        now = time.time()
        if now < self._cooldown_until:
            remaining = self._cooldown_until - now
            logger.warning(
                "Cooldown active: %d consecutive losses. %.0fs remaining.",
                self._consecutive_losses,
                remaining,
            )
            return False

        # Drawdown circuit breaker
        if now < self._circuit_breaker_until:
            remaining = self._circuit_breaker_until - now
            logger.warning(
                "Circuit breaker active. %.0fs remaining.",
                remaining,
            )
            return False

        # Trade frequency: max per hour
        cutoff = now - 3600
        recent_trades = [t for t in self._trade_timestamps if t > cutoff]
        if len(recent_trades) >= self.config.max_trades_per_hour:
            logger.warning(
                "Trade frequency limit: %d trades in last hour (max: %d)",
                len(recent_trades),
                self.config.max_trades_per_hour,
            )
            return False

        # Minimum time between trades
        if now - self._last_trade_time < self.config.min_seconds_between_trades:
            logger.debug("Too soon since last trade. Waiting.")
            return False

        # Max open positions
        if len(self._open_positions) >= self.config.max_open_positions:
            logger.warning(
                "Max open positions reached: %d (max: %d)",
                len(self._open_positions),
                self.config.max_open_positions,
            )
            return False

        return True

    def check_portfolio_heat(self, additional_risk: float = 0.0) -> bool:
        """Check if adding a new position would exceed portfolio heat limit.

        Args:
            additional_risk: Dollar risk of the proposed new position.

        Returns:
            True if within limits.
        """
        current_heat = sum(self._open_positions.values())
        total_heat = current_heat + additional_risk
        heat_pct = (total_heat / self._current_equity) * 100 if self._current_equity > 0 else 100

        if heat_pct > self.config.max_portfolio_heat_pct:
            logger.warning(
                "Portfolio heat would be %.1f%% (max: %.1f%%)",
                heat_pct,
                self.config.max_portfolio_heat_pct,
            )
            return False

        return True

    # ─── Trade Recording ───

    def record_trade(self, symbol: str = "UNKNOWN", pnl: float = 0.0) -> None:
        """Record a completed trade for risk tracking.

        Updates daily P&L, consecutive loss counter, equity tracking,
        and trade frequency counters.

        Args:
            symbol: The traded symbol.
            pnl: Realized P&L of the trade.
        """
        self._reset_daily_if_needed()

        self._daily_pnl += pnl
        self._daily_trade_count += 1

        now = time.time()
        self._trade_timestamps.append(now)
        self._last_trade_time = now

        # Trim old timestamps (keep last 2 hours)
        cutoff = now - 7200
        self._trade_timestamps = [t for t in self._trade_timestamps if t > cutoff]

        # Update equity
        self._current_equity += pnl
        if self._current_equity > self._peak_equity:
            self._peak_equity = self._current_equity

        # Consecutive loss tracking
        if pnl < 0:
            self._consecutive_losses += 1
            logger.info(
                "Loss #%d (streak): $%.2f on %s",
                self._consecutive_losses,
                pnl,
                symbol,
            )

            if self._consecutive_losses >= self.config.max_consecutive_losses:
                self._cooldown_until = now + self.config.cooldown_seconds
                logger.warning(
                    "Cooldown triggered: %d consecutive losses. Pausing %ds.",
                    self._consecutive_losses,
                    self.config.cooldown_seconds,
                )
        else:
            if self._consecutive_losses > 0:
                logger.info(
                    "Win streak reset after %d consecutive losses.",
                    self._consecutive_losses,
                )
            self._consecutive_losses = 0

        # Drawdown circuit breaker
        drawdown_pct = (
            (self._peak_equity - self._current_equity) / self._peak_equity * 100
            if self._peak_equity > 0
            else 0
        )
        if drawdown_pct >= self.config.max_drawdown_pct:
            pause_seconds = self.config.circuit_breaker_pause_hours * 3600
            self._circuit_breaker_until = now + pause_seconds
            logger.warning(
                "CIRCUIT BREAKER: Drawdown %.1f%% exceeds max %.1f%%. "
                "Pausing %.0f hours.",
                drawdown_pct,
                self.config.max_drawdown_pct,
                self.config.circuit_breaker_pause_hours,
            )

        # Record in history
        self._trade_history.append(TradeRecord(symbol=symbol, pnl=pnl))
        # FIX 9: Trim trade history to prevent unbounded growth
        if len(self._trade_history) > 1000:
            self._trade_history = self._trade_history[-500:]
        logger.info(
            "Trade recorded: %s P&L=$%.2f | Daily P&L=$%.2f | Equity=$%.2f",
            symbol,
            pnl,
            self._daily_pnl,
            self._current_equity,
        )
        self._save_state()

    # ─── Position Tracking ───

    def add_position(self, symbol: str, risk_amount: float) -> None:
        """Register an open position's risk amount.

        Args:
            symbol: Ticker symbol.
            risk_amount: Dollar amount at risk for this position.
        """
        self._open_positions[symbol] = risk_amount
        logger.info(
            "Position added: %s risk=$%.2f | Total positions: %d",
            symbol,
            risk_amount,
            len(self._open_positions),
        )
        self._save_state()

    def remove_position(self, symbol: str) -> None:
        """Remove a closed position from tracking.

        Args:
            symbol: Ticker symbol to remove.
        """
        if symbol in self._open_positions:
            del self._open_positions[symbol]
            logger.info(
                "Position removed: %s | Remaining positions: %d",
                symbol,
                len(self._open_positions),
            )
            self._save_state()

    # ─── Status ───

    def get_status(self) -> dict:
        """Get current risk manager status.

        Returns:
            Dictionary with all risk metrics.
        """
        self._reset_daily_if_needed()

        now = time.time()
        drawdown_pct = (
            (self._peak_equity - self._current_equity) / self._peak_equity * 100
            if self._peak_equity > 0
            else 0
        )
        total_heat = sum(self._open_positions.values())
        heat_pct = (total_heat / self._current_equity) * 100 if self._current_equity > 0 else 0

        recent_trades_1h = len([t for t in self._trade_timestamps if t > now - 3600])

        return {
            "can_trade": self.can_trade(),
            "current_equity": round(self._current_equity, 2),
            "peak_equity": round(self._peak_equity, 2),
            "daily_pnl": round(self._daily_pnl, 2),
            "daily_trade_count": self._daily_trade_count,
            "consecutive_losses": self._consecutive_losses,
            "cooldown_active": now < self._cooldown_until,
            "cooldown_remaining_s": max(0, round(self._cooldown_until - now)),
            "circuit_breaker_active": now < self._circuit_breaker_until,
            "drawdown_pct": round(drawdown_pct, 2),
            "open_positions": len(self._open_positions),
            "portfolio_heat_pct": round(heat_pct, 2),
            "trades_last_hour": recent_trades_1h,
            "total_trades": len(self._trade_history),
        }

    def __repr__(self) -> str:
        status = self.get_status()
        return (
            f"RiskManager(equity=${status['current_equity']:,.2f}, "
            f"daily_pnl=${status['daily_pnl']:+,.2f}, "
            f"positions={status['open_positions']}, "
            f"can_trade={status['can_trade']})"
        )
