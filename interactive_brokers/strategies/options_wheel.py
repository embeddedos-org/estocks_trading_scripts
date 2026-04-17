"""
Options Wheel Strategy for Interactive Brokers
================================================

Implements The Wheel — a systematic income strategy:
1. Sell cash-secured puts (CSP) on stocks you want to own
2. If assigned, sell covered calls (CC) on the shares
3. If called away, restart the cycle

Usage:
    wheel = OptionsWheelStrategy(connection, order_manager, notifier=dispatcher)
    wheel.sell_cash_secured_put("AAPL", target_delta=0.30, dte_range=(30, 45))
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class WheelPhase(Enum):
    """Current phase in the wheel cycle."""
    IDLE = "idle"
    CSP_OPEN = "csp_open"        # Cash-secured put sold
    ASSIGNED = "assigned"         # Put was assigned, holding shares
    CC_OPEN = "cc_open"          # Covered call sold on assigned shares
    CALLED_AWAY = "called_away"  # Shares called away, cycle complete


@dataclass
class OptionGreeks:
    """Option Greeks snapshot."""
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    implied_vol: float = 0.0
    rho: float = 0.0


@dataclass
class WheelCycle:
    """Tracks a single wheel cycle from CSP → assignment → CC → called away."""
    symbol: str
    start_date: datetime = field(default_factory=datetime.now)
    phase: WheelPhase = WheelPhase.IDLE
    put_strike: float = 0.0
    put_premium: float = 0.0
    put_expiry: Optional[date] = None
    put_quantity: int = 1
    assigned_price: float = 0.0
    assigned_shares: int = 0
    call_strike: float = 0.0
    call_premium: float = 0.0
    call_expiry: Optional[date] = None
    called_away_price: float = 0.0
    end_date: Optional[datetime] = None
    total_premium: float = 0.0
    total_pnl: float = 0.0
    num_rolls: int = 0

    @property
    def cost_basis(self) -> float:
        """Effective cost basis after premium collection."""
        if self.assigned_price > 0:
            return self.assigned_price - self.put_premium
        return 0.0

    @property
    def is_complete(self) -> bool:
        return self.phase in (WheelPhase.CALLED_AWAY, WheelPhase.IDLE)


class OptionsWheelStrategy:
    """The Wheel — systematic options income strategy.

    Implements a rules-based wheel strategy with delta targeting,
    DTE selection, roll logic, and Greeks monitoring.

    Args:
        connection: An IBAsyncConnection instance.
        order_manager: OrderManager for trade execution.
        notifier: Optional AlertDispatcher for notifications.
        capital: Capital allocated per underlying.
    """

    def __init__(
        self,
        connection: Any,
        order_manager: Any,
        notifier: Any = None,
        capital: float = 50000.0,
    ) -> None:
        self.connection = connection
        self.order_manager = order_manager
        self.notifier = notifier
        self.capital = capital

        self._cycles: Dict[str, WheelCycle] = {}
        self._completed_cycles: List[WheelCycle] = []

    def get_option_chain(
        self,
        symbol: str,
        sec_type: str = "STK",
        exchange: str = "SMART",
        currency: str = "USD",
    ) -> pd.DataFrame:
        """Fetch the full option chain for a symbol.

        Args:
            symbol: Underlying ticker symbol.
            sec_type: Security type of the underlying.
            exchange: Exchange.
            currency: Currency.

        Returns:
            DataFrame with columns: strike, expiry, right, bid, ask,
            last, volume, openInterest, impliedVol, delta, gamma,
            theta, vega.
        """
        try:
            from ib_async import Stock, Option

            underlying = Stock(symbol, exchange, currency)
            self.connection.qualifyContracts(underlying)

            chains = self.connection.ib.reqSecDefOptParams(
                underlying.symbol, "", underlying.secType, underlying.conId,
            )

            if not chains:
                logger.warning("No option chains found for %s", symbol)
                return pd.DataFrame()

            chain = chains[0]
            expirations = sorted(chain.expirations)
            strikes = sorted(chain.strikes)

            rows = []
            for exp in expirations[:6]:  # Limit to 6 nearest expirations
                for strike in strikes:
                    for right in ("P", "C"):
                        rows.append({
                            "symbol": symbol,
                            "expiry": exp,
                            "strike": strike,
                            "right": right,
                            "exchange": chain.exchange,
                        })

            df = pd.DataFrame(rows)
            logger.info(
                "Option chain for %s: %d expirations, %d strikes",
                symbol, len(expirations), len(strikes),
            )
            return df

        except ImportError:
            raise ImportError(
                "ib_async is required. Install with: pip install ib-async"
            )
        except Exception as e:
            logger.error("Failed to fetch option chain for %s: %s", symbol, e)
            raise

    async def find_strike_by_delta(
        self,
        symbol: str,
        right: str,
        target_delta: float = 0.30,
        dte_range: Tuple[int, int] = (30, 45),
        exchange: str = "SMART",
    ) -> Optional[Dict[str, Any]]:
        """Find the option strike closest to a target delta.

        Args:
            symbol: Underlying symbol.
            right: "P" for put or "C" for call.
            target_delta: Target absolute delta (e.g., 0.30).
            dte_range: (min_dte, max_dte) in days.
            exchange: Exchange for options.

        Returns:
            Dict with strike, expiry, delta, bid, ask, greeks, or None.
        """
        try:
            from ib_async import Stock, Option

            underlying = Stock(symbol, exchange, "USD")
            self.connection.qualifyContracts(underlying)

            chains = self.connection.ib.reqSecDefOptParams(
                underlying.symbol, "", underlying.secType, underlying.conId,
            )
            if not chains:
                return None

            chain = chains[0]
            today = date.today()
            min_dte, max_dte = dte_range

            valid_expirations = []
            for exp_str in chain.expirations:
                exp_date = datetime.strptime(exp_str, "%Y%m%d").date()
                dte = (exp_date - today).days
                if min_dte <= dte <= max_dte:
                    valid_expirations.append((exp_str, exp_date, dte))

            if not valid_expirations:
                logger.warning(
                    "No expirations found in DTE range %d-%d for %s",
                    min_dte, max_dte, symbol,
                )
                return None

            best_match = None
            best_delta_diff = float("inf")

            for exp_str, exp_date, dte in valid_expirations:
                for strike in chain.strikes:
                    opt = Option(symbol, exp_str, strike, right, chain.exchange)
                    try:
                        self.connection.qualifyContracts(opt)
                        ticker = self.connection.ib.reqMktData(opt, "", False, False)
                        await asyncio.sleep(2)

                        if ticker.modelGreeks:
                            opt_delta = abs(ticker.modelGreeks.delta)
                            diff = abs(opt_delta - target_delta)
                            if diff < best_delta_diff:
                                best_delta_diff = diff
                                best_match = {
                                    "symbol": symbol,
                                    "strike": strike,
                                    "expiry": exp_str,
                                    "expiry_date": exp_date,
                                    "dte": dte,
                                    "right": right,
                                    "delta": ticker.modelGreeks.delta,
                                    "gamma": ticker.modelGreeks.gamma,
                                    "theta": ticker.modelGreeks.theta,
                                    "vega": ticker.modelGreeks.vega,
                                    "implied_vol": ticker.modelGreeks.impliedVol,
                                    "bid": ticker.bid,
                                    "ask": ticker.ask,
                                    "mid": (ticker.bid + ticker.ask) / 2,
                                }

                        self.connection.ib.cancelMktData(opt)
                    except Exception:
                        continue

            if best_match:
                logger.info(
                    "Best %s strike for %s: $%.2f exp=%s delta=%.3f",
                    right, symbol, best_match["strike"],
                    best_match["expiry"], best_match["delta"],
                )
            return best_match

        except ImportError:
            raise ImportError(
                "ib_async is required. Install with: pip install ib-async"
            )

    async def sell_cash_secured_put(
        self,
        symbol: str,
        target_delta: float = 0.30,
        dte_range: Tuple[int, int] = (30, 45),
        contracts: int = 1,
    ) -> Optional[WheelCycle]:
        """Sell a cash-secured put to start or continue the wheel.

        Finds the put strike closest to the target delta within
        the DTE range, then sells to open.

        Args:
            symbol: Underlying symbol.
            target_delta: Target delta (e.g., 0.30 = ~30% ITM chance).
            dte_range: (min_dte, max_dte) in days.
            contracts: Number of contracts to sell.

        Returns:
            WheelCycle record, or None if no suitable strike found.
        """
        logger.info(
            "Wheel: looking for CSP on %s (delta~%.2f, DTE %d-%d)",
            symbol, target_delta, *dte_range,
        )

        strike_info = await self.find_strike_by_delta(
            symbol, "P", target_delta, dte_range,
        )
        if not strike_info:
            logger.warning("No suitable put strike found for %s", symbol)
            return None

        cash_required = strike_info["strike"] * 100 * contracts
        if cash_required > self.capital:
            logger.warning(
                "Insufficient capital: need $%.0f, have $%.0f",
                cash_required, self.capital,
            )
            return None

        premium = strike_info.get("mid", 0) * 100 * contracts

        try:
            self.order_manager.limit_order(
                symbol=symbol,
                action="SELL",
                quantity=contracts,
                limit_price=strike_info.get("mid", strike_info["bid"]),
                sec_type="OPT",
                expiry=strike_info["expiry"],
                strike=strike_info["strike"],
                right="P",
            )
        except Exception as e:
            logger.error("Failed to place CSP order for %s: %s", symbol, e)
            raise

        cycle = WheelCycle(
            symbol=symbol,
            phase=WheelPhase.CSP_OPEN,
            put_strike=strike_info["strike"],
            put_premium=premium,
            put_expiry=strike_info.get("expiry_date"),
            put_quantity=contracts,
        )
        self._cycles[symbol] = cycle

        msg = (
            f"Wheel CSP: SELL {contracts}x {symbol} "
            f"${strike_info['strike']:.0f}P "
            f"exp={strike_info['expiry']} "
            f"delta={strike_info['delta']:.3f} "
            f"premium=${premium:.0f}"
        )
        logger.info(msg)
        if self.notifier:
            self.notifier.info(msg)

        return cycle

    async def sell_covered_call(
        self,
        symbol: str,
        target_delta: float = 0.30,
        dte_range: Tuple[int, int] = (30, 45),
    ) -> Optional[WheelCycle]:
        """Sell a covered call on assigned shares.

        Args:
            symbol: Underlying symbol (must have assigned shares).
            target_delta: Target call delta.
            dte_range: (min_dte, max_dte) in days.

        Returns:
            Updated WheelCycle, or None if conditions not met.
        """
        cycle = self._cycles.get(symbol)
        if not cycle or cycle.phase != WheelPhase.ASSIGNED:
            logger.warning(
                "Cannot sell CC for %s: not in ASSIGNED phase", symbol,
            )
            return None

        contracts = cycle.assigned_shares // 100
        if contracts < 1:
            logger.warning("Insufficient shares for covered call: %d", cycle.assigned_shares)
            return None

        strike_info = await self.find_strike_by_delta(
            symbol, "C", target_delta, dte_range,
        )
        if not strike_info:
            logger.warning("No suitable call strike found for %s", symbol)
            return None

        premium = strike_info.get("mid", 0) * 100 * contracts

        try:
            self.order_manager.limit_order(
                symbol=symbol,
                action="SELL",
                quantity=contracts,
                limit_price=strike_info.get("mid", strike_info["bid"]),
                sec_type="OPT",
                expiry=strike_info["expiry"],
                strike=strike_info["strike"],
                right="C",
            )
        except Exception as e:
            logger.error("Failed to place CC order for %s: %s", symbol, e)
            raise

        cycle.phase = WheelPhase.CC_OPEN
        cycle.call_strike = strike_info["strike"]
        cycle.call_premium = premium
        cycle.call_expiry = strike_info.get("expiry_date")
        cycle.total_premium += premium

        msg = (
            f"Wheel CC: SELL {contracts}x {symbol} "
            f"${strike_info['strike']:.0f}C "
            f"exp={strike_info['expiry']} "
            f"delta={strike_info['delta']:.3f} "
            f"premium=${premium:.0f}"
        )
        logger.info(msg)
        if self.notifier:
            self.notifier.info(msg)

        return cycle

    def check_assignment(self, symbol: str) -> bool:
        """Check if a sold put has been assigned.

        Queries portfolio positions to detect if shares were
        assigned from a short put.

        Args:
            symbol: Underlying symbol.

        Returns:
            True if assignment detected.
        """
        cycle = self._cycles.get(symbol)
        if not cycle or cycle.phase != WheelPhase.CSP_OPEN:
            return False

        try:
            positions = self.connection.positions()
            for pos in positions:
                if (
                    pos.contract.symbol == symbol
                    and pos.contract.secType == "STK"
                    and pos.position > 0
                ):
                    cycle.phase = WheelPhase.ASSIGNED
                    cycle.assigned_price = cycle.put_strike
                    cycle.assigned_shares = int(pos.position)
                    cycle.total_premium += cycle.put_premium

                    msg = (
                        f"Wheel ASSIGNED: {symbol} {cycle.assigned_shares} shares "
                        f"@ ${cycle.assigned_price:.2f} "
                        f"(cost basis: ${cycle.cost_basis:.2f})"
                    )
                    logger.info(msg)
                    if self.notifier:
                        self.notifier.warning(msg)
                    return True
        except Exception as e:
            logger.error("Error checking assignment for %s: %s", symbol, e)

        return False

    def check_called_away(self, symbol: str) -> bool:
        """Check if shares were called away from a covered call.

        Args:
            symbol: Underlying symbol.

        Returns:
            True if shares were called away.
        """
        cycle = self._cycles.get(symbol)
        if not cycle or cycle.phase != WheelPhase.CC_OPEN:
            return False

        try:
            positions = self.connection.positions()
            has_shares = any(
                pos.contract.symbol == symbol
                and pos.contract.secType == "STK"
                and pos.position > 0
                for pos in positions
            )

            if not has_shares:
                share_pnl = (
                    (cycle.call_strike - cycle.assigned_price)
                    * cycle.assigned_shares
                )
                cycle.total_pnl = cycle.total_premium + share_pnl
                cycle.phase = WheelPhase.CALLED_AWAY
                cycle.called_away_price = cycle.call_strike
                cycle.end_date = datetime.now()

                self._completed_cycles.append(cycle)
                del self._cycles[symbol]

                msg = (
                    f"Wheel CALLED AWAY: {symbol} @ ${cycle.call_strike:.2f} "
                    f"Total P&L: ${cycle.total_pnl:.2f} "
                    f"(premium: ${cycle.total_premium:.2f})"
                )
                logger.info(msg)
                if self.notifier:
                    self.notifier.info(msg)
                return True
        except Exception as e:
            logger.error("Error checking call assignment for %s: %s", symbol, e)

        return False

    async def roll_option(
        self,
        symbol: str,
        new_dte_range: Tuple[int, int] = (30, 45),
        new_target_delta: float = 0.30,
    ) -> Optional[WheelCycle]:
        """Roll a current option position to a new expiration.

        Buys to close the current option and sells to open a new one
        at the next expiration cycle.

        Args:
            symbol: Underlying symbol.
            new_dte_range: New DTE range for the rolled option.
            new_target_delta: New target delta.

        Returns:
            Updated WheelCycle, or None if roll failed.
        """
        cycle = self._cycles.get(symbol)
        if not cycle:
            logger.warning("No active cycle for %s to roll", symbol)
            return None

        cycle.num_rolls += 1
        logger.info("Rolling %s option (roll #%d)", symbol, cycle.num_rolls)

        if cycle.phase == WheelPhase.CSP_OPEN:
            return await self.sell_cash_secured_put(
                symbol, new_target_delta, new_dte_range, cycle.put_quantity,
            )
        elif cycle.phase == WheelPhase.CC_OPEN:
            return await self.sell_covered_call(symbol, new_target_delta, new_dte_range)

        logger.warning("Cannot roll in phase %s", cycle.phase)
        return None

    def get_active_cycles(self) -> Dict[str, WheelCycle]:
        """Get all active wheel cycles."""
        return dict(self._cycles)

    def get_completed_cycles(self) -> List[WheelCycle]:
        """Get all completed wheel cycles."""
        return list(self._completed_cycles)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Calculate overall wheel strategy performance.

        Returns:
            Dictionary with total_cycles, total_premium, total_pnl,
            avg_cycle_return, win_rate.
        """
        completed = self._completed_cycles
        if not completed:
            return {
                "total_completed": 0,
                "active_cycles": len(self._cycles),
            }

        pnls = [c.total_pnl for c in completed]
        premiums = [c.total_premium for c in completed]
        wins = [p for p in pnls if p > 0]

        return {
            "total_completed": len(completed),
            "active_cycles": len(self._cycles),
            "total_premium_collected": sum(premiums),
            "total_pnl": sum(pnls),
            "avg_pnl_per_cycle": np.mean(pnls),
            "win_rate": len(wins) / len(pnls) if pnls else 0.0,
            "total_rolls": sum(c.num_rolls for c in completed),
            "avg_cycle_days": np.mean([
                (c.end_date - c.start_date).days
                for c in completed
                if c.end_date
            ]),
        }
