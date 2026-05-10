"""
Broker Simulator — Simulated Exchange Engine for CI/CD Testing
================================================================

Mimics real broker behavior without any network connections:
- Order matching (market + limit)
- Fill simulation with slippage
- TP/SL trigger evaluation
- Position tracking with real-time P&L
- Account balance management
- Simulated market data feed from OHLCV history

Each simulated broker (IB, TradeStation, Schwab) uses the same
engine but returns broker-specific response formats.

Usage:
    sim = BrokerSimulator(initial_capital=100_000)
    sim.load_price_data({"AAPL": df_aapl, "SPY": df_spy})

    # Advance to next bar
    sim.tick()

    # Place orders
    result = sim.place_market_order("AAPL", "BUY", 100)
    result = sim.place_limit_order("SPY", "SELL", 50, 510.0)

    # Check TP/SL triggers
    triggered = sim.check_triggers()

    # Get state
    positions = sim.get_positions()
    account = sim.get_account_info()
    fills = sim.get_fills()
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SimOrder:
    """A simulated order."""
    order_id: str
    symbol: str
    action: str  # BUY or SELL
    quantity: int
    order_type: str  # MARKET, LIMIT, STOP
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: str = "PENDING"  # PENDING, FILLED, CANCELLED, REJECTED
    fill_price: float = 0.0
    filled_at: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    is_tp: bool = False
    is_sl: bool = False
    parent_symbol: str = ""


@dataclass
class SimPosition:
    """A simulated position."""
    symbol: str
    quantity: int  # positive = long, negative = short
    avg_price: float
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class SimFill:
    """A simulated fill."""
    fill_id: str
    order_id: str
    symbol: str
    action: str
    quantity: int
    price: float
    commission: float
    timestamp: str


class BrokerSimulator:
    """Simulated exchange engine for testing without real connections.

    Walks through historical OHLCV data bar by bar, matching orders
    against simulated prices with realistic slippage and commission.

    Args:
        initial_capital: Starting account balance.
        commission_per_share: Commission per share traded.
        slippage_pct: Simulated slippage as fraction of price.
        broker_name: Name to identify this simulator (ib/tradestation/schwab).
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_per_share: float = 0.005,
        slippage_pct: float = 0.001,
        broker_name: str = "simulator",
    ) -> None:
        self._initial_capital = initial_capital
        self._cash = initial_capital
        self._commission = commission_per_share
        self._slippage = slippage_pct
        self._broker_name = broker_name

        # Market data
        self._price_data: Dict[str, pd.DataFrame] = {}
        self._current_bar: int = 0
        self._max_bars: int = 0

        # Orders, positions, fills
        self._pending_orders: List[SimOrder] = []
        self._filled_orders: List[SimOrder] = []
        self._positions: Dict[str, SimPosition] = {}
        self._fills: List[SimFill] = []

        # TP/SL triggers
        self._triggers: List[SimOrder] = []

        # Stats
        self._total_trades = 0
        self._total_commission = 0.0

    def load_price_data(self, data: Dict[str, pd.DataFrame]) -> None:
        """Load OHLCV data for symbols.

        Args:
            data: Dict mapping symbol to OHLCV DataFrame.
        """
        self._price_data = {}
        for symbol, df in data.items():
            df = df.copy()
            df.columns = [c.lower() for c in df.columns]
            self._price_data[symbol] = df

        if data:
            self._max_bars = min(len(df) for df in self._price_data.values())
        self._current_bar = 0
        logger.info("Loaded %d symbols, %d bars", len(data), self._max_bars)

    def load_synthetic_data(
        self,
        symbols: List[str],
        n_bars: int = 500,
        seed: int = 42,
    ) -> None:
        """Generate and load synthetic OHLCV data.

        Args:
            symbols: List of symbols to generate.
            n_bars: Number of bars.
            seed: Random seed.
        """
        rng = np.random.RandomState(seed)
        data = {}
        for i, symbol in enumerate(symbols):
            price = 100.0 + i * 50
            rows = []
            for j in range(n_bars):
                ret = 0.0003 * np.sin(2 * np.pi * j / 120) + rng.randn() * 0.015
                price *= 1 + ret
                rows.append({
                    "open": price * (1 + rng.randn() * 0.002),
                    "high": price * (1 + abs(rng.randn()) * 0.005),
                    "low": price * (1 - abs(rng.randn()) * 0.005),
                    "close": price,
                    "volume": int(rng.uniform(500_000, 2_000_000)),
                })
            data[symbol] = pd.DataFrame(rows)

        self.load_price_data(data)

    def get_current_price(self, symbol: str) -> float:
        """Get the current close price for a symbol."""
        if symbol not in self._price_data:
            return 0.0
        df = self._price_data[symbol]
        if self._current_bar >= len(df):
            return float(df["close"].iloc[-1])
        return float(df["close"].iloc[self._current_bar])

    def get_current_bar(self, symbol: str) -> Dict[str, float]:
        """Get the current OHLCV bar."""
        if symbol not in self._price_data:
            return {}
        df = self._price_data[symbol]
        idx = min(self._current_bar, len(df) - 1)
        row = df.iloc[idx]
        return {
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row.get("volume", 0)),
        }

    def get_ohlcv_history(self, symbol: str, lookback: int = 60) -> pd.DataFrame:
        """Get OHLCV history up to current bar."""
        if symbol not in self._price_data:
            return pd.DataFrame()
        df = self._price_data[symbol]
        end = min(self._current_bar + 1, len(df))
        start = max(0, end - lookback)
        return df.iloc[start:end].copy()

    # ─── Order Placement ───

    def place_market_order(self, symbol: str, action: str, quantity: int) -> SimOrder:
        """Place and immediately fill a market order."""
        price = self.get_current_price(symbol)
        if price <= 0:
            order = SimOrder(
                order_id=self._gen_id(), symbol=symbol, action=action,
                quantity=quantity, order_type="MARKET", status="REJECTED",
            )
            return order

        # Apply slippage
        if action == "BUY":
            fill_price = price * (1 + self._slippage)
        else:
            fill_price = price * (1 - self._slippage)

        order = SimOrder(
            order_id=self._gen_id(), symbol=symbol, action=action,
            quantity=quantity, order_type="MARKET", status="FILLED",
            fill_price=round(fill_price, 2),
            filled_at=datetime.now().isoformat(),
        )

        self._execute_fill(order)
        self._filled_orders.append(order)
        return order

    def place_limit_order(
        self, symbol: str, action: str, quantity: int, limit_price: float,
        is_tp: bool = False, is_sl: bool = False,
    ) -> SimOrder:
        """Place a limit order (rests until triggered)."""
        order = SimOrder(
            order_id=self._gen_id(), symbol=symbol, action=action,
            quantity=quantity, order_type="LIMIT",
            limit_price=limit_price, status="PENDING",
            is_tp=is_tp, is_sl=is_sl,
        )

        if is_tp or is_sl:
            self._triggers.append(order)
        else:
            self._pending_orders.append(order)

        return order

    def place_stop_order(
        self, symbol: str, action: str, quantity: int, stop_price: float,
    ) -> SimOrder:
        """Place a stop order."""
        order = SimOrder(
            order_id=self._gen_id(), symbol=symbol, action=action,
            quantity=quantity, order_type="STOP",
            stop_price=stop_price, status="PENDING",
            is_sl=True,
        )
        self._triggers.append(order)
        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        for orders_list in [self._pending_orders, self._triggers]:
            for order in orders_list:
                if order.order_id == order_id and order.status == "PENDING":
                    order.status = "CANCELLED"
                    orders_list.remove(order)
                    return True
        return False

    # ─── Tick / Bar Advance ───

    def tick(self) -> Dict[str, Any]:
        """Advance to next bar: match pending orders, check TP/SL triggers.

        Returns:
            Dict with: bar_index, fills, triggered_tpsl, prices.
        """
        self._current_bar += 1
        result = {
            "bar_index": self._current_bar,
            "fills": [],
            "triggered_tpsl": [],
            "prices": {},
        }

        if self._current_bar >= self._max_bars:
            return result

        # Current prices
        for symbol in self._price_data:
            result["prices"][symbol] = self.get_current_price(symbol)

        # Match pending limit orders
        for order in self._pending_orders[:]:
            bar = self.get_current_bar(order.symbol)
            if not bar:
                continue

            if order.order_type == "LIMIT" and order.limit_price:
                if order.action == "BUY" and bar["low"] <= order.limit_price:
                    order.fill_price = order.limit_price
                    order.status = "FILLED"
                    order.filled_at = datetime.now().isoformat()
                    self._execute_fill(order)
                    self._pending_orders.remove(order)
                    self._filled_orders.append(order)
                    result["fills"].append(order.order_id)

                elif order.action == "SELL" and bar["high"] >= order.limit_price:
                    order.fill_price = order.limit_price
                    order.status = "FILLED"
                    order.filled_at = datetime.now().isoformat()
                    self._execute_fill(order)
                    self._pending_orders.remove(order)
                    self._filled_orders.append(order)
                    result["fills"].append(order.order_id)

        # Check TP/SL triggers
        for trigger in self._triggers[:]:
            bar = self.get_current_bar(trigger.symbol)
            if not bar:
                continue

            triggered = False
            if trigger.is_tp and trigger.limit_price:
                if trigger.action == "SELL" and bar["high"] >= trigger.limit_price:
                    triggered = True
                    trigger.fill_price = trigger.limit_price
                elif trigger.action == "BUY" and bar["low"] <= trigger.limit_price:
                    triggered = True
                    trigger.fill_price = trigger.limit_price

            if trigger.is_sl and (trigger.stop_price or trigger.limit_price):
                stop = trigger.stop_price or trigger.limit_price
                if trigger.action == "SELL" and bar["low"] <= stop:
                    triggered = True
                    trigger.fill_price = stop
                elif trigger.action == "BUY" and bar["high"] >= stop:
                    triggered = True
                    trigger.fill_price = stop

            if triggered:
                trigger.status = "FILLED"
                trigger.filled_at = datetime.now().isoformat()
                self._execute_fill(trigger)
                self._triggers.remove(trigger)
                self._filled_orders.append(trigger)
                result["triggered_tpsl"].append({
                    "order_id": trigger.order_id,
                    "type": "TP" if trigger.is_tp else "SL",
                    "symbol": trigger.symbol,
                    "price": trigger.fill_price,
                })

        # Update position mark-to-market
        self._update_positions()

        return result

    # ─── Internal Execution ───

    def _execute_fill(self, order: SimOrder) -> None:
        """Process a filled order: update positions and cash."""
        commission = order.quantity * self._commission
        self._total_commission += commission
        self._total_trades += 1

        fill = SimFill(
            fill_id=self._gen_id(),
            order_id=order.order_id,
            symbol=order.symbol,
            action=order.action,
            quantity=order.quantity,
            price=order.fill_price,
            commission=commission,
            timestamp=datetime.now().isoformat(),
        )
        self._fills.append(fill)

        # Update position
        pos = self._positions.get(order.symbol)
        cost = order.fill_price * order.quantity + commission

        if order.action == "BUY":
            if pos is None:
                self._positions[order.symbol] = SimPosition(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    avg_price=order.fill_price,
                )
                self._cash -= cost
            elif pos.quantity >= 0:
                # Adding to long
                total_cost = pos.avg_price * pos.quantity + order.fill_price * order.quantity
                pos.quantity += order.quantity
                pos.avg_price = total_cost / pos.quantity if pos.quantity > 0 else 0
                self._cash -= cost
            else:
                # Covering short
                pnl = (pos.avg_price - order.fill_price) * min(order.quantity, abs(pos.quantity))
                pos.realized_pnl += pnl
                pos.quantity += order.quantity
                self._cash -= cost
                self._cash += pnl
                if pos.quantity == 0:
                    del self._positions[order.symbol]

        elif order.action == "SELL":
            if pos is None:
                self._positions[order.symbol] = SimPosition(
                    symbol=order.symbol,
                    quantity=-order.quantity,
                    avg_price=order.fill_price,
                )
                self._cash += order.fill_price * order.quantity - commission
            elif pos.quantity <= 0:
                # Adding to short
                total_cost = pos.avg_price * abs(pos.quantity) + order.fill_price * order.quantity
                pos.quantity -= order.quantity
                pos.avg_price = total_cost / abs(pos.quantity) if pos.quantity != 0 else 0
                self._cash += order.fill_price * order.quantity - commission
            else:
                # Selling long
                pnl = (order.fill_price - pos.avg_price) * min(order.quantity, pos.quantity)
                pos.realized_pnl += pnl
                pos.quantity -= order.quantity
                self._cash += order.fill_price * order.quantity - commission
                if pos.quantity == 0:
                    del self._positions[order.symbol]

    def _update_positions(self) -> None:
        """Mark-to-market all positions."""
        for symbol, pos in self._positions.items():
            price = self.get_current_price(symbol)
            if pos.quantity > 0:
                pos.market_value = pos.quantity * price
                pos.unrealized_pnl = (price - pos.avg_price) * pos.quantity
            elif pos.quantity < 0:
                pos.market_value = abs(pos.quantity) * price
                pos.unrealized_pnl = (pos.avg_price - price) * abs(pos.quantity)

    # ─── Queries ───

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get all current positions."""
        return [
            {
                "symbol": p.symbol,
                "quantity": p.quantity,
                "avg_price": round(p.avg_price, 2),
                "market_value": round(p.market_value, 2),
                "unrealized_pnl": round(p.unrealized_pnl, 2),
                "realized_pnl": round(p.realized_pnl, 2),
            }
            for p in self._positions.values()
        ]

    def get_account_info(self) -> Dict[str, Any]:
        """Get account summary."""
        total_unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        total_realized = sum(p.realized_pnl for p in self._positions.values())
        net_liq = self._cash + sum(p.market_value for p in self._positions.values())

        return {
            "broker": self._broker_name,
            "mode": "simulated",
            "initial_capital": self._initial_capital,
            "cash": round(self._cash, 2),
            "net_liquidation": round(net_liq, 2),
            "unrealized_pnl": round(total_unrealized, 2),
            "realized_pnl": round(total_realized, 2),
            "total_commission": round(self._total_commission, 2),
            "total_trades": self._total_trades,
            "open_positions": len(self._positions),
            "pending_orders": len(self._pending_orders),
            "pending_triggers": len(self._triggers),
            "current_bar": self._current_bar,
            "max_bars": self._max_bars,
            "total_return_pct": round((net_liq / self._initial_capital - 1) * 100, 2),
        }

    def get_fills(self, n: int = 50) -> List[Dict[str, Any]]:
        """Get recent fills."""
        return [
            {
                "fill_id": f.fill_id,
                "order_id": f.order_id,
                "symbol": f.symbol,
                "action": f.action,
                "quantity": f.quantity,
                "price": f.price,
                "commission": f.commission,
                "timestamp": f.timestamp,
            }
            for f in self._fills[-n:]
        ]

    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get pending orders + triggers."""
        return [
            {
                "order_id": o.order_id,
                "symbol": o.symbol,
                "action": o.action,
                "quantity": o.quantity,
                "type": o.order_type,
                "limit_price": o.limit_price,
                "stop_price": o.stop_price,
                "is_tp": o.is_tp,
                "is_sl": o.is_sl,
            }
            for o in self._pending_orders + self._triggers
        ]

    @property
    def is_done(self) -> bool:
        """Check if all bars have been consumed."""
        return self._current_bar >= self._max_bars - 1

    def _gen_id(self) -> str:
        return f"SIM-{self._broker_name.upper()}-{uuid.uuid4().hex[:8]}"

    def __repr__(self) -> str:
        acct = self.get_account_info()
        return (
            f"BrokerSimulator(broker='{self._broker_name}', "
            f"bar={acct['current_bar']}/{acct['max_bars']}, "
            f"net_liq=${acct['net_liquidation']:,.2f}, "
            f"positions={acct['open_positions']})"
        )
