"""
Advanced Backtesting Engine V2
================================

Full-featured backtester with short selling, multi-asset support,
slippage modeling, enhanced metrics, benchmark comparison, and
trade-level analytics. V1 is preserved for backward compatibility.

Usage:
    engine = BacktestEngineV2(initial_capital=100000)
    engine.load_data({"AAPL": df_aapl, "MSFT": df_msft})
    result = engine.run(my_strategy)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class BacktestContext:
    """Context passed to strategy function on each bar."""

    bar_index: int
    bars: Dict[str, pd.DataFrame]
    positions: Dict[str, int]  # symbol -> shares (negative = short)
    capital: float
    portfolio_value: float
    indicators: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeRecord:
    """Detailed record of a completed round-trip trade."""

    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    commission: float
    slippage: float
    hold_bars: int
    mae: float = 0.0  # Max Adverse Excursion (worst unrealized loss)
    mfe: float = 0.0  # Max Favorable Excursion (best unrealized gain)
    initial_risk: float = 0.0  # Dollar risk at entry (for R-multiple calc)
    r_multiple: float = 0.0  # P&L expressed as multiple of initial risk (Van Tharp)


@dataclass
class BacktestResultV2:
    """Extended backtest result with V2 metrics (backward-compatible)."""

    # Core metrics (compatible with V1 BacktestResult)
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    equity_curve: list[float] = field(default_factory=list)
    trade_log: list[dict[str, Any]] = field(default_factory=list)

    # V2 extended metrics
    cagr: float = 0.0
    calmar_ratio: float = 0.0
    expectancy: float = 0.0
    avg_trade_duration: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    monthly_returns: Dict[str, float] = field(default_factory=dict)

    # Benchmark metrics
    alpha: float = 0.0
    beta: float = 0.0
    information_ratio: float = 0.0
    tracking_error: float = 0.0

    # Trade-level analytics
    trades: list[TradeRecord] = field(default_factory=list)
    long_trades: int = 0
    short_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0

    # Van Tharp metrics
    avg_r_multiple: float = 0.0  # Average R-multiple across all trades
    sqn: float = 0.0  # System Quality Number = sqrt(N) * mean(R) / std(R)


@dataclass
class SlippageConfig:
    """Slippage model configuration."""

    method: str = "fixed"  # "fixed", "percentage", "volatility"
    fixed_cents: float = 1.0  # cents per share
    percentage: float = 0.05  # percentage of price
    volatility_mult: float = 0.1  # fraction of ATR


StrategyFnV2 = Callable[[BacktestContext], Dict[str, int]]


class BacktestEngineV2:
    """Advanced event-driven backtester with multi-asset + short support.

    Args:
        initial_capital: Starting portfolio value in dollars.
        commission: Commission rate as fraction of trade value.
        slippage: SlippageConfig for slippage modeling.
    """

    ANNUALIZATION_FACTOR = 252

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission: float = 0.001,
        slippage: Optional[SlippageConfig] = None,
    ) -> None:
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage or SlippageConfig()
        self._data: Dict[str, pd.DataFrame] = {}
        self._benchmark_data: Optional[pd.DataFrame] = None

    def load_data(
        self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
    ) -> None:
        """Load OHLCV data for backtesting.

        Args:
            data: Single DataFrame or dict of {symbol: DataFrame}.
                Each DataFrame must have: date, open, high, low, close, volume.
        """
        if isinstance(data, pd.DataFrame):
            data = {"DEFAULT": data}

        for symbol, df in data.items():
            df = df.copy()
            df.columns = [c.strip().lower() for c in df.columns]

            if df.index.name in ("date", "datetime"):
                df = df.reset_index()
            if "datetime" in df.columns and "date" not in df.columns:
                df.rename(columns={"datetime": "date"}, inplace=True)

            required = {"date", "open", "high", "low", "close", "volume"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"[{symbol}] Missing columns: {missing}")

            df.sort_values("date", inplace=True)
            df.reset_index(drop=True, inplace=True)
            self._data[symbol] = df

    def set_benchmark(self, df: pd.DataFrame) -> None:
        """Set benchmark data for alpha/beta calculation."""
        df = df.copy()
        df.columns = [c.strip().lower() for c in df.columns]
        if df.index.name in ("date", "datetime"):
            df = df.reset_index()
        if "datetime" in df.columns and "date" not in df.columns:
            df.rename(columns={"datetime": "date"}, inplace=True)
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        self._benchmark_data = df

    def _compute_slippage(self, price: float, atr: float = 0.0) -> float:
        """Compute slippage amount per share."""
        if self.slippage.method == "fixed":
            return self.slippage.fixed_cents / 100.0
        elif self.slippage.method == "percentage":
            return price * (self.slippage.percentage / 100.0)
        elif self.slippage.method == "volatility":
            return atr * self.slippage.volatility_mult if atr > 0 else price * 0.001
        return 0.0

    def run(self, strategy_fn: StrategyFnV2) -> BacktestResultV2:
        """Execute a backtest using the provided strategy function.

        The strategy function receives a BacktestContext and returns
        a dict of {symbol: signal} where signal is -1 (short), 0 (flat),
        or +1 (long).

        Args:
            strategy_fn: Callable that returns position signals.

        Returns:
            BacktestResultV2 with all metrics.
        """
        if not self._data:
            raise RuntimeError("No data loaded. Call load_data() first.")

        capital = self.initial_capital
        positions: Dict[str, int] = {}  # symbol -> shares (neg = short)
        entry_prices: Dict[str, float] = {}
        entry_dates: Dict[str, str] = {}
        entry_indices: Dict[str, int] = {}

        equity_curve: list[float] = []
        daily_returns: list[float] = []
        trade_log: list[dict[str, Any]] = []
        completed_trades: list[TradeRecord] = []

        # MAE/MFE tracking
        mae_tracker: Dict[str, float] = {}
        mfe_tracker: Dict[str, float] = {}

        # Determine total bars (use longest data)
        max_bars = max(len(df) for df in self._data.values())
        prev_equity = capital

        for bar_idx in range(max_bars):
            # Build current bar data
            current_bars: Dict[str, pd.DataFrame] = {}
            for sym, df in self._data.items():
                if bar_idx < len(df):
                    current_bars[sym] = df.iloc[: bar_idx + 1]

            # Calculate portfolio value
            portfolio_val = capital
            for sym, shares in positions.items():
                if sym in current_bars and len(current_bars[sym]) > 0:
                    price = float(current_bars[sym].iloc[-1]["close"])
                    if shares > 0:
                        portfolio_val += shares * price
                    else:  # short
                        portfolio_val += shares * price  # shares is negative

            # Update MAE/MFE
            for sym, shares in positions.items():
                if sym in current_bars and len(current_bars[sym]) > 0:
                    price = float(current_bars[sym].iloc[-1]["close"])
                    ep = entry_prices.get(sym, price)
                    if shares > 0:
                        unrealized_pct = (price - ep) / ep if ep > 0 else 0
                    else:
                        unrealized_pct = (ep - price) / ep if ep > 0 else 0
                    mae_tracker[sym] = min(mae_tracker.get(sym, 0), unrealized_pct)
                    mfe_tracker[sym] = max(mfe_tracker.get(sym, 0), unrealized_pct)

            context = BacktestContext(
                bar_index=bar_idx,
                bars=current_bars,
                positions=dict(positions),
                capital=capital,
                portfolio_value=portfolio_val,
            )

            signals = strategy_fn(context)

            # Process signals
            for sym, target_signal in signals.items():
                if sym not in current_bars or len(current_bars[sym]) == 0:
                    continue

                current_row = current_bars[sym].iloc[-1]
                price = float(current_row["close"])
                current_pos = positions.get(sym, 0)
                current_dir = 1 if current_pos > 0 else (-1 if current_pos < 0 else 0)

                # Calculate ATR for volatility-based slippage
                atr = 0.0
                if len(current_bars[sym]) >= 14:
                    highs = current_bars[sym]["high"].tail(14)
                    lows = current_bars[sym]["low"].tail(14)
                    closes = current_bars[sym]["close"].tail(14)
                    tr = pd.concat([
                        highs - lows,
                        (highs - closes.shift(1)).abs(),
                        (lows - closes.shift(1)).abs(),
                    ], axis=1).max(axis=1)
                    atr = float(tr.mean())

                slip = self._compute_slippage(price, atr)

                # Close existing position if direction changes
                if current_dir != 0 and target_signal != current_dir:
                    shares_to_close = abs(current_pos)
                    if current_pos > 0:  # closing long
                        exit_price = price - slip
                        proceeds = shares_to_close * exit_price
                        comm = proceeds * self.commission
                        capital += proceeds - comm
                        pnl = (exit_price - entry_prices[sym]) * shares_to_close - comm
                    else:  # closing short
                        exit_price = price + slip
                        cost = shares_to_close * exit_price
                        comm = cost * self.commission
                        capital -= cost + comm
                        # For short: profit when price drops
                        pnl = (entry_prices[sym] - exit_price) * shares_to_close - comm

                    pnl_pct = pnl / (entry_prices[sym] * shares_to_close) if entry_prices[sym] > 0 else 0
                    direction = "LONG" if current_pos > 0 else "SHORT"

                    trade_record = TradeRecord(
                        symbol=sym,
                        direction=direction,
                        entry_date=entry_dates.get(sym, ""),
                        entry_price=entry_prices[sym],
                        exit_date=str(current_row["date"]),
                        exit_price=exit_price,
                        shares=shares_to_close,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        commission=comm,
                        slippage=slip * shares_to_close,
                        hold_bars=bar_idx - entry_indices.get(sym, bar_idx),
                        mae=mae_tracker.get(sym, 0),
                        mfe=mfe_tracker.get(sym, 0),
                    )
                    completed_trades.append(trade_record)

                    trade_log.append({
                        "type": "CLOSE_" + direction,
                        "symbol": sym,
                        "date": str(current_row["date"]),
                        "price": exit_price,
                        "shares": shares_to_close,
                        "pnl": pnl,
                        "commission": comm,
                    })

                    del positions[sym]
                    del entry_prices[sym]
                    del entry_dates[sym]
                    del entry_indices[sym]
                    mae_tracker.pop(sym, None)
                    mfe_tracker.pop(sym, None)
                    current_pos = 0

                # Open new position
                if target_signal != 0 and current_pos == 0:
                    if target_signal == 1:  # go long
                        buy_price = price + slip
                        shares = int((capital * 0.95 / len(self._data)) / buy_price) if buy_price > 0 else 0
                        if shares > 0:
                            cost = shares * buy_price
                            comm = cost * self.commission
                            capital -= cost + comm
                            positions[sym] = shares
                            entry_prices[sym] = buy_price
                            entry_dates[sym] = str(current_row["date"])
                            entry_indices[sym] = bar_idx
                            mae_tracker[sym] = 0.0
                            mfe_tracker[sym] = 0.0

                            trade_log.append({
                                "type": "BUY",
                                "symbol": sym,
                                "date": str(current_row["date"]),
                                "price": buy_price,
                                "shares": shares,
                                "commission": comm,
                            })

                    elif target_signal == -1:  # go short
                        sell_price = price - slip
                        shares = int((capital * 0.95 / len(self._data)) / sell_price) if sell_price > 0 else 0
                        if shares > 0:
                            proceeds = shares * sell_price
                            comm = proceeds * self.commission
                            capital += proceeds - comm
                            positions[sym] = -shares
                            entry_prices[sym] = sell_price
                            entry_dates[sym] = str(current_row["date"])
                            entry_indices[sym] = bar_idx
                            mae_tracker[sym] = 0.0
                            mfe_tracker[sym] = 0.0

                            trade_log.append({
                                "type": "SHORT",
                                "symbol": sym,
                                "date": str(current_row["date"]),
                                "price": sell_price,
                                "shares": shares,
                                "commission": comm,
                            })

            # Recalculate equity after trades
            equity = capital
            for sym, shares in positions.items():
                if sym in current_bars and len(current_bars[sym]) > 0:
                    p = float(current_bars[sym].iloc[-1]["close"])
                    if shares > 0:
                        equity += shares * p
                    else:
                        equity += shares * p

            equity_curve.append(equity)

            if prev_equity != 0:
                daily_returns.append((equity - prev_equity) / prev_equity)
            else:
                daily_returns.append(0.0)
            prev_equity = equity

        return self._compute_metrics(
            equity_curve, daily_returns, trade_log, completed_trades
        )

    def _compute_metrics(
        self,
        equity_curve: list[float],
        daily_returns: list[float],
        trade_log: list[dict[str, Any]],
        trades: list[TradeRecord],
    ) -> BacktestResultV2:
        """Compute all risk/return metrics."""
        if not equity_curve:
            return BacktestResultV2()

        returns = np.array(daily_returns, dtype=np.float64)

        # Total return
        total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital

        # CAGR
        n_years = len(equity_curve) / self.ANNUALIZATION_FACTOR
        final_ratio = max(0.0001, equity_curve[-1] / self.initial_capital)
        cagr = final_ratio ** (1.0 / n_years) - 1 if n_years > 0 else 0

        # Sharpe
        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
        sharpe = (mean_ret / std_ret) * math.sqrt(self.ANNUALIZATION_FACTOR) if std_ret > 0 else 0.0

        # Sortino
        downside = returns[returns < 0]
        downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
        sortino = (mean_ret / downside_std) * math.sqrt(self.ANNUALIZATION_FACTOR) if downside_std > 0 else 0.0

        # Max drawdown
        peak = equity_curve[0]
        max_dd = 0.0
        for val in equity_curve:
            if val > peak:
                peak = val
            dd = (peak - val) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        # Calmar
        calmar = cagr / max_dd if max_dd > 0 else 0.0

        # Trade metrics
        total_trades = len(trades)
        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl <= 0]

        win_rate = len(winners) / total_trades if total_trades > 0 else 0.0

        gross_profit = sum(t.pnl for t in winners) if winners else 0.0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

        avg_win = gross_profit / len(winners) if winners else 0.0
        avg_loss = gross_loss / len(losers) if losers else 0.0

        expectancy = (win_rate * avg_win - (1 - win_rate) * avg_loss) if total_trades > 0 else 0.0

        avg_duration = sum(t.hold_bars for t in trades) / total_trades if total_trades > 0 else 0.0

        # Consecutive wins/losses
        max_consec_wins = 0
        max_consec_losses = 0
        current_streak = 0
        for t in trades:
            if t.pnl > 0:
                current_streak = current_streak + 1 if current_streak > 0 else 1
                max_consec_wins = max(max_consec_wins, current_streak)
            else:
                current_streak = current_streak - 1 if current_streak < 0 else -1
                max_consec_losses = max(max_consec_losses, abs(current_streak))

        long_trades = sum(1 for t in trades if t.direction == "LONG")
        short_trades = sum(1 for t in trades if t.direction == "SHORT")

        # Van Tharp R-Multiples and SQN
        r_multiples = [t.r_multiple for t in trades if t.initial_risk > 0]
        avg_r_multiple = float(np.mean(r_multiples)) if r_multiples else 0.0
        sqn = 0.0
        if len(r_multiples) >= 10:
            r_mean = float(np.mean(r_multiples))
            r_std = float(np.std(r_multiples, ddof=1))
            if r_std > 0:
                sqn = math.sqrt(len(r_multiples)) * r_mean / r_std

        # Benchmark metrics
        alpha, beta, ir, te = 0.0, 0.0, 0.0, 0.0
        if self._benchmark_data is not None and len(self._benchmark_data) > 1:
            bench_close = self._benchmark_data["close"].values.astype(float)
            bench_returns = np.diff(bench_close) / bench_close[:-1]
            min_len = min(len(returns), len(bench_returns))
            if min_len > 10:
                port_r = returns[:min_len]
                bench_r = bench_returns[:min_len]
                cov_matrix = np.cov(port_r, bench_r)
                beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 1.0
                alpha = (np.mean(port_r) - beta * np.mean(bench_r)) * self.ANNUALIZATION_FACTOR
                active = port_r - bench_r
                te = float(np.std(active, ddof=1)) * math.sqrt(self.ANNUALIZATION_FACTOR)
                ir = float(np.mean(active)) / float(np.std(active, ddof=1)) * math.sqrt(self.ANNUALIZATION_FACTOR) if np.std(active) > 0 else 0.0

        return BacktestResultV2(
            total_return=round(total_return, 6),
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(sortino, 4),
            max_drawdown=round(max_dd, 6),
            win_rate=round(win_rate, 4),
            profit_factor=round(profit_factor, 4),
            total_trades=total_trades,
            equity_curve=equity_curve,
            trade_log=trade_log,
            cagr=round(cagr, 6),
            calmar_ratio=round(calmar, 4),
            expectancy=round(expectancy, 2),
            avg_trade_duration=round(avg_duration, 1),
            max_consecutive_wins=max_consec_wins,
            max_consecutive_losses=max_consec_losses,
            alpha=round(alpha, 6),
            beta=round(beta, 4),
            information_ratio=round(ir, 4),
            tracking_error=round(te, 6),
            trades=trades,
            long_trades=long_trades,
            short_trades=short_trades,
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            avg_r_multiple=round(avg_r_multiple, 4),
            sqn=round(sqn, 4),
        )
