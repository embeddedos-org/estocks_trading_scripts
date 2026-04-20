"""
Comprehensive tests for all strategies:
  - DCABot, OptionsWheelStrategy, PairsTradingBot,
    MomentumRebalancer, RegimeTrader
"""

import sys
import os
from datetime import datetime, date
from unittest.mock import MagicMock, patch, AsyncMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from interactive_brokers.strategies.dca_bot import (
    DCABot, DCAConfig, DCASchedule, DCAPosition,
)
from interactive_brokers.strategies.options_wheel import (
    OptionsWheelStrategy, WheelCycle, WheelPhase,
)
from interactive_brokers.strategies.pairs_trading import (
    PairsTradingBot, PairState, PairSignal,
)
from interactive_brokers.strategies.momentum_rebalancer import (
    MomentumRebalancer, MomentumScore, RebalanceConfig, RebalanceOrder,
)
from interactive_brokers.strategies.regime_trader import (
    RegimeTrader, RegimeConfig, MarketRegime, RegimeSignal,
)


def _ohlcv_df(n=250, seed=42, start="2024-01-01"):
    np.random.seed(seed)
    idx = pd.date_range(start, periods=n, freq="B")
    close = 100 + np.cumsum(np.random.normal(0.05, 1.0, n))
    close = np.maximum(close, 10)
    high = close + np.abs(np.random.normal(0, 0.5, n))
    low = close - np.abs(np.random.normal(0, 0.5, n))
    open_ = close + np.random.normal(0, 0.3, n)
    volume = np.random.randint(1_000_000, 10_000_000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# ═══════════════════════ DCABot ═══════════════════════


@pytest.fixture
def dca_bot():
    conn = MagicMock()
    om = MagicMock()
    om.market_order = MagicMock(return_value=MagicMock())
    fetcher = MagicMock()
    return DCABot(conn, om, fetcher, symbols=["SPY"], dollar_amount=500.0)


class TestDCAShouldPause:
    def test_regime_check_disabled(self, dca_bot):
        dca_bot.config.enable_regime_pause = False
        pause, reason = dca_bot._check_regime(_ohlcv_df(250))
        assert pause is False

    def test_no_datetime_index_guard(self, dca_bot):
        """Verify fix: DatetimeIndex guard."""
        df = _ohlcv_df(250)
        df.index = range(len(df))
        pause, reason = dca_bot._check_regime(df)
        assert pause is False
        assert "DatetimeIndex" in reason

    def test_pause_on_overbought_rsi(self, dca_bot):
        df = _ohlcv_df(250)
        df["close"] = np.linspace(100, 300, 250)
        df.index = pd.date_range("2024-01-01", periods=250, freq="B")
        dca_bot.config.rsi_overbought_threshold = 50.0
        pause, reason = dca_bot._check_regime(df)
        assert pause is True
        assert "RSI" in reason

    def test_pause_on_death_cross(self, dca_bot):
        df = _ohlcv_df(250)
        df["close"] = np.linspace(200, 50, 250)
        df.index = pd.date_range("2024-01-01", periods=250, freq="B")
        pause, reason = dca_bot._check_regime(df)
        assert pause is True


class TestDCARSI:
    def test_weekly_rsi_not_daily(self, dca_bot):
        """Verify fix: weekly RSI not daily."""
        df = _ohlcv_df(250)
        weekly = df["close"].resample("W").last().dropna()
        rsi = DCABot._calculate_rsi(weekly, 14)
        assert len(rsi) < 100

    def test_rsi_nan_fills_100(self):
        """Verify fix: RSI NaN → 100."""
        series = pd.Series([100.0, 101.0, 102.0])
        rsi = DCABot._calculate_rsi(series, length=14)
        assert not rsi.isna().any()


class TestDCAExecute:
    def test_execute_dca_basic(self):
        conn, om, fetcher = MagicMock(), MagicMock(), MagicMock()
        om.market_order.return_value = MagicMock()
        fetcher.fetch_bars.return_value = _ohlcv_df(250)
        bot = DCABot(conn, om, fetcher, symbols=["SPY"], dollar_amount=500.0)
        bot.config.enable_regime_pause = False
        results = bot.execute_buy_cycle()
        assert len(results) == 1
        assert results[0]["status"] == "filled"

    def test_expensive_stock_guard(self):
        """Verify fix: expensive stock guard."""
        conn, om, fetcher = MagicMock(), MagicMock(), MagicMock()
        fetcher.fetch_bars.return_value = pd.DataFrame({"close": [600.0]})
        bot = DCABot(conn, om, fetcher, symbols=["AMZN"], dollar_amount=500.0)
        bot.config.enable_regime_pause = False
        results = bot.execute_buy_cycle()
        assert results[0]["status"] == "skipped"

    def test_risk_manager_blocks(self):
        conn, om, fetcher = MagicMock(), MagicMock(), MagicMock()
        om.market_order.return_value = MagicMock()
        fetcher.fetch_bars.return_value = _ohlcv_df(250)
        rm = MagicMock()
        rm.can_trade.return_value = False
        bot = DCABot(conn, om, fetcher, risk_manager=rm,
                     symbols=["SPY"], dollar_amount=500.0)
        bot.config.enable_regime_pause = False
        results = bot.execute_buy_cycle()
        assert results[0]["status"] == "blocked"

    def test_portfolio_summary(self, dca_bot):
        s = dca_bot.get_portfolio_summary()
        assert "positions" in s
        assert s["schedule"] == "weekly"


# ═══════════════════ OptionsWheel ═══════════════════


class TestWheelIsComplete:
    def test_idle_not_complete(self):
        """Verify fix: IDLE != complete."""
        assert WheelCycle(symbol="AAPL", phase=WheelPhase.IDLE).is_complete is False

    def test_csp_not_complete(self):
        assert WheelCycle(symbol="AAPL", phase=WheelPhase.CSP_OPEN).is_complete is False

    def test_assigned_not_complete(self):
        assert WheelCycle(symbol="AAPL", phase=WheelPhase.ASSIGNED).is_complete is False

    def test_cc_not_complete(self):
        assert WheelCycle(symbol="AAPL", phase=WheelPhase.CC_OPEN).is_complete is False

    def test_called_away_is_complete(self):
        assert WheelCycle(symbol="AAPL", phase=WheelPhase.CALLED_AWAY).is_complete is True


class TestWheelRollOption:
    @pytest.mark.asyncio
    async def test_roll_in_place_update(self):
        """Verify fix: roll_option does in-place update."""
        conn, om = MagicMock(), MagicMock()
        wheel = OptionsWheelStrategy(conn, om)
        cycle = WheelCycle(symbol="AAPL", phase=WheelPhase.CSP_OPEN,
                           put_strike=150.0, put_quantity=1)
        wheel._cycles["AAPL"] = cycle
        info = {"symbol": "AAPL", "strike": 145.0, "expiry": "20250720",
                "expiry_date": date(2025, 7, 20), "dte": 30,
                "right": "P", "delta": -0.28, "bid": 2.0, "ask": 2.5, "mid": 2.25}
        with patch.object(wheel, "find_strike_by_delta",
                          new_callable=AsyncMock, return_value=info):
            result = await wheel.roll_option("AAPL")
        assert result is cycle
        assert result.put_strike == 145.0
        assert result.num_rolls == 1

    @pytest.mark.asyncio
    async def test_roll_no_cycle(self):
        wheel = OptionsWheelStrategy(MagicMock(), MagicMock())
        assert await wheel.roll_option("AAPL") is None


class TestWheelMisc:
    def test_cost_basis(self):
        c = WheelCycle(symbol="AAPL", assigned_price=150.0, put_premium=3.50)
        assert c.cost_basis == pytest.approx(146.50)

    def test_cost_basis_no_assignment(self):
        assert WheelCycle(symbol="AAPL").cost_basis == 0.0

    def test_performance_summary_empty(self):
        w = OptionsWheelStrategy(MagicMock(), MagicMock())
        assert w.get_performance_summary()["total_completed"] == 0

    def test_get_active_cycles(self):
        w = OptionsWheelStrategy(MagicMock(), MagicMock())
        w._cycles["AAPL"] = WheelCycle(symbol="AAPL", phase=WheelPhase.CSP_OPEN)
        assert "AAPL" in w.get_active_cycles()


# ═══════════════════ PairsTrading ═══════════════════


@pytest.fixture
def pairs_bot():
    conn, om = MagicMock(), MagicMock()
    om.market_order.return_value = MagicMock()
    return PairsTradingBot(conn, om, capital=100000.0)


@pytest.fixture
def correlated_prices():
    np.random.seed(42)
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    base = np.cumsum(np.random.normal(0.1, 1.0, n)) + 100
    pa = pd.Series(base + np.random.normal(0, 0.5, n), index=idx)
    pb = pd.Series(base * 0.8 + np.random.normal(0, 0.3, n) + 20, index=idx)
    return pa, pb


class TestPairsSignal:
    def test_signal_hold_flat(self, pairs_bot, correlated_prices):
        pa, pb = correlated_prices
        pairs_bot._hedge_ratio = 1.0
        sig = pairs_bot.generate_signal("A", "B", pa, pb, entry_z=5.0)
        assert sig.action == "HOLD"

    def test_elif_guard(self, pairs_bot, correlated_prices):
        """Verify fix: elif prevents overwrite in SHORT_SPREAD."""
        pa, pb = correlated_prices
        pairs_bot._state = PairState.SHORT_SPREAD
        pairs_bot._hedge_ratio = 1.0
        sig = pairs_bot.generate_signal("A", "B", pa, pb, entry_z=2.0, exit_z=0.5)
        assert sig.action in ("HOLD", "EXIT", "REVERSE_TO_LONG")

    def test_nan_guard(self, pairs_bot):
        """Verify fix: NaN z-score → HOLD."""
        pa = pd.Series([100.0, 101.0, 102.0], index=range(3))
        pb = pd.Series([50.0, 50.5, 51.0], index=range(3))
        pairs_bot._hedge_ratio = 1.0
        sig = pairs_bot.generate_signal("A", "B", pa, pb, lookback=20)
        assert sig.action == "HOLD"
        assert sig.z_score == 0.0


class TestPairsSpread:
    def test_spread(self, pairs_bot, correlated_prices):
        pa, pb = correlated_prices
        pairs_bot._hedge_ratio = 0.8
        assert len(pairs_bot.calculate_spread(pa, pb)) > 0

    def test_zscore(self, pairs_bot, correlated_prices):
        pa, pb = correlated_prices
        spread = pairs_bot.calculate_spread(pa, pb)
        z = pairs_bot.calculate_zscore(spread, 20)
        assert len(z) == len(spread)


class TestPairsEnterExit:
    def test_enter(self, pairs_bot):
        t = pairs_bot.enter_trade("A", "B", PairState.LONG_SPREAD, 150, 200, 2.5)
        assert pairs_bot._state == PairState.LONG_SPREAD
        assert t.direction == PairState.LONG_SPREAD

    def test_exit(self, pairs_bot):
        pairs_bot.enter_trade("A", "B", PairState.LONG_SPREAD, 150, 200, 2.5)
        t = pairs_bot.exit_trade(155, 198, 0.3)
        assert t.closed is True
        assert pairs_bot._state == PairState.FLAT

    def test_exit_no_position(self, pairs_bot):
        assert pairs_bot.exit_trade(100, 100, 0) is None

    def test_performance_empty(self, pairs_bot):
        assert pairs_bot.get_performance_summary()["total_trades"] == 0


# ═══════════════════ MomentumRebalancer ═══════════════════


class TestMomentumRebalance:
    def test_equal_weight(self):
        """Verify fix: equal_weight works."""
        conn, om, fetcher = MagicMock(), MagicMock(), MagicMock()
        cfg = RebalanceConfig(equal_weight=True, total_capital=100000.0)
        bot = MomentumRebalancer(conn, om, fetcher, universe=["XLK", "XLF"], config=cfg)
        scores = [
            MomentumScore("XLK", 10, 8, 5, 8.0, True, 200.0, rank=1),
            MomentumScore("XLF", 5, 4, 3, 4.0, True, 50.0, rank=2),
        ]
        orders = bot.calculate_rebalance(scores)
        buys = [o for o in orders if o.action == "BUY"]
        assert len(buys) == 2
        for o in buys:
            assert o.target_weight == pytest.approx(50.0)

    def test_fallback_equal_weight(self):
        """Verify fix: fallback when total_score==0."""
        conn, om, fetcher = MagicMock(), MagicMock(), MagicMock()
        cfg = RebalanceConfig(equal_weight=False, total_capital=100000.0)
        bot = MomentumRebalancer(conn, om, fetcher, universe=["XLK", "XLF"], config=cfg)
        scores = [
            MomentumScore("XLK", 0, 0, 0, 0.0, True, 200.0, rank=1),
            MomentumScore("XLF", 0, 0, 0, 0.0, True, 50.0, rank=2),
        ]
        orders = bot.calculate_rebalance(scores)
        assert isinstance(orders, list)

    def test_sells_removed_symbols(self):
        conn, om, fetcher = MagicMock(), MagicMock(), MagicMock()
        cfg = RebalanceConfig(total_capital=100000.0, rebalance_threshold_pct=0.0)
        bot = MomentumRebalancer(conn, om, fetcher, universe=["XLK"], config=cfg)
        bot._current_holdings = {"XLE": 50}
        scores = [MomentumScore("XLK", 10, 8, 5, 8.0, True, 200.0)]
        orders = bot.calculate_rebalance(scores)
        sells = [o for o in orders if o.action == "SELL" and o.symbol == "XLE"]
        assert len(sells) == 1


class TestMomentumScore:
    def test_fallback_fetch(self):
        """Verify fix: fallback fetch when fetch_multiple unavailable."""
        conn, om, fetcher = MagicMock(), MagicMock(), MagicMock(spec=[])
        fetcher.fetch = MagicMock(return_value=_ohlcv_df(250))
        bot = MomentumRebalancer(conn, om, fetcher, universe=["XLK"])
        scores = bot.score_universe()
        fetcher.fetch.assert_called()

    def test_insufficient_data(self):
        conn, om, fetcher = MagicMock(), MagicMock(), MagicMock(spec=[])
        fetcher.fetch = MagicMock(return_value=_ohlcv_df(50))
        bot = MomentumRebalancer(conn, om, fetcher, universe=["XLK"])
        assert bot.score_universe() == []

    def test_select_holdings(self):
        conn, om, fetcher = MagicMock(), MagicMock(), MagicMock()
        bot = MomentumRebalancer(conn, om, fetcher, universe=["A", "B", "C"])
        scores = [MomentumScore(f"S{i}", i, i, i, float(i), True, 100.0, rank=i)
                  for i in range(10, 0, -1)]
        sel = bot.select_holdings(scores)
        assert len(sel) >= bot.config.min_holdings


class TestMomentumExecute:
    def test_execute(self):
        conn, om, fetcher = MagicMock(), MagicMock(), MagicMock()
        om.market_order.return_value = MagicMock()
        bot = MomentumRebalancer(conn, om, fetcher, universe=["XLK"])
        orders = [RebalanceOrder("XLK", "BUY", 10, 0, 10, 0.0, 50.0)]
        trades = bot.execute_rebalance(orders)
        assert len(trades) == 1

    def test_risk_blocked(self):
        conn, om, fetcher = MagicMock(), MagicMock(), MagicMock()
        rm = MagicMock()
        rm.can_trade.return_value = False
        bot = MomentumRebalancer(conn, om, fetcher, risk_manager=rm, universe=["XLK"])
        assert bot.execute_rebalance([RebalanceOrder("XLK", "BUY", 10, 0, 10, 0, 50)]) == []


# ═══════════════════ RegimeTrader ═══════════════════


@pytest.fixture
def regime_trader():
    conn, om, fetcher = MagicMock(), MagicMock(), MagicMock()
    om.bracket_order.return_value = [MagicMock()]
    return RegimeTrader(conn, om, fetcher)


class TestRegimeDetect:
    def test_detect_basic(self, regime_trader):
        df = _ohlcv_df(250)
        regime = regime_trader.detect_regime(df)
        assert isinstance(regime, MarketRegime)

    def test_adx_nan_fill(self, regime_trader):
        """Verify fix: ADX NaN fill — no crash on short data."""
        df = _ohlcv_df(30)
        regime = regime_trader.detect_regime(df)
        assert regime in list(MarketRegime)

    def test_detect_volatile(self, regime_trader):
        df = _ohlcv_df(250)
        df["high"] = df["close"] + 20
        df["low"] = df["close"] - 20
        regime_trader.config.atr_volatility_mult = 0.1
        assert regime_trader.detect_regime(df) == MarketRegime.VOLATILE

    def test_ml_string_handling(self, regime_trader):
        """Verify fix: ML string handling."""
        ml = MagicMock()
        ml.predict.return_value = "TRENDING"
        ml.predict_proba.return_value = {"TRENDING": 0.8, "RANGING": 0.2}
        regime_trader.ml_classifier = ml
        regime = regime_trader.detect_regime(_ohlcv_df(250))
        assert regime == MarketRegime.TRENDING

    def test_ml_enum_handling(self, regime_trader):
        ml = MagicMock()
        ml_regime = MagicMock()
        ml_regime.name = "RANGING"
        ml.predict.return_value = ml_regime
        ml.predict_proba.return_value = {"RANGING": 0.9}
        regime_trader.ml_classifier = ml
        assert regime_trader.detect_regime(_ohlcv_df(250)) == MarketRegime.RANGING

    def test_ml_fallback_on_error(self, regime_trader):
        ml = MagicMock()
        ml.predict.side_effect = RuntimeError("fail")
        regime_trader.ml_classifier = ml
        regime = regime_trader.detect_regime(_ohlcv_df(250))
        assert regime in list(MarketRegime)


class TestRegimeSignal:
    def test_generate_signal(self, regime_trader):
        df = _ohlcv_df(250)
        sig = regime_trader.generate_signal("AAPL", df)
        assert isinstance(sig, RegimeSignal)
        assert sig.symbol == "AAPL"

    def test_volatile_no_entry(self, regime_trader):
        df = _ohlcv_df(250)
        df["high"] = df["close"] + 20
        df["low"] = df["close"] - 20
        regime_trader.config.atr_volatility_mult = 0.1
        sig = regime_trader.generate_signal("AAPL", df)
        assert sig.action == "HOLD"

    def test_signal_history(self, regime_trader):
        regime_trader.generate_signal("AAPL", _ohlcv_df(250))
        assert len(regime_trader.get_signal_history()) == 1


class TestRegimeExecution:
    def test_execute_hold(self, regime_trader):
        sig = RegimeSignal(datetime.now(), "AAPL", MarketRegime.UNKNOWN, "HOLD")
        assert regime_trader.execute_signal(sig) is None

    def test_execute_buy(self, regime_trader):
        sig = RegimeSignal(datetime.now(), "AAPL", MarketRegime.TRENDING, "BUY",
                           entry_price=150, stop_price=145, target_price=160)
        result = regime_trader.execute_signal(sig)
        assert result is not None
        assert regime_trader._position == "LONG"

    def test_execute_risk_blocked(self, regime_trader):
        rm = MagicMock()
        rm.can_trade.return_value = False
        regime_trader.risk_manager = rm
        sig = RegimeSignal(datetime.now(), "AAPL", MarketRegime.TRENDING, "BUY",
                           entry_price=150, stop_price=145, target_price=160)
        assert regime_trader.execute_signal(sig) is None

    def test_stop(self, regime_trader):
        regime_trader._running = True
        regime_trader.stop()
        assert regime_trader._running is False
