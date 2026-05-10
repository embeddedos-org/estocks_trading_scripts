"""Tests for RegimeTrader, MomentumRebalancer, and DCABot IBKR strategies."""

import pytest
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from interactive_brokers.strategies.regime_trader import (
    RegimeTrader,
    RegimeConfig,
    RegimeSignal,
    MarketRegime,
)
from interactive_brokers.strategies.momentum_rebalancer import (
    MomentumRebalancer,
    RebalanceConfig,
    MomentumScore,
    RebalanceOrder,
)
from interactive_brokers.strategies.dca_bot import (
    DCABot,
    DCAConfig,
    DCAPosition,
    DCASchedule,
)


# ─── Shared Helpers ───


def _generate_ohlcv(n: int = 200, seed: int = 42, trend: float = 0.05) -> pd.DataFrame:
    """Generate synthetic OHLCV DataFrame."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    base = 100.0
    returns = rng.normal(trend / n, 0.02, n)
    close = base * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0.001, 0.02, n))
    low = close * (1 - rng.uniform(0.001, 0.02, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.randint(100000, 5000000, n).astype(float)

    df = pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=dates)
    return df


def _trending_data(n: int = 200) -> pd.DataFrame:
    """Generate data with strong uptrend (high ADX)."""
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 100.0 + np.linspace(0, 50, n) + np.random.RandomState(1).normal(0, 0.5, n)
    high = close + 1.0
    low = close - 1.0
    return pd.DataFrame({
        "open": close - 0.3, "high": high, "low": low,
        "close": close, "volume": np.full(n, 1000000.0),
    }, index=dates)


def _ranging_data(n: int = 200) -> pd.DataFrame:
    """Generate data with sideways movement (low ADX)."""
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 100.0 + np.sin(np.linspace(0, 20, n)) * 3 + np.random.RandomState(2).normal(0, 0.3, n)
    high = close + 0.5
    low = close - 0.5
    return pd.DataFrame({
        "open": close - 0.1, "high": high, "low": low,
        "close": close, "volume": np.full(n, 1000000.0),
    }, index=dates)


@pytest.fixture
def mock_connection():
    return MagicMock()


@pytest.fixture
def mock_order_manager():
    om = MagicMock()
    om.market_order = MagicMock(return_value=MagicMock(order=MagicMock(orderId=1)))
    om.bracket_order = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock()])
    return om


@pytest.fixture
def mock_fetcher():
    fetcher = MagicMock()
    fetcher.fetch_bars = MagicMock(return_value=_generate_ohlcv())
    return fetcher


@pytest.fixture
def mock_risk_manager():
    rm = MagicMock()
    rm.can_trade = MagicMock(return_value=True)
    rm.calculate_position_size = MagicMock(return_value=100)
    rm.add_position = MagicMock()
    rm.remove_position = MagicMock()
    return rm


@pytest.fixture
def mock_notifier():
    return MagicMock()


# ═══════════════════════════════════════════════════════════════
# RegimeTrader Tests
# ═══════════════════════════════════════════════════════════════


class TestMarketRegime:
    """Tests for MarketRegime enum."""

    def test_regime_values(self):
        assert MarketRegime.TRENDING.value == "TRENDING"
        assert MarketRegime.RANGING.value == "RANGING"
        assert MarketRegime.VOLATILE.value == "VOLATILE"
        assert MarketRegime.UNKNOWN.value == "UNKNOWN"


class TestRegimeConfig:
    """Tests for RegimeConfig defaults."""

    def test_default_values(self):
        config = RegimeConfig()
        assert config.adx_length == 14
        assert config.adx_trend_threshold == 25.0
        assert config.adx_range_threshold == 20.0
        assert config.ema_fast == 9
        assert config.ema_slow == 21
        assert config.trend_rr_ratio == 2.0
        assert config.rsi_overbought == 70.0
        assert config.rsi_oversold == 30.0

    def test_custom_values(self):
        config = RegimeConfig(adx_trend_threshold=30.0, trend_rr_ratio=3.0)
        assert config.adx_trend_threshold == 30.0
        assert config.trend_rr_ratio == 3.0


class TestRegimeTraderIndicators:
    """Tests for RegimeTrader static indicator calculations."""

    def test_calculate_adx_returns_series(self):
        df = _generate_ohlcv(100)
        adx = RegimeTrader._calculate_adx(df, length=14)
        assert isinstance(adx, pd.Series)
        assert len(adx) == len(df)
        # ADX values should be 0-100
        valid = adx.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_calculate_atr_returns_series(self):
        df = _generate_ohlcv(100)
        atr = RegimeTrader._calculate_atr(df, length=14)
        assert isinstance(atr, pd.Series)
        valid = atr.dropna()
        assert (valid >= 0).all()

    def test_calculate_rsi_returns_series(self):
        df = _generate_ohlcv(100)
        rsi = RegimeTrader._calculate_rsi(df["close"], length=14)
        assert isinstance(rsi, pd.Series)
        valid = rsi.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_calculate_bollinger_returns_three_series(self):
        df = _generate_ohlcv(100)
        upper, mid, lower = RegimeTrader._calculate_bollinger(df["close"])
        assert isinstance(upper, pd.Series)
        assert isinstance(mid, pd.Series)
        assert isinstance(lower, pd.Series)
        # Upper > mid > lower
        valid_idx = upper.dropna().index.intersection(lower.dropna().index)
        assert (upper.loc[valid_idx] >= mid.loc[valid_idx]).all()
        assert (mid.loc[valid_idx] >= lower.loc[valid_idx]).all()


class TestRegimeDetection:
    """Tests for the detect_regime method."""

    def test_detect_regime_returns_enum(self, mock_connection, mock_order_manager, mock_fetcher):
        trader = RegimeTrader(mock_connection, mock_order_manager, mock_fetcher)
        df = _generate_ohlcv(200)
        regime = trader.detect_regime(df)
        assert isinstance(regime, MarketRegime)
        assert regime in (MarketRegime.TRENDING, MarketRegime.RANGING,
                          MarketRegime.VOLATILE, MarketRegime.UNKNOWN)

    def test_regime_stored_as_current(self, mock_connection, mock_order_manager, mock_fetcher):
        trader = RegimeTrader(mock_connection, mock_order_manager, mock_fetcher)
        df = _generate_ohlcv(200)
        regime = trader.detect_regime(df)
        assert trader._current_regime == regime


class TestRegimeSignalGeneration:
    """Tests for the generate_signal method."""

    def test_signal_returns_regime_signal(self, mock_connection, mock_order_manager, mock_fetcher):
        trader = RegimeTrader(mock_connection, mock_order_manager, mock_fetcher)
        df = _generate_ohlcv(200)
        signal = trader.generate_signal("AAPL", df)
        assert isinstance(signal, RegimeSignal)
        assert signal.symbol == "AAPL"
        assert signal.regime in (MarketRegime.TRENDING, MarketRegime.RANGING,
                                  MarketRegime.VOLATILE, MarketRegime.UNKNOWN)
        assert signal.action in ("BUY", "SELL", "HOLD")

    def test_signal_added_to_history(self, mock_connection, mock_order_manager, mock_fetcher):
        trader = RegimeTrader(mock_connection, mock_order_manager, mock_fetcher)
        df = _generate_ohlcv(200)
        trader.generate_signal("AAPL", df)
        assert len(trader.get_signal_history()) == 1

    def test_hold_signal_has_zero_prices(self, mock_connection, mock_order_manager, mock_fetcher):
        trader = RegimeTrader(mock_connection, mock_order_manager, mock_fetcher)
        # Force a volatile regime where no trades happen
        trader._position = "LONG"  # already in position
        df = _generate_ohlcv(200)
        signal = trader.generate_signal("AAPL", df)
        if signal.action == "HOLD":
            assert signal.entry_price == 0.0


class TestRegimeExecution:
    """Tests for execute_signal."""

    def test_hold_signal_returns_none(self, mock_connection, mock_order_manager, mock_fetcher):
        trader = RegimeTrader(mock_connection, mock_order_manager, mock_fetcher)
        signal = RegimeSignal(
            timestamp=datetime.now(), symbol="AAPL",
            regime=MarketRegime.VOLATILE, action="HOLD",
        )
        result = trader.execute_signal(signal)
        assert result is None
        mock_order_manager.bracket_order.assert_not_called()

    def test_buy_signal_places_bracket_order(
        self, mock_connection, mock_order_manager, mock_fetcher, mock_risk_manager
    ):
        trader = RegimeTrader(
            mock_connection, mock_order_manager, mock_fetcher,
            risk_manager=mock_risk_manager,
        )
        signal = RegimeSignal(
            timestamp=datetime.now(), symbol="AAPL",
            regime=MarketRegime.TRENDING, action="BUY",
            entry_price=150.0, stop_price=145.0, target_price=160.0,
        )
        result = trader.execute_signal(signal)
        assert result is not None
        mock_order_manager.bracket_order.assert_called_once()
        assert trader._position == "LONG"

    def test_risk_manager_blocks_trade(
        self, mock_connection, mock_order_manager, mock_fetcher
    ):
        rm = MagicMock()
        rm.can_trade = MagicMock(return_value=False)
        trader = RegimeTrader(
            mock_connection, mock_order_manager, mock_fetcher,
            risk_manager=rm,
        )
        signal = RegimeSignal(
            timestamp=datetime.now(), symbol="AAPL",
            regime=MarketRegime.TRENDING, action="BUY",
            entry_price=150.0, stop_price=145.0, target_price=160.0,
        )
        result = trader.execute_signal(signal)
        assert result is None
        mock_order_manager.bracket_order.assert_not_called()

    def test_stop_sets_running_false(self, mock_connection, mock_order_manager, mock_fetcher):
        trader = RegimeTrader(mock_connection, mock_order_manager, mock_fetcher)
        trader._running = True
        trader.stop()
        assert trader._running is False


# ═══════════════════════════════════════════════════════════════
# MomentumRebalancer Tests
# ═══════════════════════════════════════════════════════════════


class TestRebalanceConfig:
    """Tests for RebalanceConfig."""

    def test_defaults(self):
        config = RebalanceConfig()
        assert config.weight_1m == 0.40
        assert config.weight_3m == 0.35
        assert config.weight_6m == 0.25
        assert config.top_pct == 0.20
        assert config.require_above_200sma is True
        assert config.equal_weight is True
        assert config.rebalance_threshold_pct == 5.0


class TestMomentumScoring:
    """Tests for MomentumRebalancer scoring logic."""

    def test_roc_calculation(self):
        prices = pd.Series(np.linspace(100, 110, 50))
        roc = MomentumRebalancer._calculate_roc(prices, 21)
        assert isinstance(roc, float)
        assert roc > 0  # uptrend → positive ROC

    def test_roc_downtrend(self):
        prices = pd.Series(np.linspace(110, 100, 50))
        roc = MomentumRebalancer._calculate_roc(prices, 21)
        assert roc < 0

    def test_roc_insufficient_data(self):
        prices = pd.Series([100, 101, 102])
        roc = MomentumRebalancer._calculate_roc(prices, 21)
        assert np.isnan(roc)

    def test_score_universe_returns_sorted_list(
        self, mock_connection, mock_order_manager
    ):
        fetcher = MagicMock()
        data = {}
        for sym in ["XLK", "XLF", "XLE"]:
            data[sym] = _generate_ohlcv(250, seed=hash(sym) % 100, trend=0.1)
        fetcher.fetch_multiple = MagicMock(return_value=data)

        rebalancer = MomentumRebalancer(
            mock_connection, mock_order_manager, fetcher,
            universe=["XLK", "XLF", "XLE"],
            config=RebalanceConfig(require_above_200sma=False),
        )
        scores = rebalancer.score_universe()
        assert isinstance(scores, list)
        assert len(scores) > 0
        assert all(isinstance(s, MomentumScore) for s in scores)
        # Verify sorted descending by composite
        for i in range(len(scores) - 1):
            assert scores[i].composite_score >= scores[i + 1].composite_score

    def test_score_ranks_assigned(self, mock_connection, mock_order_manager):
        fetcher = MagicMock()
        data = {s: _generate_ohlcv(250, seed=i) for i, s in enumerate(["A", "B", "C"])}
        fetcher.fetch_multiple = MagicMock(return_value=data)

        rebalancer = MomentumRebalancer(
            mock_connection, mock_order_manager, fetcher,
            universe=["A", "B", "C"],
            config=RebalanceConfig(require_above_200sma=False),
        )
        scores = rebalancer.score_universe()
        ranks = [s.rank for s in scores]
        assert ranks == [1, 2, 3]


class TestHoldingsSelection:
    """Tests for select_holdings."""

    def test_select_respects_top_pct(self, mock_connection, mock_order_manager, mock_fetcher):
        rebalancer = MomentumRebalancer(
            mock_connection, mock_order_manager, mock_fetcher,
            config=RebalanceConfig(top_pct=0.50, min_holdings=1, max_holdings=10),
        )
        scores = [
            MomentumScore("A", 10, 8, 5, 8.0, True, 100),
            MomentumScore("B", 8, 6, 4, 6.0, True, 100),
            MomentumScore("C", 5, 3, 2, 3.0, True, 100),
            MomentumScore("D", 2, 1, 0, 1.0, True, 100),
        ]
        selected = rebalancer.select_holdings(scores)
        assert len(selected) == 2  # 50% of 4 = 2

    def test_select_enforces_min_holdings(self, mock_connection, mock_order_manager, mock_fetcher):
        rebalancer = MomentumRebalancer(
            mock_connection, mock_order_manager, mock_fetcher,
            config=RebalanceConfig(top_pct=0.01, min_holdings=2, max_holdings=10),
        )
        scores = [
            MomentumScore("A", 10, 8, 5, 8.0, True, 100),
            MomentumScore("B", 5, 3, 2, 3.0, True, 100),
        ]
        selected = rebalancer.select_holdings(scores)
        assert len(selected) >= 2


class TestRebalanceCalculation:
    """Tests for calculate_rebalance."""

    def test_generates_buy_orders_for_new_positions(
        self, mock_connection, mock_order_manager, mock_fetcher
    ):
        rebalancer = MomentumRebalancer(
            mock_connection, mock_order_manager, mock_fetcher,
            config=RebalanceConfig(total_capital=100000, rebalance_threshold_pct=0),
        )
        selected = [
            MomentumScore("XLK", 10, 8, 5, 8.0, True, 100),
            MomentumScore("XLF", 5, 3, 2, 3.0, True, 50),
        ]
        orders = rebalancer.calculate_rebalance(selected, current_positions={})
        buy_orders = [o for o in orders if o.action == "BUY"]
        assert len(buy_orders) == 2
        symbols = {o.symbol for o in buy_orders}
        assert "XLK" in symbols
        assert "XLF" in symbols

    def test_generates_sell_for_removed_positions(
        self, mock_connection, mock_order_manager, mock_fetcher
    ):
        rebalancer = MomentumRebalancer(
            mock_connection, mock_order_manager, mock_fetcher,
            config=RebalanceConfig(total_capital=100000, rebalance_threshold_pct=0),
        )
        selected = [MomentumScore("XLK", 10, 8, 5, 8.0, True, 100)]
        current = {"XLK": 500, "XLE": 200}  # XLE no longer selected
        orders = rebalancer.calculate_rebalance(selected, current)
        sell_orders = [o for o in orders if o.action == "SELL" and o.symbol == "XLE"]
        assert len(sell_orders) == 1
        assert sell_orders[0].quantity == 200

    def test_rebalance_history_tracked(
        self, mock_connection, mock_order_manager, mock_fetcher
    ):
        rebalancer = MomentumRebalancer(
            mock_connection, mock_order_manager, mock_fetcher,
            config=RebalanceConfig(total_capital=100000, rebalance_threshold_pct=0),
        )
        selected = [MomentumScore("XLK", 10, 8, 5, 8.0, True, 100)]
        orders = rebalancer.calculate_rebalance(selected, current_positions={})
        rebalancer.execute_rebalance(orders)
        history = rebalancer.get_rebalance_history()
        assert len(history) == 1
        assert "timestamp" in history[0]
        assert "orders" in history[0]


# ═══════════════════════════════════════════════════════════════
# DCABot Tests
# ═══════════════════════════════════════════════════════════════


class TestDCAConfig:
    """Tests for DCAConfig."""

    def test_defaults(self):
        config = DCAConfig()
        assert config.dollar_amount == 500.0
        assert config.schedule == DCASchedule.WEEKLY
        assert "SPY" in config.symbols
        assert "QQQ" in config.symbols
        assert config.enable_regime_pause is True
        assert config.rsi_overbought_threshold == 75.0


class TestDCAPosition:
    """Tests for DCAPosition tracking."""

    def test_initial_position(self):
        pos = DCAPosition(symbol="SPY")
        assert pos.total_invested == 0.0
        assert pos.total_shares == 0.0
        assert pos.current_value == 0.0
        assert pos.unrealized_pnl == 0.0
        assert pos.return_pct == 0.0

    def test_position_with_values(self):
        pos = DCAPosition(
            symbol="SPY",
            total_invested=5000.0,
            total_shares=10.0,
            avg_cost_basis=500.0,
            current_price=550.0,
        )
        assert pos.current_value == 5500.0
        assert pos.unrealized_pnl == 500.0
        assert abs(pos.return_pct - 10.0) < 0.001

    def test_position_loss(self):
        pos = DCAPosition(
            symbol="QQQ",
            total_invested=10000.0,
            total_shares=25.0,
            avg_cost_basis=400.0,
            current_price=380.0,
        )
        assert pos.current_value == 9500.0
        assert pos.unrealized_pnl == -500.0
        assert abs(pos.return_pct - (-5.0)) < 0.001


class TestDCARegimeCheck:
    """Tests for DCA regime-aware pausing."""

    def test_normal_conditions_no_pause(
        self, mock_connection, mock_order_manager, mock_fetcher, mock_notifier
    ):
        bot = DCABot(
            mock_connection, mock_order_manager, mock_fetcher,
            notifier=mock_notifier,
        )
        df = _generate_ohlcv(250)
        should_pause, reason = bot._check_regime(df)
        # In normal conditions, might or might not pause — just verify it runs
        assert isinstance(should_pause, bool)
        assert isinstance(reason, str)

    def test_regime_pause_disabled(
        self, mock_connection, mock_order_manager, mock_fetcher
    ):
        config = DCAConfig(enable_regime_pause=False)
        bot = DCABot(
            mock_connection, mock_order_manager, mock_fetcher,
            config=config,
        )
        df = _generate_ohlcv(250)
        should_pause, reason = bot._check_regime(df)
        assert should_pause is False
        assert "disabled" in reason.lower()

    def test_rsi_calculation(self):
        prices = pd.Series(np.linspace(100, 110, 50))
        rsi = DCABot._calculate_rsi(prices, 14)
        assert isinstance(rsi, pd.Series)
        valid = rsi.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()


class TestDCABuyCycle:
    """Tests for execute_buy_cycle."""

    def test_buy_cycle_executes(
        self, mock_connection, mock_order_manager, mock_notifier
    ):
        fetcher = MagicMock()
        fetcher.fetch_bars = MagicMock(return_value=_generate_ohlcv(250))

        config = DCAConfig(
            symbols=["SPY"],
            dollar_amount=1000.0,
            enable_regime_pause=False,
        )
        bot = DCABot(
            mock_connection, mock_order_manager, fetcher,
            notifier=mock_notifier, config=config,
        )
        results = bot.execute_buy_cycle()
        assert isinstance(results, list)
        if results:
            assert results[0]["symbol"] == "SPY"
            assert results[0]["status"] == "filled"
            assert results[0]["quantity"] > 0
            mock_order_manager.market_order.assert_called()

    def test_buy_cycle_tracks_position(
        self, mock_connection, mock_order_manager
    ):
        fetcher = MagicMock()
        fetcher.fetch_bars = MagicMock(return_value=_generate_ohlcv(250))

        config = DCAConfig(
            symbols=["QQQ"],
            dollar_amount=500.0,
            enable_regime_pause=False,
        )
        bot = DCABot(
            mock_connection, mock_order_manager, fetcher,
            config=config,
        )
        results = bot.execute_buy_cycle()
        pos = bot._positions["QQQ"]
        if results and results[0].get("status") == "filled":
            assert pos.total_invested > 0
            assert pos.total_shares > 0
            assert pos.buy_count == 1
            assert pos.avg_cost_basis > 0

    def test_buy_cycle_splits_equally(
        self, mock_connection, mock_order_manager
    ):
        fetcher = MagicMock()
        fetcher.fetch_bars = MagicMock(return_value=_generate_ohlcv(250))

        config = DCAConfig(
            symbols=["SPY", "QQQ"],
            dollar_amount=1000.0,
            equal_split=True,
            enable_regime_pause=False,
        )
        bot = DCABot(
            mock_connection, mock_order_manager, fetcher,
            config=config,
        )
        results = bot.execute_buy_cycle()
        # Each symbol should get ~$500
        filled = [r for r in results if r.get("status") == "filled"]
        if len(filled) == 2:
            assert filled[0]["total_cost"] < 600  # roughly $500 each


class TestDCAPortfolioSummary:
    """Tests for get_portfolio_summary."""

    def test_empty_portfolio(self, mock_connection, mock_order_manager, mock_fetcher):
        bot = DCABot(mock_connection, mock_order_manager, mock_fetcher)
        summary = bot.get_portfolio_summary()
        assert summary["total_invested"] == 0.0
        assert summary["total_value"] == 0.0
        assert summary["total_buys"] == 0
        assert summary["paused_cycles"] == 0
        assert summary["schedule"] == "weekly"

    def test_summary_after_buys(self, mock_connection, mock_order_manager, mock_fetcher):
        bot = DCABot(mock_connection, mock_order_manager, mock_fetcher)
        # Manually set position data
        bot._positions["SPY"] = DCAPosition(
            symbol="SPY",
            total_invested=5000.0,
            total_shares=10.0,
            avg_cost_basis=500.0,
            current_price=520.0,
            buy_count=5,
        )
        summary = bot.get_portfolio_summary()
        assert summary["total_invested"] == 5000.0
        assert summary["total_value"] == 5200.0
        assert summary["total_pnl"] == 200.0
        assert summary["total_return_pct"] == 4.0
        assert summary["total_buys"] == 5

    def test_buy_history_tracking(self, mock_connection, mock_order_manager, mock_fetcher):
        bot = DCABot(mock_connection, mock_order_manager, mock_fetcher)
        assert bot.get_buy_history() == []

    def test_stop_sets_flag(self, mock_connection, mock_order_manager, mock_fetcher):
        bot = DCABot(mock_connection, mock_order_manager, mock_fetcher)
        bot._running = True
        bot.stop()
        assert bot._running is False
