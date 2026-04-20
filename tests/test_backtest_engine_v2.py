"""
Comprehensive tests for BacktestEngineV2.

Covers: __init__, load_data(), run(), _execute_bar(), position sizing
(equal allocation across assets), CAGR calculation (negative equity guarded),
Sharpe/Sortino/Calmar ratios, max drawdown, trade logging, multi-asset
support, commission handling, BacktestContext fields, and edge cases.
"""

import sys
import os
import math

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared.backtesting.backtest_engine_v2 import (
    BacktestContext,
    BacktestEngineV2,
    BacktestResultV2,
    SlippageConfig,
    TradeRecord,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_ohlcv(
    closes: list[float],
    start: str = "2023-01-01",
    spread: float = 0.5,
) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame from a list of close prices."""
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame(
        {
            "date": dates,
            "open": [c - spread for c in closes],
            "high": [c + spread for c in closes],
            "low": [c - spread for c in closes],
            "close": closes,
            "volume": [1_000_000] * len(closes),
        }
    )


def _trending_up(n: int = 60, start: float = 100.0, step: float = 0.5) -> list[float]:
    return [start + i * step for i in range(n)]


def _trending_down(n: int = 60, start: float = 150.0, step: float = 0.5) -> list[float]:
    return [start - i * step for i in range(n)]


def _flat(n: int = 60, price: float = 100.0) -> list[float]:
    return [price] * n


def _volatile(n: int = 60, base: float = 100.0, amplitude: float = 5.0) -> list[float]:
    return [base + amplitude * math.sin(i * 0.5) for i in range(n)]


def _always_long_strategy(ctx: BacktestContext) -> dict[str, int]:
    """Strategy that goes long on every asset every bar."""
    return {sym: 1 for sym in ctx.bars}


def _always_short_strategy(ctx: BacktestContext) -> dict[str, int]:
    return {sym: -1 for sym in ctx.bars}


def _always_hold_strategy(ctx: BacktestContext) -> dict[str, int]:
    return {sym: 0 for sym in ctx.bars}


def _alternating_strategy(ctx: BacktestContext) -> dict[str, int]:
    """Alternates between long and flat every 5 bars."""
    signal = 1 if (ctx.bar_index // 5) % 2 == 0 else 0
    return {sym: signal for sym in ctx.bars}


# ── Dataclass Tests ──────────────────────────────────────────────────────

class TestBacktestContext:
    def test_fields_present(self):
        ctx = BacktestContext(
            bar_index=0,
            bars={},
            positions={},
            capital=100_000.0,
            portfolio_value=100_000.0,
        )
        assert ctx.bar_index == 0
        assert ctx.capital == 100_000.0
        assert ctx.indicators == {}

    def test_indicators_default(self):
        ctx = BacktestContext(0, {}, {}, 0.0, 0.0)
        assert isinstance(ctx.indicators, dict)


class TestTradeRecord:
    def test_defaults(self):
        tr = TradeRecord(
            symbol="AAPL",
            direction="LONG",
            entry_date="2023-01-01",
            entry_price=150.0,
            exit_date="2023-01-10",
            exit_price=155.0,
            shares=10,
            pnl=50.0,
            pnl_pct=0.033,
            commission=0.15,
            slippage=0.01,
            hold_bars=9,
        )
        assert tr.mae == 0.0
        assert tr.mfe == 0.0


class TestBacktestResultV2:
    def test_defaults(self):
        r = BacktestResultV2()
        assert r.total_return == 0.0
        assert r.total_trades == 0
        assert r.equity_curve == []
        assert r.trades == []
        assert r.cagr == 0.0


class TestSlippageConfig:
    def test_defaults(self):
        sc = SlippageConfig()
        assert sc.method == "fixed"
        assert sc.fixed_cents == 1.0

    def test_custom(self):
        sc = SlippageConfig(method="percentage", percentage=0.1)
        assert sc.method == "percentage"
        assert sc.percentage == 0.1


# ── Engine Init Tests ────────────────────────────────────────────────────

class TestEngineInit:
    def test_default_values(self):
        engine = BacktestEngineV2()
        assert engine.initial_capital == 100_000.0
        assert engine.commission == 0.001
        assert engine.slippage.method == "fixed"
        assert engine._data == {}

    def test_custom_values(self):
        sc = SlippageConfig(method="percentage", percentage=0.05)
        engine = BacktestEngineV2(initial_capital=50_000, commission=0.002, slippage=sc)
        assert engine.initial_capital == 50_000
        assert engine.commission == 0.002
        assert engine.slippage.method == "percentage"


# ── load_data Tests ──────────────────────────────────────────────────────

class TestLoadData:
    def test_single_dataframe(self):
        engine = BacktestEngineV2()
        df = _make_ohlcv(_flat(10))
        engine.load_data(df)
        assert "DEFAULT" in engine._data
        assert len(engine._data["DEFAULT"]) == 10

    def test_dict_of_dataframes(self):
        engine = BacktestEngineV2()
        engine.load_data({
            "AAPL": _make_ohlcv(_flat(10)),
            "MSFT": _make_ohlcv(_flat(10)),
        })
        assert "AAPL" in engine._data
        assert "MSFT" in engine._data

    def test_missing_columns_raises(self):
        engine = BacktestEngineV2()
        df = pd.DataFrame({"date": ["2023-01-01"], "close": [100]})
        with pytest.raises(ValueError, match="Missing columns"):
            engine.load_data(df)

    def test_datetime_column_renamed(self):
        engine = BacktestEngineV2()
        df = _make_ohlcv(_flat(5))
        df.rename(columns={"date": "datetime"}, inplace=True)
        engine.load_data(df)
        assert "date" in engine._data["DEFAULT"].columns

    def test_columns_lowered_and_stripped(self):
        engine = BacktestEngineV2()
        df = _make_ohlcv(_flat(5))
        df.columns = [" Date ", " Open ", " High ", " Low ", " Close ", " Volume "]
        engine.load_data(df)
        assert "date" in engine._data["DEFAULT"].columns

    def test_data_sorted_by_date(self):
        engine = BacktestEngineV2()
        df = _make_ohlcv(_flat(5))
        df = df.iloc[::-1].reset_index(drop=True)
        engine.load_data(df)
        dates = engine._data["DEFAULT"]["date"].tolist()
        assert dates == sorted(dates)


# ── _compute_slippage Tests ──────────────────────────────────────────────

class TestComputeSlippage:
    def test_fixed_slippage(self):
        engine = BacktestEngineV2(slippage=SlippageConfig(method="fixed", fixed_cents=2.0))
        assert engine._compute_slippage(100.0) == pytest.approx(0.02)

    def test_percentage_slippage(self):
        engine = BacktestEngineV2(slippage=SlippageConfig(method="percentage", percentage=0.1))
        assert engine._compute_slippage(200.0) == pytest.approx(0.2)

    def test_volatility_slippage_with_atr(self):
        engine = BacktestEngineV2(slippage=SlippageConfig(method="volatility", volatility_mult=0.5))
        assert engine._compute_slippage(100.0, atr=2.0) == pytest.approx(1.0)

    def test_volatility_slippage_zero_atr_fallback(self):
        engine = BacktestEngineV2(slippage=SlippageConfig(method="volatility"))
        result = engine._compute_slippage(100.0, atr=0.0)
        assert result == pytest.approx(0.1)

    def test_unknown_method_returns_zero(self):
        engine = BacktestEngineV2(slippage=SlippageConfig(method="unknown"))
        assert engine._compute_slippage(100.0) == 0.0


# ── run() Tests ──────────────────────────────────────────────────────────

class TestRunBasic:
    def test_no_data_raises(self):
        engine = BacktestEngineV2()
        with pytest.raises(RuntimeError, match="No data loaded"):
            engine.run(_always_hold_strategy)

    def test_hold_strategy_no_trades(self):
        engine = BacktestEngineV2()
        engine.load_data(_make_ohlcv(_flat(20)))
        result = engine.run(_always_hold_strategy)
        assert result.total_trades == 0
        assert result.total_return == pytest.approx(0.0, abs=1e-4)
        assert len(result.equity_curve) == 20

    def test_long_trending_up_positive_return(self):
        engine = BacktestEngineV2(commission=0.0, slippage=SlippageConfig(method="fixed", fixed_cents=0.0))
        engine.load_data(_make_ohlcv(_trending_up(30)))
        result = engine.run(_always_long_strategy)
        assert result.total_return > 0

    def test_short_trending_down_positive_return(self):
        engine = BacktestEngineV2(commission=0.0, slippage=SlippageConfig(method="fixed", fixed_cents=0.0))
        engine.load_data(_make_ohlcv(_trending_down(30, start=200.0)))
        result = engine.run(_always_short_strategy)
        assert result.total_return > 0

    def test_equity_curve_length_matches_bars(self):
        engine = BacktestEngineV2()
        n = 25
        engine.load_data(_make_ohlcv(_flat(n)))
        result = engine.run(_always_hold_strategy)
        assert len(result.equity_curve) == n


# ── Position Sizing: Equal Allocation ────────────────────────────────────

class TestPositionSizing:
    def test_equal_allocation_across_assets(self):
        """Verify fix: capital * 0.95 / len(self._data) allocates per-asset."""
        engine = BacktestEngineV2(
            initial_capital=100_000,
            commission=0.0,
            slippage=SlippageConfig(method="fixed", fixed_cents=0.0),
        )
        engine.load_data({
            "A": _make_ohlcv(_flat(5, price=50.0)),
            "B": _make_ohlcv(_flat(5, price=50.0)),
        })

        result = engine.run(_always_long_strategy)
        buy_entries = [t for t in result.trade_log if t["type"] == "BUY"]
        assert len(buy_entries) >= 1
        first_shares = buy_entries[0]["shares"]
        expected = int((100_000 * 0.95 / 2) / 50.0)
        assert first_shares == expected


# ── CAGR Calculation: Negative Equity Guard ──────────────────────────────

class TestCAGRCalculation:
    def test_cagr_positive_returns(self):
        engine = BacktestEngineV2(commission=0.0, slippage=SlippageConfig(method="fixed", fixed_cents=0.0))
        engine.load_data(_make_ohlcv(_trending_up(252)))
        result = engine.run(_always_long_strategy)
        assert result.cagr > 0

    def test_cagr_negative_equity_guarded(self):
        """Verify fix: max(0.0001, equity/initial) prevents math domain error."""
        engine = BacktestEngineV2(commission=0.0, slippage=SlippageConfig(method="fixed", fixed_cents=0.0))
        engine.load_data(_make_ohlcv(_trending_down(60, start=200.0, step=3.0)))
        result = engine.run(_always_long_strategy)
        assert math.isfinite(result.cagr)

    def test_cagr_flat_market_near_zero(self):
        engine = BacktestEngineV2(commission=0.0, slippage=SlippageConfig(method="fixed", fixed_cents=0.0))
        engine.load_data(_make_ohlcv(_flat(252)))
        result = engine.run(_always_hold_strategy)
        assert abs(result.cagr) < 0.01


# ── Risk Metrics ─────────────────────────────────────────────────────────

class TestRiskMetrics:
    def _run_trending(self, closes):
        engine = BacktestEngineV2(commission=0.0, slippage=SlippageConfig(method="fixed", fixed_cents=0.0))
        engine.load_data(_make_ohlcv(closes))
        return engine.run(_always_long_strategy)

    def test_sharpe_ratio_positive_trend(self):
        result = self._run_trending(_trending_up(60))
        assert result.sharpe_ratio > 0

    def test_sortino_ratio_positive_trend(self):
        result = self._run_trending(_trending_up(60))
        assert result.sortino_ratio >= 0

    def test_calmar_ratio(self):
        result = self._run_trending(_trending_up(60))
        assert result.calmar_ratio >= 0

    def test_max_drawdown_range(self):
        result = self._run_trending(_volatile(60))
        assert 0.0 <= result.max_drawdown <= 1.0

    def test_max_drawdown_trending_up_small(self):
        result = self._run_trending(_trending_up(60))
        assert result.max_drawdown < 0.5

    def test_sharpe_zero_when_no_variation(self):
        engine = BacktestEngineV2()
        engine.load_data(_make_ohlcv(_flat(20)))
        result = engine.run(_always_hold_strategy)
        assert result.sharpe_ratio == 0.0


# ── Trade Logging & Analytics ────────────────────────────────────────────

class TestTradeLogging:
    def test_trade_log_populated(self):
        engine = BacktestEngineV2(commission=0.0, slippage=SlippageConfig(method="fixed", fixed_cents=0.0))
        engine.load_data(_make_ohlcv(_flat(20)))
        result = engine.run(_alternating_strategy)
        assert len(result.trade_log) > 0

    def test_trade_log_entry_fields(self):
        engine = BacktestEngineV2(commission=0.0, slippage=SlippageConfig(method="fixed", fixed_cents=0.0))
        engine.load_data(_make_ohlcv(_trending_up(20)))
        result = engine.run(_alternating_strategy)
        if result.trade_log:
            entry = result.trade_log[0]
            assert "type" in entry
            assert "symbol" in entry
            assert "date" in entry
            assert "price" in entry
            assert "shares" in entry

    def test_completed_trades_have_pnl(self):
        engine = BacktestEngineV2(commission=0.0, slippage=SlippageConfig(method="fixed", fixed_cents=0.0))
        engine.load_data(_make_ohlcv(_trending_up(20)))
        result = engine.run(_alternating_strategy)
        for trade in result.trades:
            assert isinstance(trade.pnl, float)
            assert isinstance(trade.pnl_pct, float)

    def test_win_rate_bounds(self):
        engine = BacktestEngineV2(commission=0.0, slippage=SlippageConfig(method="fixed", fixed_cents=0.0))
        engine.load_data(_make_ohlcv(_trending_up(40)))
        result = engine.run(_alternating_strategy)
        assert 0.0 <= result.win_rate <= 1.0

    def test_long_short_trade_counts(self):
        engine = BacktestEngineV2(commission=0.0, slippage=SlippageConfig(method="fixed", fixed_cents=0.0))
        engine.load_data(_make_ohlcv(_trending_up(40)))

        def mixed_strategy(ctx: BacktestContext) -> dict[str, int]:
            phase = ctx.bar_index // 10
            if phase % 3 == 0:
                return {sym: 1 for sym in ctx.bars}
            elif phase % 3 == 1:
                return {sym: -1 for sym in ctx.bars}
            else:
                return {sym: 0 for sym in ctx.bars}

        result = engine.run(mixed_strategy)
        assert result.long_trades + result.short_trades == result.total_trades


# ── Multi-Asset Support ──────────────────────────────────────────────────

class TestMultiAsset:
    def test_multi_asset_run(self):
        engine = BacktestEngineV2(commission=0.0, slippage=SlippageConfig(method="fixed", fixed_cents=0.0))
        engine.load_data({
            "AAPL": _make_ohlcv(_trending_up(30, start=150)),
            "MSFT": _make_ohlcv(_trending_up(30, start=300)),
        })
        result = engine.run(_always_long_strategy)
        assert result.total_return > 0

    def test_multi_asset_different_lengths(self):
        engine = BacktestEngineV2(commission=0.0, slippage=SlippageConfig(method="fixed", fixed_cents=0.0))
        engine.load_data({
            "A": _make_ohlcv(_flat(20)),
            "B": _make_ohlcv(_flat(30)),
        })
        result = engine.run(_always_long_strategy)
        assert len(result.equity_curve) == 30


# ── Commission Handling ──────────────────────────────────────────────────

class TestCommission:
    def test_commission_reduces_returns(self):
        no_comm = BacktestEngineV2(commission=0.0, slippage=SlippageConfig(method="fixed", fixed_cents=0.0))
        with_comm = BacktestEngineV2(commission=0.01, slippage=SlippageConfig(method="fixed", fixed_cents=0.0))

        data = _make_ohlcv(_trending_up(30))
        no_comm.load_data(data)
        with_comm.load_data(data.copy())

        r1 = no_comm.run(_alternating_strategy)
        r2 = with_comm.run(_alternating_strategy)

        assert r1.equity_curve[-1] >= r2.equity_curve[-1]

    def test_commission_in_trade_record(self):
        engine = BacktestEngineV2(commission=0.005, slippage=SlippageConfig(method="fixed", fixed_cents=0.0))
        engine.load_data(_make_ohlcv(_trending_up(20)))
        result = engine.run(_alternating_strategy)
        for trade in result.trades:
            assert trade.commission >= 0


# ── Edge Cases ───────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_single_bar(self):
        engine = BacktestEngineV2()
        engine.load_data(_make_ohlcv([100.0]))
        result = engine.run(_always_long_strategy)
        assert len(result.equity_curve) == 1

    def test_two_bars(self):
        engine = BacktestEngineV2(commission=0.0, slippage=SlippageConfig(method="fixed", fixed_cents=0.0))
        engine.load_data(_make_ohlcv([100.0, 110.0]))
        result = engine.run(_always_long_strategy)
        assert len(result.equity_curve) == 2

    def test_volatile_market_finite_metrics(self):
        engine = BacktestEngineV2(commission=0.0, slippage=SlippageConfig(method="fixed", fixed_cents=0.0))
        engine.load_data(_make_ohlcv(_volatile(60)))
        result = engine.run(_alternating_strategy)
        assert math.isfinite(result.sharpe_ratio)
        assert math.isfinite(result.sortino_ratio)
        assert math.isfinite(result.max_drawdown)
        assert math.isfinite(result.cagr)


# ── Benchmark / Alpha-Beta Tests ─────────────────────────────────────────

class TestBenchmark:
    def test_set_benchmark(self):
        engine = BacktestEngineV2()
        bench = _make_ohlcv(_trending_up(30))
        engine.set_benchmark(bench)
        assert engine._benchmark_data is not None

    def test_alpha_beta_computed_with_benchmark(self):
        engine = BacktestEngineV2(commission=0.0, slippage=SlippageConfig(method="fixed", fixed_cents=0.0))
        engine.load_data(_make_ohlcv(_trending_up(60)))
        engine.set_benchmark(_make_ohlcv(_trending_up(60, step=0.3)))
        result = engine.run(_always_long_strategy)
        assert math.isfinite(result.alpha)
        assert math.isfinite(result.beta)
