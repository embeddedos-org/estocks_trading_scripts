"""
Production Safety Tests
========================

Tests for all critical investment risk controls that must pass
before live trading with real money.

Covers:
- Position size caps (max % equity, max notional, max shares)
- Pre-order validation (fat-finger, price sanity)
- Short-selling limits
- Market hours gate
- Liquidity filter
- Data source health monitoring
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.risk_manager import RiskManager, RiskManagerConfig, SizingMethod


# ══════════════════════════════════════════════════════════════════════════════
# Critical 1: Max Position Size Caps
# ══════════════════════════════════════════════════════════════════════════════

class TestPositionSizeCaps:
    """Verify position sizing is capped to prevent over-leverage."""

    def _make_rm(self, **kwargs):
        config = RiskManagerConfig(
            total_capital=100_000.0,
            min_seconds_between_trades=0,
            **kwargs,
        )
        return RiskManager(config=config)

    def test_max_position_pct_equity(self):
        """Position size must not exceed max_position_pct_equity."""
        rm = self._make_rm(
            sizing_method=SizingMethod.FIXED_DOLLAR,
            fixed_dollar_amount=50_000.0,  # would be 50% of equity
            max_position_pct_equity=25.0,  # cap at 25%
        )
        shares = rm.calculate_position_size("AAPL", 100.0)
        notional = shares * 100.0
        # 25% of $100k = $25k → max 250 shares
        assert notional <= 25_000.0, f"Notional ${notional} exceeds 25% of equity"

    def test_max_position_notional(self):
        """Position must not exceed max_position_notional dollar cap."""
        rm = self._make_rm(
            sizing_method=SizingMethod.FIXED_DOLLAR,
            fixed_dollar_amount=100_000.0,
            max_position_notional=20_000.0,
        )
        shares = rm.calculate_position_size("AAPL", 100.0)
        notional = shares * 100.0
        assert notional <= 20_000.0, f"Notional ${notional} exceeds $20k cap"

    def test_max_shares_per_order(self):
        """Position must not exceed max_shares_per_order (fat-finger)."""
        rm = self._make_rm(
            sizing_method=SizingMethod.FIXED_SHARES,
            fixed_shares=50_000,
            max_shares_per_order=5_000,
        )
        shares = rm.calculate_position_size("AAPL", 1.0)
        assert shares <= 5_000, f"Shares {shares} exceeds fat-finger limit"

    def test_caps_apply_to_kelly_sizing(self):
        """Kelly criterion sizing must also be capped."""
        rm = self._make_rm(
            sizing_method=SizingMethod.KELLY,
            kelly_win_rate=0.70,
            kelly_avg_win=3.0,
            kelly_avg_loss=1.0,
            kelly_fraction=1.0,  # full Kelly (aggressive)
            max_position_pct_equity=10.0,
        )
        shares = rm.calculate_position_size("X", 10.0)
        notional = shares * 10.0
        assert notional <= 10_000.0, f"Kelly notional ${notional} exceeds 10% cap"

    def test_caps_apply_to_fixed_fractional(self):
        """Fixed fractional with tight stop must still be capped."""
        rm = self._make_rm(
            risk_per_trade_pct=2.0,
            max_position_pct_equity=25.0,
            max_shares_per_order=10_000,
        )
        # $0.10 stop → risk $2000 / $0.10 = 20,000 shares uncapped
        shares = rm.calculate_position_size("X", 50.0, stop_price=49.90)
        assert shares <= 10_000, f"Shares {shares} exceeds fat-finger limit"
        notional = shares * 50.0
        assert notional <= 25_000.0, f"Notional ${notional} exceeds 25% cap"


# ══════════════════════════════════════════════════════════════════════════════
# Critical 2: Pre-Order Validation
# ══════════════════════════════════════════════════════════════════════════════

class TestPreOrderValidation:
    """Verify validate_order() catches dangerous orders."""

    def _make_rm(self, **kwargs):
        config = RiskManagerConfig(
            total_capital=100_000.0,
            min_seconds_between_trades=0,
            max_trades_per_hour=10000,
            **kwargs,
        )
        return RiskManager(config=config)

    def test_fat_finger_rejected(self):
        rm = self._make_rm(max_shares_per_order=1000)
        ok, reason = rm.validate_order("AAPL", 5000, 150.0, 150.0)
        assert ok is False
        assert "max_shares_per_order" in reason

    def test_price_deviation_rejected(self):
        rm = self._make_rm(max_price_deviation_pct=5.0)
        ok, reason = rm.validate_order("AAPL", 100, 200.0, 150.0)
        assert ok is False
        assert "deviates" in reason

    def test_price_within_range_accepted(self):
        rm = self._make_rm(max_price_deviation_pct=10.0)
        ok, reason = rm.validate_order("AAPL", 100, 152.0, 150.0)
        assert ok is True

    def test_notional_exceeds_equity_pct(self):
        rm = self._make_rm(max_position_pct_equity=10.0)
        # 500 shares * $250 = $125k > 10% of $100k
        ok, reason = rm.validate_order("AAPL", 500, 250.0, 250.0)
        assert ok is False
        assert "equity" in reason.lower()

    def test_valid_order_accepted(self):
        rm = self._make_rm()
        ok, reason = rm.validate_order("AAPL", 50, 150.0, 150.0)
        assert ok is True
        assert reason == "OK"


# ══════════════════════════════════════════════════════════════════════════════
# Critical 3: Short-Selling Limits
# ══════════════════════════════════════════════════════════════════════════════

class TestShortSellingLimits:
    """Verify short-selling controls prevent unlimited loss exposure."""

    def _make_rm(self, **kwargs):
        defaults = dict(
            total_capital=100_000.0,
            min_seconds_between_trades=0,
            max_trades_per_hour=10000,
            max_short_positions=2,
            max_short_exposure_pct=20.0,
        )
        defaults.update(kwargs)
        config = RiskManagerConfig(**defaults)
        return RiskManager(config=config)

    def test_max_short_positions_enforced(self):
        rm = self._make_rm()
        # Register 2 short positions
        rm._open_positions["AAPL"] = -5000.0
        rm._open_positions["MSFT"] = -5000.0
        ok, reason = rm.validate_order("GOOGL", 50, 100.0, 100.0, direction="SHORT")
        assert ok is False
        assert "short positions" in reason.lower()

    def test_short_exposure_pct_enforced(self):
        rm = self._make_rm(max_short_exposure_pct=10.0)
        rm._open_positions["AAPL"] = -8000.0  # already 8% short
        # Adding $5000 more would be 13% > 10%
        ok, reason = rm.validate_order("MSFT", 50, 100.0, 100.0, direction="SHORT")
        assert ok is False
        assert "short exposure" in reason.lower()

    def test_long_order_bypasses_short_limits(self):
        rm = self._make_rm(max_short_positions=0)  # no shorts allowed
        ok, reason = rm.validate_order("AAPL", 50, 150.0, 150.0, direction="LONG")
        assert ok is True


# ══════════════════════════════════════════════════════════════════════════════
# Critical 4: Liquidity Filter
# ══════════════════════════════════════════════════════════════════════════════

class TestLiquidityFilter:
    """Verify orders are rejected for illiquid stocks."""

    def _make_rm(self, **kwargs):
        config = RiskManagerConfig(
            total_capital=100_000.0,
            min_seconds_between_trades=0,
            max_trades_per_hour=10000,
            min_avg_volume=100_000,
            max_position_pct_adv=5.0,
            **kwargs,
        )
        return RiskManager(config=config)

    def test_low_volume_rejected(self):
        rm = self._make_rm()
        ok, reason = rm.validate_order(
            "PENNY", 100, 5.0, 5.0, avg_daily_volume=50_000
        )
        assert ok is False
        assert "volume" in reason.lower()

    def test_position_too_large_for_volume(self):
        rm = self._make_rm(max_position_pct_equity=100.0, max_shares_per_order=50000)
        # 6,000 shares at $1 = $6k notional (OK for equity), but 6% of 100k ADV > 5%
        ok, reason = rm.validate_order(
            "MICRO", 6_000, 1.0, 1.0, avg_daily_volume=100_000,
            direction="LONG",
        )
        assert ok is False
        assert "daily volume" in reason.lower()

    def test_adequate_liquidity_passes(self):
        rm = self._make_rm()
        ok, reason = rm.validate_order(
            "AAPL", 100, 150.0, 150.0, avg_daily_volume=5_000_000
        )
        assert ok is True


# ══════════════════════════════════════════════════════════════════════════════
# High 5: Market Hours Gate
# ══════════════════════════════════════════════════════════════════════════════

class TestMarketHoursGate:
    """Verify market hours enforcement."""

    def test_disabled_allows_trading(self):
        rm = RiskManager(config=RiskManagerConfig(enforce_market_hours=False))
        allowed, reason = rm.check_market_hours()
        assert allowed is True

    def test_config_defaults_safe(self):
        """Default config has market hours enforcement OFF (for backtesting)."""
        config = RiskManagerConfig()
        assert config.enforce_market_hours is False
        assert config.max_short_positions == 5
        assert config.max_shares_per_order == 10000
        assert config.max_position_pct_equity == 25.0


# ══════════════════════════════════════════════════════════════════════════════
# High 6: Data Source Health
# ══════════════════════════════════════════════════════════════════════════════

class TestDataSourceHealth:
    """Verify data source health monitoring."""

    def test_health_check_returns_dict(self):
        from shared.data.public_data_fetcher import PublicDataFetcher
        pf = PublicDataFetcher(cache_enabled=False)
        health = pf.get_data_health()
        assert isinstance(health, dict)
        assert "ohlcv_failures" in health
        assert "fundamentals_failures" in health
        assert "circuit_breaker_open" in health
        assert health["ohlcv_failures"] == 0
        assert health["circuit_breaker_open"] is False

    def test_separate_circuit_breakers(self):
        from shared.data.public_data_fetcher import PublicDataFetcher
        pf = PublicDataFetcher(cache_enabled=False)
        pf._fundamentals_failures = 10
        health = pf.get_data_health()
        assert health["fundamentals_cb_open"] is True
        assert health["circuit_breaker_open"] is False  # OHLCV still fine


# ══════════════════════════════════════════════════════════════════════════════
# Integration: Full Production Safety Check
# ══════════════════════════════════════════════════════════════════════════════

class TestProductionReadiness:
    """End-to-end safety checks for production deployment."""

    def test_all_safety_fields_exist_in_config(self):
        """All critical production safety fields must exist in config."""
        config = RiskManagerConfig()
        required_fields = [
            "max_position_pct_equity",
            "max_position_notional",
            "max_shares_per_order",
            "max_price_deviation_pct",
            "max_short_positions",
            "max_short_exposure_pct",
            "require_short_stop",
            "enforce_market_hours",
            "min_avg_volume",
            "max_position_pct_adv",
        ]
        for field in required_fields:
            assert hasattr(config, field), f"Missing production safety field: {field}"

    def test_all_safety_methods_exist(self):
        """All critical production safety methods must exist."""
        rm = RiskManager()
        required_methods = [
            "validate_order",
            "check_market_hours",
            "can_trade",
            "can_pyramid",
            "check_portfolio_heat",
            "get_status",
            "get_monthly_pnl",
        ]
        for method in required_methods:
            assert hasattr(rm, method), f"Missing production safety method: {method}"
            assert callable(getattr(rm, method)), f"{method} is not callable"

    def test_status_includes_production_fields(self):
        """get_status() must expose production safety state."""
        rm = RiskManager(config=RiskManagerConfig(
            min_seconds_between_trades=0,
            max_trades_per_hour=10000,
        ))
        status = rm.get_status()
        assert "max_position_pct_equity" in status
        assert "max_shares_per_order" in status
        assert "enforce_market_hours" in status
        assert "monthly_pnl" in status

    def test_default_config_is_conservative(self):
        """Default config values must be conservative for safety."""
        config = RiskManagerConfig()
        assert config.risk_per_trade_pct <= 2.0, "Default risk too high"
        assert config.max_daily_loss <= 5000.0, "Default daily loss too high"
        assert config.max_drawdown_pct <= 15.0, "Default drawdown too high"
        assert config.max_open_positions <= 15, "Default positions too many"
        assert config.max_shares_per_order <= 10000, "Default fat-finger too high"
        assert config.max_position_pct_equity <= 30.0, "Default position % too high"
        assert config.enable_pyramiding is False, "Pyramiding should be off by default"
        assert config.kelly_fraction <= 0.5, "Kelly fraction too aggressive"
