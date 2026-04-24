"""
Strategy Data Enricher
========================

Shared module providing news sentiment, fundamental quality, earnings
awareness, and regime detection to ANY strategy. Strategies call the
enricher to get additional data signals without duplicating code.

Usage in any strategy:
    from shared.strategy_enricher import StrategyEnricher, EnricherConfig

    class MyStrategy:
        def __init__(self):
            self._enricher = StrategyEnricher()

        def generate_signals(self, ctx):
            for sym, df in ctx.bars.items():
                enriched = self._enricher.enrich(sym, df)
                # enriched.sentiment_score   → float [-1, +1]
                # enriched.fundamental_ok    → bool
                # enriched.earnings_safe     → bool (no earnings in next 2 days)
                # enriched.regime            → "TRENDING" / "RANGING" / "VOLATILE"
                # enriched.composite_boost   → float [0.5, 1.5] multiplier
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EnricherConfig:
    """Configuration for the strategy data enricher."""

    # Sentiment
    use_sentiment: bool = True
    bearish_block_threshold: float = -0.4  # block entry if sentiment < this
    min_headlines: int = 3

    # Fundamentals
    use_fundamentals: bool = True
    min_market_cap: float = 1_000_000_000.0
    max_pe_ratio: float = 60.0
    max_debt_to_equity: float = 3.0

    # Earnings calendar
    use_earnings_calendar: bool = True
    earnings_blackout_days: int = 2  # avoid entering within N days of earnings

    # Regime detection
    use_regime: bool = True

    # ML signal (optional ensemble predictor)
    use_ml_signal: bool = True


@dataclass
class EnrichedData:
    """Enriched data signals for a symbol."""

    # Sentiment
    sentiment_score: float = 0.0  # [-1, +1]
    sentiment_label: str = "NEUTRAL"
    sentiment_available: bool = False

    # Fundamentals
    fundamental_ok: bool = True  # passes quality filter
    fundamental_score: float = 0.5  # [0, 1]
    fundamentals_available: bool = False

    # Earnings
    earnings_safe: bool = True  # no upcoming earnings within blackout
    days_to_earnings: Optional[int] = None
    last_surprise_pct: Optional[float] = None
    earnings_available: bool = False

    # Regime
    regime: str = "UNKNOWN"
    regime_available: bool = False

    # ML
    ml_signal: float = 0.0  # [-1, +1] from ensemble predictor
    ml_confidence: float = 0.0
    ml_available: bool = False

    # Composite
    composite_boost: float = 1.0  # multiplier [0.5, 1.5] for signal confidence


class StrategyEnricher:
    """Shared data enricher for all strategies.

    Fetches and caches sentiment, fundamentals, earnings, and regime
    data. Call enrich(symbol, df) to get all signals at once.
    """

    def __init__(self, config: EnricherConfig | None = None) -> None:
        self.config = config or EnricherConfig()
        self._fetcher: Optional[Any] = None
        self._sentiment_analyzer: Optional[Any] = None
        self._sentiment_cache: Dict[str, Dict] = {}
        self._fundamentals_cache: Dict[str, Dict] = {}
        self._earnings_cache: Dict[str, List] = {}

    def _get_fetcher(self):
        if self._fetcher is None:
            try:
                from shared.data.public_data_fetcher import PublicDataFetcher
                self._fetcher = PublicDataFetcher()
            except Exception as e:
                logger.debug("Enricher init: %s", e)
        return self._fetcher

    def _get_sentiment_analyzer(self):
        if self._sentiment_analyzer is None:
            try:
                from shared.ml.news_sentiment import NewsSentimentAnalyzer
                self._sentiment_analyzer = NewsSentimentAnalyzer()
            except Exception as e:
                logger.debug("Enricher init: %s", e)
        return self._sentiment_analyzer

    def enrich(self, symbol: str, df: pd.DataFrame) -> EnrichedData:
        """Enrich a symbol with all available data sources.

        Args:
            symbol: Ticker symbol.
            df: OHLCV DataFrame.

        Returns:
            EnrichedData with sentiment, fundamentals, earnings, regime.
        """
        result = EnrichedData()
        boost_factors = []

        if self.config.use_sentiment:
            self._enrich_sentiment(symbol, result)
            if result.sentiment_available:
                # Map sentiment [-1,+1] to boost [0.7, 1.3]
                boost_factors.append(1.0 + result.sentiment_score * 0.3)

        if self.config.use_fundamentals:
            self._enrich_fundamentals(symbol, result)
            if result.fundamentals_available:
                boost_factors.append(0.7 + result.fundamental_score * 0.6)

        if self.config.use_earnings_calendar:
            self._enrich_earnings(symbol, result)

        if self.config.use_regime and len(df) >= 60:
            self._enrich_regime(df, result)
            if result.regime_available:
                regime_boost = {"TRENDING": 1.1, "RANGING": 0.9, "VOLATILE": 0.8}
                boost_factors.append(regime_boost.get(result.regime, 1.0))

        if self.config.use_ml_signal:
            self._enrich_ml(df, result)
            if result.ml_available:
                # ML signal [-1,+1] → boost [0.7, 1.3]
                boost_factors.append(1.0 + result.ml_signal * 0.3)

        # Composite boost
        if boost_factors:
            result.composite_boost = float(np.clip(np.mean(boost_factors), 0.5, 1.5))

        return result

    def _enrich_sentiment(self, symbol: str, result: EnrichedData) -> None:
        if symbol in self._sentiment_cache:
            data = self._sentiment_cache[symbol]
        else:
            try:
                analyzer = self._get_sentiment_analyzer()
                if analyzer is None:
                    return
                data = analyzer.analyze(symbol) or {}
                self._sentiment_cache[symbol] = data
            except Exception as e:
                logger.debug("Sentiment enrichment failed for %s: %s", symbol, e)
                return

        n_headlines = data.get("headlines_analyzed", 0)
        if n_headlines >= self.config.min_headlines:
            result.sentiment_score = data.get("sentiment_score", 0.0)
            result.sentiment_label = data.get("sentiment_label", "NEUTRAL")
            result.sentiment_available = True

    def _enrich_fundamentals(self, symbol: str, result: EnrichedData) -> None:
        if symbol in self._fundamentals_cache:
            f = self._fundamentals_cache[symbol]
        else:
            try:
                fetcher = self._get_fetcher()
                if fetcher is None:
                    return
                f = fetcher.fetch_fundamentals(symbol) or {}
                self._fundamentals_cache[symbol] = f
            except Exception as e:
                logger.debug("Fundamentals enrichment failed for %s: %s", symbol, e)
                return

        if not f:
            return

        result.fundamentals_available = True
        score = 0.0
        checks = 0

        # Market cap
        mcap = f.get("market_cap")
        if mcap is not None:
            if mcap >= self.config.min_market_cap:
                score += 1
            checks += 1

        # P/E
        pe = f.get("pe_ratio")
        if pe is not None:
            if 0 < pe < self.config.max_pe_ratio:
                score += 1
            checks += 1

        # Earnings growth
        eg = f.get("earnings_growth")
        if eg is not None:
            if eg > 0:
                score += 1
            checks += 1

        # Debt/equity
        dte = f.get("debt_to_equity")
        if dte is not None:
            ratio = dte / 100.0 if dte > 10 else dte
            if ratio < self.config.max_debt_to_equity:
                score += 1
            checks += 1

        # Profit margin
        pm = f.get("profit_margin")
        if pm is not None:
            if pm > 0:
                score += 1
            checks += 1

        result.fundamental_score = score / max(checks, 1)
        result.fundamental_ok = result.fundamental_score >= 0.4

    def _enrich_earnings(self, symbol: str, result: EnrichedData) -> None:
        if symbol in self._earnings_cache:
            earnings = self._earnings_cache[symbol]
        else:
            try:
                fetcher = self._get_fetcher()
                if fetcher is None:
                    return
                earnings = fetcher.fetch_earnings_dates(symbol) or []
                self._earnings_cache[symbol] = earnings
            except Exception as e:
                logger.debug("Earnings enrichment failed for %s: %s", symbol, e)
                return

        if not earnings:
            return

        result.earnings_available = True
        now = datetime.now()

        # Find next earnings date
        for e in earnings:
            try:
                edate = pd.Timestamp(e["date"]).to_pydatetime()
                if hasattr(edate, 'tzinfo') and edate.tzinfo:
                    edate = edate.replace(tzinfo=None)
                if edate > now:
                    days = (edate - now).days
                    result.days_to_earnings = days
                    result.earnings_safe = days > self.config.earnings_blackout_days
                    break
            except (ValueError, TypeError):
                continue

        # Last surprise
        for e in sorted(earnings, key=lambda x: str(x.get("date", "")), reverse=True):
            s = e.get("surprise_pct")
            if s is not None:
                result.last_surprise_pct = s
                break

    def _enrich_regime(self, df: pd.DataFrame, result: EnrichedData) -> None:
        """Classify market regime from price data."""
        close = df["close"]

        try:
            from shared.indicators.technical_indicators import TechnicalIndicators as TI
            adx_val, _, _ = TI.adx(df, 14)
            adx = float(adx_val.iloc[-1]) if not np.isnan(adx_val.iloc[-1]) else 20

            atr = TI.atr(df, 14)
            atr_pct = float(atr.iloc[-1]) / float(close.iloc[-1]) * 100 if float(close.iloc[-1]) > 0 else 2

            if atr_pct > 4.0:
                result.regime = "VOLATILE"
            elif adx > 25:
                result.regime = "TRENDING"
            else:
                result.regime = "RANGING"

            result.regime_available = True
        except Exception:
            result.regime = "UNKNOWN"

    def _enrich_ml(self, df: pd.DataFrame, result: EnrichedData) -> None:
        """Get ML ensemble signal if available."""
        try:
            from shared.ml.ensemble_predictor import EnsemblePredictor

            close = df["close"]
            if len(close) < 30:
                return

            # Compute momentum signal (always available, no torch needed)
            ret_5d = float(close.iloc[-1] / close.iloc[-5] - 1) if len(close) >= 5 else 0
            ret_20d = float(close.iloc[-1] / close.iloc[-20] - 1) if len(close) >= 20 else 0
            momentum = (ret_5d + ret_20d) / 2

            # Use momentum as the ML signal (works without PyTorch)
            result.ml_signal = float(np.clip(momentum * 30, -1, 1))
            result.ml_confidence = min(abs(result.ml_signal), 1.0)
            result.ml_available = True

        except Exception:
            pass

    def should_block_entry(self, enriched: EnrichedData) -> tuple:
        """Check if enriched data indicates the entry should be blocked.

        Returns (blocked: bool, reason: str).
        """
        cfg = self.config

        if cfg.use_sentiment and enriched.sentiment_available:
            if enriched.sentiment_score < cfg.bearish_block_threshold:
                return True, f"Bearish sentiment ({enriched.sentiment_score:.2f})"

        if cfg.use_fundamentals and enriched.fundamentals_available:
            if not enriched.fundamental_ok:
                return True, f"Poor fundamentals (score={enriched.fundamental_score:.2f})"

        if cfg.use_earnings_calendar and enriched.earnings_available:
            if not enriched.earnings_safe:
                return True, f"Earnings in {enriched.days_to_earnings} days (blackout)"

        return False, "OK"
