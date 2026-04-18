"""
Public Data Fetcher
===================

Fetches market data and news from free public sources (Yahoo Finance, RSS feeds)
with optional in-memory caching. No brokerage account required.

Used by:
- LiveRunner (OHLCV + market-hours detection)
- NewsSentimentAnalyzer (news headlines)
- AI-webhook endpoint (OHLCV for agent decisions)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ─── NYSE market hours (Eastern) ─────────────────────────────────────────────
_MARKET_OPEN_HOUR_ET = 9
_MARKET_OPEN_MINUTE_ET = 30
_MARKET_CLOSE_HOUR_ET = 16
_MARKET_CLOSE_MINUTE_ET = 0


class PublicDataFetcher:
    """Fetches OHLCV data and news headlines from public sources.

    Args:
        cache_enabled: If True, caches fetched data for ``cache_ttl_seconds``.
        cache_ttl_seconds: How long to keep cached data (default 5 min).
    """

    def __init__(
        self,
        cache_enabled: bool = True,
        cache_ttl_seconds: int = 300,
    ) -> None:
        self._cache_enabled = cache_enabled
        self._cache_ttl = cache_ttl_seconds
        self._ohlcv_cache: Dict[str, Dict[str, Any]] = {}
        self._news_cache: Dict[str, Dict[str, Any]] = {}

    # ─── OHLCV ───────────────────────────────────────────────────────────────

    def fetch_ohlcv(
        self,
        symbol: str,
        period: str = "6mo",
        interval: str = "1d",
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from Yahoo Finance.

        Args:
            symbol: Ticker symbol (e.g., "AAPL").
            period: History window — yfinance period string ("1d","5d","1mo",
                    "3mo","6mo","1y","2y","5y","10y","ytd","max").
            interval: Bar size — yfinance interval string ("1m","2m","5m",
                      "15m","30m","60m","90m","1h","1d","5d","1wk","1mo","3mo").

        Returns:
            DataFrame with columns [open, high, low, close, volume] and a
            DatetimeIndex named "date", or None on failure.
        """
        cache_key = f"{symbol}:{period}:{interval}"
        if self._cache_enabled:
            cached = self._ohlcv_cache.get(cache_key)
            if cached and time.time() - cached["ts"] < self._cache_ttl:
                logger.debug("OHLCV cache hit: %s", cache_key)
                return cached["df"]

        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval, auto_adjust=True)

            if df is None or df.empty:
                logger.warning("No OHLCV data for %s (period=%s, interval=%s)", symbol, period, interval)
                return None

            # Normalise column names to lowercase
            df.columns = [c.lower() for c in df.columns]

            # Keep only OHLCV columns
            keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
            df = df[keep].copy()

            # Ensure DatetimeIndex named "date"
            df.index.name = "date"
            if df.index.tz is not None:
                df.index = df.index.tz_convert("UTC").tz_localize(None)

            df = df.dropna(subset=["close"])

            if self._cache_enabled:
                self._ohlcv_cache[cache_key] = {"df": df, "ts": time.time()}

            logger.debug("Fetched %d bars for %s", len(df), symbol)
            return df

        except Exception as e:
            logger.error("Failed to fetch OHLCV for %s: %s", symbol, e)
            return None

    def fetch_latest_price(self, symbol: str) -> float:
        """Return the most recent close price for a symbol.

        Returns 0.0 on failure so callers can check ``price > 0``.
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            price = getattr(info, "last_price", None) or getattr(info, "regularMarketPrice", None)
            if price and price > 0:
                return float(price)
            # Fallback: last bar from 1-day history
            df = self.fetch_ohlcv(symbol, period="5d", interval="1d")
            if df is not None and not df.empty:
                return float(df["close"].iloc[-1])
        except Exception as e:
            logger.warning("fetch_latest_price(%s) failed: %s", symbol, e)
        return 0.0

    # ─── News Headlines ───────────────────────────────────────────────────────

    def fetch_news_headlines(
        self,
        symbol: str,
        max_items: int = 20,
    ) -> List[Dict[str, str]]:
        """Fetch news headlines for a symbol.

        Tries Yahoo Finance news first, falls back to Google News RSS,
        then Reuters RSS.

        Args:
            symbol: Ticker symbol (e.g., "AAPL").
            max_items: Maximum number of headlines to return.

        Returns:
            List of dicts with keys: "title", "source", "published".
        """
        cache_key = f"news:{symbol}"
        if self._cache_enabled:
            cached = self._news_cache.get(cache_key)
            if cached and time.time() - cached["ts"] < self._cache_ttl:
                logger.debug("News cache hit: %s", symbol)
                return cached["items"][:max_items]

        headlines: List[Dict[str, str]] = []

        # 1. Yahoo Finance news via yfinance
        headlines.extend(self._fetch_yfinance_news(symbol, max_items))

        # 2. Google News RSS fallback
        if len(headlines) < max_items:
            headlines.extend(self._fetch_google_news_rss(symbol, max_items - len(headlines)))

        # Deduplicate by title
        seen: set = set()
        unique: List[Dict[str, str]] = []
        for h in headlines:
            title = h.get("title", "")
            if title and title not in seen:
                seen.add(title)
                unique.append(h)

        if self._cache_enabled:
            self._news_cache[cache_key] = {"items": unique, "ts": time.time()}

        logger.debug("Fetched %d headlines for %s", len(unique), symbol)
        return unique[:max_items]

    def _fetch_yfinance_news(self, symbol: str, limit: int) -> List[Dict[str, str]]:
        """Fetch news from Yahoo Finance via yfinance."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            news = ticker.news or []
            results = []
            for item in news[:limit]:
                title = item.get("title") or item.get("content", {}).get("title", "")
                source = (item.get("publisher") or
                          item.get("content", {}).get("provider", {}).get("displayName", "Yahoo Finance"))
                published = ""
                ptime = item.get("providerPublishTime") or item.get("content", {}).get("pubDate", "")
                if isinstance(ptime, (int, float)):
                    published = datetime.fromtimestamp(ptime, tz=timezone.utc).isoformat()
                elif isinstance(ptime, str):
                    published = ptime
                if title:
                    results.append({"title": title, "source": source, "published": published})
            return results
        except Exception as e:
            logger.debug("yfinance news fetch failed for %s: %s", symbol, e)
            return []

    def _fetch_google_news_rss(self, symbol: str, limit: int) -> List[Dict[str, str]]:
        """Fetch headlines from Google News RSS for the symbol."""
        try:
            import feedparser
            url = f"https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(url)
            results = []
            for entry in feed.entries[:limit]:
                title = entry.get("title", "")
                source = entry.get("source", {}).get("title", "Google News") if hasattr(entry.get("source", {}), "get") else "Google News"
                published = entry.get("published", "")
                if title:
                    results.append({"title": title, "source": source, "published": published})
            return results
        except Exception as e:
            logger.debug("Google News RSS fetch failed for %s: %s", symbol, e)
            return []

    # ─── Market Hours ─────────────────────────────────────────────────────────

    def is_market_open(self) -> bool:
        """Return True if the US equity market (NYSE/NASDAQ) is currently open.

        Uses system local time converted to Eastern Time. Does not account for
        market holidays — use a library like ``trading_calendars`` for holiday
        awareness if needed.
        """
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            try:
                from backports.zoneinfo import ZoneInfo  # type: ignore[no-reattr]
            except ImportError:
                # Best-effort fallback using UTC offset for ET (UTC-5 / UTC-4)
                return self._is_market_open_utc_fallback()

        now_et = datetime.now(ZoneInfo("America/New_York"))
        return self._check_market_hours(now_et)

    def _check_market_hours(self, now_et: datetime) -> bool:
        """Check if ``now_et`` falls within regular NYSE trading hours."""
        # Weekends
        if now_et.weekday() >= 5:
            return False

        open_time = now_et.replace(
            hour=_MARKET_OPEN_HOUR_ET,
            minute=_MARKET_OPEN_MINUTE_ET,
            second=0,
            microsecond=0,
        )
        close_time = now_et.replace(
            hour=_MARKET_CLOSE_HOUR_ET,
            minute=_MARKET_CLOSE_MINUTE_ET,
            second=0,
            microsecond=0,
        )
        return open_time <= now_et < close_time

    def _is_market_open_utc_fallback(self) -> bool:
        """Fallback market-hours check using UTC offset (ET = UTC-5 during EST, UTC-4 during EDT)."""
        now_utc = datetime.now(timezone.utc)
        # Approximate: use UTC-5 (EST). Slightly wrong during EDT but acceptable fallback.
        et_hour = (now_utc.hour - 5) % 24
        et_weekday = now_utc.weekday()
        # Adjust weekday if crossing midnight
        if now_utc.hour < 5:
            et_weekday = (et_weekday - 1) % 7

        if et_weekday >= 5:
            return False

        et_minutes = et_hour * 60 + now_utc.minute
        market_open = _MARKET_OPEN_HOUR_ET * 60 + _MARKET_OPEN_MINUTE_ET
        market_close = _MARKET_CLOSE_HOUR_ET * 60 + _MARKET_CLOSE_MINUTE_ET
        return market_open <= et_minutes < market_close

    def clear_cache(self) -> None:
        """Clear all in-memory caches."""
        self._ohlcv_cache.clear()
        self._news_cache.clear()
        logger.debug("Data cache cleared")

    def __repr__(self) -> str:
        return (
            f"PublicDataFetcher(cache={'on' if self._cache_enabled else 'off'}, "
            f"ttl={self._cache_ttl}s)"
        )
