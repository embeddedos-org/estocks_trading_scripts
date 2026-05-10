"""
Historical Market Data Fetcher for Interactive Brokers
=======================================================

Fetches historical OHLCV bars from IB with proper pacing,
contract resolution, and DataFrame output.

Usage:
    fetcher = HistoricalDataFetcher(connection)
    df = fetcher.fetch_bars("AAPL", duration="1 Y", bar_size="1 day")
    fetcher.save_to_csv(df, "aapl_daily.csv")
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_PACING_INTERVAL = 11.0  # seconds between requests (conservative)
_MAX_REQUESTS_PER_10MIN = 60


class HistoricalDataFetcher:
    """Fetches historical market data from Interactive Brokers.

    Handles contract creation, IB pacing limits, and returns
    data as pandas DataFrames. Optionally caches results to
    avoid redundant API calls.

    Args:
        connection: An IBInsyncConnection instance.
        cache: Optional MarketDataCache for local caching.
    """

    VALID_BAR_SIZES = [
        "1 secs", "5 secs", "10 secs", "15 secs", "30 secs",
        "1 min", "2 mins", "3 mins", "5 mins", "10 mins", "15 mins",
        "20 mins", "30 mins", "1 hour", "2 hours", "3 hours",
        "4 hours", "8 hours", "1 day", "1 week", "1 month",
    ]

    VALID_WHAT_TO_SHOW = [
        "TRADES", "MIDPOINT", "BID", "ASK", "BID_ASK",
        "ADJUSTED_LAST", "HISTORICAL_VOLATILITY",
        "OPTION_IMPLIED_VOLATILITY",
    ]

    def __init__(self, connection: Any, cache: Any = None) -> None:
        self.connection = connection
        self.cache = cache  # Optional MarketDataCache instance
        self._last_request_time: float = 0.0
        self._request_count: int = 0
        self._request_window_start: float = 0.0

    def _create_contract(
        self,
        symbol: str,
        sec_type: str = "STK",
        exchange: str = "SMART",
        currency: str = "USD",
        **kwargs: Any,
    ) -> Any:
        """Create and qualify an IB contract.

        Args:
            symbol: Ticker symbol.
            sec_type: Security type (STK, OPT, FUT, CASH).
            exchange: Exchange (default SMART).
            currency: Currency code.

        Returns:
            A qualified Contract object.
        """
        try:
            from ib_insync import Stock, Forex, Future, Contract

            if sec_type == "STK":
                contract = Stock(symbol, exchange, currency)
            elif sec_type == "CASH":
                contract = Forex(symbol)
            elif sec_type == "FUT":
                contract = Future(
                    symbol,
                    kwargs.get("expiry", ""),
                    exchange,
                    currency=currency,
                )
            else:
                contract = Contract(
                    symbol=symbol,
                    secType=sec_type,
                    exchange=exchange,
                    currency=currency,
                )

            self.connection.qualifyContracts(contract)
            return contract
        except ImportError:
            raise ImportError(
                "ib_insync is required for HistoricalDataFetcher. "
                "Install with: pip install ib_insync"
            )

    def _respect_pacing(self) -> None:
        """Enforce IB historical data pacing rules.

        IB allows a maximum of 60 historical data requests in any
        10-minute window. This method sleeps if necessary to stay
        within limits.
        """
        now = time.time()

        if now - self._request_window_start > 600:
            self._request_count = 0
            self._request_window_start = now

        if self._request_count >= _MAX_REQUESTS_PER_10MIN:
            wait = 600 - (now - self._request_window_start) + 1
            logger.warning(
                "Pacing limit reached (%d requests). Waiting %.0fs...",
                self._request_count, wait,
            )
            time.sleep(wait)
            self._request_count = 0
            self._request_window_start = time.time()

        elapsed = now - self._last_request_time
        if elapsed < _PACING_INTERVAL:
            sleep_time = _PACING_INTERVAL - elapsed
            logger.debug("Pacing: sleeping %.1fs", sleep_time)
            time.sleep(sleep_time)

        self._last_request_time = time.time()
        self._request_count += 1

    def fetch_bars(
        self,
        symbol: str,
        duration: str = "1 Y",
        bar_size: str = "1 day",
        what_to_show: str = "TRADES",
        end_date: str = "",
        use_rth: bool = True,
        sec_type: str = "STK",
        exchange: str = "SMART",
        currency: str = "USD",
        use_cache: bool = True,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fetch historical bars for a single symbol.

        If a cache is configured and use_cache is True, checks cache
        first and only fetches from IB on cache miss or stale data.

        Args:
            symbol: Ticker symbol (e.g., "AAPL").
            duration: Duration string (e.g., "1 Y", "6 M", "30 D").
            bar_size: Bar size (e.g., "1 day", "1 hour", "5 mins").
            what_to_show: Data type ("TRADES", "MIDPOINT", "BID", "ASK").
            end_date: End date/time string (empty = now).
            use_rth: If True, only return Regular Trading Hours data.
            sec_type: Security type.
            exchange: Exchange name.
            currency: Currency code.
            use_cache: If True and cache is configured, use cached data.

        Returns:
            DataFrame with columns: date, open, high, low, close, volume,
            barCount, average.

        Raises:
            ValueError: If invalid bar_size or what_to_show is provided.
        """
        if bar_size not in self.VALID_BAR_SIZES:
            raise ValueError(
                f"Invalid bar_size '{bar_size}'. Valid: {self.VALID_BAR_SIZES}"
            )
        if what_to_show not in self.VALID_WHAT_TO_SHOW:
            raise ValueError(
                f"Invalid what_to_show '{what_to_show}'. "
                f"Valid: {self.VALID_WHAT_TO_SHOW}"
            )

        # Check cache first
        if use_cache and self.cache is not None and not self.cache.is_stale(symbol, bar_size):
            cached = self.cache.get_bars(symbol, bar_size)
            if cached is not None and not cached.empty:
                logger.info("Using cached data for %s [%s] (%d bars)", symbol, bar_size, len(cached))
                return cached

        contract = self._create_contract(
            symbol, sec_type, exchange, currency, **kwargs
        )
        self._respect_pacing()

        logger.info(
            "Fetching %s bars for %s [duration=%s, bar_size=%s]",
            what_to_show, symbol, duration, bar_size,
        )

        try:
            bars = self.connection.reqHistoricalData(
                contract,
                endDateTime=end_date,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=use_rth,
                formatDate=1,
            )

            if not bars:
                logger.warning("No data returned for %s", symbol)
                return pd.DataFrame()

            from ib_insync import util
            df = util.df(bars)

            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)

            df.attrs["symbol"] = symbol
            df.attrs["duration"] = duration
            df.attrs["bar_size"] = bar_size

            logger.info(
                "Fetched %d bars for %s (%s to %s)",
                len(df), symbol,
                df.index[0] if len(df) > 0 else "N/A",
                df.index[-1] if len(df) > 0 else "N/A",
            )

            # Store to cache
            if self.cache is not None and not df.empty:
                try:
                    self.cache.store_bars(symbol, bar_size, df)
                except Exception as cache_err:
                    logger.warning("Failed to cache bars for %s: %s", symbol, cache_err)

            return df

        except Exception as e:
            logger.error("Failed to fetch bars for %s: %s", symbol, e)
            raise

    def fetch_multiple(
        self,
        symbols: List[str],
        duration: str = "1 Y",
        bar_size: str = "1 day",
        what_to_show: str = "TRADES",
        **kwargs: Any,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch historical bars for multiple symbols.

        Respects IB pacing rules between requests.

        Args:
            symbols: List of ticker symbols.
            duration: Duration string.
            bar_size: Bar size string.
            what_to_show: Data type.

        Returns:
            Dictionary mapping symbol → DataFrame.
        """
        results: Dict[str, pd.DataFrame] = {}

        logger.info(
            "Fetching data for %d symbols: %s",
            len(symbols), ", ".join(symbols),
        )

        for i, symbol in enumerate(symbols):
            try:
                logger.info(
                    "Fetching %s (%d/%d)...",
                    symbol, i + 1, len(symbols),
                )
                df = self.fetch_bars(
                    symbol,
                    duration=duration,
                    bar_size=bar_size,
                    what_to_show=what_to_show,
                    **kwargs,
                )
                results[symbol] = df
            except Exception as e:
                logger.error("Failed to fetch %s: %s", symbol, e)
                results[symbol] = pd.DataFrame()

        logger.info(
            "Completed: %d/%d symbols fetched successfully",
            sum(1 for df in results.values() if not df.empty),
            len(symbols),
        )
        return results

    @staticmethod
    def save_to_csv(
        df: pd.DataFrame,
        filepath: str | Path,
        include_metadata: bool = True,
    ) -> Path:
        """Save a DataFrame to CSV.

        Args:
            df: The DataFrame to save.
            filepath: Output file path.
            include_metadata: If True, write metadata as CSV comments.

        Returns:
            The resolved Path of the saved file.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if include_metadata and hasattr(df, "attrs") and df.attrs:
            with open(filepath, "w", newline="") as f:
                for key, value in df.attrs.items():
                    f.write(f"# {key}: {value}\n")
                df.to_csv(f)
        else:
            df.to_csv(filepath)

        logger.info("Saved %d rows to %s", len(df), filepath)
        return filepath
