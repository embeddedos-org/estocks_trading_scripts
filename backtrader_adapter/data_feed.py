"""
Backtrader Data Feed Adapters
==============================

Custom data feeds that bridge pandas DataFrames and MarketDataCache
into Backtrader's Cerebro engine.

Usage:
    feed = DataFrameFeed(dataname=df)
    cerebro.adddata(feed)
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import backtrader as bt  # type: ignore[import-untyped]
    _HAS_BT = True
except ImportError:
    _HAS_BT = False
    logger.warning("backtrader not installed. Install: pip install backtrader")

if _HAS_BT:

    class DataFrameFeed(bt.feeds.PandasData):
        """Custom Backtrader feed from a pandas DataFrame.

        Expects columns: open, high, low, close, volume.
        Index must be DatetimeIndex.
        """

        params = (
            ("open", "open"),
            ("high", "high"),
            ("low", "low"),
            ("close", "close"),
            ("volume", "volume"),
            ("openinterest", None),
        )

    class CacheFeed:
        """Reads from MarketDataCache and wraps as a Backtrader DataFrameFeed.

        Usage:
            from shared.data.market_data_cache import MarketDataCache
            cache = MarketDataCache()
            feed = CacheFeed.from_cache(cache, "AAPL", "1D")
            cerebro.adddata(feed)
        """

        @staticmethod
        def from_cache(
            cache,
            symbol: str,
            bar_size: str = "1 day",
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
        ) -> DataFrameFeed:
            """Create a Backtrader feed from MarketDataCache.

            Args:
                cache: MarketDataCache instance
                symbol: Ticker symbol
                bar_size: Bar size string
                start_date: Optional start date filter
                end_date: Optional end date filter

            Returns:
                DataFrameFeed ready for cerebro.adddata()
            """
            df = cache.get_bars(symbol, bar_size, start_date=start_date, end_date=end_date)
            if df is None or df.empty:
                raise ValueError(f"No cached data for {symbol} ({bar_size})")

            if not isinstance(df.index, pd.DatetimeIndex):
                if "date" in df.columns:
                    df = df.set_index("date")
                df.index = pd.to_datetime(df.index)

            required = ["open", "high", "low", "close", "volume"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")

            return DataFrameFeed(dataname=df)

else:
    DataFrameFeed = None  # type: ignore[assignment,misc]
    CacheFeed = None  # type: ignore[assignment,misc]
