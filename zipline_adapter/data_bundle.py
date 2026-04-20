"""
Zipline Data Bundle Loader
=============================

Registers a custom zipline data bundle that loads from
MarketDataCache (SQLite) or CSV files.

Requires: pip install zipline-reloaded

Usage:
    loader = CacheBundleLoader()
    loader.register_bundle("stocks_plugin")
    loader.ingest_from_cache(["AAPL", "MSFT"], "2020-01-01", "2024-01-01")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from zipline.data.bundles import register  # type: ignore[import-untyped]
    from zipline.utils.calendar_utils import get_calendar  # type: ignore[import-untyped]
    _HAS_ZIPLINE = True
except ImportError:
    _HAS_ZIPLINE = False
    logger.debug("zipline-reloaded not installed — bundle loader unavailable")

import sys
import os
_parent_path = os.path.join(os.path.dirname(__file__), "..")
if _parent_path not in sys.path:
    sys.path.insert(0, _parent_path)

try:
    from shared.data.market_data_cache import MarketDataCache
except ImportError:
    MarketDataCache = None  # type: ignore[assignment,misc]
    logger.warning("shared.data.market_data_cache not available — MarketDataCache will be None")


def _require_zipline() -> None:
    if not _HAS_ZIPLINE:
        raise ImportError(
            "zipline-reloaded is required for data bundles. "
            "Install with: pip install zipline-reloaded>=3.0"
        )


class CacheBundleLoader:
    """Load market data into zipline from MarketDataCache or CSV files.

    Bridges stocks_plugin's SQLite cache and the zipline data
    ingestion pipeline.

    Args:
        cache: Optional MarketDataCache instance (creates default if None).
        bar_size: Bar size string used in cache (default: "1 day").
    """

    def __init__(
        self,
        cache: Optional[MarketDataCache] = None,
        bar_size: str = "1 day",
    ) -> None:
        self._cache = cache or MarketDataCache()
        self._bar_size = bar_size

    def register_bundle(self, bundle_name: str = "stocks_plugin") -> None:
        """Register a custom zipline data bundle backed by MarketDataCache.

        Args:
            bundle_name: Name for the zipline bundle.
        """
        _require_zipline()

        cache = self._cache
        bar_size = self._bar_size

        def ingest(environ: Any, asset_db_writer: Any, minute_bar_writer: Any,
                   daily_bar_writer: Any, adjustment_writer: Any,
                   calendar: Any, start_session: Any, end_session: Any,
                   cache_obj: MarketDataCache = cache,
                   bs: str = bar_size) -> None:

            stats = cache_obj.get_cache_stats()
            symbols = stats.get("symbols", [])

            if not symbols:
                logger.warning("No symbols in cache — nothing to ingest")
                return

            metadata = pd.DataFrame(columns=[
                "start_date", "end_date", "auto_close_date",
                "symbol", "exchange",
            ])

            data_dict: Dict[int, pd.DataFrame] = {}

            for sid, sym in enumerate(symbols):
                df = cache_obj.get_bars(sym, bs)
                if df is None or df.empty:
                    continue

                if isinstance(df.index, pd.DatetimeIndex):
                    df = df.reset_index()
                    if "date" not in df.columns:
                        df.rename(columns={df.columns[0]: "date"}, inplace=True)

                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date")

                ohlcv = pd.DataFrame({
                    "open": df["open"].values,
                    "high": df["high"].values,
                    "low": df["low"].values,
                    "close": df["close"].values,
                    "volume": df["volume"].values,
                }, index=pd.DatetimeIndex(df["date"].values, tz="UTC"))

                data_dict[sid] = ohlcv

                meta_row = pd.DataFrame([{
                    "start_date": ohlcv.index[0],
                    "end_date": ohlcv.index[-1],
                    "auto_close_date": ohlcv.index[-1] + pd.Timedelta(days=1),
                    "symbol": sym,
                    "exchange": "NYSE",
                }])
                metadata = pd.concat([metadata, meta_row], ignore_index=True)

            if data_dict:
                daily_bar_writer.write(
                    ((sid, df) for sid, df in data_dict.items()),
                    show_progress=True,
                )

            asset_db_writer.write(equities=metadata)
            adjustment_writer.write()

            logger.info(
                "Bundle '%s' ingested: %d symbols",
                bundle_name,
                len(data_dict),
            )

        register(bundle_name, ingest)
        logger.info("Zipline bundle '%s' registered", bundle_name)

    def ingest_from_cache(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Read data from MarketDataCache for specified symbols.

        This does NOT require zipline — it simply returns DataFrames
        that can be used with BacktestEngineV2 or passed to zipline.

        Args:
            symbols: List of ticker symbols.
            start_date: Optional start date (YYYY-MM-DD).
            end_date: Optional end date (YYYY-MM-DD).

        Returns:
            Dict of {symbol: OHLCV DataFrame}.
        """
        data: Dict[str, pd.DataFrame] = {}

        for sym in symbols:
            df = self._cache.get_bars(sym, self._bar_size, start_date, end_date)
            if df is not None and not df.empty:
                data[sym] = df
                logger.info("Loaded %d bars for %s from cache", len(df), sym)
            else:
                logger.warning("No cached data for %s", sym)

        return data

    @staticmethod
    def ingest_from_csv(
        csv_dir: str,
        date_column: str = "date",
    ) -> Dict[str, pd.DataFrame]:
        """Load OHLCV data from CSV files in a directory.

        Each CSV file should be named {SYMBOL}.csv and contain
        columns: date, open, high, low, close, volume.

        Compatible with HistoricalDataFetcher.save_to_csv() output.

        Args:
            csv_dir: Path to directory containing CSV files.
            date_column: Name of the date column.

        Returns:
            Dict of {symbol: OHLCV DataFrame}.
        """
        csv_path = Path(csv_dir)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV directory not found: {csv_dir}")

        data: Dict[str, pd.DataFrame] = {}

        for csv_file in sorted(csv_path.glob("*.csv")):
            symbol = csv_file.stem.upper()

            try:
                df = pd.read_csv(csv_file)
                df.columns = [c.strip().lower() for c in df.columns]

                if date_column.lower() in df.columns:
                    df[date_column.lower()] = pd.to_datetime(df[date_column.lower()])
                    df.set_index(date_column.lower(), inplace=True)

                required = {"open", "high", "low", "close"}
                if not required.issubset(set(df.columns)):
                    logger.warning(
                        "Skipping %s: missing required columns %s",
                        csv_file.name,
                        required - set(df.columns),
                    )
                    continue

                if "volume" not in df.columns:
                    df["volume"] = 0

                df = df.sort_index()
                data[symbol] = df

                logger.info("Loaded %d bars for %s from %s", len(df), symbol, csv_file.name)

            except Exception as e:
                logger.warning("Failed to load %s: %s", csv_file.name, e)

        logger.info("Loaded %d symbols from %s", len(data), csv_dir)
        return data
