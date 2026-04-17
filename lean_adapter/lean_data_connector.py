"""
LEAN Data Connector
=====================

Bridges market data between stocks_plugin and LEAN CSV format.

Usage:
    from lean_adapter.lean_data_connector import LEANDataBridge
    bridge = LEANDataBridge()
    df = bridge.lean_csv_to_dataframe("path/to/lean/data.csv")
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LEANDataBridge:
    """Convert between LEAN CSV format and pandas DataFrames."""

    @staticmethod
    def lean_csv_to_dataframe(path: str) -> pd.DataFrame:
        """Read a LEAN-format CSV into a standard OHLCV DataFrame.

        LEAN daily format: Date(yyyyMMdd HH:mm), Open*10000, High*10000,
        Low*10000, Close*10000, Volume
        """
        df = pd.read_csv(path, header=None,
                        names=["date", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(), format="%Y%m%d 00:00")
        df = df.set_index("date")
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col] / 10000.0
        return df

    @staticmethod
    def dataframe_to_lean_csv(df: pd.DataFrame, output_path: str) -> str:
        """Write a DataFrame to LEAN daily CSV format."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        lean_df = pd.DataFrame()
        if isinstance(df.index, pd.DatetimeIndex):
            lean_df["date"] = df.index.strftime("%Y%m%d 00:00")
        else:
            lean_df["date"] = df.index
        lean_df["open"] = (df["open"] * 10000).astype(int)
        lean_df["high"] = (df["high"] * 10000).astype(int)
        lean_df["low"] = (df["low"] * 10000).astype(int)
        lean_df["close"] = (df["close"] * 10000).astype(int)
        lean_df["volume"] = df["volume"].astype(int)
        lean_df.to_csv(output_path, index=False, header=False)
        logger.info("Written LEAN CSV: %s (%d bars)", output_path, len(lean_df))
        return output_path

    @staticmethod
    def from_market_data_cache(cache, symbol: str, output_dir: str, bar_size: str = "1 day") -> str:
        """Export MarketDataCache data to LEAN CSV format."""
        df = cache.get_bars(symbol, bar_size)
        if df is None or df.empty:
            raise ValueError(f"No cached data for {symbol}")
        output_path = os.path.join(output_dir, f"{symbol.lower()}.csv")
        return LEANDataBridge.dataframe_to_lean_csv(df, output_path)
