"""
Unified Sector Map — Single Source of Truth
=============================================

Maps US stock ticker symbols to GICS sectors.
Used by webhook_server.py, risk_manager_unified.py, and any module
that needs symbol→sector classification.

Import:
    from shared.config.sector_map import SECTOR_MAP, get_sector
"""

from __future__ import annotations

from typing import Dict

# 200+ common US stock tickers mapped to GICS sectors
SECTOR_MAP: Dict[str, str] = {
    # ── Technology ──
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "GOOG": "Technology", "META": "Technology", "NVDA": "Technology",
    "TSM": "Technology", "AVGO": "Technology", "ORCL": "Technology",
    "CRM": "Technology", "ADBE": "Technology", "AMD": "Technology",
    "INTC": "Technology", "CSCO": "Technology", "QCOM": "Technology",
    "IBM": "Technology", "TXN": "Technology", "AMAT": "Technology",
    "NOW": "Technology", "INTU": "Technology", "MU": "Technology",
    "LRCX": "Technology", "KLAC": "Technology", "SNPS": "Technology",
    "CDNS": "Technology", "MRVL": "Technology", "ADSK": "Technology",
    "PANW": "Technology", "CRWD": "Technology", "FTNT": "Technology",
    "WDAY": "Technology", "TEAM": "Technology", "ZS": "Technology",
    "DDOG": "Technology", "NET": "Technology", "SNOW": "Technology",
    "PLTR": "Technology", "SHOP": "Technology", "SQ": "Technology",
    "MELI": "Technology", "UBER": "Technology", "DASH": "Technology",
    "COIN": "Technology", "RBLX": "Technology", "U": "Technology",
    "HUBS": "Technology", "DOCU": "Technology", "OKTA": "Technology",
    "TWLO": "Technology", "ZM": "Technology", "TTD": "Technology",
    "BILL": "Technology", "MDB": "Technology", "ESTC": "Technology",
    "VEEV": "Technology", "ANSS": "Technology", "CPRT": "Technology",

    # ── Financials ──
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "C": "Financials",
    "BLK": "Financials", "SCHW": "Financials", "AXP": "Financials",
    "V": "Financials", "MA": "Financials", "PYPL": "Financials",
    "USB": "Financials", "PNC": "Financials", "TFC": "Financials",
    "BK": "Financials", "STT": "Financials", "COF": "Financials",
    "AIG": "Financials", "MET": "Financials", "PRU": "Financials",
    "AFL": "Financials", "ALL": "Financials", "TRV": "Financials",
    "CME": "Financials", "ICE": "Financials", "SPGI": "Financials",
    "MCO": "Financials", "MSCI": "Financials", "FIS": "Financials",
    "FISV": "Financials", "GPN": "Financials", "SYF": "Financials",

    # ── Healthcare ──
    "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare",
    "ABBV": "Healthcare", "MRK": "Healthcare", "LLY": "Healthcare",
    "TMO": "Healthcare", "ABT": "Healthcare", "DHR": "Healthcare",
    "BMY": "Healthcare", "AMGN": "Healthcare", "GILD": "Healthcare",
    "ISRG": "Healthcare", "MDT": "Healthcare", "SYK": "Healthcare",
    "BSX": "Healthcare", "VRTX": "Healthcare", "REGN": "Healthcare",
    "ZTS": "Healthcare", "CI": "Healthcare", "ELV": "Healthcare",
    "HCA": "Healthcare", "DXCM": "Healthcare", "IQV": "Healthcare",
    "IDXX": "Healthcare", "MRNA": "Healthcare", "BIIB": "Healthcare",
    "A": "Healthcare", "BDX": "Healthcare", "EW": "Healthcare",
    "GEHC": "Healthcare", "HUM": "Healthcare", "CNC": "Healthcare",

    # ── Consumer Discretionary ──
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "NKE": "Consumer Discretionary",
    "MCD": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "LOW": "Consumer Discretionary", "TJX": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary", "ABNB": "Consumer Discretionary",
    "MAR": "Consumer Discretionary", "HLT": "Consumer Discretionary",
    "GM": "Consumer Discretionary", "F": "Consumer Discretionary",
    "ROST": "Consumer Discretionary", "DHI": "Consumer Discretionary",
    "LEN": "Consumer Discretionary", "ORLY": "Consumer Discretionary",
    "AZO": "Consumer Discretionary", "CMG": "Consumer Discretionary",
    "YUM": "Consumer Discretionary", "DPZ": "Consumer Discretionary",
    "LULU": "Consumer Discretionary", "DECK": "Consumer Discretionary",
    "RCL": "Consumer Discretionary", "CCL": "Consumer Discretionary",
    "EBAY": "Consumer Discretionary", "ETSY": "Consumer Discretionary",

    # ── Consumer Staples ──
    "PG": "Consumer Staples", "KO": "Consumer Staples", "PEP": "Consumer Staples",
    "COST": "Consumer Staples", "WMT": "Consumer Staples", "PM": "Consumer Staples",
    "MO": "Consumer Staples", "CL": "Consumer Staples", "MDLZ": "Consumer Staples",
    "KHC": "Consumer Staples", "GIS": "Consumer Staples", "K": "Consumer Staples",
    "SJM": "Consumer Staples", "HSY": "Consumer Staples", "STZ": "Consumer Staples",
    "KDP": "Consumer Staples", "MNST": "Consumer Staples", "EL": "Consumer Staples",
    "TGT": "Consumer Staples", "DG": "Consumer Staples", "DLTR": "Consumer Staples",
    "KR": "Consumer Staples", "SYY": "Consumer Staples", "ADM": "Consumer Staples",

    # ── Energy ──
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "SLB": "Energy", "EOG": "Energy", "OXY": "Energy",
    "MPC": "Energy", "VLO": "Energy", "PSX": "Energy",
    "PXD": "Energy", "DVN": "Energy", "HES": "Energy",
    "FANG": "Energy", "HAL": "Energy", "BKR": "Energy",
    "WMB": "Energy", "KMI": "Energy", "OKE": "Energy",

    # ── Industrials ──
    "BA": "Industrials", "CAT": "Industrials", "GE": "Industrials",
    "HON": "Industrials", "UPS": "Industrials", "RTX": "Industrials",
    "LMT": "Industrials", "NOC": "Industrials", "GD": "Industrials",
    "DE": "Industrials", "MMM": "Industrials", "ITW": "Industrials",
    "EMR": "Industrials", "FDX": "Industrials", "WM": "Industrials",
    "RSG": "Industrials", "CSX": "Industrials", "UNP": "Industrials",
    "NSC": "Industrials", "TDG": "Industrials", "CTAS": "Industrials",
    "PCAR": "Industrials", "FAST": "Industrials", "VRSK": "Industrials",

    # ── Utilities ──
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    "D": "Utilities", "SRE": "Utilities", "AEP": "Utilities",
    "EXC": "Utilities", "XEL": "Utilities", "ED": "Utilities",
    "WEC": "Utilities", "ES": "Utilities", "AWK": "Utilities",

    # ── Communication Services ──
    "DIS": "Communication Services", "NFLX": "Communication Services",
    "CMCSA": "Communication Services", "T": "Communication Services",
    "VZ": "Communication Services", "TMUS": "Communication Services",
    "CHTR": "Communication Services", "EA": "Communication Services",
    "TTWO": "Communication Services", "MTCH": "Communication Services",
    "WBD": "Communication Services", "PARA": "Communication Services",
    "LYV": "Communication Services", "SNAP": "Communication Services",
    "PINS": "Communication Services", "ROKU": "Communication Services",

    # ── Materials ──
    "LIN": "Materials", "APD": "Materials", "SHW": "Materials",
    "ECL": "Materials", "NEM": "Materials", "FCX": "Materials",
    "NUE": "Materials", "STLD": "Materials", "VMC": "Materials",
    "MLM": "Materials", "DOW": "Materials", "DD": "Materials",
    "PPG": "Materials", "ALB": "Materials", "CF": "Materials",

    # ── Real Estate ──
    "AMT": "Real Estate", "PLD": "Real Estate", "CCI": "Real Estate",
    "EQIX": "Real Estate", "PSA": "Real Estate", "SPG": "Real Estate",
    "O": "Real Estate", "WELL": "Real Estate", "DLR": "Real Estate",
    "AVB": "Real Estate", "EQR": "Real Estate", "VICI": "Real Estate",
    "IRM": "Real Estate", "SBAC": "Real Estate", "ARE": "Real Estate",

    # ── ETFs ──
    "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF", "DIA": "ETF",
    "VOO": "ETF", "VTI": "ETF", "XLF": "ETF", "XLK": "ETF",
    "XLE": "ETF", "XLV": "ETF", "XLI": "ETF", "XLY": "ETF",
    "XLP": "ETF", "XLU": "ETF", "XLB": "ETF", "XLRE": "ETF",
    "XLC": "ETF", "ARKK": "ETF", "SOXL": "ETF", "TQQQ": "ETF",
    "SQQQ": "ETF", "GLD": "ETF", "SLV": "ETF", "USO": "ETF",
    "TLT": "ETF", "HYG": "ETF", "LQD": "ETF", "EEM": "ETF",
    "EFA": "ETF", "VWO": "ETF", "IEMG": "ETF", "VEA": "ETF",
}


def get_sector(symbol: str) -> str:
    """Return the sector for a given ticker symbol.

    Args:
        symbol: Ticker symbol (case-insensitive).

    Returns:
        Sector name string, or "Unknown" if not mapped.
    """
    return SECTOR_MAP.get(symbol.upper(), "Unknown")
