"""
Schwab API Client (for thinkorswim)
=====================================

OAuth2-authenticated REST client for the Charles Schwab Trader API.
Replaces the old TDA API. Provides market data, order placement,
account info, and position management.

API Docs: https://developer.schwab.com

Setup:
    1. Register at https://developer.schwab.com
    2. Create an app to get client_id and client_secret
    3. Complete OAuth2 flow to get initial refresh_token
    4. Pass credentials to SchwabClient

Usage:
    client = SchwabClient({
        "client_id": "your_app_key",
        "client_secret": "your_secret",
        "refresh_token": "your_refresh_token",
        "account_id": "your_account_hash",
    })
    quote = client.get_quote("AAPL")
    order_id = client.place_market_order("AAPL", "BUY", 100)
    positions = client.get_positions()
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class SchwabAPIError(Exception):
    """Custom exception for Schwab API errors."""

    def __init__(self, message: str, status_code: int = 0, body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class SchwabClient:
    """Charles Schwab Trader API client with OAuth2.

    Handles token management, rate limiting, and provides methods
    for market data, order placement, account queries, and positions.

    Args:
        config: Dict with:
            - client_id: OAuth2 app key
            - client_secret: OAuth2 app secret
            - refresh_token: OAuth2 refresh token
            - account_id: Account hash (encrypted account number)
            - redirect_uri: OAuth2 redirect URI (default: https://127.0.0.1)
    """

    AUTH_URL = "https://api.schwabapi.com/v1/oauth/token"
    BASE_URL = "https://api.schwabapi.com"
    TRADER_URL = "https://api.schwabapi.com/trader/v1"
    MARKETDATA_URL = "https://api.schwabapi.com/marketdata/v1"

    MAX_REQUESTS_PER_SECOND = 2

    def __init__(self, config: Dict[str, str]) -> None:
        self._client_id = config["client_id"]
        self._client_secret = config["client_secret"]
        self._refresh_token = config["refresh_token"]
        self._account_id = config.get("account_id", "")
        self._redirect_uri = config.get("redirect_uri", "https://127.0.0.1")

        self._access_token: Optional[str] = None
        self._token_expiry: float = 0.0
        self._request_times: deque = deque()
        self._session = requests.Session()

        self._authenticate()
        logger.info("SchwabClient initialized (account=%s)", self._account_id[:8] + "..." if self._account_id else "?")

    # ─── OAuth2 Authentication ───

    def _authenticate(self) -> None:
        """Exchange refresh_token for access_token."""
        import base64
        credentials = base64.b64encode(
            f"{self._client_id}:{self._client_secret}".encode()
        ).decode()

        headers = {
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {
            "grant_type": "refresh_token",
            "refresh_token": self._refresh_token,
        }

        resp = requests.post(self.AUTH_URL, headers=headers, data=data, timeout=30)
        if resp.status_code != 200:
            raise SchwabAPIError(
                f"Authentication failed: {resp.text}",
                status_code=resp.status_code, body=resp.text,
            )

        token_data = resp.json()
        self._access_token = token_data["access_token"]
        self._token_expiry = time.time() + token_data.get("expires_in", 1800) - 60

        if "refresh_token" in token_data:
            self._refresh_token = token_data["refresh_token"]

        logger.info("Schwab authentication successful")

    def _ensure_token(self) -> None:
        """Refresh token if expired."""
        if time.time() >= self._token_expiry:
            logger.info("Schwab token expired, refreshing...")
            self._authenticate()

    def _headers(self) -> Dict[str, str]:
        """Get authorization headers."""
        self._ensure_token()
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    # ─── Rate Limiting ───

    def _rate_limit(self) -> None:
        """Enforce rate limit (2 requests/second)."""
        now = time.time()
        while self._request_times and self._request_times[0] < now - 1:
            self._request_times.popleft()

        if len(self._request_times) >= self.MAX_REQUESTS_PER_SECOND:
            sleep_for = 1.0 - (now - self._request_times[0])
            if sleep_for > 0:
                time.sleep(sleep_for)

        self._request_times.append(time.time())

    # ─── HTTP ───

    def _request(self, method: str, url: str, json_body: Any = None, params: Any = None) -> Any:
        """Make an authenticated API request."""
        self._rate_limit()
        headers = self._headers()

        try:
            resp = self._session.request(
                method, url, headers=headers, json=json_body, params=params, timeout=30,
            )
        except requests.RequestException as e:
            raise SchwabAPIError(f"Network error: {e}")

        if resp.status_code == 401:
            self._authenticate()
            headers = self._headers()
            resp = self._session.request(
                method, url, headers=headers, json=json_body, params=params, timeout=30,
            )

        if resp.status_code not in (200, 201):
            raise SchwabAPIError(
                f"API error {resp.status_code}: {resp.text}",
                status_code=resp.status_code, body=resp.text,
            )

        if resp.text:
            return resp.json()
        return {}

    # ─── Market Data ───

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get a real-time quote for a symbol.

        Returns:
            Dict with: lastPrice, bidPrice, askPrice, totalVolume, etc.
        """
        resp = self._request("GET", f"{self.MARKETDATA_URL}/{symbol}/quotes")
        if symbol in resp:
            quote = resp[symbol].get("quote", resp[symbol])
            return {
                "symbol": symbol,
                "lastPrice": quote.get("lastPrice", 0),
                "bidPrice": quote.get("bidPrice", 0),
                "askPrice": quote.get("askPrice", 0),
                "openPrice": quote.get("openPrice", 0),
                "highPrice": quote.get("highPrice", 0),
                "lowPrice": quote.get("lowPrice", 0),
                "totalVolume": quote.get("totalVolume", 0),
                "netChange": quote.get("netChange", 0),
                "netPercentChange": quote.get("netPercentChangeInDouble", 0),
            }
        return {"symbol": symbol, "lastPrice": 0}

    def get_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get quotes for multiple symbols."""
        symbol_str = ",".join(symbols)
        resp = self._request("GET", f"{self.MARKETDATA_URL}/quotes", params={"symbols": symbol_str})
        results = {}
        for sym in symbols:
            if sym in resp:
                quote = resp[sym].get("quote", resp[sym])
                results[sym] = {
                    "lastPrice": quote.get("lastPrice", 0),
                    "bidPrice": quote.get("bidPrice", 0),
                    "askPrice": quote.get("askPrice", 0),
                    "totalVolume": quote.get("totalVolume", 0),
                }
        return results

    def get_price_history(
        self,
        symbol: str,
        period_type: str = "month",
        period: int = 6,
        frequency_type: str = "daily",
        frequency: int = 1,
    ) -> List[Dict[str, Any]]:
        """Get historical price data (OHLCV candles).

        Args:
            symbol: Ticker symbol.
            period_type: "day", "month", "year", "ytd"
            period: Number of periods.
            frequency_type: "minute", "daily", "weekly", "monthly"
            frequency: Frequency interval.

        Returns:
            List of candle dicts with: open, high, low, close, volume, datetime.
        """
        params = {
            "periodType": period_type,
            "period": period,
            "frequencyType": frequency_type,
            "frequency": frequency,
        }
        resp = self._request("GET", f"{self.MARKETDATA_URL}/pricehistory", params={"symbol": symbol, **params})
        candles = resp.get("candles", [])
        return candles

    # ─── Orders ───

    def place_market_order(self, symbol: str, action: str, quantity: int) -> str:
        """Place a market order.

        Args:
            symbol: Ticker symbol.
            action: "BUY" or "SELL".
            quantity: Number of shares.

        Returns:
            Order ID.
        """
        instruction = action.upper()
        if instruction == "SELL":
            instruction = "SELL"
        elif instruction == "BUY":
            instruction = "BUY"

        order_body = {
            "orderType": "MARKET",
            "session": "NORMAL",
            "duration": "DAY",
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": instruction,
                    "quantity": quantity,
                    "instrument": {
                        "symbol": symbol,
                        "assetType": "EQUITY",
                    },
                }
            ],
        }

        logger.info("Schwab: placing market %s order for %d %s", action, quantity, symbol)
        resp = self._request(
            "POST",
            f"{self.TRADER_URL}/accounts/{self._account_id}/orders",
            json_body=order_body,
        )
        order_id = resp.get("orderId", str(id(order_body)))
        logger.info("Schwab market order placed: %s", order_id)
        return str(order_id)

    def place_limit_order(
        self, symbol: str, action: str, quantity: int, price: float,
    ) -> str:
        """Place a limit order.

        Returns:
            Order ID.
        """
        order_body = {
            "orderType": "LIMIT",
            "session": "NORMAL",
            "price": str(price),
            "duration": "DAY",
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": action.upper(),
                    "quantity": quantity,
                    "instrument": {
                        "symbol": symbol,
                        "assetType": "EQUITY",
                    },
                }
            ],
        }

        logger.info("Schwab: placing limit %s %d %s @ $%.2f", action, quantity, symbol, price)
        resp = self._request(
            "POST",
            f"{self.TRADER_URL}/accounts/{self._account_id}/orders",
            json_body=order_body,
        )
        order_id = resp.get("orderId", str(id(order_body)))
        logger.info("Schwab limit order placed: %s", order_id)
        return str(order_id)

    def cancel_order(self, order_id: str) -> None:
        """Cancel an order."""
        logger.info("Schwab: cancelling order %s", order_id)
        self._request(
            "DELETE",
            f"{self.TRADER_URL}/accounts/{self._account_id}/orders/{order_id}",
        )

    def get_orders(self, status: str = "WORKING") -> List[Dict[str, Any]]:
        """Get orders for the account.

        Args:
            status: Filter by status — "WORKING", "FILLED", "CANCELED", etc.

        Returns:
            List of order dicts.
        """
        resp = self._request(
            "GET",
            f"{self.TRADER_URL}/accounts/{self._account_id}/orders",
            params={"status": status},
        )
        return resp if isinstance(resp, list) else resp.get("orders", [])

    # ─── Account ───

    def get_account_info(self) -> Dict[str, Any]:
        """Get account balances and summary."""
        resp = self._request(
            "GET",
            f"{self.TRADER_URL}/accounts/{self._account_id}",
            params={"fields": "positions"},
        )
        account = resp.get("securitiesAccount", resp)
        balances = account.get("currentBalances", {})
        return {
            "account_id": self._account_id,
            "account_type": account.get("type", ""),
            "net_liquidation": balances.get("liquidationValue", 0),
            "cash_balance": balances.get("cashBalance", 0),
            "buying_power": balances.get("buyingPower", 0),
            "equity": balances.get("equity", 0),
        }

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions.

        Returns:
            List of position dicts with: symbol, quantity, avgPrice, marketValue, pnl.
        """
        resp = self._request(
            "GET",
            f"{self.TRADER_URL}/accounts/{self._account_id}",
            params={"fields": "positions"},
        )
        account = resp.get("securitiesAccount", resp)
        positions = account.get("positions", [])

        return [
            {
                "symbol": p.get("instrument", {}).get("symbol", ""),
                "quantity": p.get("longQuantity", 0) - p.get("shortQuantity", 0),
                "avg_price": p.get("averagePrice", 0),
                "market_value": p.get("marketValue", 0),
                "current_price": p.get("currentDayProfitLossPercentage", 0),
                "pnl": p.get("currentDayProfitLoss", 0),
            }
            for p in positions
        ]

    def __repr__(self) -> str:
        return f"SchwabClient(account={self._account_id[:8]}...)"
