"""TradeStation Order Router — OAuth2-authenticated order management via the v3 REST API."""

import time
import logging
from collections import deque
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class TradeStationAPIError(Exception):
    """Custom exception for TradeStation API errors."""

    def __init__(self, message: str, status_code: int = 0, response_body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class TradeStationOrderRouter:
    """Handles order placement, cancellation, and status queries against TradeStation v3 API.

    Includes OAuth2 token management with automatic refresh and rate limiting
    (max 120 requests/minute).
    """

    BASE_URL = "https://api.tradestation.com/v3"
    TOKEN_URL = "https://signin.tradestation.com/oauth/token"
    MAX_REQUESTS_PER_MINUTE = 120

    def __init__(self, config: dict):
        """Initialize the order router.

        Args:
            config: Dictionary containing:
                - client_id (str): OAuth2 client ID
                - client_secret (str): OAuth2 client secret
                - redirect_uri (str): OAuth2 redirect URI
                - refresh_token (str): OAuth2 refresh token for token exchange
        """
        self.client_id = config["client_id"]
        self.client_secret = config["client_secret"]
        self.redirect_uri = config["redirect_uri"]
        self.refresh_token = config["refresh_token"]

        self.access_token: Optional[str] = None
        self.token_expiry: float = 0.0
        self._request_timestamps: deque = deque()
        self._session = requests.Session()

        self._authenticate()

    # ── OAuth2 ──────────────────────────────────────────────────────────

    def _authenticate(self):
        """Exchange the refresh token for an access token."""
        payload = {
            "grant_type": "refresh_token",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": self.refresh_token,
            "redirect_uri": self.redirect_uri,
        }
        resp = requests.post(self.TOKEN_URL, data=payload, timeout=30)
        if resp.status_code != 200:
            raise TradeStationAPIError(
                f"Authentication failed: {resp.text}",
                status_code=resp.status_code,
                response_body=resp.text,
            )

        data = resp.json()
        self.access_token = data["access_token"]
        self.token_expiry = time.time() + data.get("expires_in", 1200) - 60
        if "refresh_token" in data:
            self.refresh_token = data["refresh_token"]
        logger.info("TradeStation authentication successful")

    def _refresh_access_token(self):
        """Refresh the access token if it is expired or about to expire."""
        if time.time() >= self.token_expiry:
            logger.info("Access token expired, refreshing...")
            self._authenticate()

    def _get_headers(self) -> dict:
        """Return authorization headers, refreshing the token if needed."""
        self._refresh_access_token()
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

    # ── Rate Limiting ───────────────────────────────────────────────────

    def _enforce_rate_limit(self):
        """Block until a request slot is available (120 req/min sliding window)."""
        now = time.time()
        while self._request_timestamps and self._request_timestamps[0] < now - 60:
            self._request_timestamps.popleft()

        if len(self._request_timestamps) >= self.MAX_REQUESTS_PER_MINUTE:
            sleep_for = 60 - (now - self._request_timestamps[0])
            if sleep_for > 0:
                logger.warning("Rate limit reached, sleeping %.2fs", sleep_for)
                time.sleep(sleep_for)

        self._request_timestamps.append(time.time())

    # ── HTTP helpers ────────────────────────────────────────────────────

    def _request(self, method: str, path: str, json_body: dict = None) -> dict:
        """Execute an API request with rate limiting and error handling."""
        self._enforce_rate_limit()
        url = f"{self.BASE_URL}{path}"
        headers = self._get_headers()

        try:
            resp = self._session.request(
                method, url, headers=headers, json=json_body, timeout=30
            )
        except requests.RequestException as exc:
            raise TradeStationAPIError(f"Network error: {exc}") from exc

        if resp.status_code == 401:
            logger.info("Received 401, attempting token refresh and retry")
            self._authenticate()
            headers = self._get_headers()
            resp = self._session.request(
                method, url, headers=headers, json=json_body, timeout=30
            )

        if resp.status_code not in (200, 201):
            raise TradeStationAPIError(
                f"API error {resp.status_code}: {resp.text}",
                status_code=resp.status_code,
                response_body=resp.text,
            )

        if resp.text:
            return resp.json()
        return {}

    # ── Order Methods ───────────────────────────────────────────────────

    def place_market_order(
        self, account_id: str, symbol: str, action: str, quantity: int
    ) -> str:
        """Place a market order.

        Args:
            account_id: TradeStation account ID.
            symbol: Ticker symbol (e.g. "AAPL").
            action: "BUY" or "SELL".
            quantity: Number of shares.

        Returns:
            The order ID assigned by TradeStation.
        """
        body = {
            "AccountID": account_id,
            "Symbol": symbol,
            "Quantity": str(quantity),
            "OrderType": "Market",
            "TradeAction": action,
            "TimeInForce": {"Duration": "DAY"},
            "Route": "Intelligent",
        }
        logger.info("Placing market %s order: %s x%d", action, symbol, quantity)
        result = self._request("POST", "/orderexecution/orders", json_body=body)
        order_id = result["Orders"][0]["OrderID"]
        logger.info("Market order placed — OrderID: %s", order_id)
        return order_id

    def place_limit_order(
        self,
        account_id: str,
        symbol: str,
        action: str,
        quantity: int,
        limit_price: float,
    ) -> str:
        """Place a limit order.

        Returns:
            The order ID assigned by TradeStation.
        """
        body = {
            "AccountID": account_id,
            "Symbol": symbol,
            "Quantity": str(quantity),
            "OrderType": "Limit",
            "LimitPrice": str(limit_price),
            "TradeAction": action,
            "TimeInForce": {"Duration": "DAY"},
            "Route": "Intelligent",
        }
        logger.info(
            "Placing limit %s order: %s x%d @ %.2f",
            action,
            symbol,
            quantity,
            limit_price,
        )
        result = self._request("POST", "/orderexecution/orders", json_body=body)
        order_id = result["Orders"][0]["OrderID"]
        logger.info("Limit order placed — OrderID: %s", order_id)
        return order_id

    def place_bracket_order(
        self,
        account_id: str,
        symbol: str,
        action: str,
        quantity: int,
        limit_price: float,
        profit_target: float,
        stop_loss: float,
    ) -> dict:
        """Place a bracket (OCO) order with profit target and stop loss.

        Returns:
            Dict with OrderIDs for the entry, target, and stop orders.
        """
        body = {
            "AccountID": account_id,
            "Symbol": symbol,
            "Quantity": str(quantity),
            "OrderType": "Limit",
            "LimitPrice": str(limit_price),
            "TradeAction": action,
            "TimeInForce": {"Duration": "GTC"},
            "Route": "Intelligent",
            "OSOs": [
                {
                    "Type": "BRK",
                    "Orders": [
                        {
                            "AccountID": account_id,
                            "Symbol": symbol,
                            "Quantity": str(quantity),
                            "OrderType": "Limit",
                            "LimitPrice": str(profit_target),
                            "TradeAction": "SELL" if action == "BUY" else "BUYTOCOVER",
                            "TimeInForce": {"Duration": "GTC"},
                            "Route": "Intelligent",
                        },
                        {
                            "AccountID": account_id,
                            "Symbol": symbol,
                            "Quantity": str(quantity),
                            "OrderType": "StopMarket",
                            "StopPrice": str(stop_loss),
                            "TradeAction": "SELL" if action == "BUY" else "BUYTOCOVER",
                            "TimeInForce": {"Duration": "GTC"},
                            "Route": "Intelligent",
                        },
                    ],
                }
            ],
        }
        logger.info(
            "Placing bracket %s order: %s x%d @ %.2f  TP=%.2f  SL=%.2f",
            action,
            symbol,
            quantity,
            limit_price,
            profit_target,
            stop_loss,
        )
        result = self._request("POST", "/orderexecution/orders", json_body=body)
        orders = result.get("Orders", [])
        return {
            "entry_order_id": orders[0]["OrderID"] if len(orders) > 0 else None,
            "bracket_orders": [o["OrderID"] for o in orders[1:]] if len(orders) > 1 else [],
        }

    def cancel_order(self, account_id: str, order_id: str):
        """Cancel an open order.

        Args:
            account_id: TradeStation account ID.
            order_id: The order to cancel.
        """
        logger.info("Cancelling order %s on account %s", order_id, account_id)
        self._request("DELETE", f"/orderexecution/orders/{order_id}")
        logger.info("Order %s cancelled", order_id)

    def get_order_status(self, account_id: str, order_id: str) -> dict:
        """Get the status of a specific order.

        Returns:
            Dict with order status details.
        """
        result = self._request(
            "GET", f"/brokerage/accounts/{account_id}/orders/{order_id}"
        )
        return result

    def get_orders(self, account_id: str) -> list:
        """Get all orders for an account.

        Returns:
            List of order dicts.
        """
        result = self._request(
            "GET", f"/brokerage/accounts/{account_id}/orders"
        )
        return result.get("Orders", [])
