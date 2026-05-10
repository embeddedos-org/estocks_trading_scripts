"""Comprehensive tests for Schwab API Client (thinkorswim)."""

import sys
import os
import json
import time
import base64
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from thinkorswim.api.schwab_client import SchwabClient, SchwabAPIError


# ── Helpers ──────────────────────────────────────────────────────────────────

VALID_CONFIG = {
    "client_id": "test_app_key",
    "client_secret": "test_app_secret",
    "refresh_token": "test_refresh_tok",
    "account_id": "ENCRYPTED_ACCOUNT_HASH_1234",
    "redirect_uri": "https://127.0.0.1",
}

AUTH_RESPONSE = {
    "access_token": "schwab_tok_abc",
    "expires_in": 1800,
    "refresh_token": "schwab_refresh_new",
}


def _make_mock_session(auth_json=None, request_json=None, request_status=200):
    if auth_json is None:
        auth_json = AUTH_RESPONSE

    mock_session = MagicMock()

    auth_resp = MagicMock()
    auth_resp.status_code = 200
    auth_resp.json.return_value = auth_json
    auth_resp.text = json.dumps(auth_json)
    mock_session.post.return_value = auth_resp

    api_resp = MagicMock()
    api_resp.status_code = request_status
    api_resp.json.return_value = request_json or {}
    api_resp.text = json.dumps(request_json) if request_json else ""
    api_resp.headers = {"Location": ""}
    mock_session.request.return_value = api_resp

    return mock_session


def _build_client(mock_session=None):
    if mock_session is None:
        mock_session = _make_mock_session()
    with patch("thinkorswim.api.schwab_client.requests.Session", return_value=mock_session):
        client = SchwabClient(VALID_CONFIG)
    return client, mock_session


# ══════════════════════════════════════════════════════════════════════════════
#  SchwabClient — Initialization
# ══════════════════════════════════════════════════════════════════════════════


class TestSchwabInit:
    def test_init_stores_config(self):
        client, _ = _build_client()
        assert client._client_id == "test_app_key"
        assert client._client_secret == "test_app_secret"
        assert client._account_id == "ENCRYPTED_ACCOUNT_HASH_1234"
        assert client._redirect_uri == "https://127.0.0.1"

    def test_init_calls_authenticate(self):
        mock_session = _make_mock_session()
        with patch("thinkorswim.api.schwab_client.requests.Session", return_value=mock_session):
            client = SchwabClient(VALID_CONFIG)
        assert client._access_token == "schwab_tok_abc"
        mock_session.post.assert_called_once()

    def test_init_default_redirect_uri(self):
        config = {k: v for k, v in VALID_CONFIG.items() if k != "redirect_uri"}
        mock_session = _make_mock_session()
        with patch("thinkorswim.api.schwab_client.requests.Session", return_value=mock_session):
            client = SchwabClient(config)
        assert client._redirect_uri == "https://127.0.0.1"

    def test_init_missing_key_raises(self):
        bad_config = {"client_id": "x"}
        with pytest.raises(KeyError):
            with patch("thinkorswim.api.schwab_client.requests.Session", return_value=MagicMock()):
                SchwabClient(bad_config)

    def test_repr(self):
        client, _ = _build_client()
        r = repr(client)
        assert "SchwabClient" in r
        assert "ENCRYPTE" in r


# ══════════════════════════════════════════════════════════════════════════════
#  SchwabClient — _authenticate
# ══════════════════════════════════════════════════════════════════════════════


class TestSchwabAuthenticate:
    def test_authenticate_uses_basic_auth_header(self):
        mock_session = _make_mock_session()
        with patch("thinkorswim.api.schwab_client.requests.Session", return_value=mock_session):
            SchwabClient(VALID_CONFIG)
        call_kwargs = mock_session.post.call_args
        headers = call_kwargs.kwargs.get("headers", {})
        assert "Basic" in headers.get("Authorization", "")

    def test_authenticate_sends_correct_credentials(self):
        mock_session = _make_mock_session()
        with patch("thinkorswim.api.schwab_client.requests.Session", return_value=mock_session):
            SchwabClient(VALID_CONFIG)
        call_kwargs = mock_session.post.call_args
        headers = call_kwargs.kwargs.get("headers", {})
        expected_creds = base64.b64encode(b"test_app_key:test_app_secret").decode()
        assert expected_creds in headers.get("Authorization", "")

    def test_authenticate_sets_token(self):
        client, _ = _build_client()
        assert client._access_token == "schwab_tok_abc"

    def test_authenticate_sets_expiry(self):
        client, _ = _build_client()
        assert client._token_expiry > time.time()

    def test_authenticate_updates_refresh_token(self):
        client, _ = _build_client()
        assert client._refresh_token == "schwab_refresh_new"

    def test_authenticate_no_refresh_in_response(self):
        auth_resp = {"access_token": "tok_only", "expires_in": 600}
        mock_session = _make_mock_session(auth_json=auth_resp)
        with patch("thinkorswim.api.schwab_client.requests.Session", return_value=mock_session):
            client = SchwabClient(VALID_CONFIG)
        assert client._refresh_token == "test_refresh_tok"

    def test_authenticate_failure_raises(self):
        """Verify fix: token error handling raises SchwabAPIError."""
        mock_session = MagicMock()
        fail_resp = MagicMock()
        fail_resp.status_code = 401
        fail_resp.text = "Unauthorized"
        mock_session.post.return_value = fail_resp
        with patch("thinkorswim.api.schwab_client.requests.Session", return_value=mock_session):
            with pytest.raises(SchwabAPIError, match="Authentication failed"):
                SchwabClient(VALID_CONFIG)

    def test_authenticate_no_access_token_raises(self):
        """Verify fix: missing access_token in response raises."""
        auth_resp = {"token_type": "Bearer"}
        mock_session = _make_mock_session(auth_json=auth_resp)
        with patch("thinkorswim.api.schwab_client.requests.Session", return_value=mock_session):
            with pytest.raises(SchwabAPIError, match="Auth failed"):
                SchwabClient(VALID_CONFIG)


# ══════════════════════════════════════════════════════════════════════════════
#  SchwabClient — _request
# ══════════════════════════════════════════════════════════════════════════════


class TestSchwabRequest:
    def test_request_success_200(self):
        client, mock_session = _build_client()
        api_resp = MagicMock()
        api_resp.status_code = 200
        api_resp.text = '{"ok": true}'
        api_resp.json.return_value = {"ok": True}
        mock_session.request.return_value = api_resp

        result = client._request("GET", "https://api.schwabapi.com/test")
        assert result == {"ok": True}

    def test_request_accepts_204_no_content(self):
        """Verify fix: 204 accepted as success."""
        client, mock_session = _build_client()
        api_resp = MagicMock()
        api_resp.status_code = 204
        api_resp.text = ""
        mock_session.request.return_value = api_resp

        result = client._request("DELETE", "https://api.schwabapi.com/test")
        assert result == {}

    def test_request_json_decode_error_returns_empty(self):
        """Verify fix: JSONDecodeError handling."""
        client, mock_session = _build_client()
        api_resp = MagicMock()
        api_resp.status_code = 200
        api_resp.text = "not-json-at-all"
        api_resp.json.side_effect = ValueError("No JSON")
        mock_session.request.return_value = api_resp

        result = client._request("GET", "https://api.schwabapi.com/bad-json")
        assert result == {}

    def test_request_network_error_with_exception_chaining(self):
        """Verify fix: exception chaining (from e)."""
        import requests as req
        client, mock_session = _build_client()
        original_exc = req.ConnectionError("connection refused")
        mock_session.request.side_effect = original_exc

        with pytest.raises(SchwabAPIError, match="Network error") as exc_info:
            client._request("GET", "https://api.schwabapi.com/fail")
        assert exc_info.value.__cause__ is original_exc

    def test_request_raises_on_500(self):
        client, mock_session = _build_client()
        api_resp = MagicMock()
        api_resp.status_code = 500
        api_resp.text = "Internal Server Error"
        mock_session.request.return_value = api_resp

        with pytest.raises(SchwabAPIError, match="API error 500"):
            client._request("GET", "https://api.schwabapi.com/fail")

    def test_request_401_triggers_reauthentication(self):
        client, mock_session = _build_client()

        first_resp = MagicMock()
        first_resp.status_code = 401

        second_resp = MagicMock()
        second_resp.status_code = 200
        second_resp.text = '{"retried": true}'
        second_resp.json.return_value = {"retried": True}

        mock_session.request.side_effect = [first_resp, second_resp]

        auth_resp = MagicMock()
        auth_resp.status_code = 200
        auth_resp.json.return_value = AUTH_RESPONSE
        auth_resp.text = json.dumps(AUTH_RESPONSE)
        mock_session.post.return_value = auth_resp

        result = client._request("GET", "https://api.schwabapi.com/needs-auth")
        assert result == {"retried": True}
        assert mock_session.request.call_count == 2


# ══════════════════════════════════════════════════════════════════════════════
#  SchwabClient — _submit_order
# ══════════════════════════════════════════════════════════════════════════════


class TestSubmitOrder:
    def test_submit_order_extracts_id_from_location_header(self):
        """Verify fix: order ID extracted from Location header."""
        client, mock_session = _build_client()
        order_resp = MagicMock()
        order_resp.status_code = 201
        order_resp.text = ""
        order_resp.headers = {
            "Location": "https://api.schwabapi.com/trader/v1/accounts/ACC/orders/98765"
        }
        mock_session.post.return_value = order_resp

        order_id = client._submit_order({"orderType": "MARKET"})
        assert order_id == "98765"

    def test_submit_order_fallback_uuid_when_no_location(self):
        client, mock_session = _build_client()
        order_resp = MagicMock()
        order_resp.status_code = 201
        order_resp.text = ""
        order_resp.headers = {"Location": ""}
        mock_session.post.return_value = order_resp

        order_id = client._submit_order({"orderType": "MARKET"})
        assert len(order_id) == 12

    def test_submit_order_network_error(self):
        """Verify fix: exception chaining on network error."""
        import requests as req
        client, mock_session = _build_client()
        original_exc = req.Timeout("timed out")
        mock_session.post.side_effect = original_exc

        with pytest.raises(SchwabAPIError, match="Network error") as exc_info:
            client._submit_order({"orderType": "LIMIT"})
        assert exc_info.value.__cause__ is original_exc

    def test_submit_order_401_retries(self):
        client, mock_session = _build_client()

        first_resp = MagicMock()
        first_resp.status_code = 401

        second_resp = MagicMock()
        second_resp.status_code = 201
        second_resp.text = ""
        second_resp.headers = {"Location": "/orders/RETRY123"}

        mock_session.post.side_effect = [
            mock_session.post.return_value,  # initial auth in __init__
            first_resp,
            mock_session.post.return_value,  # re-auth
            second_resp,
        ]

        mock_session.post.side_effect = None
        mock_session.post.return_value = second_resp

        auth_resp = MagicMock()
        auth_resp.status_code = 200
        auth_resp.json.return_value = AUTH_RESPONSE
        auth_resp.text = json.dumps(AUTH_RESPONSE)

        call_count = [0]
        def post_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return first_resp
            elif call_count[0] == 2:
                return auth_resp
            else:
                return second_resp

        mock_session.post.side_effect = post_side_effect

        order_id = client._submit_order({"orderType": "MARKET"})
        assert order_id == "RETRY123"

    def test_submit_order_failure_raises(self):
        client, mock_session = _build_client()
        fail_resp = MagicMock()
        fail_resp.status_code = 400
        fail_resp.text = "Bad order"
        mock_session.post.return_value = fail_resp

        with pytest.raises(SchwabAPIError, match="Order placement failed"):
            client._submit_order({"orderType": "MARKET"})


# ══════════════════════════════════════════════════════════════════════════════
#  SchwabClient — place_market_order / place_limit_order
# ══════════════════════════════════════════════════════════════════════════════


class TestSchwabMarketOrder:
    def test_place_market_order_returns_id(self):
        client, mock_session = _build_client()
        order_resp = MagicMock()
        order_resp.status_code = 201
        order_resp.text = ""
        order_resp.headers = {"Location": "/orders/MKT-001"}
        mock_session.post.return_value = order_resp

        oid = client.place_market_order("AAPL", "BUY", 100)
        assert oid == "MKT-001"

    def test_place_market_order_body_structure(self):
        client, mock_session = _build_client()
        order_resp = MagicMock()
        order_resp.status_code = 201
        order_resp.text = ""
        order_resp.headers = {"Location": "/orders/MKT-002"}
        mock_session.post.return_value = order_resp

        client.place_market_order("TSLA", "SELL", 50)
        call_kwargs = mock_session.post.call_args
        body = call_kwargs.kwargs.get("json", {})
        assert body["orderType"] == "MARKET"
        assert body["orderLegCollection"][0]["instruction"] == "SELL"
        assert body["orderLegCollection"][0]["quantity"] == 50
        assert body["orderLegCollection"][0]["instrument"]["symbol"] == "TSLA"


class TestSchwabLimitOrder:
    def test_place_limit_order_returns_id(self):
        client, mock_session = _build_client()
        order_resp = MagicMock()
        order_resp.status_code = 201
        order_resp.text = ""
        order_resp.headers = {"Location": "/orders/LMT-001"}
        mock_session.post.return_value = order_resp

        oid = client.place_limit_order("MSFT", "BUY", 25, 350.50)
        assert oid == "LMT-001"

    def test_place_limit_order_price_as_float(self):
        """Verify fix: price is float, not string."""
        client, mock_session = _build_client()
        order_resp = MagicMock()
        order_resp.status_code = 201
        order_resp.text = ""
        order_resp.headers = {"Location": "/orders/LMT-002"}
        mock_session.post.return_value = order_resp

        client.place_limit_order("GOOG", "BUY", 10, "175.25")
        call_kwargs = mock_session.post.call_args
        body = call_kwargs.kwargs.get("json", {})
        assert isinstance(body["price"], float)
        assert body["price"] == 175.25


# ══════════════════════════════════════════════════════════════════════════════
#  SchwabClient — cancel_order
# ══════════════════════════════════════════════════════════════════════════════


class TestSchwabCancelOrder:
    def test_cancel_order_sends_delete(self):
        client, mock_session = _build_client()
        api_resp = MagicMock()
        api_resp.status_code = 204
        api_resp.text = ""
        mock_session.request.return_value = api_resp

        client.cancel_order("ORD-999")
        call_args = mock_session.request.call_args
        assert call_args[0][0] == "DELETE"
        assert "ORD-999" in call_args[0][1]

    def test_cancel_order_uses_account_id_in_path(self):
        client, mock_session = _build_client()
        api_resp = MagicMock()
        api_resp.status_code = 204
        api_resp.text = ""
        mock_session.request.return_value = api_resp

        client.cancel_order("ORD-123")
        call_args = mock_session.request.call_args
        url = call_args[0][1]
        assert "ENCRYPTED_ACCOUNT_HASH_1234" in url


# ══════════════════════════════════════════════════════════════════════════════
#  SchwabClient — get_positions
# ══════════════════════════════════════════════════════════════════════════════


class TestSchwabPositions:
    def test_get_positions_parses_response(self):
        resp_json = {
            "securitiesAccount": {
                "positions": [
                    {
                        "instrument": {"symbol": "AAPL"},
                        "longQuantity": 100,
                        "shortQuantity": 0,
                        "averagePrice": 150.0,
                        "marketValue": 17500.0,
                        "currentPrice": 175.0,
                        "currentDayProfitLossPercentage": 2.5,
                        "currentDayProfitLoss": 250.0,
                    }
                ]
            }
        }
        client, mock_session = _build_client()
        api_resp = MagicMock()
        api_resp.status_code = 200
        api_resp.text = json.dumps(resp_json)
        api_resp.json.return_value = resp_json
        mock_session.request.return_value = api_resp

        positions = client.get_positions()
        assert len(positions) == 1
        assert positions[0]["symbol"] == "AAPL"
        assert positions[0]["quantity"] == 100

    def test_get_positions_current_price_correct(self):
        """Verify fix: current_price uses currentPrice field."""
        resp_json = {
            "securitiesAccount": {
                "positions": [
                    {
                        "instrument": {"symbol": "MSFT"},
                        "longQuantity": 50,
                        "shortQuantity": 0,
                        "averagePrice": 300.0,
                        "marketValue": 17500.0,
                        "currentPrice": 350.0,
                        "currentDayProfitLossPercentage": 1.5,
                        "currentDayProfitLoss": 2500.0,
                    }
                ]
            }
        }
        client, mock_session = _build_client()
        api_resp = MagicMock()
        api_resp.status_code = 200
        api_resp.text = json.dumps(resp_json)
        api_resp.json.return_value = resp_json
        mock_session.request.return_value = api_resp

        positions = client.get_positions()
        assert positions[0]["current_price"] == 350.0

    def test_get_positions_daily_pnl_pct_separate(self):
        """Verify fix: daily_pnl_pct is a separate field from pnl."""
        resp_json = {
            "securitiesAccount": {
                "positions": [
                    {
                        "instrument": {"symbol": "GOOG"},
                        "longQuantity": 20,
                        "shortQuantity": 0,
                        "averagePrice": 140.0,
                        "marketValue": 3000.0,
                        "currentPrice": 150.0,
                        "currentDayProfitLossPercentage": 3.2,
                        "currentDayProfitLoss": 96.0,
                    }
                ]
            }
        }
        client, mock_session = _build_client()
        api_resp = MagicMock()
        api_resp.status_code = 200
        api_resp.text = json.dumps(resp_json)
        api_resp.json.return_value = resp_json
        mock_session.request.return_value = api_resp

        positions = client.get_positions()
        assert positions[0]["daily_pnl_pct"] == 3.2
        assert positions[0]["pnl"] == 96.0
        assert positions[0]["daily_pnl_pct"] != positions[0]["pnl"]

    def test_get_positions_short_quantity(self):
        resp_json = {
            "securitiesAccount": {
                "positions": [
                    {
                        "instrument": {"symbol": "SPY"},
                        "longQuantity": 0,
                        "shortQuantity": 50,
                        "averagePrice": 450.0,
                        "marketValue": -22500.0,
                        "currentPrice": 445.0,
                        "currentDayProfitLossPercentage": 1.1,
                        "currentDayProfitLoss": 250.0,
                    }
                ]
            }
        }
        client, mock_session = _build_client()
        api_resp = MagicMock()
        api_resp.status_code = 200
        api_resp.text = json.dumps(resp_json)
        api_resp.json.return_value = resp_json
        mock_session.request.return_value = api_resp

        positions = client.get_positions()
        assert positions[0]["quantity"] == -50

    def test_get_positions_empty(self):
        resp_json = {"securitiesAccount": {"positions": []}}
        client, mock_session = _build_client()
        api_resp = MagicMock()
        api_resp.status_code = 200
        api_resp.text = json.dumps(resp_json)
        api_resp.json.return_value = resp_json
        mock_session.request.return_value = api_resp

        positions = client.get_positions()
        assert positions == []


# ══════════════════════════════════════════════════════════════════════════════
#  SchwabClient — get_account_info
# ══════════════════════════════════════════════════════════════════════════════


class TestSchwabAccount:
    def test_get_account_info_parses_balances(self):
        resp_json = {
            "securitiesAccount": {
                "type": "MARGIN",
                "currentBalances": {
                    "liquidationValue": 100000.0,
                    "cashBalance": 25000.0,
                    "buyingPower": 50000.0,
                    "equity": 75000.0,
                },
            }
        }
        client, mock_session = _build_client()
        api_resp = MagicMock()
        api_resp.status_code = 200
        api_resp.text = json.dumps(resp_json)
        api_resp.json.return_value = resp_json
        mock_session.request.return_value = api_resp

        info = client.get_account_info()
        assert info["account_id"] == "ENCRYPTED_ACCOUNT_HASH_1234"
        assert info["account_type"] == "MARGIN"
        assert info["net_liquidation"] == 100000.0
        assert info["cash_balance"] == 25000.0
        assert info["buying_power"] == 50000.0
        assert info["equity"] == 75000.0

    def test_get_account_info_missing_balances(self):
        resp_json = {"securitiesAccount": {"type": "CASH"}}
        client, mock_session = _build_client()
        api_resp = MagicMock()
        api_resp.status_code = 200
        api_resp.text = json.dumps(resp_json)
        api_resp.json.return_value = resp_json
        mock_session.request.return_value = api_resp

        info = client.get_account_info()
        assert info["net_liquidation"] == 0
        assert info["cash_balance"] == 0


# ══════════════════════════════════════════════════════════════════════════════
#  SchwabClient — get_orders / get_quote / get_quotes / get_price_history
# ══════════════════════════════════════════════════════════════════════════════


class TestSchwabOrders:
    def test_get_orders_list_response(self):
        resp_json = [{"orderId": "1", "status": "WORKING"}, {"orderId": "2", "status": "FILLED"}]
        client, mock_session = _build_client()
        api_resp = MagicMock()
        api_resp.status_code = 200
        api_resp.text = json.dumps(resp_json)
        api_resp.json.return_value = resp_json
        mock_session.request.return_value = api_resp

        orders = client.get_orders()
        assert len(orders) == 2

    def test_get_orders_dict_response(self):
        resp_json = {"orders": [{"orderId": "1"}]}
        client, mock_session = _build_client()
        api_resp = MagicMock()
        api_resp.status_code = 200
        api_resp.text = json.dumps(resp_json)
        api_resp.json.return_value = resp_json
        mock_session.request.return_value = api_resp

        orders = client.get_orders(status="FILLED")
        assert len(orders) == 1


class TestSchwabQuotes:
    def test_get_quote_parses_nested(self):
        resp_json = {
            "AAPL": {
                "quote": {
                    "lastPrice": 175.50,
                    "bidPrice": 175.45,
                    "askPrice": 175.55,
                    "openPrice": 174.00,
                    "highPrice": 176.00,
                    "lowPrice": 173.50,
                    "totalVolume": 50000000,
                    "netChange": 1.50,
                    "netPercentChangeInDouble": 0.86,
                }
            }
        }
        client, mock_session = _build_client()
        api_resp = MagicMock()
        api_resp.status_code = 200
        api_resp.text = json.dumps(resp_json)
        api_resp.json.return_value = resp_json
        mock_session.request.return_value = api_resp

        quote = client.get_quote("AAPL")
        assert quote["symbol"] == "AAPL"
        assert quote["lastPrice"] == 175.50
        assert quote["netPercentChange"] == 0.86

    def test_get_quote_symbol_not_found(self):
        client, mock_session = _build_client()
        api_resp = MagicMock()
        api_resp.status_code = 200
        api_resp.text = "{}"
        api_resp.json.return_value = {}
        mock_session.request.return_value = api_resp

        quote = client.get_quote("UNKNOWN")
        assert quote == {"symbol": "UNKNOWN", "lastPrice": 0}

    def test_get_quotes_multiple_symbols(self):
        resp_json = {
            "AAPL": {"quote": {"lastPrice": 175, "bidPrice": 174, "askPrice": 176, "totalVolume": 1000}},
            "MSFT": {"quote": {"lastPrice": 400, "bidPrice": 399, "askPrice": 401, "totalVolume": 2000}},
        }
        client, mock_session = _build_client()
        api_resp = MagicMock()
        api_resp.status_code = 200
        api_resp.text = json.dumps(resp_json)
        api_resp.json.return_value = resp_json
        mock_session.request.return_value = api_resp

        quotes = client.get_quotes(["AAPL", "MSFT"])
        assert "AAPL" in quotes
        assert "MSFT" in quotes

    def test_get_price_history(self):
        resp_json = {
            "candles": [
                {"open": 170, "high": 175, "low": 169, "close": 174, "volume": 1000, "datetime": 1700000000},
            ]
        }
        client, mock_session = _build_client()
        api_resp = MagicMock()
        api_resp.status_code = 200
        api_resp.text = json.dumps(resp_json)
        api_resp.json.return_value = resp_json
        mock_session.request.return_value = api_resp

        candles = client.get_price_history("AAPL")
        assert len(candles) == 1
        assert candles[0]["close"] == 174


# ══════════════════════════════════════════════════════════════════════════════
#  SchwabClient — Rate Limiting
# ══════════════════════════════════════════════════════════════════════════════


class TestSchwabRateLimit:
    def test_rate_limit_clears_old(self):
        client, _ = _build_client()
        client._request_times.clear()
        old = time.time() - 5
        for _ in range(5):
            client._request_times.append(old)
        client._rate_limit()
        assert len(client._request_times) == 1

    @patch("thinkorswim.api.schwab_client.time.sleep")
    def test_rate_limit_sleeps_when_full(self, mock_sleep):
        client, _ = _build_client()
        client._request_times.clear()
        now = time.time()
        for _ in range(2):
            client._request_times.append(now)
        client._rate_limit()
        mock_sleep.assert_called_once()
