"""Comprehensive tests for TradeStation API — OrderRouter and AccountMonitor."""

import sys
import os
import json
import time
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tradestation.api.order_router import TradeStationOrderRouter, TradeStationAPIError
from tradestation.api.account_monitor import AccountMonitor


# ── Helpers ──────────────────────────────────────────────────────────────────

VALID_CONFIG = {
    "client_id": "test_id",
    "client_secret": "test_secret",
    "redirect_uri": "https://localhost/callback",
    "refresh_token": "test_refresh_token",
}

AUTH_RESPONSE = {
    "access_token": "tok_abc123",
    "expires_in": 1200,
    "refresh_token": "new_refresh_tok",
}


def _make_mock_session(auth_json=None, request_json=None, request_status=200):
    """Build a mock requests.Session pre-wired with auth and request responses."""
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
    mock_session.request.return_value = api_resp

    return mock_session


def _build_router(mock_session=None):
    """Instantiate a TradeStationOrderRouter with mocked session."""
    if mock_session is None:
        mock_session = _make_mock_session()
    with patch("tradestation.api.order_router.requests.Session", return_value=mock_session):
        router = TradeStationOrderRouter(VALID_CONFIG)
    return router, mock_session


# ══════════════════════════════════════════════════════════════════════════════
#  OrderRouter — Initialization
# ══════════════════════════════════════════════════════════════════════════════


class TestOrderRouterInit:
    def test_init_stores_config(self):
        router, _ = _build_router()
        assert router.client_id == "test_id"
        assert router.client_secret == "test_secret"
        assert router.redirect_uri == "https://localhost/callback"

    def test_init_calls_authenticate(self):
        mock_session = _make_mock_session()
        with patch("tradestation.api.order_router.requests.Session", return_value=mock_session):
            router = TradeStationOrderRouter(VALID_CONFIG)
        assert router.access_token == "tok_abc123"
        mock_session.post.assert_called_once()

    def test_init_missing_key_raises(self):
        bad_config = {"client_id": "x"}
        with pytest.raises(KeyError):
            with patch("tradestation.api.order_router.requests.Session", return_value=MagicMock()):
                TradeStationOrderRouter(bad_config)


# ══════════════════════════════════════════════════════════════════════════════
#  OrderRouter — _authenticate
# ══════════════════════════════════════════════════════════════════════════════


class TestAuthenticate:
    def test_authenticate_uses_session_post_not_bare_requests(self):
        """Verify fix: session.post is used, not bare requests.post."""
        mock_session = _make_mock_session()
        with patch("tradestation.api.order_router.requests.Session", return_value=mock_session):
            router = TradeStationOrderRouter(VALID_CONFIG)
        mock_session.post.assert_called_once()

    def test_authenticate_sets_access_token(self):
        router, _ = _build_router()
        assert router.access_token == "tok_abc123"

    def test_authenticate_sets_token_expiry(self):
        router, _ = _build_router()
        assert router.token_expiry > time.time()

    def test_authenticate_updates_refresh_token(self):
        router, _ = _build_router()
        assert router.refresh_token == "new_refresh_tok"

    def test_authenticate_no_refresh_token_in_response(self):
        auth_resp = {"access_token": "tok_xyz", "expires_in": 600}
        mock_session = _make_mock_session(auth_json=auth_resp)
        with patch("tradestation.api.order_router.requests.Session", return_value=mock_session):
            router = TradeStationOrderRouter(VALID_CONFIG)
        assert router.refresh_token == "test_refresh_token"

    def test_authenticate_failure_raises(self):
        mock_session = MagicMock()
        fail_resp = MagicMock()
        fail_resp.status_code = 400
        fail_resp.text = "Bad Request"
        mock_session.post.return_value = fail_resp

        with patch("tradestation.api.order_router.requests.Session", return_value=mock_session):
            with pytest.raises(TradeStationAPIError, match="Authentication failed"):
                TradeStationOrderRouter(VALID_CONFIG)

    def test_authenticate_no_access_token_in_body_raises(self):
        auth_resp = {"some_key": "some_value"}
        mock_session = _make_mock_session(auth_json=auth_resp)
        with patch("tradestation.api.order_router.requests.Session", return_value=mock_session):
            with pytest.raises(TradeStationAPIError, match="Auth failed"):
                TradeStationOrderRouter(VALID_CONFIG)


# ══════════════════════════════════════════════════════════════════════════════
#  OrderRouter — _request
# ══════════════════════════════════════════════════════════════════════════════


class TestRequest:
    def test_request_success_200(self):
        router, mock_session = _build_router()
        api_resp = MagicMock()
        api_resp.status_code = 200
        api_resp.text = '{"ok": true}'
        api_resp.json.return_value = {"ok": True}
        mock_session.request.return_value = api_resp

        result = router._request("GET", "/test")
        assert result == {"ok": True}

    def test_request_accepts_204_no_content(self):
        """Verify fix: 204 is accepted as success, not just 200/201."""
        router, mock_session = _build_router()
        api_resp = MagicMock()
        api_resp.status_code = 204
        api_resp.text = ""
        mock_session.request.return_value = api_resp

        result = router._request("DELETE", "/test")
        assert result == {}

    def test_request_accepts_201_created(self):
        router, mock_session = _build_router()
        api_resp = MagicMock()
        api_resp.status_code = 201
        api_resp.text = '{"id": "123"}'
        api_resp.json.return_value = {"id": "123"}
        mock_session.request.return_value = api_resp

        result = router._request("POST", "/test")
        assert result == {"id": "123"}

    def test_request_raises_on_500(self):
        router, mock_session = _build_router()
        api_resp = MagicMock()
        api_resp.status_code = 500
        api_resp.text = "Internal Server Error"
        mock_session.request.return_value = api_resp

        with pytest.raises(TradeStationAPIError, match="API error 500"):
            router._request("GET", "/fail")

    def test_request_json_decode_error_returns_empty_dict(self):
        """Verify fix: JSONDecodeError is caught and returns {}."""
        router, mock_session = _build_router()
        api_resp = MagicMock()
        api_resp.status_code = 200
        api_resp.text = "not-json"
        api_resp.json.side_effect = ValueError("No JSON")
        mock_session.request.return_value = api_resp

        result = router._request("GET", "/bad-json")
        assert result == {}

    def test_request_network_error_raises(self):
        import requests as req
        router, mock_session = _build_router()
        mock_session.request.side_effect = req.ConnectionError("timeout")

        with pytest.raises(TradeStationAPIError, match="Network error"):
            router._request("GET", "/network-fail")

    def test_request_401_triggers_retry(self):
        """Verify fix: 401 triggers token refresh + rate-limited retry."""
        router, mock_session = _build_router()

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

        result = router._request("GET", "/needs-auth")
        assert result == {"retried": True}
        assert mock_session.request.call_count == 2


# ══════════════════════════════════════════════════════════════════════════════
#  OrderRouter — _enforce_rate_limit
# ══════════════════════════════════════════════════════════════════════════════


class TestRateLimit:
    def test_rate_limit_allows_under_threshold(self):
        router, _ = _build_router()
        router._request_timestamps.clear()
        router._enforce_rate_limit()
        assert len(router._request_timestamps) == 1

    def test_rate_limit_clears_old_timestamps(self):
        router, _ = _build_router()
        router._request_timestamps.clear()
        old_time = time.time() - 120
        for _ in range(10):
            router._request_timestamps.append(old_time)
        router._enforce_rate_limit()
        assert len(router._request_timestamps) == 1

    @patch("tradestation.api.order_router.time.sleep")
    def test_rate_limit_sleeps_when_full(self, mock_sleep):
        router, _ = _build_router()
        router._request_timestamps.clear()
        now = time.time()
        for _ in range(120):
            router._request_timestamps.append(now)
        router._enforce_rate_limit()
        mock_sleep.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
#  OrderRouter — Order Methods
# ══════════════════════════════════════════════════════════════════════════════


class TestMarketOrder:
    def test_place_market_order_returns_order_id(self):
        resp_json = {"Orders": [{"OrderID": "MKT-001"}]}
        router, mock_session = _build_router(_make_mock_session(request_json=resp_json))
        mock_session.request.return_value.text = json.dumps(resp_json)
        mock_session.request.return_value.json.return_value = resp_json

        order_id = router.place_market_order("ACC1", "AAPL", "BUY", 100)
        assert order_id == "MKT-001"

    def test_place_market_order_safe_get_for_order_id(self):
        """Verify fix: uses .get('OrderID') instead of ['OrderID']."""
        resp_json = {"Orders": [{}]}
        router, mock_session = _build_router(_make_mock_session(request_json=resp_json))
        mock_session.request.return_value.text = json.dumps(resp_json)
        mock_session.request.return_value.json.return_value = resp_json

        order_id = router.place_market_order("ACC1", "AAPL", "BUY", 50)
        assert order_id is None

    def test_place_market_order_empty_orders_list(self):
        resp_json = {"Orders": []}
        router, mock_session = _build_router(_make_mock_session(request_json=resp_json))
        mock_session.request.return_value.text = json.dumps(resp_json)
        mock_session.request.return_value.json.return_value = resp_json

        order_id = router.place_market_order("ACC1", "AAPL", "SELL", 10)
        assert order_id is None

    def test_place_market_order_body_structure(self):
        resp_json = {"Orders": [{"OrderID": "MKT-002"}]}
        router, mock_session = _build_router(_make_mock_session(request_json=resp_json))
        mock_session.request.return_value.text = json.dumps(resp_json)
        mock_session.request.return_value.json.return_value = resp_json

        router.place_market_order("ACC1", "TSLA", "BUY", 50)
        call_args = mock_session.request.call_args
        body = call_args[1]["json"]
        assert body["OrderType"] == "Market"
        assert body["TradeAction"] == "BUY"
        assert body["Symbol"] == "TSLA"
        assert body["Quantity"] == "50"


class TestLimitOrder:
    def test_place_limit_order_returns_order_id(self):
        resp_json = {"Orders": [{"OrderID": "LMT-001"}]}
        router, mock_session = _build_router(_make_mock_session(request_json=resp_json))
        mock_session.request.return_value.text = json.dumps(resp_json)
        mock_session.request.return_value.json.return_value = resp_json

        order_id = router.place_limit_order("ACC1", "MSFT", "BUY", 25, 350.50)
        assert order_id == "LMT-001"

    def test_place_limit_order_body_has_limit_price(self):
        resp_json = {"Orders": [{"OrderID": "LMT-002"}]}
        router, mock_session = _build_router(_make_mock_session(request_json=resp_json))
        mock_session.request.return_value.text = json.dumps(resp_json)
        mock_session.request.return_value.json.return_value = resp_json

        router.place_limit_order("ACC1", "GOOG", "SELL", 10, 175.25)
        call_args = mock_session.request.call_args
        body = call_args[1]["json"]
        assert body["OrderType"] == "Limit"
        assert body["LimitPrice"] == "175.25"


class TestBracketOrder:
    def test_place_bracket_order_returns_entry_and_brackets(self):
        resp_json = {
            "Orders": [
                {"OrderID": "BRK-ENTRY"},
                {"OrderID": "BRK-TP"},
                {"OrderID": "BRK-SL"},
            ]
        }
        router, mock_session = _build_router(_make_mock_session(request_json=resp_json))
        mock_session.request.return_value.text = json.dumps(resp_json)
        mock_session.request.return_value.json.return_value = resp_json

        result = router.place_bracket_order("ACC1", "NVDA", "BUY", 100, 800.0, 850.0, 780.0)
        assert result["entry_order_id"] == "BRK-ENTRY"
        assert result["bracket_orders"] == ["BRK-TP", "BRK-SL"]

    def test_bracket_order_exit_action_mapping_buy(self):
        """Verify fix: BUY entry → SELL exit legs."""
        resp_json = {"Orders": [{"OrderID": "X"}]}
        router, mock_session = _build_router(_make_mock_session(request_json=resp_json))
        mock_session.request.return_value.text = json.dumps(resp_json)
        mock_session.request.return_value.json.return_value = resp_json

        router.place_bracket_order("ACC1", "AMD", "BUY", 50, 150.0, 160.0, 140.0)
        call_body = mock_session.request.call_args[1]["json"]
        oso_orders = call_body["OSOs"][0]["Orders"]
        assert oso_orders[0]["TradeAction"] == "SELL"
        assert oso_orders[1]["TradeAction"] == "SELL"

    def test_bracket_order_exit_action_mapping_sellshort(self):
        """Verify fix: SELLSHORT entry → BUYTOCOVER exit legs."""
        resp_json = {"Orders": [{"OrderID": "X"}]}
        router, mock_session = _build_router(_make_mock_session(request_json=resp_json))
        mock_session.request.return_value.text = json.dumps(resp_json)
        mock_session.request.return_value.json.return_value = resp_json

        router.place_bracket_order("ACC1", "AMD", "SELLSHORT", 50, 150.0, 140.0, 160.0)
        call_body = mock_session.request.call_args[1]["json"]
        oso_orders = call_body["OSOs"][0]["Orders"]
        assert oso_orders[0]["TradeAction"] == "BUYTOCOVER"
        assert oso_orders[1]["TradeAction"] == "BUYTOCOVER"

    def test_bracket_order_empty_orders_response(self):
        resp_json = {"Orders": []}
        router, mock_session = _build_router(_make_mock_session(request_json=resp_json))
        mock_session.request.return_value.text = json.dumps(resp_json)
        mock_session.request.return_value.json.return_value = resp_json

        result = router.place_bracket_order("ACC1", "X", "BUY", 10, 100, 110, 90)
        assert result["entry_order_id"] is None
        assert result["bracket_orders"] == []


class TestCancelOrder:
    def test_cancel_order_sends_delete(self):
        """Verify fix: cancel uses DELETE, no unused param."""
        router, mock_session = _build_router()
        api_resp = MagicMock()
        api_resp.status_code = 204
        api_resp.text = ""
        mock_session.request.return_value = api_resp

        router.cancel_order("ORD-999")
        call_args = mock_session.request.call_args
        assert call_args[0][0] == "DELETE"
        assert "/orderexecution/orders/ORD-999" in call_args[0][1]

    def test_cancel_order_no_unused_params(self):
        """Verify fix: cancel_order only takes order_id, no account_id."""
        import inspect
        sig = inspect.signature(TradeStationOrderRouter.cancel_order)
        params = list(sig.parameters.keys())
        assert params == ["self", "order_id"]


# ══════════════════════════════════════════════════════════════════════════════
#  OrderRouter — get_order_status, get_orders, get_quote(s)
# ══════════════════════════════════════════════════════════════════════════════


class TestOrderQueries:
    def test_get_order_status(self):
        resp_json = {"Status": "Filled", "OrderID": "123"}
        router, mock_session = _build_router(_make_mock_session(request_json=resp_json))
        mock_session.request.return_value.text = json.dumps(resp_json)
        mock_session.request.return_value.json.return_value = resp_json

        result = router.get_order_status("ACC1", "123")
        assert result["Status"] == "Filled"

    def test_get_orders(self):
        resp_json = {"Orders": [{"OrderID": "A"}, {"OrderID": "B"}]}
        router, mock_session = _build_router(_make_mock_session(request_json=resp_json))
        mock_session.request.return_value.text = json.dumps(resp_json)
        mock_session.request.return_value.json.return_value = resp_json

        orders = router.get_orders("ACC1")
        assert len(orders) == 2

    def test_get_quote_returns_first_quote(self):
        resp_json = {"Quotes": [{"Last": "150.00", "Symbol": "AAPL"}]}
        router, mock_session = _build_router(_make_mock_session(request_json=resp_json))
        mock_session.request.return_value.text = json.dumps(resp_json)
        mock_session.request.return_value.json.return_value = resp_json

        quote = router.get_quote("AAPL")
        assert quote["Last"] == "150.00"

    def test_get_quote_api_error_returns_empty(self):
        router, mock_session = _build_router()
        mock_session.request.side_effect = TradeStationAPIError("fail")

        quote = router.get_quote("BAD")
        assert quote == {}

    def test_get_quotes_joins_symbols(self):
        resp_json = {"Quotes": [{"Symbol": "A"}, {"Symbol": "B"}]}
        router, mock_session = _build_router(_make_mock_session(request_json=resp_json))
        mock_session.request.return_value.text = json.dumps(resp_json)
        mock_session.request.return_value.json.return_value = resp_json

        quotes = router.get_quotes(["A", "B"])
        assert len(quotes) == 2

    def test_get_quotes_empty_list(self):
        router, _ = _build_router()
        assert router.get_quotes([]) == []


# ══════════════════════════════════════════════════════════════════════════════
#  AccountMonitor
# ══════════════════════════════════════════════════════════════════════════════


class TestAccountMonitorBalances:
    def test_get_balances_parses_response(self):
        router, mock_session = _build_router()
        bal_data = {
            "Balances": [
                {
                    "CashBalance": "50000",
                    "Equity": "100000",
                    "MarketValue": "80000",
                    "MarginUsed": "20000",
                    "MarginAvailable": "60000",
                }
            ]
        }
        mock_session.request.return_value.status_code = 200
        mock_session.request.return_value.text = json.dumps(bal_data)
        mock_session.request.return_value.json.return_value = bal_data

        monitor = AccountMonitor(router, {})
        balances = monitor.get_balances("ACC1")
        assert balances["cash_balance"] == 50000.0
        assert balances["equity"] == 100000.0

    def test_get_balances_empty_balances_guard(self):
        """Verify fix: handles empty Balances list without IndexError."""
        router, mock_session = _build_router()
        empty_data = {"Balances": []}
        mock_session.request.return_value.status_code = 200
        mock_session.request.return_value.text = json.dumps(empty_data)
        mock_session.request.return_value.json.return_value = empty_data

        monitor = AccountMonitor(router, {})
        balances = monitor.get_balances("ACC1")
        assert balances["cash_balance"] == 0.0
        assert balances["equity"] == 0.0


class TestAccountMonitorPositions:
    def test_get_positions_parses_multiple(self):
        router, mock_session = _build_router()
        pos_data = {
            "Positions": [
                {
                    "Symbol": "AAPL",
                    "AveragePrice": "150",
                    "Last": "160",
                    "Quantity": "100",
                    "UnrealizedProfitLoss": "1000",
                    "MarketValue": "16000",
                    "AssetType": "STOCK",
                },
                {
                    "Symbol": "TSLA",
                    "AveragePrice": "200",
                    "Last": "190",
                    "Quantity": "50",
                    "UnrealizedProfitLoss": "-500",
                    "MarketValue": "9500",
                    "AssetType": "STOCK",
                },
            ]
        }
        mock_session.request.return_value.status_code = 200
        mock_session.request.return_value.text = json.dumps(pos_data)
        mock_session.request.return_value.json.return_value = pos_data

        monitor = AccountMonitor(router, {})
        positions = monitor.get_positions("ACC1")
        assert len(positions) == 2
        assert positions[0]["symbol"] == "AAPL"
        assert positions[0]["quantity"] == 100.0

    def test_get_positions_empty(self):
        router, mock_session = _build_router()
        mock_session.request.return_value.status_code = 200
        mock_session.request.return_value.text = json.dumps({"Positions": []})
        mock_session.request.return_value.json.return_value = {"Positions": []}

        monitor = AccountMonitor(router, {})
        assert monitor.get_positions("ACC1") == []

    def test_get_positions_pnl_pct_calc(self):
        router, mock_session = _build_router()
        pos_data = {
            "Positions": [
                {
                    "Symbol": "X",
                    "AveragePrice": "100",
                    "Last": "110",
                    "Quantity": "10",
                    "UnrealizedProfitLoss": "100",
                    "MarketValue": "1100",
                }
            ]
        }
        mock_session.request.return_value.status_code = 200
        mock_session.request.return_value.text = json.dumps(pos_data)
        mock_session.request.return_value.json.return_value = pos_data

        monitor = AccountMonitor(router, {})
        positions = monitor.get_positions("ACC1")
        assert positions[0]["pnl_pct"] == 10.0


class TestAccountMonitorOrders:
    def test_get_orders_delegates_to_router(self):
        router, mock_session = _build_router()
        order_data = {"Orders": [{"OrderID": "A", "Status": "Open"}]}
        mock_session.request.return_value.status_code = 200
        mock_session.request.return_value.text = json.dumps(order_data)
        mock_session.request.return_value.json.return_value = order_data

        monitor = AccountMonitor(router, {})
        orders = monitor.get_orders("ACC1")
        assert len(orders) == 1

    def test_get_pending_count_via_daily_summary(self):
        router, mock_session = _build_router()

        bal_data = {"Balances": [{"CashBalance": "10000", "Equity": "50000", "MarketValue": "40000", "MarginUsed": "0", "MarginAvailable": "50000"}]}
        pos_data = {"Positions": []}
        order_data = {"Orders": [{"OrderID": "1", "Status": "Open"}, {"OrderID": "2", "Status": "Filled"}, {"OrderID": "3", "Status": "Queued"}]}

        call_count = [0]
        def side_effect(*args, **kwargs):
            result = MagicMock()
            result.status_code = 200
            if call_count[0] == 0:
                result.json.return_value = bal_data
                result.text = json.dumps(bal_data)
            elif call_count[0] == 1:
                result.json.return_value = pos_data
                result.text = json.dumps(pos_data)
            else:
                result.json.return_value = order_data
                result.text = json.dumps(order_data)
            call_count[0] += 1
            return result

        mock_session.request.side_effect = side_effect

        monitor = AccountMonitor(router, {})
        summary = monitor.generate_daily_summary("ACC1")
        assert "Pending Orders:" in summary


class TestAccountMonitorAlerts:
    def test_margin_alert_fires(self):
        router, _ = _build_router()
        notifier = MagicMock()
        monitor = AccountMonitor(router, {"margin_warning_pct": 50.0}, notifier=notifier)

        monitor._check_margin({"equity": 100000, "margin_used": 60000})
        notifier.send.assert_called_once()
        assert "Margin" in notifier.send.call_args[0][0]

    def test_drawdown_alert_fires(self):
        router, _ = _build_router()
        notifier = MagicMock()
        monitor = AccountMonitor(router, {"max_drawdown_pct": 5.0}, notifier=notifier)
        monitor._peak_equity = 100000

        monitor._check_drawdown({"equity": 90000})
        notifier.send.assert_called_once()

    def test_concentration_alert_fires(self):
        router, _ = _build_router()
        notifier = MagicMock()
        monitor = AccountMonitor(router, {"position_concentration_pct": 20.0}, notifier=notifier)

        positions = [{"symbol": "AAPL", "market_value": 30000}]
        monitor._check_concentration(positions, 100000)
        notifier.send.assert_called_once()

    def test_no_alert_when_below_threshold(self):
        router, _ = _build_router()
        notifier = MagicMock()
        monitor = AccountMonitor(router, {"margin_warning_pct": 90.0}, notifier=notifier)

        monitor._check_margin({"equity": 100000, "margin_used": 10000})
        notifier.send.assert_not_called()

    def test_send_alert_no_notifier(self):
        router, _ = _build_router()
        monitor = AccountMonitor(router, {})
        monitor._send_alert("Test", "msg")  # should not raise


class TestAccountMonitorMonitoring:
    def test_start_and_stop_monitoring(self):
        router, mock_session = _build_router()
        bal_data = {"Balances": [{"CashBalance": "10000", "Equity": "50000", "MarketValue": "40000", "MarginUsed": "0", "MarginAvailable": "50000"}]}
        pos_data = {"Positions": []}

        mock_session.request.return_value.status_code = 200
        mock_session.request.return_value.text = json.dumps(bal_data)
        mock_session.request.return_value.json.return_value = bal_data

        monitor = AccountMonitor(router, {})
        monitor.start_monitoring("ACC1", interval_seconds=1)
        assert monitor._monitor_thread is not None
        assert monitor._monitor_thread.is_alive()

        monitor.stop_monitoring()
        assert monitor._monitor_thread is None
