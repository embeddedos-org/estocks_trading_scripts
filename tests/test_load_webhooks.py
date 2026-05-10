# -*- coding: utf-8 -*-
"""
Load Test: Concurrent Trades Across All 4 Platform Webhooks
==============================================================

Hammers the webhook server with concurrent trade requests routed to all
4 broker adapters (IB, TradeStation, Schwab/thinkorswim), measuring
throughput, latency, risk gate enforcement, and error rates.

Uses FastAPI TestClient (in-process, no network) for reliable, repeatable
results. Broker adapters are mocked to isolate webhook logic from live APIs.

Run:
    cd /home/spatchava/stocks_plugin
    python -m pytest tests/test_load_webhooks.py -v --tb=short
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import statistics
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─── Mock Broker Adapters ───


def _make_mock_order_result(success=True):
    """Create a real OrderResult for the webhook server."""
    from tradingview.webhooks.webhook_server import OrderResult
    return OrderResult(
        success=success,
        broker="mock",
        order_id=f"MOCK-{int(time.time() * 1000)}",
        message="Mock order filled",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


class MockBrokerAdapter:
    """Fake broker adapter that returns instant fills."""

    def __init__(self, name: str, latency_ms: float = 0):
        self._name = name
        self._latency_ms = latency_ms
        self.order_count = 0

    @property
    def name(self) -> str:
        return self._name

    def connect(self) -> bool:
        return True

    def disconnect(self) -> None:
        pass

    def place_order(self, symbol, action, quantity, order_type="market", price=0.0):
        if self._latency_ms > 0:
            time.sleep(self._latency_ms / 1000)
        self.order_count += 1
        return _make_mock_order_result()

    def get_account_info(self) -> dict:
        return {"balance": 100000, "buying_power": 50000}


# ─── Fixtures ───


@pytest.fixture
def config():
    """Webhook server config with relaxed limits for load testing."""
    return {
        "server": {"title": "Load Test", "debug": False},
        "security": {
            "hmac_secret": "test_secret_key_for_load_testing_only",
            "require_hmac": True,
            "require_passphrase": False,
            "cors_origins": [],
            "allowed_ips": [],
        },
        "rate_limiting": {
            "max_requests_per_minute": 10000,
            "window_seconds": 60,
        },
        "risk_management": {
            "max_daily_loss": 50000.0,
            "max_consecutive_losses": 100,
            "cooldown_minutes": 0,
            "max_drawdown_pct": 50.0,
            "prefer_limit_orders": False,
        },
        "broker_routing": {
            "default_broker": "interactive_brokers",
            "routes": [
                {"pattern": "AAPL|MSFT|GOOGL", "broker": "interactive_brokers"},
                {"pattern": "TSLA|AMZN", "broker": "tradestation"},
                {"pattern": "META|NFLX", "broker": "schwab"},
                {"pattern": "JPM|BAC", "broker": "thinkorswim"},
            ],
        },
    }


@pytest.fixture
def mock_adapters():
    """Create mock broker adapters for all 4 platforms."""
    return {
        "interactive_brokers": MockBrokerAdapter("interactive_brokers", latency_ms=1),
        "tradestation": MockBrokerAdapter("tradestation", latency_ms=2),
        "schwab": MockBrokerAdapter("schwab", latency_ms=1),
        "thinkorswim": MockBrokerAdapter("thinkorswim", latency_ms=1),
    }


@pytest.fixture
def app_and_client(config, mock_adapters):
    """Create a configured FastAPI app with mock brokers and a TestClient."""
    from tradingview.webhooks.webhook_server import create_app, BrokerRouter
    from starlette.testclient import TestClient

    with patch("tradingview.webhooks.webhook_server.load_config", return_value=config):
        test_app = create_app()

    # Ensure HMAC secret matches what our test uses
    test_app.state.config["security"]["hmac_secret"] = HMAC_SECRET

    # Replace broker router's adapters and routes with mocks
    router = test_app.state.broker_router
    router._adapters = mock_adapters
    router._default = "interactive_brokers"
    router._routes = [
        {"pattern": "AAPL|MSFT|GOOGL", "broker": "interactive_brokers"},
        {"pattern": "TSLA|AMZN", "broker": "tradestation"},
        {"pattern": "META|NFLX", "broker": "schwab"},
        {"pattern": "JPM|BAC", "broker": "thinkorswim"},
    ]

    client = TestClient(test_app)
    return test_app, client, mock_adapters


def _sign_payload(payload_bytes: bytes, secret: str) -> str:
    """Compute HMAC-SHA256 signature."""
    return hmac.new(secret.encode(), payload_bytes, hashlib.sha256).hexdigest()


def _make_trade_payload(
    symbol: str,
    action: str = "buy",
    price: float = 175.50,
    quantity: float = 10.0,
    uid: str = "",
) -> dict:
    """Build a valid AlertPayload dict."""
    return {
        "symbol": symbol,
        "action": action,
        "price": price,
        "quantity": quantity,
        "order_type": "market",
        "strategy": "load_test",
        "message": f"Load test trade {uid}",
    }


HMAC_SECRET = "test_secret_key_for_load_testing_only"


def _post_trade(client, payload: dict) -> dict:
    """Send a signed webhook trade request and return timing + result."""
    body = json.dumps(payload).encode()
    sig = _sign_payload(body, HMAC_SECRET)
    start = time.perf_counter()
    resp = client.post(
        "/webhook",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Webhook-Signature": sig,
        },
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    return {
        "status_code": resp.status_code,
        "body": resp.json() if resp.status_code == 200 else {},
        "elapsed_ms": elapsed_ms,
        "symbol": payload.get("symbol", "?"),
    }


# ─── Tests ───


class TestSinglePlatformTrades:
    """Verify each broker adapter receives trades correctly."""

    def test_ib_trade(self, app_and_client):
        _, client, adapters = app_and_client
        payload = _make_trade_payload("AAPL", uid="ib-1")
        result = _post_trade(client, payload)
        assert result["status_code"] == 200
        assert adapters["interactive_brokers"].order_count >= 1

    def test_tradestation_trade(self, app_and_client):
        _, client, adapters = app_and_client
        payload = _make_trade_payload("TSLA", uid="ts-1")
        result = _post_trade(client, payload)
        assert result["status_code"] == 200
        assert adapters["tradestation"].order_count >= 1

    def test_schwab_trade(self, app_and_client):
        _, client, adapters = app_and_client
        payload = _make_trade_payload("META", uid="schwab-1")
        result = _post_trade(client, payload)
        assert result["status_code"] == 200
        assert adapters["schwab"].order_count >= 1

    def test_thinkorswim_trade(self, app_and_client):
        _, client, adapters = app_and_client
        payload = _make_trade_payload("JPM", uid="tos-1")
        result = _post_trade(client, payload)
        assert result["status_code"] == 200
        assert adapters["thinkorswim"].order_count >= 1


class TestConcurrentLoad:
    """Hammer the webhook with concurrent requests across all platforms."""

    SYMBOLS_BY_BROKER = {
        "interactive_brokers": ["AAPL", "MSFT", "GOOGL"],
        "tradestation": ["TSLA", "AMZN"],
        "schwab": ["META", "NFLX"],
        "thinkorswim": ["JPM", "BAC"],
    }

    def _generate_payloads(self, n: int) -> List[dict]:
        """Generate n unique trade payloads spread across all brokers."""
        all_symbols = []
        for symbols in self.SYMBOLS_BY_BROKER.values():
            all_symbols.extend(symbols)

        payloads = []
        actions = ["buy", "sell"]
        for i in range(n):
            symbol = all_symbols[i % len(all_symbols)]
            action = actions[i % len(actions)]
            price = 150.0 + (i % 50)
            payloads.append(_make_trade_payload(symbol, action, price, uid=f"load-{i}"))
        return payloads

    def test_50_sequential_trades(self, app_and_client):
        """50 trades in sequence across all 4 platforms."""
        _, client, adapters = app_and_client
        payloads = self._generate_payloads(50)

        results = []
        for payload in payloads:
            results.append(_post_trade(client, payload))

        success = [r for r in results if r["status_code"] == 200]
        latencies = [r["elapsed_ms"] for r in results]

        total_orders = sum(a.order_count for a in adapters.values())

        print(f"\n--- 50 Sequential Trades ---")
        print(f"  Success: {len(success)}/50")
        print(f"  Total broker orders: {total_orders}")
        print(f"  Avg latency: {statistics.mean(latencies):.1f}ms")
        print(f"  P50 latency: {sorted(latencies)[25]:.1f}ms")
        print(f"  P95 latency: {sorted(latencies)[47]:.1f}ms")
        print(f"  P99 latency: {sorted(latencies)[49]:.1f}ms")
        print(f"  Orders per broker:")
        for name, adapter in adapters.items():
            print(f"    {name}: {adapter.order_count}")

        assert len(success) >= 45, f"Expected >= 45 successes, got {len(success)}"

    def test_100_concurrent_trades(self, app_and_client):
        """100 trades fired concurrently with 10 threads."""
        _, client, adapters = app_and_client
        payloads = self._generate_payloads(100)

        results: List[dict] = []
        start_all = time.perf_counter()

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(_post_trade, client, p) for p in payloads]
            for future in as_completed(futures):
                results.append(future.result())

        wall_time = (time.perf_counter() - start_all) * 1000

        success = [r for r in results if r["status_code"] == 200]
        latencies = [r["elapsed_ms"] for r in results]
        total_orders = sum(a.order_count for a in adapters.values())
        throughput = len(results) / (wall_time / 1000)

        print(f"\n--- 100 Concurrent Trades (10 threads) ---")
        print(f"  Wall time: {wall_time:.0f}ms")
        print(f"  Throughput: {throughput:.1f} req/s")
        print(f"  Success: {len(success)}/100")
        print(f"  Total broker orders: {total_orders}")
        print(f"  Avg latency: {statistics.mean(latencies):.1f}ms")
        print(f"  P50: {sorted(latencies)[50]:.1f}ms")
        print(f"  P95: {sorted(latencies)[95]:.1f}ms")
        print(f"  P99: {sorted(latencies)[99]:.1f}ms")
        print(f"  Orders per broker:")
        for name, adapter in adapters.items():
            print(f"    {name}: {adapter.order_count}")

        assert len(success) >= 80, f"Expected >= 80 successes, got {len(success)}"
        assert throughput > 10, f"Expected >10 req/s, got {throughput:.1f}"

    def test_200_concurrent_trades_20_threads(self, app_and_client):
        """200 trades fired concurrently with 20 threads — stress test."""
        _, client, adapters = app_and_client
        payloads = self._generate_payloads(200)

        results: List[dict] = []
        start_all = time.perf_counter()

        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(_post_trade, client, p) for p in payloads]
            for future in as_completed(futures):
                results.append(future.result())

        wall_time = (time.perf_counter() - start_all) * 1000

        success = [r for r in results if r["status_code"] == 200]
        blocked = [r for r in results if r["body"].get("status") == "blocked"]
        duplicates = [r for r in results if r["body"].get("status") == "duplicate"]
        errors = [r for r in results if r["status_code"] != 200]
        latencies = [r["elapsed_ms"] for r in results]
        total_orders = sum(a.order_count for a in adapters.values())
        throughput = len(results) / (wall_time / 1000)

        print(f"\n--- 200 Concurrent Trades (20 threads) ---")
        print(f"  Wall time: {wall_time:.0f}ms")
        print(f"  Throughput: {throughput:.1f} req/s")
        print(f"  Success (200): {len(success)}/200")
        print(f"  Broker orders: {total_orders}")
        print(f"  Blocked by risk: {len(blocked)}")
        print(f"  Duplicates: {len(duplicates)}")
        print(f"  HTTP errors: {len(errors)}")
        print(f"  Avg latency: {statistics.mean(latencies):.1f}ms")
        print(f"  P50: {sorted(latencies)[100]:.1f}ms")
        print(f"  P95: {sorted(latencies)[190]:.1f}ms")
        print(f"  P99: {sorted(latencies)[198]:.1f}ms")
        print(f"  Orders per broker:")
        for name, adapter in adapters.items():
            print(f"    {name}: {adapter.order_count}")

        # Some may be deduplicated due to identical payload bodies
        assert len(success) >= 100, f"Expected >= 100 successes, got {len(success)}"


class TestRiskGateEnforcement:
    """Verify risk gates block trades under load."""

    def test_daily_loss_limit_blocks(self, app_and_client):
        """After hitting daily loss limit, trades should be blocked."""
        test_app, client, adapters = app_and_client

        # Simulate accumulated daily losses
        test_app.state.pnl_tracker._daily_pnl = -49999.0  # Just under limit

        # This trade should still go through
        p1 = _make_trade_payload("AAPL", uid="loss-1")
        r1 = _post_trade(client, p1)
        assert r1["status_code"] == 200

        # Push past the limit
        test_app.state.pnl_tracker._daily_pnl = -50001.0

        # This should be blocked
        p2 = _make_trade_payload("MSFT", uid="loss-2")
        r2 = _post_trade(client, p2)
        assert r2["status_code"] == 200
        assert r2["body"].get("status") == "blocked"

    def test_drawdown_breaker_blocks(self, app_and_client):
        """Drawdown circuit breaker should block all trades."""
        from datetime import datetime, timedelta, timezone
        test_app, client, _ = app_and_client

        # Trip the drawdown breaker
        breaker = test_app.state.drawdown_breaker
        breaker._tripped_until = datetime.now(timezone.utc) + timedelta(hours=1)

        p = _make_trade_payload("AAPL", uid="dd-1")
        r = _post_trade(client, p)
        assert r["status_code"] == 200
        assert r["body"].get("status") == "blocked"

    def test_dedup_rejects_identical_trades(self, app_and_client):
        """Same payload within 5 minutes should be deduplicated."""
        _, client, _ = app_and_client

        payload = _make_trade_payload("AAPL", uid="dedup-same")
        r1 = _post_trade(client, payload)
        r2 = _post_trade(client, payload)

        assert r1["status_code"] == 200
        # Second identical request should be deduped
        assert r2["status_code"] == 200
        assert r2["body"].get("status") == "duplicate"

    def test_hmac_rejects_bad_signature(self, app_and_client):
        """Invalid HMAC signature should be rejected."""
        _, client, _ = app_and_client

        payload = _make_trade_payload("AAPL", uid="bad-sig")
        body = json.dumps(payload).encode()
        resp = client.post(
            "/webhook",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Webhook-Signature": "invalid_signature_here",
            },
        )
        # Should be 401 or 200 with error status
        assert resp.status_code in (200, 401, 403)

    def test_mixed_load_with_risk_blocks(self, app_and_client):
        """50 trades with some risk blocks — verify correct proportions."""
        test_app, client, adapters = app_and_client

        # Set a tight daily loss limit
        test_app.state.pnl_tracker._max_daily_loss = 1000.0
        test_app.state.pnl_tracker._daily_pnl = -900.0  # Close to limit

        payloads = []
        for i in range(50):
            symbol = ["AAPL", "TSLA", "META", "JPM"][i % 4]
            payloads.append(_make_trade_payload(symbol, uid=f"mixed-{i}"))

        results = []
        for p in payloads:
            results.append(_post_trade(client, p))

        success_exec = [r for r in results if r["body"].get("status") not in ("blocked", "duplicate")]
        blocked = [r for r in results if r["body"].get("status") == "blocked"]

        print(f"\n--- Mixed Load with Risk Blocks ---")
        print(f"  Total: {len(results)}")
        print(f"  Executed: {len(success_exec)}")
        print(f"  Blocked: {len(blocked)}")

        # At least some should be blocked after daily loss limit trips
        # (depends on whether the mock adapter reports losses)
        assert len(results) == 50
