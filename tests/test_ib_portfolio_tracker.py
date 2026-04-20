"""
Comprehensive tests for PortfolioTracker (interactive_brokers.analytics.portfolio_tracker).
"""

import sys
import os
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from interactive_brokers.analytics.portfolio_tracker import (
    Position,
    PortfolioSnapshot,
    PortfolioTracker,
    SECTOR_MAP,
)


# ── Helpers ──


def _make_account_summary():
    items = []
    values = {
        "AccountCode": "DU12345",
        "EquityWithLoanValue": "100000",
        "TotalCashValue": "25000",
        "BuyingPower": "200000",
        "NetLiquidation": "100000",
        "InitMarginReq": "15000",
        "MaintMarginReq": "10000",
        "AvailableFunds": "85000",
        "ExcessLiquidity": "90000",
        "Cushion": "0.90",
    }
    for tag, value in values.items():
        item = MagicMock()
        item.tag = tag
        item.value = value
        items.append(item)
    return items


def _make_portfolio_item(symbol, sec_type="STK", position=100,
                         avg_cost=150.0, market_price=160.0,
                         market_value=16000.0, unrealized_pnl=1000.0,
                         realized_pnl=0.0, currency="USD",
                         account="DU12345", multiplier=None):
    item = MagicMock()
    item.contract = MagicMock()
    item.contract.symbol = symbol
    item.contract.secType = sec_type
    item.contract.currency = currency
    item.position = position
    item.averageCost = avg_cost
    item.marketPrice = market_price
    item.marketValue = market_value
    item.unrealizedPNL = unrealized_pnl
    item.realizedPNL = realized_pnl
    item.account = account
    if multiplier is not None:
        item.contract.multiplier = str(multiplier)
    else:
        item.contract.multiplier = None
    return item


# ── Fixtures ──


@pytest.fixture
def mock_connection():
    conn = MagicMock()
    conn.accountSummary = MagicMock(return_value=[])
    conn.portfolio = MagicMock(return_value=[])
    return conn


@pytest.fixture
def mock_connection_with_ib():
    conn = MagicMock()
    conn.accountSummary = MagicMock(return_value=[])
    conn.portfolio = MagicMock(return_value=[])
    conn.ib = MagicMock()
    return conn


@pytest.fixture
def tracker(mock_connection):
    return PortfolioTracker(mock_connection)


@pytest.fixture
def tracker_with_ib(mock_connection_with_ib):
    return PortfolioTracker(mock_connection_with_ib)


# ── Position Dataclass ──


class TestPosition:
    def test_total_cost(self):
        pos = Position(symbol="AAPL", quantity=100, avg_cost=150.0)
        assert pos.total_cost == 15000.0

    def test_return_pct_positive(self):
        pos = Position(symbol="AAPL", quantity=100, avg_cost=100.0, unrealized_pnl=500.0)
        assert pos.return_pct == pytest.approx(5.0)

    def test_return_pct_zero_cost(self):
        pos = Position(symbol="AAPL", quantity=0, avg_cost=0.0)
        assert pos.return_pct == 0.0

    def test_weight_is_dataclass_field(self):
        """Verify fix: Position.weight is a dataclass field, not a property."""
        pos = Position(symbol="AAPL")
        assert hasattr(pos, "weight")
        pos.weight = 0.25
        assert pos.weight == 0.25

    def test_weight_default_zero(self):
        pos = Position(symbol="AAPL")
        assert pos.weight == 0.0

    def test_return_pct_negative(self):
        pos = Position(symbol="AAPL", quantity=100, avg_cost=100.0, unrealized_pnl=-300.0)
        assert pos.return_pct == pytest.approx(-3.0)


# ── PortfolioSnapshot Dataclass ──


class TestPortfolioSnapshot:
    def test_num_positions(self):
        snap = PortfolioSnapshot(positions=[Position(symbol="AAPL"), Position(symbol="MSFT")])
        assert snap.num_positions == 2

    def test_margin_utilization(self):
        snap = PortfolioSnapshot(net_liquidation=100000.0, initial_margin=15000.0)
        assert snap.margin_utilization == pytest.approx(15.0)

    def test_margin_utilization_zero_nlv(self):
        snap = PortfolioSnapshot(net_liquidation=0.0, initial_margin=15000.0)
        assert snap.margin_utilization == 0.0

    def test_total_market_value(self):
        snap = PortfolioSnapshot(positions=[
            Position(symbol="AAPL", market_value=10000.0),
            Position(symbol="MSFT", market_value=5000.0),
        ])
        assert snap.total_market_value == 15000.0


# ── PortfolioTracker Init ──


class TestPortfolioTrackerInit:
    def test_default_sector_map(self, mock_connection):
        t = PortfolioTracker(mock_connection)
        assert t.sector_map == SECTOR_MAP

    def test_custom_sector_map(self, mock_connection):
        custom = {"AAPL": "CustomSector"}
        t = PortfolioTracker(mock_connection, sector_map=custom)
        assert t.sector_map == custom


# ── get_positions ──


class TestGetPositions:
    def test_get_positions_stock(self, tracker, mock_connection):
        mock_connection.portfolio.return_value = [_make_portfolio_item("AAPL")]
        positions = tracker.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "AAPL"
        assert positions[0].quantity == 100
        assert positions[0].market_price == 160.0
        assert positions[0].sector == "Technology"

    def test_get_positions_unknown_sector(self, tracker, mock_connection):
        mock_connection.portfolio.return_value = [_make_portfolio_item("ZZZZ")]
        positions = tracker.get_positions()
        assert positions[0].sector == "Unknown"

    def test_get_positions_multiple(self, tracker, mock_connection):
        mock_connection.portfolio.return_value = [
            _make_portfolio_item("AAPL"),
            _make_portfolio_item("JPM", unrealized_pnl=-200),
        ]
        positions = tracker.get_positions()
        assert len(positions) == 2
        assert {p.symbol for p in positions} == {"AAPL", "JPM"}

    def test_get_positions_empty(self, tracker, mock_connection):
        mock_connection.portfolio.return_value = []
        assert tracker.get_positions() == []

    def test_get_positions_exception(self, tracker, mock_connection):
        mock_connection.portfolio.side_effect = RuntimeError("API error")
        assert tracker.get_positions() == []


# ── Greeks (options) ──


class TestGetPositionsGreeks:
    def test_multiplier_from_contract(self, tracker_with_ib, mock_connection_with_ib):
        """Verify fix: multiplier from contract, not hardcoded 100."""
        item = _make_portfolio_item("AAPL", sec_type="OPT", position=5, multiplier=50)
        mock_connection_with_ib.portfolio.return_value = [item]

        ticker = MagicMock()
        ticker.modelGreeks = MagicMock()
        ticker.modelGreeks.delta = 0.5
        ticker.modelGreeks.gamma = 0.02
        ticker.modelGreeks.theta = -0.05
        ticker.modelGreeks.vega = 0.10
        mock_connection_with_ib.ib.reqMktData.return_value = ticker

        positions = tracker_with_ib.get_positions()
        pos = positions[0]
        expected_mult = 5 * 50  # position * contract.multiplier
        assert pos.delta == pytest.approx(0.5 * expected_mult)
        assert pos.gamma == pytest.approx(0.02 * expected_mult)
        assert pos.theta == pytest.approx(-0.05 * expected_mult)
        assert pos.vega == pytest.approx(0.10 * expected_mult)

    def test_multiplier_default_100(self, tracker_with_ib, mock_connection_with_ib):
        """When multiplier is None, falls back to 100."""
        item = _make_portfolio_item("AAPL", sec_type="OPT", position=2, multiplier=None)
        mock_connection_with_ib.portfolio.return_value = [item]

        ticker = MagicMock()
        ticker.modelGreeks = MagicMock()
        ticker.modelGreeks.delta = 0.3
        ticker.modelGreeks.gamma = 0.01
        ticker.modelGreeks.theta = -0.02
        ticker.modelGreeks.vega = 0.05
        mock_connection_with_ib.ib.reqMktData.return_value = ticker

        positions = tracker_with_ib.get_positions()
        assert positions[0].delta == pytest.approx(0.3 * 200)  # 2 * 100

    def test_hasattr_guard_no_ib(self, tracker, mock_connection):
        """Verify fix: hasattr guard for .ib — no crash when .ib missing."""
        del mock_connection.ib
        mock_connection.portfolio.return_value = [
            _make_portfolio_item("AAPL", sec_type="OPT", position=5),
        ]
        positions = tracker.get_positions()
        assert len(positions) == 1
        assert positions[0].delta == 0.0

    def test_no_model_greeks(self, tracker_with_ib, mock_connection_with_ib):
        item = _make_portfolio_item("AAPL", sec_type="OPT", position=5, multiplier=100)
        mock_connection_with_ib.portfolio.return_value = [item]
        ticker = MagicMock()
        ticker.modelGreeks = None
        mock_connection_with_ib.ib.reqMktData.return_value = ticker

        positions = tracker_with_ib.get_positions()
        assert positions[0].delta == 0.0

    def test_greeks_exception_handled(self, tracker_with_ib, mock_connection_with_ib):
        item = _make_portfolio_item("AAPL", sec_type="OPT", position=5, multiplier=100)
        mock_connection_with_ib.portfolio.return_value = [item]
        mock_connection_with_ib.ib.reqMktData.side_effect = RuntimeError("fail")

        positions = tracker_with_ib.get_positions()
        assert len(positions) == 1
        assert positions[0].delta == 0.0


# ── get_snapshot ──


class TestGetSnapshot:
    def test_snapshot_values(self, tracker, mock_connection):
        mock_connection.accountSummary.return_value = _make_account_summary()
        mock_connection.portfolio.return_value = [
            _make_portfolio_item("AAPL", unrealized_pnl=1000, realized_pnl=200),
        ]
        snap = tracker.get_snapshot()
        assert snap.equity == 100000.0
        assert snap.cash == 25000.0
        assert snap.unrealized_pnl == 1000.0
        assert snap.realized_pnl == 200.0
        assert snap.num_positions == 1

    def test_snapshot_stored_in_history(self, tracker, mock_connection):
        mock_connection.accountSummary.return_value = _make_account_summary()
        mock_connection.portfolio.return_value = []
        tracker.get_snapshot()
        tracker.get_snapshot()
        assert len(tracker._snapshots) == 2

    def test_snapshot_aggregate_greeks(self, tracker, mock_connection):
        mock_connection.accountSummary.return_value = []
        with patch.object(tracker, "get_positions") as mock_pos:
            p1 = Position(symbol="AAPL", delta=10.0, gamma=1.0, theta=-2.0, vega=5.0)
            p2 = Position(symbol="MSFT", delta=5.0, gamma=0.5, theta=-1.0, vega=3.0)
            mock_pos.return_value = [p1, p2]
            snap = tracker.get_snapshot()
            assert snap.total_delta == 15.0
            assert snap.total_gamma == 1.5
            assert snap.total_theta == -3.0
            assert snap.total_vega == 8.0

    def test_snapshot_empty_account_summary(self, tracker, mock_connection):
        mock_connection.accountSummary.return_value = []
        mock_connection.portfolio.return_value = []
        snap = tracker.get_snapshot()
        assert snap.equity == 0.0
        assert snap.cash == 0.0


# ── PnL Summary ──


class TestPnlSummary:
    def test_pnl_summary_empty(self, tracker, mock_connection):
        mock_connection.portfolio.return_value = []
        summary = tracker.get_pnl_summary()
        assert summary["total_positions"] == 0

    def test_pnl_summary_with_positions(self, tracker, mock_connection):
        mock_connection.portfolio.return_value = [
            _make_portfolio_item("AAPL", unrealized_pnl=500, realized_pnl=100),
            _make_portfolio_item("MSFT", unrealized_pnl=-200, realized_pnl=50),
        ]
        summary = tracker.get_pnl_summary()
        assert summary["total_positions"] == 2
        assert summary["num_winners"] == 1
        assert summary["num_losers"] == 1
        assert summary["total_unrealized_pnl"] == 300.0
        assert summary["best_performer"]["symbol"] == "AAPL"
        assert summary["worst_performer"]["symbol"] == "MSFT"


# ── Sector Exposure ──


class TestSectorExposure:
    def test_sector_exposure(self, tracker, mock_connection):
        mock_connection.portfolio.return_value = [
            _make_portfolio_item("AAPL", market_value=10000, unrealized_pnl=500),
            _make_portfolio_item("JPM", market_value=5000, unrealized_pnl=-100),
        ]
        exposure = tracker.sector_exposure()
        assert "Technology" in exposure
        assert "Financials" in exposure
        assert exposure["Technology"]["value"] == 10000
        assert exposure["Technology"]["num_positions"] == 1
        total_weight = sum(s["weight"] for s in exposure.values())
        assert total_weight == pytest.approx(100.0)

    def test_sector_exposure_empty(self, tracker, mock_connection):
        mock_connection.portfolio.return_value = []
        exposure = tracker.sector_exposure()
        assert exposure == {}


# ── Snapshot History ──


class TestSnapshotHistory:
    def test_get_snapshot_history(self, tracker, mock_connection):
        mock_connection.accountSummary.return_value = []
        mock_connection.portfolio.return_value = []
        tracker.get_snapshot()
        tracker.get_snapshot()
        history = tracker.get_snapshot_history()
        assert len(history) == 2
        assert all(isinstance(s, PortfolioSnapshot) for s in history)
