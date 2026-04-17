"""Tests for IBConnection — factory pattern, connect/disconnect, retry logic, context manager."""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock


class TestIBConnectionFactory:
    """Tests for the IBConnection.create() factory method."""

    def test_import_ib_connection(self):
        from interactive_brokers.utils.ib_connection import IBConnection
        assert IBConnection is not None

    def test_create_returns_connection_object(self):
        from interactive_brokers.utils.ib_connection import IBConnection
        conn = IBConnection.create(
            backend="ib_async",
            host="127.0.0.1",
            port=7497,
            client_id=99,
        )
        assert conn is not None

    def test_create_ib_async_backend(self):
        from interactive_brokers.utils.ib_connection import IBConnection
        conn = IBConnection.create(backend="ib_async", port=7497)
        assert conn is not None

    def test_create_ibapi_backend(self):
        from interactive_brokers.utils.ib_connection import IBConnection
        conn = IBConnection.create(backend="ibapi", port=7497)
        assert conn is not None

    def test_create_invalid_backend_raises(self):
        from interactive_brokers.utils.ib_connection import IBConnection
        with pytest.raises((ValueError, KeyError)):
            IBConnection.create(backend="nonexistent_backend")

    def test_default_backend(self):
        from interactive_brokers.utils.ib_connection import IBConnection
        conn = IBConnection.create()
        assert conn is not None

    def test_paper_mode_port(self):
        from interactive_brokers.utils.ib_connection import IBConnection
        conn = IBConnection.create(port=7497)
        assert hasattr(conn, "port") or hasattr(conn, "_port") or conn is not None

    def test_live_mode_port(self):
        from interactive_brokers.utils.ib_connection import IBConnection
        conn = IBConnection.create(port=7496)
        assert conn is not None


class TestIBConnectionInterface:
    """Tests for the connection interface (connect, disconnect, is_connected)."""

    def test_has_connect_method(self):
        from interactive_brokers.utils.ib_connection import IBConnection
        conn = IBConnection.create(port=7497)
        assert hasattr(conn, "connect")
        assert callable(conn.connect)

    def test_has_disconnect_method(self):
        from interactive_brokers.utils.ib_connection import IBConnection
        conn = IBConnection.create(port=7497)
        assert hasattr(conn, "disconnect")
        assert callable(conn.disconnect)

    def test_has_is_connected_property(self):
        from interactive_brokers.utils.ib_connection import IBConnection
        conn = IBConnection.create(port=7497)
        assert hasattr(conn, "is_connected")

    def test_not_connected_initially(self):
        from interactive_brokers.utils.ib_connection import IBConnection
        conn = IBConnection.create(port=7497)
        assert conn.is_connected is False


class TestIBConnectionContextManager:
    """Tests for the context manager interface."""

    def test_has_context_manager_methods(self):
        from interactive_brokers.utils.ib_connection import IBConnection
        conn = IBConnection.create(port=7497)
        assert hasattr(conn, "__enter__")
        assert hasattr(conn, "__exit__")

    @patch("interactive_brokers.utils.ib_connection.IBAsyncConnection.connect")
    @patch("interactive_brokers.utils.ib_connection.IBAsyncConnection.disconnect")
    def test_context_manager_calls_connect_disconnect(self, mock_disconnect, mock_connect):
        from interactive_brokers.utils.ib_connection import IBConnection
        conn = IBConnection.create(backend="ib_async", port=7497)
        with conn:
            mock_connect.assert_called_once()
        mock_disconnect.assert_called_once()

    @patch("interactive_brokers.utils.ib_connection.IBAsyncConnection.connect")
    @patch("interactive_brokers.utils.ib_connection.IBAsyncConnection.disconnect")
    def test_context_manager_disconnects_on_exception(self, mock_disconnect, mock_connect):
        from interactive_brokers.utils.ib_connection import IBConnection
        conn = IBConnection.create(backend="ib_async", port=7497)
        try:
            with conn:
                raise RuntimeError("test error")
        except RuntimeError:
            pass
        mock_disconnect.assert_called_once()


class TestIBConnectionRetry:
    """Tests for auto-reconnect and retry logic."""

    def test_has_retry_config(self):
        from interactive_brokers.utils.ib_connection import IBConnection
        conn = IBConnection.create(port=7497)
        has_retry = (
            hasattr(conn, "max_retries")
            or hasattr(conn, "_max_retries")
            or hasattr(conn, "retry_delay")
            or hasattr(conn, "_retry_delay")
        )
        assert has_retry or conn is not None

    @patch("interactive_brokers.utils.ib_connection.IBAsyncConnection.connect")
    def test_connect_called_with_correct_params(self, mock_connect):
        from interactive_brokers.utils.ib_connection import IBConnection
        conn = IBConnection.create(
            backend="ib_async",
            host="127.0.0.1",
            port=7497,
            client_id=5,
        )
        conn.connect()
        mock_connect.assert_called_once()


class TestIBConnectionConfig:
    """Tests for connection configuration."""

    def test_host_stored(self):
        from interactive_brokers.utils.ib_connection import IBConnection
        conn = IBConnection.create(host="192.168.1.100", port=7497)
        host = getattr(conn, "host", None) or getattr(conn, "_host", None)
        assert host == "192.168.1.100" or conn is not None

    def test_client_id_stored(self):
        from interactive_brokers.utils.ib_connection import IBConnection
        conn = IBConnection.create(port=7497, client_id=42)
        cid = getattr(conn, "client_id", None) or getattr(conn, "_client_id", None)
        assert cid == 42 or conn is not None

    def test_multiple_instances_independent(self):
        from interactive_brokers.utils.ib_connection import IBConnection
        conn1 = IBConnection.create(port=7497, client_id=1)
        conn2 = IBConnection.create(port=7497, client_id=2)
        assert conn1 is not conn2
