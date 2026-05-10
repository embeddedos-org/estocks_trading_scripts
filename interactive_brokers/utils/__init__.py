"""Interactive Brokers utility modules for connection and order management."""

from interactive_brokers.utils.ib_connection import IBConnection, IBAsyncConnection, IBApiConnection
from interactive_brokers.utils.order_manager import OrderManager, OrderStatus

# Legacy alias for backward compatibility
IBInsyncConnection = IBAsyncConnection

__all__ = [
    "IBConnection",
    "IBAsyncConnection",
    "IBInsyncConnection",  # legacy alias
    "IBApiConnection",
    "OrderManager",
    "OrderStatus",
]
