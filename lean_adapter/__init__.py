"""QuantConnect LEAN integration adapter."""
try:
    from lean_adapter.lean_bridge import LEANProjectGenerator
    __all__ = ["LEANProjectGenerator"]
except ImportError:
    __all__ = []
