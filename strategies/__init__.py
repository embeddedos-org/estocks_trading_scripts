"""
Strategies Package
====================

Unified strategy examples and CLI runner for the stocks_plugin framework.

Provides a registry of all example strategies for use with BacktestEngineV2.

Usage:
    from strategies import STRATEGY_REGISTRY
    strategy_cls = STRATEGY_REGISTRY["trend_following"]
    strategy = strategy_cls()
"""

from __future__ import annotations

from typing import Any, Dict, Type

STRATEGY_REGISTRY: Dict[str, Type[Any]] = {}


def register_strategy(name: str):
    """Decorator to register a strategy class in the global registry."""

    def decorator(cls):
        STRATEGY_REGISTRY[name] = cls
        return cls

    return decorator


def list_strategies() -> Dict[str, str]:
    """Return dict of {name: description} for all registered strategies."""
    return {
        name: (cls.__doc__ or "").strip().split("\n")[0]
        for name, cls in STRATEGY_REGISTRY.items()
    }
