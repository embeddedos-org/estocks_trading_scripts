"""Configuration loader for stocks_plugin.

Loads YAML config files with environment variable overlay via python-dotenv.
Environment variables always take precedence over YAML values.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, TypedDict

import yaml
from dotenv import load_dotenv


class BrokerIBConfig(TypedDict, total=False):
    host: str
    port: int
    client_id: int
    account: str
    mode: str


class BrokerTSConfig(TypedDict, total=False):
    client_id: str
    client_secret: str
    redirect_uri: str
    refresh_token: str


class BrokerSchwabConfig(TypedDict, total=False):
    app_key: str
    app_secret: str


class BrokersConfig(TypedDict, total=False):
    ib: BrokerIBConfig
    tradestation: BrokerTSConfig
    schwab: BrokerSchwabConfig


class DiscordConfig(TypedDict, total=False):
    webhook_url: str


class EmailConfig(TypedDict, total=False):
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    smtp_from: str
    smtp_to: str


class SMSConfig(TypedDict, total=False):
    account_sid: str
    auth_token: str
    from_number: str
    to_number: str


class ConsoleConfig(TypedDict, total=False):
    enabled: bool


class NotificationsConfig(TypedDict, total=False):
    enabled_channels: list[str]
    discord: DiscordConfig
    email: EmailConfig
    sms: SMSConfig
    console: ConsoleConfig


class StrategiesConfig(TypedDict, total=False):
    default_position_size_pct: float
    max_daily_loss_pct: float
    max_position_pct: float


class WebhookConfig(TypedDict, total=False):
    port: int
    hmac_secret: str
    rate_limit_per_minute: int


class AppConfig(TypedDict, total=False):
    brokers: BrokersConfig
    notifications: NotificationsConfig
    strategies: StrategiesConfig
    webhook: WebhookConfig


_ENV_OVERLAY: dict[str, list[tuple[str, str, type]]] = {
    "brokers.ib": [
        ("IB_HOST", "host", str),
        ("IB_PORT", "port", int),
        ("IB_CLIENT_ID", "client_id", int),
        ("IB_ACCOUNT", "account", str),
    ],
    "brokers.tradestation": [
        ("TS_CLIENT_ID", "client_id", str),
        ("TS_CLIENT_SECRET", "client_secret", str),
        ("TS_REDIRECT_URI", "redirect_uri", str),
        ("TS_REFRESH_TOKEN", "refresh_token", str),
    ],
    "brokers.schwab": [
        ("SCHWAB_APP_KEY", "app_key", str),
        ("SCHWAB_APP_SECRET", "app_secret", str),
    ],
    "notifications.discord": [
        ("DISCORD_WEBHOOK_URL", "webhook_url", str),
    ],
    "notifications.email": [
        ("SMTP_HOST", "smtp_host", str),
        ("SMTP_PORT", "smtp_port", int),
        ("SMTP_USER", "smtp_user", str),
        ("SMTP_PASSWORD", "smtp_password", str),
        ("SMTP_FROM", "smtp_from", str),
        ("SMTP_TO", "smtp_to", str),
    ],
    "notifications.sms": [
        ("TWILIO_ACCOUNT_SID", "account_sid", str),
        ("TWILIO_AUTH_TOKEN", "auth_token", str),
        ("TWILIO_FROM_NUMBER", "from_number", str),
        ("TWILIO_TO_NUMBER", "to_number", str),
    ],
    "webhook": [
        ("WEBHOOK_PORT", "port", int),
        ("WEBHOOK_HMAC_SECRET", "hmac_secret", str),
    ],
}


def _deep_set(cfg: dict, dotted_key: str, field: str, value: Any) -> None:
    """Set a nested dict value using a dotted path."""
    parts = dotted_key.split(".")
    node = cfg
    for part in parts:
        node = node.setdefault(part, {})
    node[field] = value


def _apply_env_overlay(cfg: dict) -> None:
    """Overlay environment variables onto the config dict.

    Environment variables take precedence over YAML values.
    """
    for dotted_path, mappings in _ENV_OVERLAY.items():
        for env_var, field_name, cast_type in mappings:
            raw = os.environ.get(env_var)
            if raw is not None and raw != "":
                try:
                    _deep_set(cfg, dotted_path, field_name, cast_type(raw))
                except (ValueError, TypeError) as e:
                    logging.getLogger(__name__).warning(
                        "Failed to cast env var %s=%s: %s", env_var, raw, e
                    )


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load application configuration from YAML with env-var overlay.

    Args:
        path: Path to a YAML config file. Defaults to
              ``shared/config/config.example.yaml`` relative to this module.

    Returns:
        An AppConfig typed dict with sections: brokers, notifications,
        strategies, webhook.
    """
    load_dotenv()

    if path is None:
        path = Path(__file__).parent / "config.example.yaml"
    else:
        path = Path(path)

    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            cfg: dict = yaml.safe_load(fh) or {}
    else:
        cfg = {}

    _apply_env_overlay(cfg)

    # FIX 12: Validate config schema
    _validate_config(cfg)

    return cfg  # type: ignore[return-value]


# ─── FIX 12: Config Schema Validation ───

_REQUIRED_KEYS = {
    "brokers": "Broker configuration is required (ib, tradestation, or schwab)",
}

_KNOWN_TOP_KEYS = {
    "brokers", "notifications", "strategies", "webhook",
    "symbols", "risk_management", "logging", "server",
    "health_monitor", "broker_configs", "broker_routing",
    "security", "rate_limiting",
}


def _validate_config(config: dict) -> None:
    """Validate loaded config for required keys and warn on unknowns.

    Args:
        config: The loaded configuration dictionary.
    """
    log = logging.getLogger(__name__)

    # Check required keys
    for key, message in _REQUIRED_KEYS.items():
        if key not in config:
            log.warning("Config validation: missing required key '%s' — %s", key, message)

    # Warn on unknown top-level keys (possible typos)
    for key in config:
        if key not in _KNOWN_TOP_KEYS:
            log.warning(
                "Config validation: unknown top-level key '%s' — possible typo? "
                "Known keys: %s",
                key, ", ".join(sorted(_KNOWN_TOP_KEYS)),
            )

    # Validate IB port matches mode if both are specified
    ib_config = config.get("brokers", {}).get("ib", {})
    if ib_config:
        port = ib_config.get("port")
        mode = ib_config.get("mode", "")
        if port and mode:
            if mode == "paper" and port not in (7497, 4002):
                log.warning(
                    "Config validation: IB mode='paper' but port=%d "
                    "(expected 7497 or 4002)",
                    port,
                )
            elif mode == "live" and port not in (7496, 4001):
                log.warning(
                    "Config validation: IB mode='live' but port=%d "
                    "(expected 7496 or 4001)",
                    port,
                )
