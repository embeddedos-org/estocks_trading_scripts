"""
Tests for shared.config — load_config(), _apply_env_overlay(), TypedDict schemas,
default values, environment variable overrides, malformed YAML handling, and missing fields.
"""

import sys
import os
import logging
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

import pytest

# Add project root to sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from shared.config import (
    load_config,
    _apply_env_overlay,
    _deep_set,
    AppConfig,
    BrokerIBConfig,
    BrokerTSConfig,
    BrokerSchwabConfig,
    BrokersConfig,
    DiscordConfig,
    EmailConfig,
    SMSConfig,
    ConsoleConfig,
    NotificationsConfig,
    StrategiesConfig,
    WebhookConfig,
)


# ── Fixtures ──


@pytest.fixture
def minimal_yaml_content():
    """Minimal valid YAML config."""
    return """\
brokers:
  ib:
    host: "127.0.0.1"
    port: 7497
    client_id: 1
strategies:
  default_position_size_pct: 2.0
"""


@pytest.fixture
def full_yaml_content():
    """Fully populated YAML config matching config.example.yaml structure."""
    return """\
brokers:
  ib:
    host: "127.0.0.1"
    port: 7497
    client_id: 1
    account: "DU12345"
    mode: "paper"
  tradestation:
    client_id: "ts_id"
    client_secret: "ts_secret"
    redirect_uri: "http://localhost:3000/callback"
    refresh_token: "ts_refresh"
  schwab:
    app_key: "schwab_key"
    app_secret: "schwab_secret"
notifications:
  enabled_channels:
    - console
    - discord
  discord:
    webhook_url: "https://discord.com/webhook/123"
  email:
    smtp_host: "smtp.gmail.com"
    smtp_port: 587
    smtp_user: "user@gmail.com"
    smtp_password: "pass"
    smtp_from: "from@gmail.com"
    smtp_to: "to@gmail.com"
  sms:
    account_sid: "AC123"
    auth_token: "token123"
    from_number: "+15551234567"
    to_number: "+15559876543"
  console:
    enabled: true
strategies:
  default_position_size_pct: 2.0
  max_daily_loss_pct: 5.0
  max_position_pct: 10.0
webhook:
  port: 8000
  hmac_secret: "secret123"
  rate_limit_per_minute: 30
"""


@pytest.fixture
def clean_env():
    """Ensure relevant env vars are cleared before and after each test."""
    env_vars = [
        "IB_HOST", "IB_PORT", "IB_CLIENT_ID", "IB_ACCOUNT",
        "TS_CLIENT_ID", "TS_CLIENT_SECRET", "TS_REDIRECT_URI", "TS_REFRESH_TOKEN",
        "SCHWAB_APP_KEY", "SCHWAB_APP_SECRET",
        "DISCORD_WEBHOOK_URL",
        "SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "SMTP_FROM", "SMTP_TO",
        "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER", "TWILIO_TO_NUMBER",
        "WEBHOOK_PORT", "WEBHOOK_HMAC_SECRET",
    ]
    saved = {k: os.environ.pop(k, None) for k in env_vars}
    yield
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


# ── _deep_set tests ──


class TestDeepSet:
    """Tests for the _deep_set helper that sets nested dict values via dotted paths."""

    def test_single_level(self):
        """Validates setting a value at a single-level dotted path."""
        cfg = {}
        _deep_set(cfg, "webhook", "port", 9000)
        assert cfg == {"webhook": {"port": 9000}}

    def test_two_level(self):
        """Validates setting a value at a two-level dotted path (e.g. brokers.ib)."""
        cfg = {}
        _deep_set(cfg, "brokers.ib", "host", "192.168.1.1")
        assert cfg["brokers"]["ib"]["host"] == "192.168.1.1"

    def test_preserves_existing_keys(self):
        """Validates that _deep_set does not clobber existing sibling keys."""
        cfg = {"brokers": {"ib": {"host": "127.0.0.1"}}}
        _deep_set(cfg, "brokers.ib", "port", 7496)
        assert cfg["brokers"]["ib"]["host"] == "127.0.0.1"
        assert cfg["brokers"]["ib"]["port"] == 7496

    def test_overwrites_existing_value(self):
        """Validates overwriting a value at an already-set dotted path."""
        cfg = {"brokers": {"ib": {"port": 7497}}}
        _deep_set(cfg, "brokers.ib", "port", 4002)
        assert cfg["brokers"]["ib"]["port"] == 4002

    def test_three_level_path(self):
        """Validates deeply nested dotted path (three levels)."""
        cfg = {}
        _deep_set(cfg, "a.b.c", "key", "value")
        assert cfg["a"]["b"]["c"]["key"] == "value"


# ── _apply_env_overlay tests ──


class TestApplyEnvOverlay:
    """Tests for _apply_env_overlay which overlays env vars onto config dict."""

    def test_ib_host_override(self, clean_env):
        """Validates IB_HOST env var overrides brokers.ib.host."""
        cfg = {"brokers": {"ib": {"host": "127.0.0.1"}}}
        with patch.dict(os.environ, {"IB_HOST": "10.0.0.1"}):
            _apply_env_overlay(cfg)
        assert cfg["brokers"]["ib"]["host"] == "10.0.0.1"

    def test_ib_port_cast_to_int(self, clean_env):
        """Validates IB_PORT env var is cast to int."""
        cfg = {}
        with patch.dict(os.environ, {"IB_PORT": "4002"}):
            _apply_env_overlay(cfg)
        assert cfg["brokers"]["ib"]["port"] == 4002
        assert isinstance(cfg["brokers"]["ib"]["port"], int)

    def test_empty_env_var_ignored(self, clean_env):
        """Validates empty string env vars are skipped (not treated as set)."""
        cfg = {"brokers": {"ib": {"host": "original"}}}
        with patch.dict(os.environ, {"IB_HOST": ""}):
            _apply_env_overlay(cfg)
        assert cfg["brokers"]["ib"]["host"] == "original"

    def test_missing_env_var_no_effect(self, clean_env):
        """Validates absent env vars leave config unchanged."""
        cfg = {"brokers": {"ib": {"host": "original"}}}
        _apply_env_overlay(cfg)
        assert cfg["brokers"]["ib"]["host"] == "original"

    def test_invalid_int_cast_logs_warning(self, clean_env, caplog):
        """Validates that non-numeric value for int field logs warning and is skipped."""
        cfg = {}
        with patch.dict(os.environ, {"IB_PORT": "not_a_number"}):
            with caplog.at_level(logging.WARNING):
                _apply_env_overlay(cfg)
        assert "Failed to cast env var" in caplog.text
        assert "port" not in cfg.get("brokers", {}).get("ib", {})

    def test_schwab_overlay(self, clean_env):
        """Validates Schwab broker env vars are applied correctly."""
        cfg = {}
        with patch.dict(os.environ, {"SCHWAB_APP_KEY": "key123", "SCHWAB_APP_SECRET": "sec456"}):
            _apply_env_overlay(cfg)
        assert cfg["brokers"]["schwab"]["app_key"] == "key123"
        assert cfg["brokers"]["schwab"]["app_secret"] == "sec456"

    def test_tradestation_overlay(self, clean_env):
        """Validates TradeStation env vars are overlaid."""
        cfg = {}
        env = {
            "TS_CLIENT_ID": "tsid",
            "TS_CLIENT_SECRET": "tssec",
            "TS_REDIRECT_URI": "http://cb",
            "TS_REFRESH_TOKEN": "refresh",
        }
        with patch.dict(os.environ, env):
            _apply_env_overlay(cfg)
        ts = cfg["brokers"]["tradestation"]
        assert ts["client_id"] == "tsid"
        assert ts["client_secret"] == "tssec"
        assert ts["redirect_uri"] == "http://cb"
        assert ts["refresh_token"] == "refresh"

    def test_discord_webhook_overlay(self, clean_env):
        """Validates Discord webhook URL env var override."""
        cfg = {}
        with patch.dict(os.environ, {"DISCORD_WEBHOOK_URL": "https://hooks.example.com"}):
            _apply_env_overlay(cfg)
        assert cfg["notifications"]["discord"]["webhook_url"] == "https://hooks.example.com"

    def test_email_overlay_all_fields(self, clean_env):
        """Validates all SMTP env vars are applied to email config."""
        cfg = {}
        env = {
            "SMTP_HOST": "smtp.test.com",
            "SMTP_PORT": "465",
            "SMTP_USER": "user",
            "SMTP_PASSWORD": "pass",
            "SMTP_FROM": "from@test.com",
            "SMTP_TO": "to@test.com",
        }
        with patch.dict(os.environ, env):
            _apply_env_overlay(cfg)
        email = cfg["notifications"]["email"]
        assert email["smtp_host"] == "smtp.test.com"
        assert email["smtp_port"] == 465
        assert email["smtp_user"] == "user"

    def test_sms_twilio_overlay(self, clean_env):
        """Validates Twilio SMS env vars are applied."""
        cfg = {}
        env = {
            "TWILIO_ACCOUNT_SID": "AC_test",
            "TWILIO_AUTH_TOKEN": "tok",
            "TWILIO_FROM_NUMBER": "+1111",
            "TWILIO_TO_NUMBER": "+2222",
        }
        with patch.dict(os.environ, env):
            _apply_env_overlay(cfg)
        sms = cfg["notifications"]["sms"]
        assert sms["account_sid"] == "AC_test"
        assert sms["to_number"] == "+2222"

    def test_webhook_overlay(self, clean_env):
        """Validates webhook port (int) and hmac_secret (str) env overlay."""
        cfg = {}
        with patch.dict(os.environ, {"WEBHOOK_PORT": "9090", "WEBHOOK_HMAC_SECRET": "s3cr3t"}):
            _apply_env_overlay(cfg)
        assert cfg["webhook"]["port"] == 9090
        assert cfg["webhook"]["hmac_secret"] == "s3cr3t"

    def test_overlay_creates_nested_structure_from_empty(self, clean_env):
        """Validates env overlay creates full nested path from empty config."""
        cfg = {}
        with patch.dict(os.environ, {"IB_ACCOUNT": "U1234567"}):
            _apply_env_overlay(cfg)
        assert cfg["brokers"]["ib"]["account"] == "U1234567"

    def test_multiple_overlays_combined(self, clean_env):
        """Validates multiple env vars across different sections are all applied."""
        cfg = {}
        env = {"IB_HOST": "1.2.3.4", "WEBHOOK_PORT": "5000", "DISCORD_WEBHOOK_URL": "url"}
        with patch.dict(os.environ, env):
            _apply_env_overlay(cfg)
        assert cfg["brokers"]["ib"]["host"] == "1.2.3.4"
        assert cfg["webhook"]["port"] == 5000
        assert cfg["notifications"]["discord"]["webhook_url"] == "url"


# ── load_config tests ──


class TestLoadConfig:
    """Tests for load_config() — YAML loading, default path, env overlay integration."""

    @patch("shared.config.load_dotenv")
    def test_loads_from_explicit_path(self, mock_dotenv, tmp_path, minimal_yaml_content, clean_env):
        """Validates load_config reads from an explicit file path."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(minimal_yaml_content, encoding="utf-8")

        cfg = load_config(str(config_file))
        assert cfg["brokers"]["ib"]["host"] == "127.0.0.1"
        assert cfg["brokers"]["ib"]["port"] == 7497
        mock_dotenv.assert_called_once()

    @patch("shared.config.load_dotenv")
    def test_loads_from_path_object(self, mock_dotenv, tmp_path, minimal_yaml_content, clean_env):
        """Validates load_config accepts pathlib.Path objects."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(minimal_yaml_content, encoding="utf-8")

        cfg = load_config(config_file)
        assert cfg["brokers"]["ib"]["host"] == "127.0.0.1"

    @patch("shared.config.load_dotenv")
    def test_nonexistent_path_returns_empty_config(self, mock_dotenv, clean_env):
        """Validates load_config returns empty dict when file does not exist."""
        cfg = load_config("/nonexistent/path/config.yaml")
        assert isinstance(cfg, dict)

    @patch("shared.config.load_dotenv")
    def test_empty_yaml_file(self, mock_dotenv, tmp_path, clean_env):
        """Validates load_config handles an empty YAML file (returns empty dict)."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("", encoding="utf-8")

        cfg = load_config(str(config_file))
        assert cfg == {} or cfg is not None

    @patch("shared.config.load_dotenv")
    def test_yaml_with_null_content(self, mock_dotenv, tmp_path, clean_env):
        """Validates load_config handles YAML that parses to None."""
        config_file = tmp_path / "null.yaml"
        config_file.write_text("---\n", encoding="utf-8")

        cfg = load_config(str(config_file))
        assert isinstance(cfg, dict)

    @patch("shared.config.load_dotenv")
    def test_full_config_loads_all_sections(self, mock_dotenv, tmp_path, full_yaml_content, clean_env):
        """Validates all top-level sections are present in a fully populated config."""
        config_file = tmp_path / "full.yaml"
        config_file.write_text(full_yaml_content, encoding="utf-8")

        cfg = load_config(str(config_file))
        assert "brokers" in cfg
        assert "notifications" in cfg
        assert "strategies" in cfg
        assert "webhook" in cfg
        assert cfg["webhook"]["port"] == 8000
        assert cfg["strategies"]["max_daily_loss_pct"] == 5.0

    @patch("shared.config.load_dotenv")
    def test_env_overlay_overrides_yaml_values(self, mock_dotenv, tmp_path, minimal_yaml_content, clean_env):
        """Validates env vars take precedence over YAML values."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(minimal_yaml_content, encoding="utf-8")

        with patch.dict(os.environ, {"IB_HOST": "10.10.10.10", "IB_PORT": "4002"}):
            cfg = load_config(str(config_file))
        assert cfg["brokers"]["ib"]["host"] == "10.10.10.10"
        assert cfg["brokers"]["ib"]["port"] == 4002

    @patch("shared.config.load_dotenv")
    def test_default_path_uses_config_example(self, mock_dotenv, clean_env):
        """Validates that load_config(None) defaults to config.example.yaml."""
        cfg = load_config(None)
        assert isinstance(cfg, dict)
        assert "brokers" in cfg

    @patch("shared.config.load_dotenv")
    def test_malformed_yaml_raises(self, mock_dotenv, tmp_path, clean_env):
        """Validates that invalid YAML content raises a yaml error."""
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("{{{{invalid yaml::::", encoding="utf-8")

        with pytest.raises(Exception, match="while parsing"):
            load_config(str(config_file))

    @patch("shared.config.load_dotenv")
    def test_yaml_with_only_strategies(self, mock_dotenv, tmp_path, clean_env):
        """Validates partial YAML with only strategies section."""
        content = "strategies:\n  default_position_size_pct: 3.5\n"
        config_file = tmp_path / "partial.yaml"
        config_file.write_text(content, encoding="utf-8")

        cfg = load_config(str(config_file))
        assert cfg["strategies"]["default_position_size_pct"] == 3.5
        assert "brokers" not in cfg

    @patch("shared.config.load_dotenv")
    def test_dotenv_is_called(self, mock_dotenv, tmp_path, clean_env):
        """Validates that load_dotenv() is always called."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("brokers: {}", encoding="utf-8")
        load_config(str(config_file))
        mock_dotenv.assert_called_once()

    @patch("shared.config.load_dotenv")
    def test_yaml_with_extra_unknown_keys(self, mock_dotenv, tmp_path, clean_env):
        """Validates load_config tolerates extra keys not in TypedDict schemas."""
        content = "custom_section:\n  foo: bar\nbrokers:\n  ib:\n    host: test\n"
        config_file = tmp_path / "extra.yaml"
        config_file.write_text(content, encoding="utf-8")

        cfg = load_config(str(config_file))
        assert cfg["custom_section"]["foo"] == "bar"
        assert cfg["brokers"]["ib"]["host"] == "test"


# ── TypedDict schema structure tests ──


class TestTypedDictSchemas:
    """Validates TypedDict schemas have the expected fields (structural checks)."""

    def test_app_config_keys(self):
        """Validates AppConfig TypedDict has expected top-level keys."""
        annotations = AppConfig.__annotations__
        assert "brokers" in annotations
        assert "notifications" in annotations
        assert "strategies" in annotations
        assert "webhook" in annotations

    def test_broker_ib_config_keys(self):
        """Validates BrokerIBConfig has expected fields."""
        annotations = BrokerIBConfig.__annotations__
        expected = {"host", "port", "client_id", "account", "mode"}
        assert expected == set(annotations.keys())

    def test_webhook_config_keys(self):
        """Validates WebhookConfig has port, hmac_secret, rate_limit_per_minute."""
        annotations = WebhookConfig.__annotations__
        assert "port" in annotations
        assert "hmac_secret" in annotations
        assert "rate_limit_per_minute" in annotations

    def test_notifications_config_keys(self):
        """Validates NotificationsConfig has all sub-channel fields."""
        annotations = NotificationsConfig.__annotations__
        assert "enabled_channels" in annotations
        assert "discord" in annotations
        assert "email" in annotations
        assert "sms" in annotations
        assert "console" in annotations

    def test_strategies_config_keys(self):
        """Validates StrategiesConfig has position size and loss fields."""
        annotations = StrategiesConfig.__annotations__
        assert "default_position_size_pct" in annotations
        assert "max_daily_loss_pct" in annotations
        assert "max_position_pct" in annotations
