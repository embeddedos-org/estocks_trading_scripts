"""Multi-channel alert dispatcher for trading notifications.

Sends alerts to Discord, email, SMS, and console. Each channel is
independently wrapped in try/except so one failure never blocks others.
"""

from __future__ import annotations

import json
import logging
import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

import requests

logger = logging.getLogger(__name__)

PRIORITY_COLORS = {
    "INFO": "\033[94m",       # blue
    "WARNING": "\033[93m",    # yellow
    "CRITICAL": "\033[91m",   # red
}
PRIORITY_DISCORD_COLORS = {
    "INFO": 3447003,          # blue
    "WARNING": 16776960,      # yellow
    "CRITICAL": 15158332,     # red
}
RESET_COLOR = "\033[0m"


class AlertDispatcher:
    """Dispatches alerts to all enabled notification channels.

    Args:
        config: The ``notifications`` section of the app config dict.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.enabled_channels: list[str] = config.get("enabled_channels", ["console"])

    def dispatch(self, title: str, message: str, priority: str = "INFO") -> None:
        """Send an alert to every enabled channel.

        Args:
            title: Short summary headline.
            message: Detailed alert body.
            priority: One of ``INFO``, ``WARNING``, ``CRITICAL``.
        """
        priority = priority.upper()
        if priority not in ("INFO", "WARNING", "CRITICAL"):
            priority = "INFO"

        channel_methods = {
            "discord": self._send_discord,
            "email": self._send_email,
            "sms": self._send_sms,
            "console": self._send_console,
        }

        for channel in self.enabled_channels:
            method = channel_methods.get(channel)
            if method is None:
                logger.warning("Unknown notification channel: %s", channel)
                continue
            try:
                method(title, message, priority)
                logger.info("Alert sent via %s: %s", channel, title)
            except Exception:
                logger.exception("Failed to send alert via %s: %s", channel, title)

    # ------------------------------------------------------------------
    # Channel implementations
    # ------------------------------------------------------------------

    def _send_discord(self, title: str, message: str, priority: str) -> None:
        """POST an embed to the configured Discord webhook URL."""
        discord_cfg = self.config.get("discord", {})
        webhook_url = discord_cfg.get("webhook_url", "")
        if not webhook_url:
            logger.warning("Discord webhook URL not configured; skipping.")
            return

        embed = {
            "title": f"[{priority}] {title}",
            "description": message,
            "color": PRIORITY_DISCORD_COLORS.get(priority, 3447003),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        payload = {"embeds": [embed]}
        resp = requests.post(
            webhook_url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()

    def _send_email(self, title: str, message: str, priority: str) -> None:
        """Send an alert email via SMTP."""
        email_cfg = self.config.get("email", {})
        smtp_host = email_cfg.get("smtp_host", "")
        smtp_port = int(email_cfg.get("smtp_port", 587))
        smtp_user = email_cfg.get("smtp_user", "")
        smtp_password = email_cfg.get("smtp_password", "")
        smtp_from = email_cfg.get("smtp_from", "")
        smtp_to = email_cfg.get("smtp_to", "")

        if not all([smtp_host, smtp_user, smtp_password, smtp_from, smtp_to]):
            logger.warning("Email SMTP settings incomplete; skipping.")
            return

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[{priority}] {title}"
        msg["From"] = smtp_from
        msg["To"] = smtp_to

        body = (
            f"Priority: {priority}\n"
            f"Time: {datetime.now(timezone.utc).isoformat()}\n\n"
            f"{message}"
        )
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_from, [smtp_to], msg.as_string())

    def _send_sms(self, title: str, message: str, priority: str) -> None:
        """Send an SMS via the Twilio REST API."""
        sms_cfg = self.config.get("sms", {})
        account_sid = sms_cfg.get("account_sid", "")
        auth_token = sms_cfg.get("auth_token", "")
        from_number = sms_cfg.get("from_number", "")
        to_number = sms_cfg.get("to_number", "")

        if not all([account_sid, auth_token, from_number, to_number]):
            logger.warning("Twilio SMS settings incomplete; skipping.")
            return

        url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
        body_text = f"[{priority}] {title}\n{message}"
        resp = requests.post(
            url,
            data={"From": from_number, "To": to_number, "Body": body_text},
            auth=(account_sid, auth_token),
            timeout=10,
        )
        resp.raise_for_status()

    def _send_console(self, title: str, message: str, priority: str) -> None:
        """Print a formatted alert to the console with color-coded priority."""
        console_cfg = self.config.get("console", {})
        if not console_cfg.get("enabled", True):
            return

        color = PRIORITY_COLORS.get(priority, "")
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(
            f"{color}[{priority}]{RESET_COLOR} {timestamp} | {title}\n"
            f"  {message}"
        )
