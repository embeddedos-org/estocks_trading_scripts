"""TradeStation Account Monitor — periodic balance/position surveillance with alerting."""

import time
import logging
import threading
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class AccountMonitor:
    """Monitors a TradeStation account for balance, position, and risk changes.

    Runs a background thread that periodically checks account state against
    configurable alert thresholds and dispatches notifications via an optional
    notifier (e.g. AlertDispatcher).

    When drawdown or margin thresholds are breached, trading is **blocked**
    until the condition clears or a manual override is issued.
    """

    def __init__(self, order_router, config: dict, notifier=None):
        """Initialize the account monitor.

        Args:
            order_router: A TradeStationOrderRouter instance used for API calls.
            config: Configuration dict with optional keys:
                - margin_warning_pct (float): Margin usage alert threshold (default 80).
                - max_drawdown_pct (float): Max drawdown alert threshold (default 5).
                - position_concentration_pct (float): Single-position concentration alert (default 25).
                - daily_summary_time (str): HH:MM UTC time for daily summary (default "21:00").
                - auto_unblock_after_hours (int): Hours after which a block auto-expires (default 24).
            notifier: Optional object with a send(subject, message) method.
        """
        self._router = order_router
        self._notifier = notifier

        self.margin_warning_pct: float = config.get("margin_warning_pct", 80.0)
        self.max_drawdown_pct: float = config.get("max_drawdown_pct", 5.0)
        self.position_concentration_pct: float = config.get("position_concentration_pct", 25.0)
        self._daily_summary_time: str = config.get("daily_summary_time", "21:00")
        self.auto_unblock_after_hours: int = config.get("auto_unblock_after_hours", 24)

        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._peak_equity: float = 0.0
        self._monitored_account: Optional[str] = None

        self._trading_blocked: bool = False
        self._block_reason: str = ""
        self._blocked_at: Optional[datetime] = None

    # ── Trading Block Gate ───────────────────────────────────────────────

    def is_trading_blocked(self) -> Tuple[bool, str]:
        """Check whether trading is currently blocked.

        Auto-unblocks if the block has been active longer than
        ``auto_unblock_after_hours``.

        Returns:
            Tuple of (is_blocked, reason).  reason is "" when not blocked.
        """
        if self._trading_blocked and self._blocked_at is not None:
            elapsed = datetime.now(timezone.utc) - self._blocked_at
            if elapsed >= timedelta(hours=self.auto_unblock_after_hours):
                logger.info(
                    "Auto-unblocking trading after %d hours (reason was: %s)",
                    self.auto_unblock_after_hours,
                    self._block_reason,
                )
                self._trading_blocked = False
                self._block_reason = ""
                self._blocked_at = None
        return self._trading_blocked, self._block_reason

    def unblock_trading(self) -> None:
        """Manually override and clear the trading block."""
        if self._trading_blocked:
            logger.warning(
                "Manual trading unblock — previous reason: %s", self._block_reason
            )
        self._trading_blocked = False
        self._block_reason = ""
        self._blocked_at = None

    def _block_trading(self, reason: str) -> None:
        """Internal helper to set the trading block."""
        if not self._trading_blocked:
            self._trading_blocked = True
            self._block_reason = reason
            self._blocked_at = datetime.now(timezone.utc)
            logger.critical("TRADING BLOCKED — reason: %s", reason)

    # ── Data Retrieval ──────────────────────────────────────────────────

    def get_balances(self, account_id: str) -> dict:
        """Fetch account balances.

        Returns:
            Dict with keys: cash_balance, equity, market_value, margin_used, margin_available.
        """
        data = self._router._request(
            "GET", f"/brokerage/accounts/{account_id}/balances"
        )
        balances_list = data.get("Balances", [])
        balances = balances_list[0] if balances_list else {}
        return {
            "cash_balance": float(balances.get("CashBalance", 0)),
            "equity": float(balances.get("Equity", 0)),
            "market_value": float(balances.get("MarketValue", 0)),
            "margin_used": float(balances.get("MarginUsed", 0)),
            "margin_available": float(balances.get("MarginAvailable", 0)),
        }

    def get_positions(self, account_id: str) -> list:
        """Fetch all open positions.

        Returns:
            List of dicts, each containing symbol, quantity, avg_price, market_value,
            unrealized_pnl, and pnl_pct.
        """
        data = self._router._request(
            "GET", f"/brokerage/accounts/{account_id}/positions"
        )
        raw_positions = data.get("Positions", [])
        positions = []
        for p in raw_positions:
            avg_price = float(p.get("AveragePrice", 0))
            last = float(p.get("Last", avg_price))
            qty = float(p.get("Quantity", 0))
            unrealized = float(p.get("UnrealizedProfitLoss", (last - avg_price) * qty))
            market_val = float(p.get("MarketValue", last * qty))
            pnl_pct = (unrealized / (avg_price * qty) * 100) if avg_price * qty != 0 else 0
            positions.append({
                "symbol": p.get("Symbol", ""),
                "quantity": qty,
                "avg_price": avg_price,
                "last_price": last,
                "market_value": market_val,
                "unrealized_pnl": unrealized,
                "pnl_pct": round(pnl_pct, 2),
                "asset_type": p.get("AssetType", ""),
            })
        return positions

    def get_orders(self, account_id: str) -> list:
        """Fetch all orders for the account.

        Returns:
            List of order dicts.
        """
        return self._router.get_orders(account_id)

    # ── Alert Checks ────────────────────────────────────────────────────

    def _check_margin(self, balances: dict):
        """Block trading if margin usage exceeds the configured threshold."""
        equity = balances.get("equity", 0)
        margin_used = balances.get("margin_used", 0)
        if equity > 0:
            usage_pct = (margin_used / equity) * 100
            if usage_pct >= self.margin_warning_pct:
                msg = (
                    f"⚠️ Margin usage at {usage_pct:.1f}% "
                    f"(threshold {self.margin_warning_pct:.0f}%). "
                    f"Margin used: ${margin_used:,.2f}, Equity: ${equity:,.2f}"
                )
                logger.warning(msg)
                self._send_alert("Margin Warning — TRADING BLOCKED", msg)
                self._block_trading("margin exceeded")

    def _check_drawdown(self, balances: dict):
        """Block trading if drawdown from peak equity exceeds the configured threshold."""
        equity = balances.get("equity", 0)
        if equity > self._peak_equity:
            self._peak_equity = equity

        if self._peak_equity > 0:
            drawdown_pct = ((self._peak_equity - equity) / self._peak_equity) * 100
            if drawdown_pct >= self.max_drawdown_pct:
                msg = (
                    f"🔻 Drawdown alert: {drawdown_pct:.2f}% from peak "
                    f"(threshold {self.max_drawdown_pct:.0f}%). "
                    f"Peak: ${self._peak_equity:,.2f}, Current: ${equity:,.2f}"
                )
                logger.warning(msg)
                self._send_alert("Drawdown Alert — TRADING BLOCKED", msg)
                self._block_trading("drawdown exceeded")

    def _check_concentration(self, positions: list, equity: float):
        """Alert if any single position exceeds the concentration threshold."""
        if equity <= 0:
            return
        for pos in positions:
            mkt_val = abs(pos.get("market_value", 0))
            concentration = (mkt_val / equity) * 100
            if concentration >= self.position_concentration_pct:
                msg = (
                    f"📊 Position concentration alert: {pos['symbol']} is "
                    f"{concentration:.1f}% of equity "
                    f"(threshold {self.position_concentration_pct:.0f}%). "
                    f"Value: ${mkt_val:,.2f}"
                )
                logger.warning(msg)
                self._send_alert("Concentration Alert", msg)

    def _send_alert(self, subject: str, message: str):
        """Dispatch an alert through the notifier if available."""
        if self._notifier and hasattr(self._notifier, "send"):
            try:
                self._notifier.send(subject, message)
            except Exception as exc:
                logger.error("Failed to send alert '%s': %s", subject, exc)

    # ── Background Monitoring ───────────────────────────────────────────

    def start_monitoring(self, account_id: str, interval_seconds: int = 60):
        """Start the background monitoring loop.

        Args:
            account_id: The TradeStation account to monitor.
            interval_seconds: Polling interval in seconds (default 60).
        """
        if self._monitor_thread and self._monitor_thread.is_alive():
            logger.warning("Monitoring already active")
            return

        self._monitored_account = account_id
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(account_id, interval_seconds),
            daemon=True,
            name="TradeStationMonitor",
        )
        self._monitor_thread.start()
        logger.info(
            "Account monitoring started for %s (interval=%ds)",
            account_id,
            interval_seconds,
        )

    def stop_monitoring(self):
        """Signal the background monitoring thread to stop and wait for it."""
        self._stop_event.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=10)
            logger.info("Account monitoring stopped")
        self._monitor_thread = None

    def _monitor_loop(self, account_id: str, interval_seconds: int):
        """Main monitoring loop — runs in a background thread."""
        last_summary_date: Optional[str] = None

        while not self._stop_event.is_set():
            try:
                balances = self.get_balances(account_id)
                positions = self.get_positions(account_id)
                equity = balances.get("equity", 0)

                self._check_margin(balances)
                self._check_drawdown(balances)
                self._check_concentration(positions, equity)

                last_summary_date = self._schedule_daily_summary(
                    account_id, last_summary_date
                )

            except Exception as exc:
                logger.error("Monitor loop error: %s", exc)

            self._stop_event.wait(interval_seconds)

    def _schedule_daily_summary(
        self, account_id: str, last_summary_date: Optional[str]
    ) -> Optional[str]:
        """Send a daily summary at the configured time (once per day).

        Returns:
            The date string of the last summary sent.
        """
        now = datetime.now(timezone.utc)
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M")

        if current_time >= self._daily_summary_time and last_summary_date != current_date:
            try:
                summary = self.generate_daily_summary(account_id)
                self._send_alert("Daily Account Summary", summary)
                logger.info("Daily summary sent for %s", current_date)
                return current_date
            except Exception as exc:
                logger.error("Failed to generate daily summary: %s", exc)

        return last_summary_date

    # ── Daily Summary ───────────────────────────────────────────────────

    def generate_daily_summary(self, account_id: str) -> str:
        """Generate a formatted daily account summary.

        Args:
            account_id: The TradeStation account ID.

        Returns:
            Multi-line formatted string with account overview.
        """
        balances = self.get_balances(account_id)
        positions = self.get_positions(account_id)
        orders = self.get_orders(account_id)

        equity = balances.get("equity", 0)
        cash = balances.get("cash_balance", 0)
        margin_used = balances.get("margin_used", 0)

        total_unrealized = sum(p.get("unrealized_pnl", 0) for p in positions)
        open_order_count = len([o for o in orders if o.get("Status", "") in ("Queued", "Sent", "Open")])

        drawdown_pct = 0.0
        if self._peak_equity > 0:
            drawdown_pct = ((self._peak_equity - equity) / self._peak_equity) * 100

        margin_usage_pct = (margin_used / equity * 100) if equity > 0 else 0

        blocked, block_reason = self.is_trading_blocked()

        lines = [
            "═══════════════════════════════════════════",
            f"  DAILY ACCOUNT SUMMARY — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            f"  Account: {account_id}",
            f"  Trading Blocked: {'YES — ' + block_reason if blocked else 'No'}",
            "═══════════════════════════════════════════",
            "",
            f"  Equity:           ${equity:>14,.2f}",
            f"  Cash Balance:     ${cash:>14,.2f}",
            f"  Margin Used:      ${margin_used:>14,.2f}  ({margin_usage_pct:.1f}%)",
            f"  Peak Equity:      ${self._peak_equity:>14,.2f}",
            f"  Current Drawdown: {drawdown_pct:>14.2f}%",
            "",
            f"  Open Positions:   {len(positions):>14d}",
            f"  Unrealized P&L:   ${total_unrealized:>14,.2f}",
            f"  Pending Orders:   {open_order_count:>14d}",
            "",
        ]

        if positions:
            lines.append("  ── Positions ──────────────────────────────")
            lines.append(f"  {'Symbol':<8} {'Qty':>8} {'AvgPx':>10} {'Last':>10} {'P&L':>12} {'%':>7}")
            lines.append("  " + "─" * 57)
            for p in sorted(positions, key=lambda x: abs(x.get("unrealized_pnl", 0)), reverse=True):
                lines.append(
                    f"  {p['symbol']:<8} {p['quantity']:>8.0f} "
                    f"{p['avg_price']:>10.2f} {p['last_price']:>10.2f} "
                    f"${p['unrealized_pnl']:>11,.2f} {p['pnl_pct']:>6.1f}%"
                )
            lines.append("")

        lines.append("═══════════════════════════════════════════")
        return "\n".join(lines)
