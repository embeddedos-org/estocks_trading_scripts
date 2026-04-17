"""
Python Charting & Visualization
==================================

Provides candlestick charts, equity curves, drawdown analysis,
monthly return heatmaps, indicator panels, trade PnL analysis,
and a comprehensive dashboard — all from backtest results.

Uses mplfinance for candlestick charts with matplotlib fallback.

Usage:
    from shared.visualization.chart_renderer import ChartRenderer
    ChartRenderer.plot_equity_curve(result, save_path="equity.png")
    ChartRenderer.plot_candlestick(df, indicators=["sma_20", "bb"])
    ChartRenderer.dashboard(result, save_path="dashboard.png")
"""

from __future__ import annotations

import calendar
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import FuncFormatter
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    logger.debug("matplotlib not installed — charting unavailable")

try:
    import mplfinance as mpf  # type: ignore[import-untyped]
    _HAS_MPF = True
except ImportError:
    _HAS_MPF = False
    logger.debug("mplfinance not installed — candlestick charts will use basic matplotlib")

try:
    import seaborn as sns  # type: ignore[import-untyped]
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False
    logger.debug("seaborn not installed — heatmaps will use basic matplotlib")


def _require_matplotlib() -> None:
    if not _HAS_MPL:
        raise ImportError(
            "matplotlib is required for charting. "
            "Install with: pip install matplotlib"
        )


def _save_or_show(fig: Any, save_path: Optional[str]) -> None:
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Chart saved to %s", save_path)
        plt.close(fig)
    else:
        plt.show()


class ChartRenderer:
    """Static-method library for trading visualizations.

    All methods accept BacktestResultV2 objects or raw DataFrames
    and produce publication-quality charts.
    """

    @staticmethod
    def plot_candlestick(
        df: pd.DataFrame,
        indicators: Optional[List[str]] = None,
        trades: Optional[List[Dict[str, Any]]] = None,
        title: str = "Candlestick Chart",
        save_path: Optional[str] = None,
        volume: bool = True,
    ) -> None:
        """Plot OHLCV candlestick chart with optional indicator overlays and trade markers.

        Args:
            df: DataFrame with columns: date/open/high/low/close/volume.
                If index is DatetimeIndex, uses it directly.
            indicators: List of indicator names to overlay (e.g. ["sma_20", "ema_9", "bb"]).
            trades: List of trade dicts with 'date', 'type' (BUY/SELL), 'price' keys.
            title: Chart title.
            save_path: If provided, saves to file instead of displaying.
            volume: Whether to show volume subplot.
        """
        _require_matplotlib()

        ohlcv = df.copy()
        ohlcv.columns = [c.strip().lower() for c in ohlcv.columns]

        if not isinstance(ohlcv.index, pd.DatetimeIndex):
            if "date" in ohlcv.columns:
                ohlcv["date"] = pd.to_datetime(ohlcv["date"])
                ohlcv.set_index("date", inplace=True)
            elif "datetime" in ohlcv.columns:
                ohlcv["datetime"] = pd.to_datetime(ohlcv["datetime"])
                ohlcv.set_index("datetime", inplace=True)

        add_plots = []
        if indicators:
            from shared.indicators.technical_indicators import TechnicalIndicators as TI

            for ind in indicators:
                ind_lower = ind.lower()
                if ind_lower.startswith("sma"):
                    length = int(ind_lower.replace("sma_", "").replace("sma", "20"))
                    series = TI.sma(ohlcv["close"], length)
                    add_plots.append(mpf.make_addplot(series, label=f"SMA({length})") if _HAS_MPF else None)
                elif ind_lower.startswith("ema"):
                    length = int(ind_lower.replace("ema_", "").replace("ema", "20"))
                    series = TI.ema(ohlcv["close"], length)
                    add_plots.append(mpf.make_addplot(series, label=f"EMA({length})") if _HAS_MPF else None)
                elif ind_lower == "bb":
                    bb = TI.bbands(ohlcv["close"])
                    if _HAS_MPF:
                        add_plots.append(mpf.make_addplot(bb["BBU"], color="gray", linestyle="--"))
                        add_plots.append(mpf.make_addplot(bb["BBL"], color="gray", linestyle="--"))
                        add_plots.append(mpf.make_addplot(bb["BBM"], color="orange", linestyle=":"))

            add_plots = [p for p in add_plots if p is not None]

        if trades and _HAS_MPF:
            buy_markers = pd.Series(np.nan, index=ohlcv.index)
            sell_markers = pd.Series(np.nan, index=ohlcv.index)

            for trade in trades:
                trade_date = pd.to_datetime(trade.get("date"))
                trade_type = trade.get("type", "")
                trade_price = trade.get("price", 0)

                if trade_date in ohlcv.index:
                    if "BUY" in trade_type.upper():
                        buy_markers.loc[trade_date] = trade_price
                    elif "SELL" in trade_type.upper() or "SHORT" in trade_type.upper():
                        sell_markers.loc[trade_date] = trade_price

            if buy_markers.notna().any():
                add_plots.append(mpf.make_addplot(
                    buy_markers, type="scatter", marker="^", markersize=80, color="green"
                ))
            if sell_markers.notna().any():
                add_plots.append(mpf.make_addplot(
                    sell_markers, type="scatter", marker="v", markersize=80, color="red"
                ))

        if _HAS_MPF:
            kwargs: Dict[str, Any] = {
                "type": "candle",
                "title": title,
                "style": "charles",
                "volume": volume and "volume" in ohlcv.columns,
                "figsize": (14, 8),
                "returnfig": True,
            }
            if add_plots:
                kwargs["addplot"] = add_plots
            if save_path:
                kwargs["savefig"] = save_path

            fig, axes = mpf.plot(ohlcv, **kwargs)
            if not save_path:
                plt.show()
            else:
                plt.close(fig)
                logger.info("Candlestick chart saved to %s", save_path)
        else:
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.plot(ohlcv.index, ohlcv["close"], color="black", linewidth=1, label="Close")
            ax.set_title(title)
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            _save_or_show(fig, save_path)

    @staticmethod
    def plot_equity_curve(
        result: Any,
        benchmark: Optional[List[float]] = None,
        save_path: Optional[str] = None,
        title: str = "Equity Curve",
    ) -> None:
        """Plot equity curve with drawdown shading and optional benchmark overlay.

        Args:
            result: BacktestResultV2 with equity_curve list.
            benchmark: Optional benchmark equity curve for comparison.
            save_path: If provided, saves to file.
            title: Chart title.
        """
        _require_matplotlib()

        equity = np.array(result.equity_curve, dtype=float)
        if len(equity) == 0:
            logger.warning("Empty equity curve — nothing to plot")
            return

        x = np.arange(len(equity))

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 8), height_ratios=[3, 1], sharex=True
        )

        ax1.plot(x, equity, color="steelblue", linewidth=1.5, label="Strategy")

        peak = np.maximum.accumulate(equity)
        ax1.fill_between(x, equity, peak, alpha=0.15, color="red", label="Drawdown")

        if benchmark is not None:
            bench = np.array(benchmark, dtype=float)
            if len(bench) == len(equity):
                ax1.plot(x, bench, color="gray", linewidth=1, alpha=0.7, label="Benchmark")

        dd_idx = np.argmax(peak - equity)
        if dd_idx > 0:
            ax1.annotate(
                f"Max DD: {(equity[dd_idx] - peak[dd_idx]) / peak[dd_idx] * 100:.1f}%",
                xy=(dd_idx, equity[dd_idx]),
                fontsize=9, color="red",
                arrowprops=dict(arrowstyle="->", color="red"),
                xytext=(dd_idx + len(equity) * 0.05, equity[dd_idx] + (peak[0] - equity[dd_idx]) * 0.3),
            )

        ax1.set_title(title, fontsize=14)
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        drawdown = (equity - peak) / peak * 100
        ax2.fill_between(x, drawdown, 0, color="red", alpha=0.4)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Bars")
        ax2.grid(True, alpha=0.3)

        metrics_text = (
            f"Return: {result.total_return * 100:.2f}%  |  "
            f"Sharpe: {result.sharpe_ratio:.2f}  |  "
            f"Max DD: {result.max_drawdown * 100:.2f}%  |  "
            f"Trades: {result.total_trades}"
        )
        fig.text(0.5, 0.01, metrics_text, ha="center", fontsize=10, style="italic")

        plt.tight_layout()
        _save_or_show(fig, save_path)

    @staticmethod
    def plot_drawdown(
        result: Any,
        save_path: Optional[str] = None,
        title: str = "Drawdown Analysis",
    ) -> None:
        """Plot standalone drawdown chart with max drawdown period highlighted.

        Args:
            result: BacktestResultV2 with equity_curve list.
            save_path: If provided, saves to file.
            title: Chart title.
        """
        _require_matplotlib()

        equity = np.array(result.equity_curve, dtype=float)
        if len(equity) == 0:
            return

        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak * 100

        fig, ax = plt.subplots(figsize=(14, 5))

        ax.fill_between(range(len(drawdown)), drawdown, 0, color="red", alpha=0.4)
        ax.plot(drawdown, color="darkred", linewidth=1)

        max_dd_idx = np.argmin(drawdown)
        peak_before_dd = np.argmax(equity[:max_dd_idx + 1]) if max_dd_idx > 0 else 0
        ax.axvspan(peak_before_dd, max_dd_idx, alpha=0.1, color="red", label="Max DD Period")

        ax.set_title(title, fontsize=14)
        ax.set_ylabel("Drawdown (%)")
        ax.set_xlabel("Bars")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        _save_or_show(fig, save_path)

    @staticmethod
    def plot_monthly_returns(
        result: Any,
        save_path: Optional[str] = None,
        title: str = "Monthly Returns Heatmap",
    ) -> None:
        """Plot monthly returns heatmap with rows=years, columns=months.

        Args:
            result: BacktestResultV2 with equity_curve list.
            save_path: If provided, saves to file.
            title: Chart title.
        """
        _require_matplotlib()

        equity = result.equity_curve
        if len(equity) < 22:
            logger.warning("Not enough data for monthly returns")
            return

        returns_daily = np.diff(equity) / np.array(equity[:-1])

        n_months = max(1, len(returns_daily) // 21)
        monthly_returns = []
        for i in range(n_months):
            start = i * 21
            end = min((i + 1) * 21, len(returns_daily))
            month_ret = np.prod(1 + returns_daily[start:end]) - 1
            monthly_returns.append(month_ret * 100)

        n_years = max(1, (n_months + 11) // 12)
        data = np.full((n_years, 12), np.nan)

        for i, ret in enumerate(monthly_returns):
            year_idx = i // 12
            month_idx = i % 12
            if year_idx < n_years:
                data[year_idx, month_idx] = ret

        fig, ax = plt.subplots(figsize=(12, max(3, n_years + 1)))

        month_labels = [calendar.month_abbr[m] for m in range(1, 13)]
        year_labels = [f"Year {y + 1}" for y in range(n_years)]

        if _HAS_SNS:
            sns.heatmap(
                data, ax=ax, annot=True, fmt=".1f", center=0,
                cmap="RdYlGn", linewidths=0.5,
                xticklabels=month_labels, yticklabels=year_labels,
                cbar_kws={"label": "Return (%)"},
                mask=np.isnan(data),
            )
        else:
            im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-5, vmax=5)
            ax.set_xticks(range(12))
            ax.set_xticklabels(month_labels)
            ax.set_yticks(range(n_years))
            ax.set_yticklabels(year_labels)
            plt.colorbar(im, ax=ax, label="Return (%)")

            for i in range(n_years):
                for j in range(12):
                    if not np.isnan(data[i, j]):
                        ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", fontsize=8)

        ax.set_title(title, fontsize=14)
        plt.tight_layout()
        _save_or_show(fig, save_path)

    @staticmethod
    def plot_indicators(
        df: pd.DataFrame,
        indicators: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        title: str = "Technical Indicators",
    ) -> None:
        """Plot multi-panel chart with price on top and indicators below.

        Args:
            df: OHLCV DataFrame.
            indicators: List of indicator names (default: ["rsi", "macd", "stochastic"]).
            save_path: If provided, saves to file.
            title: Chart title.
        """
        _require_matplotlib()
        from shared.indicators.technical_indicators import TechnicalIndicators as TI

        if indicators is None:
            indicators = ["rsi", "macd", "stochastic"]

        n_panels = 1 + len(indicators)
        fig, axes = plt.subplots(
            n_panels, 1, figsize=(14, 3 * n_panels), sharex=True,
            gridspec_kw={"height_ratios": [2] + [1] * len(indicators)},
        )
        if n_panels == 1:
            axes = [axes]

        ohlcv = df.copy()
        ohlcv.columns = [c.strip().lower() for c in ohlcv.columns]
        close = ohlcv["close"]
        x = range(len(close))

        axes[0].plot(x, close, color="black", linewidth=1)
        axes[0].set_title(title, fontsize=14)
        axes[0].set_ylabel("Price")
        axes[0].grid(True, alpha=0.3)

        for i, ind in enumerate(indicators):
            ax = axes[i + 1]
            ind_lower = ind.lower()

            if ind_lower == "rsi":
                rsi = TI.rsi(close, 14)
                ax.plot(x, rsi, color="purple", linewidth=1)
                ax.axhline(70, color="red", linestyle="--", alpha=0.5)
                ax.axhline(30, color="green", linestyle="--", alpha=0.5)
                ax.fill_between(x, 30, 70, alpha=0.05, color="gray")
                ax.set_ylabel("RSI (14)")
                ax.set_ylim(0, 100)

            elif ind_lower == "macd":
                macd_line, signal_line, hist = TI.macd(close)
                ax.plot(x, macd_line, color="blue", linewidth=1, label="MACD")
                ax.plot(x, signal_line, color="red", linewidth=1, label="Signal")
                colors = ["green" if h >= 0 else "red" for h in hist]
                ax.bar(x, hist, color=colors, alpha=0.5, width=1)
                ax.axhline(0, color="black", linewidth=0.5)
                ax.set_ylabel("MACD")
                ax.legend(fontsize=8)

            elif ind_lower == "stochastic":
                k, d = TI.stochastic(ohlcv)
                ax.plot(x, k, color="blue", linewidth=1, label="%K")
                ax.plot(x, d, color="red", linewidth=1, label="%D")
                ax.axhline(80, color="red", linestyle="--", alpha=0.5)
                ax.axhline(20, color="green", linestyle="--", alpha=0.5)
                ax.set_ylabel("Stochastic")
                ax.set_ylim(0, 100)
                ax.legend(fontsize=8)

            elif ind_lower == "adx":
                adx_val, plus_di, minus_di = TI.adx(ohlcv)
                ax.plot(x, adx_val, color="black", linewidth=1.5, label="ADX")
                ax.plot(x, plus_di, color="green", linewidth=1, label="+DI")
                ax.plot(x, minus_di, color="red", linewidth=1, label="-DI")
                ax.axhline(25, color="gray", linestyle="--", alpha=0.5)
                ax.set_ylabel("ADX")
                ax.legend(fontsize=8)

            elif ind_lower == "atr":
                atr_val = TI.atr(ohlcv, 14)
                ax.plot(x, atr_val, color="orange", linewidth=1)
                ax.set_ylabel("ATR (14)")

            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Bars")
        plt.tight_layout()
        _save_or_show(fig, save_path)

    @staticmethod
    def plot_trade_analysis(
        result: Any,
        save_path: Optional[str] = None,
        title: str = "Trade Analysis",
    ) -> None:
        """Plot trade PnL distribution, hold duration vs return, and MAE/MFE scatter.

        Args:
            result: BacktestResultV2 with trades list.
            save_path: If provided, saves to file.
            title: Chart title.
        """
        _require_matplotlib()

        trades = getattr(result, "trades", [])
        if not trades:
            logger.warning("No trades to analyze")
            return

        pnls = [t.pnl for t in trades]
        pnl_pcts = [t.pnl_pct * 100 for t in trades]
        hold_bars = [t.hold_bars for t in trades]
        maes = [t.mae * 100 for t in trades]
        mfes = [t.mfe * 100 for t in trades]

        has_mae_mfe = any(m != 0 for m in maes) or any(m != 0 for m in mfes)
        n_cols = 3 if has_mae_mfe else 2

        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
        fig.suptitle(title, fontsize=14)

        colors = ["green" if p > 0 else "red" for p in pnls]
        axes[0].bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
        axes[0].axhline(0, color="black", linewidth=0.5)
        axes[0].set_title("Trade PnL Distribution")
        axes[0].set_xlabel("Trade #")
        axes[0].set_ylabel("PnL ($)")

        win_colors = ["green" if p > 0 else "red" for p in pnl_pcts]
        axes[1].scatter(hold_bars, pnl_pcts, c=win_colors, alpha=0.6, edgecolors="gray", s=40)
        axes[1].axhline(0, color="black", linewidth=0.5)
        axes[1].set_title("Hold Duration vs Return")
        axes[1].set_xlabel("Hold Duration (bars)")
        axes[1].set_ylabel("Return (%)")

        if has_mae_mfe and n_cols == 3:
            axes[2].scatter(maes, mfes, c=win_colors, alpha=0.6, edgecolors="gray", s=40)
            axes[2].axhline(0, color="black", linewidth=0.5)
            axes[2].axvline(0, color="black", linewidth=0.5)
            max_val = max(max(abs(v) for v in maes + mfes), 1)
            axes[2].plot([-max_val, max_val], [-max_val, max_val], "k--", alpha=0.3)
            axes[2].set_title("MAE vs MFE")
            axes[2].set_xlabel("Max Adverse Excursion (%)")
            axes[2].set_ylabel("Max Favorable Excursion (%)")

        for ax in axes:
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        _save_or_show(fig, save_path)

    @staticmethod
    def dashboard(
        result: Any,
        save_path: Optional[str] = None,
        title: str = "Backtest Dashboard",
    ) -> None:
        """Comprehensive 2×3 dashboard combining all key visualizations.

        Panels: equity curve, drawdown, monthly returns, trade PnL histogram,
        key metrics table, win/loss streaks.

        Args:
            result: BacktestResultV2 with full metrics.
            save_path: If provided, saves to file.
            title: Chart title.
        """
        _require_matplotlib()

        equity = np.array(result.equity_curve, dtype=float)
        if len(equity) == 0:
            logger.warning("Empty equity curve — cannot render dashboard")
            return

        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # Panel 1: Equity Curve
        ax1 = fig.add_subplot(gs[0, 0])
        x = np.arange(len(equity))
        peak = np.maximum.accumulate(equity)
        ax1.plot(x, equity, color="steelblue", linewidth=1.2)
        ax1.fill_between(x, equity, peak, alpha=0.15, color="red")
        ax1.set_title("Equity Curve")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.grid(True, alpha=0.3)

        # Panel 2: Drawdown
        ax2 = fig.add_subplot(gs[0, 1])
        drawdown = (equity - peak) / peak * 100
        ax2.fill_between(x, drawdown, 0, color="red", alpha=0.4)
        ax2.set_title("Drawdown")
        ax2.set_ylabel("Drawdown (%)")
        ax2.grid(True, alpha=0.3)

        # Panel 3: Monthly Returns (mini heatmap)
        ax3 = fig.add_subplot(gs[0, 2])
        if len(equity) >= 22:
            returns_daily = np.diff(equity) / np.array(equity[:-1])
            n_months = max(1, len(returns_daily) // 21)
            monthly_rets = []
            for i in range(min(n_months, 36)):
                start = i * 21
                end = min((i + 1) * 21, len(returns_daily))
                month_ret = (np.prod(1 + returns_daily[start:end]) - 1) * 100
                monthly_rets.append(month_ret)
            colors = ["green" if r > 0 else "red" for r in monthly_rets]
            ax3.bar(range(len(monthly_rets)), monthly_rets, color=colors, alpha=0.7)
            ax3.axhline(0, color="black", linewidth=0.5)
        ax3.set_title("Monthly Returns")
        ax3.set_ylabel("Return (%)")
        ax3.grid(True, alpha=0.3)

        # Panel 4: Trade PnL Histogram
        ax4 = fig.add_subplot(gs[1, 0])
        trades = getattr(result, "trades", [])
        if trades:
            pnls = [t.pnl for t in trades]
            n_bins = min(50, max(10, len(pnls) // 3))
            ax4.hist(pnls, bins=n_bins, color="steelblue", alpha=0.7, edgecolor="black")
            ax4.axvline(0, color="red", linewidth=1)
            ax4.axvline(np.mean(pnls), color="green", linewidth=1.5, linestyle="--", label=f"Mean: ${np.mean(pnls):.0f}")
            ax4.legend(fontsize=8)
        ax4.set_title("Trade PnL Distribution")
        ax4.set_xlabel("PnL ($)")
        ax4.set_ylabel("Frequency")
        ax4.grid(True, alpha=0.3)

        # Panel 5: Key Metrics Table
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.axis("off")
        metrics_data = [
            ["Total Return", f"{result.total_return * 100:.2f}%"],
            ["CAGR", f"{getattr(result, 'cagr', 0) * 100:.2f}%"],
            ["Sharpe Ratio", f"{result.sharpe_ratio:.2f}"],
            ["Sortino Ratio", f"{result.sortino_ratio:.2f}"],
            ["Max Drawdown", f"{result.max_drawdown * 100:.2f}%"],
            ["Calmar Ratio", f"{getattr(result, 'calmar_ratio', 0):.2f}"],
            ["Win Rate", f"{result.win_rate * 100:.1f}%"],
            ["Profit Factor", f"{result.profit_factor:.2f}"],
            ["Total Trades", f"{result.total_trades}"],
            ["Avg Win", f"${result.avg_win:.2f}"],
            ["Avg Loss", f"${result.avg_loss:.2f}"],
            ["Expectancy", f"${getattr(result, 'expectancy', 0):.2f}"],
        ]
        table = ax5.table(
            cellText=metrics_data,
            colLabels=["Metric", "Value"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.4)
        ax5.set_title("Key Metrics", fontsize=12, pad=20)

        # Panel 6: Win/Loss Streaks
        ax6 = fig.add_subplot(gs[1, 2])
        if trades:
            streak_data = []
            current_streak = 0
            for t in trades:
                if t.pnl > 0:
                    current_streak = current_streak + 1 if current_streak > 0 else 1
                else:
                    current_streak = current_streak - 1 if current_streak < 0 else -1
                streak_data.append(current_streak)

            colors = ["green" if s > 0 else "red" for s in streak_data]
            ax6.bar(range(len(streak_data)), streak_data, color=colors, alpha=0.7)
            ax6.axhline(0, color="black", linewidth=0.5)
        ax6.set_title("Win/Loss Streaks")
        ax6.set_xlabel("Trade #")
        ax6.set_ylabel("Streak Length")
        ax6.grid(True, alpha=0.3)

        _save_or_show(fig, save_path)
