"""
Live Trading Runner — 24/7 Autonomous Daemon
================================================

Runs the SelfLearningAgent continuously, handling:
- Market hours detection (trade during open, learn after close)
- Public data fetching (Yahoo Finance, no brokerage needed)
- News sentiment analysis before every decision
- Trade memory accumulation across sessions
- Model health monitoring and retrain alerts
- Logging everything for audit trail
- Graceful shutdown (Ctrl+C)

Platform Integration:
- Standalone: runs independently, fetches its own data
- TradingView: receives webhook alerts and routes through the agent
- IB Gateway: (future) can route orders to Interactive Brokers

Modes:
- "monitor": Watch and learn only — no trades, just log decisions
- "paper": Simulated trading — track P&L without real orders
- "live": Real trading via IB Gateway (requires IB connection)

Usage:
    # Monitor mode (safe — just watches and learns):
    python -m shared.daemon.live_runner --mode monitor --symbols AAPL,MSFT,GOOGL

    # Paper trading:
    python -m shared.daemon.live_runner --mode paper --symbols SPY,QQQ --interval 300

    # With news sentiment:
    python -m shared.daemon.live_runner --mode paper --symbols AAPL --news
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Configure logging
LOG_DIR = os.path.join(os.path.expanduser("~"), ".stocks_plugin", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "live_runner.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("live_runner")


class LiveRunner:
    """24/7 autonomous trading daemon.

    Continuously monitors markets, fetches data, analyzes news,
    and makes trading decisions using the SelfLearningAgent.

    Args:
        symbols: List of symbols to monitor.
        mode: "monitor" (watch only), "paper" (simulated), or "live" (real orders).
        interval_seconds: Seconds between decision cycles.
        use_news: Whether to analyze news sentiment.
        db_path: Path to trade memory database.
        models: ML models to train ("regime", "lstm", "transformer", "rl").
    """

    def __init__(
        self,
        symbols: List[str],
        mode: str = "monitor",
        interval_seconds: int = 300,
        use_news: bool = True,
        db_path: Optional[str] = None,
        models: Optional[List[str]] = None,
        broker: Optional[str] = None,
        broker_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._symbols = symbols
        self._mode = mode
        self._interval = interval_seconds
        self._use_news = use_news
        self._running = False
        self._cycle_count = 0
        self._broker_name = broker

        # Default paths
        if db_path is None:
            db_dir = os.path.join(os.path.expanduser("~"), ".stocks_plugin", "data")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "trade_memory.db")

        self._models_to_train = models or ["regime"]

        # Initialize components
        from shared.data.public_data_fetcher import PublicDataFetcher
        self._data_fetcher = PublicDataFetcher()

        self._sentiment_analyzer = None
        if use_news:
            try:
                from shared.ml.news_sentiment import NewsSentimentAnalyzer
                self._sentiment_analyzer = NewsSentimentAnalyzer()
            except Exception as e:
                logger.warning("News sentiment unavailable: %s", e)

        from shared.ml.self_learning_agent import SelfLearningAgent, AgentConfig
        self._agent = SelfLearningAgent(AgentConfig(
            db_path=db_path,
            adaptive_thresholds=True,
        ))

        # Broker Bridge (for live/paper trading with real brokers)
        self._broker_bridge = None
        if broker and mode in ("paper", "live"):
            try:
                from shared.daemon.broker_bridge import BrokerBridge
                self._broker_bridge = BrokerBridge(
                    broker=broker,
                    config=broker_config or {},
                    mode=mode,
                )
                logger.info("BrokerBridge initialized: %s (%s mode)", broker, mode)
            except Exception as e:
                logger.warning("BrokerBridge init failed: %s. Falling back to paper simulation.", e)

        # Paper trading state (fallback when no broker bridge)
        self._paper_positions: Dict[str, Dict[str, Any]] = {}
        self._paper_capital = 100_000.0
        self._paper_pnl = 0.0

        # Signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Optional LLM reasoning layer
        self._llm_reasoner = None

        logger.info(
            "LiveRunner initialized: symbols=%s, mode=%s, interval=%ds, news=%s",
            symbols, mode, interval_seconds, use_news,
        )

    def start(self, train_first: bool = True) -> None:
        """Start the 24/7 daemon loop.

        Args:
            train_first: If True, train models on 1 year of history before starting.
        """
        self._running = True

        self._print_banner()

        if train_first:
            self._initial_training()

        # Connect to broker if configured
        if self._broker_bridge and self._mode in ("paper", "live"):
            logger.info("Connecting to broker: %s", self._broker_name)
            if self._broker_bridge.connect():
                logger.info("Broker connected: %s", self._broker_bridge)
                acct = self._broker_bridge.get_account_info()
                logger.info("Account info: %s", acct)
            else:
                logger.warning("Broker connection failed. Running in paper-simulation mode.")
                self._broker_bridge = None

        logger.info("=" * 60)
        logger.info("DAEMON STARTED — Mode: %s", self._mode.upper())
        logger.info("Monitoring: %s", ", ".join(self._symbols))
        logger.info("Interval: %d seconds", self._interval)
        logger.info("Press Ctrl+C to stop gracefully")
        logger.info("=" * 60)

        while self._running:
            try:
                self._run_cycle()
                self._cycle_count += 1

                if self._running:
                    self._smart_sleep()

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error("Cycle error: %s", e, exc_info=True)
                time.sleep(60)  # wait 1 minute on error

        self._shutdown()

    def _run_cycle(self) -> None:
        """Execute one decision cycle for all symbols."""
        market_open = self._data_fetcher.is_market_open()
        now = datetime.now()

        logger.info(
            "─── Cycle #%d | %s | Market: %s ───",
            self._cycle_count + 1,
            now.strftime("%Y-%m-%d %H:%M:%S"),
            "OPEN" if market_open else "CLOSED",
        )

        for symbol in self._symbols:
            try:
                self._process_symbol(symbol, market_open)
            except Exception as e:
                logger.error("Error processing %s: %s", symbol, e)

        # Position reconciliation + force-close (from hyperliquid-trading-agent patterns)
        if self._broker_bridge and self._broker_bridge.is_connected():
            try:
                self._broker_bridge.reconcile_positions()
                force_closed = self._broker_bridge.check_and_force_close(agent=self._agent)
                if force_closed:
                    logger.warning("Force-closed %d positions exceeding max loss", len(force_closed))
            except Exception as e:
                logger.debug("Reconciliation/force-close check: %s", e)

        # Periodic status report
        if self._cycle_count % 12 == 0:  # every 12 cycles
            self._print_status_report()

    def _process_symbol(self, symbol: str, market_open: bool) -> None:
        """Process a single symbol: fetch data → analyze → decide."""

        # Step 1: Fetch latest data
        df = self._data_fetcher.fetch_ohlcv(symbol, period="6mo", interval="1d")
        if df is None or df.empty or len(df) < 60:
            logger.warning("Insufficient data for %s (%d bars)", symbol, len(df) if df is not None else 0)
            return

        # Step 2: News sentiment (if enabled)
        sentiment_score = 0.0
        if self._sentiment_analyzer:
            try:
                sentiment = self._sentiment_analyzer.analyze(symbol, max_headlines=10)
                sentiment_score = sentiment.get("sentiment_score", 0.0)
                logger.info(
                    "  %s sentiment: %.3f (%s) | %d headlines | %d bull, %d bear",
                    symbol, sentiment_score, sentiment.get("sentiment_label", "?"),
                    sentiment.get("headlines_analyzed", 0),
                    sentiment.get("bullish_count", 0),
                    sentiment.get("bearish_count", 0),
                )
            except Exception as e:
                logger.debug("Sentiment analysis failed for %s: %s", symbol, e)

        # Step 3: Agent decision
        decision = self._agent.decide(df, symbol=symbol)

        action = decision["action"]
        confidence = decision["confidence"]
        regime = decision["regime"]
        price = decision["price"]

        # Incorporate sentiment into action
        if sentiment_score != 0:
            action = self._apply_sentiment_filter(action, sentiment_score, confidence)

        logger.info(
            "  %s: %s @ $%.2f | confidence=%.2f | regime=%s | sentiment=%.3f",
            symbol, action, price, confidence, regime, sentiment_score,
        )

        # Step 4b: Optional LLM reasoning layer (like hyperliquid-trading-agent)
        if hasattr(self, '_llm_reasoner') and self._llm_reasoner is not None:
            try:
                llm_result = self._llm_reasoner.reason(
                    symbol=symbol,
                    price=price,
                    regime=regime,
                    regime_confidence=decision.get("regime_probabilities", {}).get(regime, 0),
                    predictions=decision.get("predictions", {}),
                    ensemble_signal=decision.get("ensemble_signal", {}),
                    sentiment={"sentiment_score": sentiment_score} if sentiment_score else None,
                    risk_status=decision.get("risk_status"),
                )
                action = llm_result.get("action", action)
                confidence = llm_result.get("confidence", confidence)
                # Pass TP/SL from LLM to broker bridge
                decision["tp_price"] = llm_result.get("tp_price")
                decision["sl_price"] = llm_result.get("sl_price")
                decision["exit_plan"] = llm_result.get("exit_plan", "")
                decision["reasoning"] = llm_result.get("reasoning", "")[:500]
                logger.info(
                    "  LLM override: %s (confidence=%.2f) | TP=%s SL=%s",
                    action, confidence, llm_result.get("tp_price"), llm_result.get("sl_price"),
                )
            except Exception as e:
                logger.debug("LLM reasoning skipped: %s", e)

        # Step 4: Execute based on mode
        if self._mode == "monitor":
            self._log_decision(symbol, action, price, confidence, regime)
        elif self._mode == "paper":
            if self._broker_bridge and self._broker_bridge.is_connected():
                self._execute_live_trade(symbol, action, price, confidence)
            else:
                self._execute_paper_trade(symbol, action, price, confidence)
        elif self._mode == "live" and market_open:
            self._execute_live_trade(symbol, action, price, confidence)

    def _apply_sentiment_filter(
        self, action: str, sentiment: float, confidence: float,
    ) -> str:
        """Adjust trading action based on news sentiment.

        Strong negative sentiment can block buys.
        Strong positive sentiment can block sells.
        """
        if action == "BUY" and sentiment < -0.3:
            logger.info("  Sentiment override: blocking BUY (sentiment=%.2f)", sentiment)
            return "HOLD"
        if action == "SELL" and sentiment > 0.3:
            logger.info("  Sentiment override: blocking SELL (sentiment=%.2f)", sentiment)
            return "HOLD"
        return action

    # ─── Paper Trading ───

    def _execute_paper_trade(
        self, symbol: str, action: str, price: float, confidence: float,
    ) -> None:
        """Simulate a trade in paper mode."""
        has_position = symbol in self._paper_positions

        if action == "BUY" and not has_position:
            # Open long position
            shares = int(self._paper_capital * 0.1 / price)  # 10% of capital per position
            if shares > 0:
                cost = shares * price
                self._paper_positions[symbol] = {
                    "shares": shares,
                    "entry_price": price,
                    "entry_time": datetime.now().isoformat(),
                    "direction": "long",
                }
                logger.info(
                    "  📈 PAPER BUY: %d shares of %s @ $%.2f ($%.2f)",
                    shares, symbol, price, cost,
                )

        elif action == "SELL" and has_position:
            # Close position
            pos = self._paper_positions.pop(symbol)
            pnl = (price - pos["entry_price"]) * pos["shares"]
            pnl_pct = (price - pos["entry_price"]) / pos["entry_price"]
            self._paper_pnl += pnl

            emoji = "✅" if pnl > 0 else "❌"
            logger.info(
                "  %s PAPER SELL: %d shares of %s @ $%.2f | P&L: $%.2f (%.2f%%)",
                emoji, pos["shares"], symbol, price, pnl, pnl_pct * 100,
            )

            # Record outcome in agent memory
            self._agent.record_outcome(
                exit_price=price,
                pnl=pnl,
                holding_period_bars=1,
            )

        elif action == "SELL" and not has_position:
            # Open short position (paper)
            shares = int(self._paper_capital * 0.1 / price)
            if shares > 0:
                self._paper_positions[symbol] = {
                    "shares": shares,
                    "entry_price": price,
                    "entry_time": datetime.now().isoformat(),
                    "direction": "short",
                }
                logger.info(
                    "  📉 PAPER SHORT: %d shares of %s @ $%.2f",
                    shares, symbol, price,
                )

    def _execute_live_trade(
        self, symbol: str, action: str, price: float, confidence: float,
    ) -> None:
        """Execute a live trade via the connected BrokerBridge."""
        if not self._broker_bridge:
            logger.warning(
                "LIVE TRADING: no broker connected. Use --broker ib|tradestation|schwab"
            )
            return

        if not self._broker_bridge.is_connected():
            logger.info("Connecting to broker...")
            if not self._broker_bridge.connect():
                logger.error("Broker connection failed. Skipping trade.")
                return

        # Build a decision dict for the bridge
        decision = {"action": action, "confidence": confidence, "price": price}
        result = self._broker_bridge.execute_decision(decision, symbol, agent=self._agent)

        if result and result.success:
            logger.info(
                "🔴 LIVE TRADE: %s %d %s @ $%.2f via %s [order=%s]",
                action, result.shares, symbol, price,
                result.broker, result.order_id,
            )
        elif result:
            logger.error("LIVE TRADE FAILED: %s | %s", symbol, result.message)

    def _log_decision(
        self, symbol: str, action: str, price: float, confidence: float, regime: str,
    ) -> None:
        """Log decision in monitor mode (no execution)."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": action,
            "price": price,
            "confidence": confidence,
            "regime": regime,
            "mode": "monitor",
        }

        log_file = os.path.join(LOG_DIR, "decisions.jsonl")
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    # ─── Training ───

    def _initial_training(self) -> None:
        """Train models on historical data before starting the live loop."""
        logger.info("=" * 60)
        logger.info("INITIAL TRAINING PHASE")
        logger.info("=" * 60)

        # Use the first symbol's data for training
        primary_symbol = self._symbols[0]
        logger.info("Fetching 2 years of %s data for training...", primary_symbol)

        df = self._data_fetcher.fetch_ohlcv(primary_symbol, period="2y", interval="1d")
        if df is None or df.empty:
            logger.warning("Could not fetch training data for %s. Using untrained agent.", primary_symbol)
            return

        logger.info("Training on %d bars of %s...", len(df), primary_symbol)
        try:
            results = self._agent.train(
                df.reset_index() if "date" not in df.columns else df,
                models=self._models_to_train,
                verbose=True,
            )
            logger.info("Training complete: %s", list(results.keys()))
        except Exception as e:
            logger.error("Training failed: %s. Agent will use momentum fallback.", e)

    # ─── Sleep & Scheduling ───

    def _smart_sleep(self) -> None:
        """Sleep with market-aware intervals.

        During market hours: use configured interval
        After hours: check less frequently (15 min)
        Weekends: check very infrequently (1 hour)
        """
        now = datetime.now()
        is_weekend = now.weekday() >= 5

        if is_weekend:
            sleep_time = 3600  # 1 hour on weekends
        elif self._data_fetcher.is_market_open():
            sleep_time = self._interval  # configured interval during market
        else:
            sleep_time = 900  # 15 minutes after hours

        logger.debug("Sleeping %d seconds...", sleep_time)

        # Sleep in small chunks for responsive shutdown
        elapsed = 0
        while elapsed < sleep_time and self._running:
            time.sleep(min(10, sleep_time - elapsed))
            elapsed += 10

    # ─── Status Reporting ───

    def _print_status_report(self) -> None:
        """Print periodic status report."""
        logger.info("=" * 50)
        logger.info("STATUS REPORT — Cycle #%d", self._cycle_count)
        logger.info("=" * 50)

        # Paper positions
        if self._paper_positions:
            logger.info("Open positions:")
            for sym, pos in self._paper_positions.items():
                logger.info("  %s: %d shares @ $%.2f (%s)",
                           sym, pos["shares"], pos["entry_price"], pos["direction"])
        else:
            logger.info("No open positions")

        logger.info("Paper P&L: $%.2f", self._paper_pnl)

        # Agent stats
        try:
            perf = self._agent.get_performance(lookback_days=7)
            logger.info("Agent (7d): %d trades, win_rate=%.0f%%",
                       perf.get("total_trades", 0),
                       perf.get("win_rate", 0) * 100)
        except Exception:
            pass

        # Weights
        try:
            weights = self._agent.get_weight_summary()
            for model, w in weights.items():
                logger.info("  %s weight: %.3f", model, w.get("effective_weight", 0))
        except Exception:
            pass

        logger.info("=" * 50)

    def _print_banner(self) -> None:
        """Print startup banner."""
        print()
        print("╔══════════════════════════════════════════════════╗")
        print("║     🧠 SELF-LEARNING TRADING AGENT — LIVE       ║")
        print("║                                                  ║")
        print(f"║  Mode:    {self._mode.upper():<40s}║")
        print(f"║  Symbols: {', '.join(self._symbols):<40s}║")
        print(f"║  Interval: {self._interval}s{' ' * (38 - len(str(self._interval)))}║")
        print(f"║  News:    {'ON' if self._use_news else 'OFF':<40s}║")
        broker_str = self._broker_name.upper() if self._broker_name else "NONE"
        print(f"║  Broker:  {broker_str:<40s}║")
        print("║                                                  ║")
        print("║  Press Ctrl+C to stop                            ║")
        print("╚══════════════════════════════════════════════════╝")
        print()

    # ─── Shutdown ───

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals gracefully."""
        logger.info("Shutdown signal received (signal=%d)", signum)
        self._running = False

    def _shutdown(self) -> None:
        """Graceful shutdown: close positions, save state, report."""
        logger.info("=" * 60)
        logger.info("SHUTTING DOWN")
        logger.info("=" * 60)

        # Report final state
        if self._paper_positions:
            logger.info("OPEN POSITIONS at shutdown:")
            for sym, pos in self._paper_positions.items():
                logger.info("  %s: %d shares @ $%.2f", sym, pos["shares"], pos["entry_price"])
            logger.warning("Paper positions NOT auto-closed. Run again to manage them.")

        logger.info("Total paper P&L: $%.2f", self._paper_pnl)
        logger.info("Total cycles: %d", self._cycle_count)

        # Save models
        try:
            model_dir = os.path.join(os.path.expanduser("~"), ".stocks_plugin", "models")
            self._agent.save_models(model_dir)
            logger.info("Models saved to %s", model_dir)
        except Exception as e:
            logger.warning("Failed to save models: %s", e)

        self._agent.close()
        logger.info("Daemon stopped.")


# ─── CLI Entry Point ───

def main() -> None:
    """CLI entry point for the live runner daemon."""
    parser = argparse.ArgumentParser(
        prog="live_runner",
        description="24/7 Self-Learning Trading Agent Daemon",
    )
    parser.add_argument(
        "--symbols", "-s", required=True,
        help="Comma-separated list of symbols (e.g., AAPL,MSFT,GOOGL)",
    )
    parser.add_argument(
        "--mode", "-m", default="monitor",
        choices=["monitor", "paper", "live"],
        help="Trading mode: monitor (watch only), paper (simulated), live (real). Default: monitor",
    )
    parser.add_argument(
        "--interval", "-i", type=int, default=300,
        help="Seconds between decision cycles (default: 300 = 5 min)",
    )
    parser.add_argument(
        "--news", action="store_true",
        help="Enable news sentiment analysis",
    )
    parser.add_argument(
        "--no-train", action="store_true",
        help="Skip initial model training",
    )
    parser.add_argument(
        "--models", default="regime",
        help="Comma-separated models to train (default: regime). Options: regime,lstm,transformer,rl",
    )
    parser.add_argument(
        "--db", default=None,
        help="Path to trade memory database",
    )
    parser.add_argument(
        "--broker", "-b", default=None,
        choices=["ib", "tradestation", "schwab"],
        help="Broker for live/paper trading: ib, tradestation, or schwab",
    )
    parser.add_argument(
        "--broker-port", type=int, default=None,
        help="Broker connection port (IB: 7497=paper, 7496=live; default: auto)",
    )
    parser.add_argument(
        "--broker-host", default="127.0.0.1",
        help="Broker host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--llm", default=None,
        choices=["anthropic", "openai"],
        help="Enable LLM reasoning layer: anthropic (Claude) or openai (GPT)",
    )
    parser.add_argument(
        "--llm-key", default=None,
        help="API key for the LLM provider (or set ANTHROPIC_API_KEY / OPENAI_API_KEY env var)",
    )

    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    models = [m.strip() for m in args.models.split(",")]

    broker_config = {}
    if args.broker:
        broker_config["host"] = args.broker_host
        if args.broker_port:
            broker_config["port"] = args.broker_port

    runner = LiveRunner(
        symbols=symbols,
        mode=args.mode,
        interval_seconds=args.interval,
        use_news=args.news,
        db_path=args.db,
        models=models,
        broker=args.broker,
        broker_config=broker_config if broker_config else None,
    )

    runner.start(train_first=not args.no_train)

    # Initialize LLM reasoning if requested
    if args.llm:
        try:
            from shared.ml.llm_reasoning import LLMReasoner
            api_key = args.llm_key or os.environ.get(
                "ANTHROPIC_API_KEY" if args.llm == "anthropic" else "OPENAI_API_KEY"
            )
            if api_key:
                runner._llm_reasoner = LLMReasoner(provider=args.llm, api_key=api_key)
                logger.info("LLM reasoning enabled: %s", args.llm)
            else:
                logger.warning("No API key for %s. Set --llm-key or env var.", args.llm)
        except ImportError as e:
            logger.warning("LLM provider not installed: %s", e)


if __name__ == "__main__":
    main()
