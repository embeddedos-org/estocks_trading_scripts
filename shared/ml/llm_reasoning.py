"""
LLM Reasoning Layer — AI-Powered Trade Decisions
====================================================

Inspired by hyperliquid-trading-agent's decision_maker.py.
Uses Claude or GPT to make first-principles trade decisions
based on technical indicators, ML ensemble signals, news sentiment,
regime classification, and trade history.

This sits ON TOP of the ML ensemble — the LLM receives all model
predictions and decides whether to agree, override, or refine them.

Key features (ported from hyperliquid-trading-agent):
- System prompt with quantitative trader persona
- Take-profit and stop-loss price recommendations
- Exit plan with explicit invalidation conditions
- Cooldown logic to prevent overtrading
- Position-aware: knows current holdings
- Reasoning chain logged for audit

Usage:
    reasoner = LLMReasoner(provider="anthropic", api_key="sk-...")
    result = reasoner.reason(context)
    print(result["action"])       # "BUY" / "SELL" / "HOLD"
    print(result["tp_price"])     # take-profit price
    print(result["sl_price"])     # stop-loss price
    print(result["exit_plan"])    # "close if 4h RSI > 70"
    print(result["reasoning"])    # full reasoning chain

Requires: pip install anthropic  (or)  pip install openai
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_HAS_ANTHROPIC = False
try:
    import anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    pass

_HAS_OPENAI = False
try:
    import openai
    _HAS_OPENAI = True
except ImportError:
    pass


_SYSTEM_PROMPT = """You are a rigorous QUANTITATIVE TRADER optimizing risk-adjusted returns for equities and ETFs.

You receive market context for an asset including:
- Current price, regime (TRENDING/RANGING/VOLATILE), and regime confidence
- ML model predictions: LSTM, Transformer, RL, momentum, ensemble score
- News sentiment score and top headlines
- Technical indicators: RSI, MACD, ADX, Bollinger Bands, volume
- Recent trade history and win rate
- Current position (if any)
- Risk manager status

Your goal: make decisive, first-principles decisions that maximize risk-adjusted returns.

Decision discipline:
- Choose one: BUY, SELL, or HOLD
- Set tp_price (take-profit) and sl_price (stop-loss) for every trade
  - BUY: tp_price > current_price, sl_price < current_price
  - SELL: tp_price < current_price, sl_price > current_price
- Provide an exit_plan with explicit invalidation conditions
- confidence: 0.0 to 1.0 reflecting conviction
- allocation_pct: what % of capital to use (1-20%)

Core policy:
1) Respect ML ensemble: if ensemble confidence > 0.7 and agrees with your analysis, follow it
2) Override ML: if your analysis contradicts the ensemble with strong evidence, explain why
3) News matters: strong negative sentiment can block buys; strong positive can boost confidence
4) Regime-aware: trending markets favor momentum; ranging markets favor mean reversion
5) Risk first: never risk more than 2% of capital on a single trade's stop-loss distance
6) Cooldown: after a loss, wait for clear setup before re-entering

Output ONLY a JSON object with these keys:
{
    "action": "BUY" | "SELL" | "HOLD",
    "confidence": 0.0-1.0,
    "tp_price": number or null,
    "sl_price": number or null,
    "allocation_pct": 1-20,
    "exit_plan": "string describing when to exit",
    "reasoning": "detailed step-by-step reasoning"
}

Do not output markdown, code fences, or any extra text."""


class LLMReasoner:
    """LLM-powered reasoning layer for trade decisions.

    Sends market context to Claude or GPT and receives structured
    trade decisions with TP/SL, exit plans, and reasoning.

    Args:
        provider: "anthropic" or "openai".
        api_key: API key for the provider.
        model: Model name (default: claude-sonnet-4-20250514 or gpt-4o).
        max_tokens: Max response tokens.
        temperature: Sampling temperature (0 = deterministic).
    """

    def __init__(
        self,
        provider: str = "anthropic",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> None:
        self._provider = provider.lower()
        self._max_tokens = max_tokens
        self._temperature = temperature

        if self._provider == "anthropic":
            if not _HAS_ANTHROPIC:
                raise ImportError("anthropic not installed. pip install anthropic")
            self._client = anthropic.Anthropic(api_key=api_key)
            self._model = model or "claude-sonnet-4-20250514"
        elif self._provider == "openai":
            if not _HAS_OPENAI:
                raise ImportError("openai not installed. pip install openai")
            self._client = openai.OpenAI(api_key=api_key)
            self._model = model or "gpt-4o"
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'anthropic' or 'openai'.")

        logger.info("LLMReasoner initialized: provider=%s, model=%s", self._provider, self._model)

    def reason(
        self,
        symbol: str,
        price: float,
        regime: str,
        regime_confidence: float,
        predictions: Dict[str, float],
        ensemble_signal: Dict[str, Any],
        sentiment: Optional[Dict[str, Any]] = None,
        indicators: Optional[Dict[str, float]] = None,
        position: Optional[Dict[str, Any]] = None,
        trade_history: Optional[Dict[str, Any]] = None,
        risk_status: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get LLM-reasoned trade decision.

        Args:
            symbol: Ticker symbol.
            price: Current price.
            regime: Market regime (TRENDING/RANGING/VOLATILE).
            regime_confidence: Regime classification confidence.
            predictions: Dict of model predictions (lstm, transformer, rl, momentum).
            ensemble_signal: Ensemble output (direction, raw_score, confidence, agreement).
            sentiment: News sentiment result (sentiment_score, headlines, etc).
            indicators: Technical indicator values (rsi, macd, adx, etc).
            position: Current position info (if any).
            trade_history: Recent trade performance summary.
            risk_status: Risk manager status.

        Returns:
            Dict with: action, confidence, tp_price, sl_price, allocation_pct,
                       exit_plan, reasoning, llm_used.
        """
        context = self._build_context(
            symbol, price, regime, regime_confidence,
            predictions, ensemble_signal, sentiment,
            indicators, position, trade_history, risk_status,
        )

        try:
            response_text = self._call_llm(context)
            parsed = self._parse_response(response_text)

            logger.info(
                "LLM decision for %s: %s (confidence=%.2f, tp=%s, sl=%s)",
                symbol, parsed["action"], parsed["confidence"],
                parsed.get("tp_price"), parsed.get("sl_price"),
            )

            parsed["llm_used"] = True
            parsed["llm_model"] = self._model
            return parsed

        except Exception as e:
            logger.error("LLM reasoning failed: %s. Falling back to ensemble.", e)
            return self._fallback(ensemble_signal)

    def _build_context(
        self, symbol, price, regime, regime_confidence,
        predictions, ensemble_signal, sentiment,
        indicators, position, trade_history, risk_status,
    ) -> str:
        """Build the context message for the LLM."""
        context = {
            "current_time": datetime.now().isoformat(),
            "symbol": symbol,
            "current_price": price,
            "regime": {
                "classification": regime,
                "confidence": regime_confidence,
            },
            "ml_predictions": predictions,
            "ensemble": ensemble_signal,
        }

        if sentiment:
            context["news_sentiment"] = {
                "score": sentiment.get("sentiment_score", 0),
                "label": sentiment.get("sentiment_label", "NEUTRAL"),
                "headlines_analyzed": sentiment.get("headlines_analyzed", 0),
                "top_bullish": [h.get("title", "") for h in sentiment.get("top_bullish", [])[:2]],
                "top_bearish": [h.get("title", "") for h in sentiment.get("top_bearish", [])[:2]],
            }

        if indicators:
            context["technical_indicators"] = indicators

        if position:
            context["current_position"] = position

        if trade_history:
            context["recent_performance"] = trade_history

        if risk_status:
            context["risk_status"] = {
                k: v for k, v in risk_status.items()
                if k in ("can_trade", "daily_pnl", "consecutive_losses",
                         "drawdown_pct", "open_positions")
            }

        return json.dumps(context, indent=2, default=str)

    def _call_llm(self, context: str) -> str:
        """Call the LLM API and return raw text response."""
        if self._provider == "anthropic":
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": context}],
            )
            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text
            return text

        elif self._provider == "openai":
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                ],
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content

        return ""

    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Parse LLM JSON response into structured decision."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            first_nl = cleaned.find("\n")
            if first_nl == -1:
                cleaned = ""
            else:
                cleaned = cleaned[first_nl + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].rstrip()

        parsed = json.loads(cleaned)

        return {
            "action": str(parsed.get("action", "HOLD")).upper(),
            "confidence": float(parsed.get("confidence", 0.5)),
            "tp_price": parsed.get("tp_price"),
            "sl_price": parsed.get("sl_price"),
            "allocation_pct": float(parsed.get("allocation_pct", 5)),
            "exit_plan": str(parsed.get("exit_plan", "")),
            "reasoning": str(parsed.get("reasoning", "")),
        }

    @staticmethod
    def _fallback(ensemble_signal: Dict[str, Any]) -> Dict[str, Any]:
        """Fall back to ensemble when LLM fails."""
        direction = ensemble_signal.get("direction", 0)
        action = "BUY" if direction == 1 else ("SELL" if direction == -1 else "HOLD")
        return {
            "action": action,
            "confidence": ensemble_signal.get("confidence", 0),
            "tp_price": None,
            "sl_price": None,
            "allocation_pct": 5,
            "exit_plan": "fallback to ensemble — no LLM reasoning",
            "reasoning": "LLM call failed, using ensemble signal directly",
            "llm_used": False,
        }

    def __repr__(self) -> str:
        return f"LLMReasoner(provider='{self._provider}', model='{self._model}')"
