"""
Tests for shared.ml.llm_reasoning — LLMReasoner
===================================================

Covers:
- __init__(): provider selection, missing deps
- reason(): full pipeline with mocked LLM
- _build_context(): JSON context construction with all optional fields
- _parse_response(): clean JSON, code-fenced JSON, malformed JSON
- Verify fix: .find() instead of .index() for newline detection
- _fallback(): ensemble fallback when LLM fails
- _call_llm(): anthropic and openai routing (mocked)
- Edge cases: missing fields, extra fields, various output formats
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from unittest.mock import MagicMock, patch

import pytest


# ─── Helpers ───


def _valid_llm_json(**overrides):
    """Return a valid LLM response JSON string."""
    data = {
        "action": "BUY",
        "confidence": 0.85,
        "tp_price": 160.0,
        "sl_price": 145.0,
        "allocation_pct": 10,
        "exit_plan": "Close if RSI > 70",
        "reasoning": "Strong trend with ML agreement",
    }
    data.update(overrides)
    return json.dumps(data)


def _make_reasoner(provider="anthropic"):
    """Create an LLMReasoner with fully mocked API clients.

    Because anthropic/openai may not be installed, we must use create=True
    when patching the module-level names.
    """
    with patch("shared.ml.llm_reasoning._HAS_ANTHROPIC", True), \
         patch("shared.ml.llm_reasoning._HAS_OPENAI", True), \
         patch("shared.ml.llm_reasoning.anthropic", create=True) as mock_anthropic, \
         patch("shared.ml.llm_reasoning.openai", create=True) as mock_openai:

        from shared.ml.llm_reasoning import LLMReasoner

        mock_client = MagicMock()
        if provider == "anthropic":
            mock_anthropic.Anthropic.return_value = mock_client
        else:
            mock_openai.OpenAI.return_value = mock_client

        reasoner = LLMReasoner(provider=provider, api_key="test-key")
        return reasoner


# ─── Fixtures ───


@pytest.fixture
def anthropic_reasoner():
    return _make_reasoner("anthropic")


@pytest.fixture
def openai_reasoner():
    return _make_reasoner("openai")


# ─── __init__() ───


class TestInit:
    def test_anthropic_provider(self):
        r = _make_reasoner("anthropic")
        assert r._provider == "anthropic"
        assert "claude" in r._model

    def test_openai_provider(self):
        r = _make_reasoner("openai")
        assert r._provider == "openai"
        assert "gpt" in r._model

    def test_unknown_provider_raises(self):
        with patch("shared.ml.llm_reasoning._HAS_ANTHROPIC", True), \
             patch("shared.ml.llm_reasoning._HAS_OPENAI", True), \
             patch("shared.ml.llm_reasoning.anthropic", create=True), \
             patch("shared.ml.llm_reasoning.openai", create=True):
            from shared.ml.llm_reasoning import LLMReasoner
            with pytest.raises(ValueError, match="Unknown provider"):
                LLMReasoner(provider="gemini", api_key="key")

    def test_missing_anthropic_raises(self):
        with patch("shared.ml.llm_reasoning._HAS_ANTHROPIC", False), \
             patch("shared.ml.llm_reasoning._HAS_OPENAI", True):
            from shared.ml.llm_reasoning import LLMReasoner
            with pytest.raises(ImportError, match="anthropic"):
                LLMReasoner(provider="anthropic", api_key="key")

    def test_missing_openai_raises(self):
        with patch("shared.ml.llm_reasoning._HAS_ANTHROPIC", True), \
             patch("shared.ml.llm_reasoning._HAS_OPENAI", False):
            from shared.ml.llm_reasoning import LLMReasoner
            with pytest.raises(ImportError, match="openai"):
                LLMReasoner(provider="openai", api_key="key")

    def test_custom_model_and_params(self):
        with patch("shared.ml.llm_reasoning._HAS_ANTHROPIC", True), \
             patch("shared.ml.llm_reasoning.anthropic", create=True) as mock_a:
            mock_a.Anthropic.return_value = MagicMock()
            from shared.ml.llm_reasoning import LLMReasoner
            r = LLMReasoner(provider="anthropic", api_key="k",
                            model="custom-model", max_tokens=1024, temperature=0.5)
            assert r._model == "custom-model"
            assert r._max_tokens == 1024
            assert r._temperature == 0.5


# ─── _parse_response() ───


class TestParseResponse:
    def _get_reasoner(self):
        return _make_reasoner("anthropic")

    def test_clean_json(self):
        r = self._get_reasoner()
        result = r._parse_response(_valid_llm_json())
        assert result["action"] == "BUY"
        assert result["confidence"] == 0.85
        assert result["tp_price"] == 160.0
        assert result["sl_price"] == 145.0
        assert result["allocation_pct"] == 10
        assert result["exit_plan"] == "Close if RSI > 70"

    def test_json_with_code_fences(self):
        """Verify fix: .find() handles code blocks correctly."""
        r = self._get_reasoner()
        text = '```json\n' + _valid_llm_json() + '\n```'
        result = r._parse_response(text)
        assert result["action"] == "BUY"

    def test_code_fence_no_language_tag(self):
        r = self._get_reasoner()
        text = '```\n' + _valid_llm_json() + '\n```'
        result = r._parse_response(text)
        assert result["action"] == "BUY"

    def test_code_fence_no_newline_after_backticks(self):
        """Verify fix: .find() returns -1 when no newline found, not ValueError."""
        r = self._get_reasoner()
        text = '```'
        with pytest.raises(json.JSONDecodeError):
            r._parse_response(text)

    def test_find_vs_index_fix(self):
        """Verify that .find() is used (returns -1) instead of .index() (raises ValueError)."""
        r = self._get_reasoner()
        text = '```'
        try:
            r._parse_response(text)
        except json.JSONDecodeError:
            pass  # Expected — find() returned -1, cleaned became empty
        except ValueError:
            pytest.fail(".index() raised ValueError — should use .find() instead")

    def test_whitespace_around_json(self):
        r = self._get_reasoner()
        text = "  \n\n" + _valid_llm_json() + "  \n\n"
        result = r._parse_response(text)
        assert result["action"] == "BUY"

    def test_missing_action_defaults_to_hold(self):
        r = self._get_reasoner()
        text = json.dumps({"confidence": 0.5})
        result = r._parse_response(text)
        assert result["action"] == "HOLD"

    def test_missing_confidence_defaults_to_half(self):
        r = self._get_reasoner()
        text = json.dumps({"action": "BUY"})
        result = r._parse_response(text)
        assert result["confidence"] == 0.5

    def test_missing_allocation_defaults_to_five(self):
        r = self._get_reasoner()
        text = json.dumps({"action": "SELL"})
        result = r._parse_response(text)
        assert result["allocation_pct"] == 5

    def test_action_uppercased(self):
        r = self._get_reasoner()
        text = json.dumps({"action": "buy"})
        result = r._parse_response(text)
        assert result["action"] == "BUY"

    def test_tp_sl_can_be_none(self):
        r = self._get_reasoner()
        text = json.dumps({"action": "HOLD", "tp_price": None, "sl_price": None})
        result = r._parse_response(text)
        assert result["tp_price"] is None
        assert result["sl_price"] is None

    def test_malformed_json_raises(self):
        r = self._get_reasoner()
        with pytest.raises(json.JSONDecodeError):
            r._parse_response("not json at all")

    def test_extra_fields_ignored(self):
        r = self._get_reasoner()
        text = json.dumps({"action": "BUY", "confidence": 0.9, "extra_field": "ignored"})
        result = r._parse_response(text)
        assert "extra_field" not in result
        assert result["action"] == "BUY"

    def test_nested_code_fence_with_trailing_whitespace(self):
        r = self._get_reasoner()
        text = '```json\n' + _valid_llm_json(action="SELL") + '\n```  \n'
        result = r._parse_response(text)
        assert result["action"] == "SELL"


# ─── _fallback() ───


class TestFallback:
    def test_buy_fallback(self):
        from shared.ml.llm_reasoning import LLMReasoner
        result = LLMReasoner._fallback({"direction": 1, "confidence": 0.8})
        assert result["action"] == "BUY"
        assert result["confidence"] == 0.8
        assert result["llm_used"] is False

    def test_sell_fallback(self):
        from shared.ml.llm_reasoning import LLMReasoner
        result = LLMReasoner._fallback({"direction": -1, "confidence": 0.6})
        assert result["action"] == "SELL"

    def test_hold_fallback(self):
        from shared.ml.llm_reasoning import LLMReasoner
        result = LLMReasoner._fallback({"direction": 0, "confidence": 0.3})
        assert result["action"] == "HOLD"

    def test_fallback_missing_direction(self):
        from shared.ml.llm_reasoning import LLMReasoner
        result = LLMReasoner._fallback({})
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0

    def test_fallback_has_no_tp_sl(self):
        from shared.ml.llm_reasoning import LLMReasoner
        result = LLMReasoner._fallback({"direction": 1, "confidence": 0.5})
        assert result["tp_price"] is None
        assert result["sl_price"] is None

    def test_fallback_reasoning_mentions_ensemble(self):
        from shared.ml.llm_reasoning import LLMReasoner
        result = LLMReasoner._fallback({"direction": 0})
        assert "ensemble" in result["reasoning"].lower()


# ─── _build_context() ───


class TestBuildContext:
    def test_basic_context(self):
        r = _make_reasoner("anthropic")
        ctx_str = r._build_context(
            symbol="AAPL", price=150.0, regime="TRENDING",
            regime_confidence=0.9,
            predictions={"lstm": 0.02},
            ensemble_signal={"direction": 1, "confidence": 0.8},
            sentiment=None, indicators=None,
            position=None, trade_history=None, risk_status=None,
        )
        ctx = json.loads(ctx_str)
        assert ctx["symbol"] == "AAPL"
        assert ctx["current_price"] == 150.0
        assert ctx["regime"]["classification"] == "TRENDING"
        assert ctx["ml_predictions"]["lstm"] == 0.02

    def test_context_with_sentiment(self):
        r = _make_reasoner("anthropic")
        sentiment = {
            "sentiment_score": 0.7,
            "sentiment_label": "BULLISH",
            "headlines_analyzed": 10,
            "top_bullish": [{"title": "surge"}, {"title": "rally"}, {"title": "jump"}],
            "top_bearish": [{"title": "dip"}],
        }
        ctx_str = r._build_context(
            "AAPL", 150.0, "TRENDING", 0.9, {}, {},
            sentiment=sentiment,
            indicators=None, position=None, trade_history=None, risk_status=None,
        )
        ctx = json.loads(ctx_str)
        assert ctx["news_sentiment"]["score"] == 0.7
        assert len(ctx["news_sentiment"]["top_bullish"]) == 2  # capped at 2

    def test_context_with_indicators(self):
        r = _make_reasoner("anthropic")
        indicators = {"rsi": 65, "macd": 0.5, "adx": 30}
        ctx_str = r._build_context(
            "AAPL", 150.0, "TRENDING", 0.9, {}, {},
            sentiment=None, indicators=indicators,
            position=None, trade_history=None, risk_status=None,
        )
        ctx = json.loads(ctx_str)
        assert ctx["technical_indicators"]["rsi"] == 65

    def test_context_with_position(self):
        r = _make_reasoner("anthropic")
        position = {"symbol": "AAPL", "quantity": 100, "avg_price": 148.0}
        ctx_str = r._build_context(
            "AAPL", 150.0, "TRENDING", 0.9, {}, {},
            sentiment=None, indicators=None,
            position=position, trade_history=None, risk_status=None,
        )
        ctx = json.loads(ctx_str)
        assert ctx["current_position"]["quantity"] == 100

    def test_context_with_risk_status_filters_keys(self):
        r = _make_reasoner("anthropic")
        risk_status = {
            "can_trade": True,
            "daily_pnl": 500,
            "consecutive_losses": 2,
            "drawdown_pct": 0.05,
            "open_positions": 3,
            "internal_detail": "should be filtered out",
        }
        ctx_str = r._build_context(
            "AAPL", 150.0, "TRENDING", 0.9, {}, {},
            sentiment=None, indicators=None,
            position=None, trade_history=None, risk_status=risk_status,
        )
        ctx = json.loads(ctx_str)
        assert ctx["risk_status"]["can_trade"] is True
        assert "internal_detail" not in ctx["risk_status"]

    def test_context_excludes_none_optionals(self):
        r = _make_reasoner("anthropic")
        ctx_str = r._build_context(
            "AAPL", 150.0, "TRENDING", 0.9, {}, {},
            sentiment=None, indicators=None,
            position=None, trade_history=None, risk_status=None,
        )
        ctx = json.loads(ctx_str)
        assert "news_sentiment" not in ctx
        assert "technical_indicators" not in ctx
        assert "current_position" not in ctx
        assert "recent_performance" not in ctx
        assert "risk_status" not in ctx

    def test_context_with_trade_history(self):
        r = _make_reasoner("anthropic")
        ctx_str = r._build_context(
            "AAPL", 150.0, "TRENDING", 0.9, {}, {},
            sentiment=None, indicators=None,
            position=None, trade_history={"win_rate": 0.65, "total_trades": 50},
            risk_status=None,
        )
        ctx = json.loads(ctx_str)
        assert ctx["recent_performance"]["win_rate"] == 0.65


# ─── reason() full pipeline ───


class TestReason:
    def test_successful_reason(self):
        r = _make_reasoner("anthropic")
        with patch.object(r, "_call_llm", return_value=_valid_llm_json()):
            result = r.reason(
                symbol="AAPL", price=150.0, regime="TRENDING",
                regime_confidence=0.9,
                predictions={"lstm": 0.02},
                ensemble_signal={"direction": 1, "confidence": 0.8},
            )
        assert result["action"] == "BUY"
        assert result["llm_used"] is True
        assert result["llm_model"] == r._model

    def test_reason_with_all_optional_params(self):
        r = _make_reasoner("anthropic")
        with patch.object(r, "_call_llm", return_value=_valid_llm_json()):
            result = r.reason(
                symbol="AAPL", price=150.0, regime="TRENDING",
                regime_confidence=0.9,
                predictions={"lstm": 0.02},
                ensemble_signal={"direction": 1},
                sentiment={"sentiment_score": 0.5},
                indicators={"rsi": 60},
                position={"quantity": 100},
                trade_history={"win_rate": 0.6},
                risk_status={"can_trade": True},
            )
        assert result["action"] == "BUY"

    def test_reason_llm_failure_falls_back(self):
        r = _make_reasoner("anthropic")
        with patch.object(r, "_call_llm", side_effect=RuntimeError("API down")):
            result = r.reason(
                symbol="AAPL", price=150.0, regime="TRENDING",
                regime_confidence=0.9,
                predictions={},
                ensemble_signal={"direction": -1, "confidence": 0.7},
            )
        assert result["action"] == "SELL"
        assert result["llm_used"] is False

    def test_reason_parse_failure_falls_back(self):
        r = _make_reasoner("anthropic")
        with patch.object(r, "_call_llm", return_value="not json"):
            result = r.reason(
                symbol="AAPL", price=150.0, regime="TRENDING",
                regime_confidence=0.9,
                predictions={},
                ensemble_signal={"direction": 0, "confidence": 0.3},
            )
        assert result["action"] == "HOLD"
        assert result["llm_used"] is False


# ─── _call_llm() ───


class TestCallLLM:
    def test_anthropic_call(self):
        r = _make_reasoner("anthropic")
        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = _valid_llm_json()
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        r._client.messages.create.return_value = mock_response

        result = r._call_llm("test context")
        assert "BUY" in result
        r._client.messages.create.assert_called_once()

    def test_openai_call(self):
        r = _make_reasoner("openai")
        mock_message = MagicMock()
        mock_message.content = _valid_llm_json()
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        r._client.chat.completions.create.return_value = mock_response

        result = r._call_llm("test context")
        assert "BUY" in result
        r._client.chat.completions.create.assert_called_once()


# ─── repr ───


class TestRepr:
    def test_repr_anthropic(self):
        r = _make_reasoner("anthropic")
        s = repr(r)
        assert "LLMReasoner" in s
        assert "anthropic" in s

    def test_repr_openai(self):
        r = _make_reasoner("openai")
        s = repr(r)
        assert "openai" in s
