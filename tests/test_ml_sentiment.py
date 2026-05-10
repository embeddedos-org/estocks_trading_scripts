"""
Tests for shared.ml.news_sentiment — NewsSentimentAnalyzer
============================================================

Covers:
- analyze(): full pipeline with provided headlines
- _score_keywords(): bullish, bearish, neutral, multi-word phrases
- _score_vader(): mocked VADER scoring
- _score_finbert(): mocked FinBERT pipeline
- _score_headline(): method routing
- analyze_multiple(): multi-symbol batch
- _empty_result(): neutral fallback
- Verify fix: agreement = max(0.0, 1.0 - std_score)
- Confidence calculation: agreement * sample_factor
- Edge cases: empty headlines, all neutral, single headline
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from shared.ml.news_sentiment import (
    NewsSentimentAnalyzer,
    _BULLISH_KEYWORDS,
    _BEARISH_KEYWORDS,
)


# ─── Fixtures ───


@pytest.fixture
def analyzer():
    """Keyword-based analyzer (no external NLP deps required)."""
    return NewsSentimentAnalyzer(method="keyword")


def _headlines(titles):
    """Helper: convert list of strings to list of headline dicts."""
    return [{"title": t, "source": "test"} for t in titles]


# ─── _score_keywords() static method ───


class TestScoreKeywords:
    def test_bullish_keyword(self):
        score = NewsSentimentAnalyzer._score_keywords("Stock surges to record high")
        assert score > 0

    def test_bearish_keyword(self):
        score = NewsSentimentAnalyzer._score_keywords("Stock crashes after fraud scandal")
        assert score < 0

    def test_neutral_no_keywords(self):
        score = NewsSentimentAnalyzer._score_keywords("Company announces quarterly report date")
        assert score == 0.0

    def test_mixed_keywords_balanced(self):
        # "surges" is bullish, "crashes" is bearish → balanced
        score = NewsSentimentAnalyzer._score_keywords("Stock surges then crashes")
        assert -0.1 <= score <= 0.1

    def test_multi_word_bullish_phrase(self):
        score = NewsSentimentAnalyzer._score_keywords("Company hits all-time high today")
        assert score > 0

    def test_multi_word_bearish_phrase(self):
        score = NewsSentimentAnalyzer._score_keywords("Market sees major sell-off today")
        assert score < 0

    def test_score_bounded_negative_one_to_one(self):
        # Pack lots of bullish keywords
        text = "surge rally jump gain rise beat upgrade bullish optimistic strong growth buy"
        score = NewsSentimentAnalyzer._score_keywords(text)
        assert -1.0 <= score <= 1.0

    def test_case_insensitive(self):
        score = NewsSentimentAnalyzer._score_keywords("STOCK SURGES DRAMATICALLY")
        assert score > 0

    def test_empty_string(self):
        score = NewsSentimentAnalyzer._score_keywords("")
        assert score == 0.0

    def test_purely_bullish(self):
        score = NewsSentimentAnalyzer._score_keywords("surge rally jump gain rise")
        assert score == pytest.approx(1.0)

    def test_purely_bearish(self):
        score = NewsSentimentAnalyzer._score_keywords("crash plunge drop fall decline")
        assert score == pytest.approx(-1.0)


# ─── analyze() with provided headlines ───


class TestAnalyze:
    def test_bullish_headlines(self, analyzer):
        headlines = _headlines([
            "Apple surges on strong earnings beat",
            "AAPL rallies to record high",
            "Apple gains momentum with new product",
        ])
        result = analyzer.analyze("AAPL", headlines=headlines)
        assert result["sentiment_score"] > 0
        assert result["sentiment_label"] == "BULLISH"
        assert result["headlines_analyzed"] == 3
        assert result["bullish_count"] >= 1
        assert result["method"] == "keyword"

    def test_bearish_headlines(self, analyzer):
        headlines = _headlines([
            "Tesla crashes after disappointing earnings miss",
            "TSLA drops on fraud investigation",
            "Tesla faces major lawsuit and declining sales",
        ])
        result = analyzer.analyze("TSLA", headlines=headlines)
        assert result["sentiment_score"] < 0
        assert result["sentiment_label"] == "BEARISH"
        assert result["bearish_count"] >= 1

    def test_neutral_headlines(self, analyzer):
        headlines = _headlines([
            "Company to hold annual meeting next week",
            "Board of directors scheduled to convene",
            "Quarterly report date announced",
        ])
        result = analyzer.analyze("XYZ", headlines=headlines)
        assert result["sentiment_label"] == "NEUTRAL"
        assert result["sentiment_score"] == pytest.approx(0.0, abs=0.1)

    def test_empty_headlines_returns_empty_result(self, analyzer):
        result = analyzer.analyze("AAPL", headlines=[])
        assert result["sentiment_score"] == 0.0
        assert result["sentiment_label"] == "NEUTRAL"
        assert result["confidence"] == 0.0
        assert result["headlines_analyzed"] == 0

    def test_none_headlines_fetches_then_empty(self, analyzer):
        with patch.object(analyzer, "_fetch_headlines", return_value=[]):
            result = analyzer.analyze("AAPL", headlines=None)
        assert result["headlines_analyzed"] == 0
        assert result["sentiment_label"] == "NEUTRAL"

    def test_headlines_with_empty_titles_skipped(self, analyzer):
        headlines = [{"title": "", "source": "x"}, {"title": "Stock surges", "source": "y"}]
        result = analyzer.analyze("AAPL", headlines=headlines)
        assert result["headlines_analyzed"] == 1

    def test_max_headlines_limit(self, analyzer):
        headlines = _headlines([f"headline {i}" for i in range(50)])
        result = analyzer.analyze("AAPL", headlines=headlines, max_headlines=5)
        assert result["headlines_analyzed"] <= 5

    def test_result_has_all_keys(self, analyzer):
        headlines = _headlines(["Stock surges on earnings"])
        result = analyzer.analyze("AAPL", headlines=headlines)
        expected_keys = {
            "symbol", "sentiment_score", "sentiment_label", "confidence",
            "headlines_analyzed", "bullish_count", "bearish_count", "neutral_count",
            "top_bullish", "top_bearish", "method", "analyzed_at",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_top_bullish_sorted_descending(self, analyzer):
        headlines = _headlines([
            "surge rally jump gain rise",
            "slightly positive upgrade",
            "massive surge record high breakout momentum",
        ])
        result = analyzer.analyze("AAPL", headlines=headlines)
        if len(result["top_bullish"]) >= 2:
            scores = [h["score"] for h in result["top_bullish"]]
            assert scores == sorted(scores, reverse=True)

    def test_top_bearish_sorted_ascending(self, analyzer):
        headlines = _headlines([
            "crash plunge drop",
            "slight decline",
            "massive crash bankruptcy fraud scandal",
        ])
        result = analyzer.analyze("AAPL", headlines=headlines)
        if len(result["top_bearish"]) >= 2:
            scores = [h["score"] for h in result["top_bearish"]]
            assert scores == sorted(scores)


# ─── Verify fix: agreement = max(0.0, 1.0 - std_score) ───


class TestAgreementFix:
    def test_agreement_clamped_to_zero(self, analyzer):
        """If std_score > 1.0, agreement should be max(0.0, ...) = 0.0, not negative."""
        # Create headlines with wildly different scores to get high std
        headlines = _headlines([
            "surge rally jump gain rise beat upgrade bullish",  # very positive
            "crash plunge drop fall decline sink bankruptcy fraud",  # very negative
        ])
        result = analyzer.analyze("TEST", headlines=headlines)
        assert result["confidence"] >= 0.0

    def test_high_agreement_uniform_scores(self, analyzer):
        """All same-direction headlines → low std → high agreement."""
        headlines = _headlines([
            "surges on earnings",
            "rallies after upgrade",
            "gains momentum",
            "rises sharply",
        ])
        result = analyzer.analyze("TEST", headlines=headlines)
        assert result["confidence"] > 0

    def test_confidence_bounded_zero_to_one(self, analyzer):
        """Confidence should always be in [0.0, 1.0]."""
        for titles in [
            ["surge"] * 20,
            ["crash", "surge"],
            ["neutral sentence about nothing"],
        ]:
            headlines = _headlines(titles)
            result = analyzer.analyze("T", headlines=headlines)
            assert 0.0 <= result["confidence"] <= 1.0

    def test_sample_factor_scales_confidence(self, analyzer):
        """More headlines → higher sample_factor → higher confidence."""
        few = _headlines(["surge", "rally"])
        many = _headlines(["surge"] * 15)
        r_few = analyzer.analyze("T", headlines=few)
        r_many = analyzer.analyze("T", headlines=many)
        # More headlines should give higher or equal confidence
        assert r_many["confidence"] >= r_few["confidence"]


# ─── _score_headline() method routing ───


class TestScoreHeadlineRouting:
    def test_keyword_method(self, analyzer):
        score = analyzer._score_headline("Stock surges")
        assert score > 0

    def test_vader_method_mocked(self):
        with patch("shared.ml.news_sentiment._HAS_VADER", True):
            a = NewsSentimentAnalyzer.__new__(NewsSentimentAnalyzer)
            a._active_method = "vader"
            mock_vader = MagicMock()
            mock_vader.polarity_scores.return_value = {"compound": 0.75}
            a._vader = mock_vader
            a._finbert_pipeline = None
            score = a._score_headline("Great earnings")
            assert score == pytest.approx(0.75)
            mock_vader.polarity_scores.assert_called_once()

    def test_finbert_method_mocked(self):
        a = NewsSentimentAnalyzer.__new__(NewsSentimentAnalyzer)
        a._active_method = "finbert"
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "positive", "score": 0.92}]
        a._finbert_pipeline = mock_pipeline
        a._vader = None
        score = a._score_headline("Earnings beat expectations")
        assert score == pytest.approx(0.92)

    def test_finbert_negative_label(self):
        a = NewsSentimentAnalyzer.__new__(NewsSentimentAnalyzer)
        a._active_method = "finbert"
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "negative", "score": 0.88}]
        a._finbert_pipeline = mock_pipeline
        a._vader = None
        score = a._score_headline("Stock plummets")
        assert score == pytest.approx(-0.88)

    def test_finbert_neutral_label(self):
        a = NewsSentimentAnalyzer.__new__(NewsSentimentAnalyzer)
        a._active_method = "finbert"
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "neutral", "score": 0.5}]
        a._finbert_pipeline = mock_pipeline
        a._vader = None
        score = a._score_headline("Quarterly report")
        assert score == 0.0

    def test_finbert_exception_falls_back_to_keywords(self):
        a = NewsSentimentAnalyzer.__new__(NewsSentimentAnalyzer)
        a._active_method = "finbert"
        mock_pipeline = MagicMock(side_effect=RuntimeError("model error"))
        a._finbert_pipeline = mock_pipeline
        a._vader = None
        score = a._score_headline("Stock surges")
        assert score > 0  # fell back to keyword scoring


# ─── analyze_multiple() ───


class TestAnalyzeMultiple:
    def test_returns_all_symbols(self, analyzer):
        with patch.object(analyzer, "_fetch_headlines", return_value=[]):
            results = analyzer.analyze_multiple(["AAPL", "GOOG", "TSLA"])
        assert set(results.keys()) == {"AAPL", "GOOG", "TSLA"}

    def test_exception_returns_empty_result(self, analyzer):
        with patch.object(analyzer, "analyze", side_effect=[
            {"sentiment_score": 0.5, "sentiment_label": "BULLISH"},
            RuntimeError("oops"),
        ]):
            with patch.object(analyzer, "_empty_result", return_value={"sentiment_score": 0.0}):
                results = analyzer.analyze_multiple(["AAPL", "BAD"])
                assert results["BAD"]["sentiment_score"] == 0.0


# ─── _empty_result() ───


class TestEmptyResult:
    def test_empty_result_structure(self):
        result = NewsSentimentAnalyzer._empty_result("AAPL")
        assert result["symbol"] == "AAPL"
        assert result["sentiment_score"] == 0.0
        assert result["sentiment_label"] == "NEUTRAL"
        assert result["confidence"] == 0.0
        assert result["headlines_analyzed"] == 0
        assert result["method"] == "none"

    def test_empty_result_has_analyzed_at(self):
        result = NewsSentimentAnalyzer._empty_result("X")
        assert "analyzed_at" in result


# ─── repr ───


class TestRepr:
    def test_repr(self, analyzer):
        r = repr(analyzer)
        assert "NewsSentimentAnalyzer" in r
        assert "keyword" in r
