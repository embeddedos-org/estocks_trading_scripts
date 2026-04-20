"""
News Sentiment Analyzer
=========================

Fetches financial news and analyzes sentiment using multiple methods:
1. VADER (rule-based, fast, no dependencies beyond nltk)
2. FinBERT (transformer-based, accurate, requires transformers + torch)
3. Simple keyword scoring (zero-dependency fallback)

Produces a sentiment score [-1.0, +1.0] that feeds into the
EnsemblePredictor as the "sentiment" model signal.

Usage:
    analyzer = NewsSentimentAnalyzer()
    result = analyzer.analyze("AAPL")
    print(f"Sentiment: {result['sentiment_score']:.2f}")
    print(f"Headlines analyzed: {result['headlines_analyzed']}")
    print(f"Bullish: {result['bullish_count']}, Bearish: {result['bearish_count']}")

Requires: pip install yfinance feedparser
Optional: pip install nltk vaderSentiment transformers torch
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─── Optional NLP Dependencies ───

_HAS_VADER = False
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderAnalyzer
    _HAS_VADER = True
except ImportError:
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer as VaderAnalyzer
        _HAS_VADER = True
    except ImportError:
        logger.debug("VADER not available. Install: pip install vaderSentiment")

_HAS_TRANSFORMERS = False
_finbert_pipeline = None
try:
    from transformers import pipeline as hf_pipeline
    _HAS_TRANSFORMERS = True
except ImportError:
    logger.debug("transformers not available. Install: pip install transformers torch")


# ─── Keyword-Based Fallback ───

_BULLISH_KEYWORDS = {
    "surge", "surges", "surging", "soar", "soars", "soaring",
    "rally", "rallies", "rallying", "jump", "jumps", "jumping",
    "gain", "gains", "gaining", "rise", "rises", "rising",
    "beat", "beats", "beating", "exceed", "exceeds",
    "upgrade", "upgrades", "upgraded", "outperform",
    "bullish", "optimistic", "positive", "strong", "growth",
    "record high", "all-time high", "breakout", "momentum",
    "buy", "accumulate", "overweight", "upside",
    "profit", "revenue growth", "earnings beat",
    "expansion", "recovery", "rebound",
}

_BEARISH_KEYWORDS = {
    "crash", "crashes", "crashing", "plunge", "plunges", "plunging",
    "drop", "drops", "dropping", "fall", "falls", "falling",
    "decline", "declines", "declining", "sink", "sinks", "sinking",
    "miss", "misses", "missing", "disappoint", "disappoints",
    "downgrade", "downgrades", "downgraded", "underperform",
    "bearish", "pessimistic", "negative", "weak", "slowdown",
    "record low", "sell-off", "selloff", "breakdown",
    "sell", "reduce", "underweight", "downside",
    "loss", "revenue decline", "earnings miss",
    "contraction", "recession", "layoff", "layoffs",
    "investigation", "lawsuit", "fraud", "scandal",
    "warning", "guidance cut", "bankruptcy",
}


class NewsSentimentAnalyzer:
    """Multi-method news sentiment analyzer.

    Fetches headlines and scores them using the best available method:
    1. FinBERT (if transformers installed) — most accurate
    2. VADER (if nltk/vaderSentiment installed) — good general-purpose
    3. Keyword scoring (always available) — simple but reliable fallback

    Args:
        method: "auto" (best available), "finbert", "vader", or "keyword".
        finbert_model: HuggingFace model name for FinBERT.
    """

    def __init__(
        self,
        method: str = "auto",
        finbert_model: str = "ProsusAI/finbert",
    ) -> None:
        self._method = method
        self._finbert_model = finbert_model
        self._finbert_pipeline = None
        self._vader = None

        # Initialize the best available analyzer
        if method == "auto":
            if _HAS_VADER:
                self._vader = VaderAnalyzer()
                self._active_method = "vader"
                logger.info("Sentiment analyzer: VADER (rule-based)")
            else:
                self._active_method = "keyword"
                logger.info("Sentiment analyzer: keyword scoring (fallback)")
        elif method == "finbert":
            self._init_finbert()
        elif method == "vader":
            if not _HAS_VADER:
                raise ImportError("VADER not available. Install: pip install vaderSentiment")
            self._vader = VaderAnalyzer()
            self._active_method = "vader"
        else:
            self._active_method = "keyword"

    def _init_finbert(self) -> None:
        """Initialize FinBERT pipeline (lazy, downloads model on first use)."""
        if not _HAS_TRANSFORMERS:
            raise ImportError(
                "transformers not available. Install: pip install transformers torch"
            )
        try:
            self._finbert_pipeline = hf_pipeline(
                "sentiment-analysis",
                model=self._finbert_model,
                tokenizer=self._finbert_model,
            )
            self._active_method = "finbert"
            logger.info("Sentiment analyzer: FinBERT (%s)", self._finbert_model)
        except Exception as e:
            logger.warning("FinBERT init failed: %s. Falling back to VADER/keyword.", e)
            if _HAS_VADER:
                self._vader = VaderAnalyzer()
                self._active_method = "vader"
            else:
                self._active_method = "keyword"

    # ─── Main Analysis ───

    def analyze(
        self,
        symbol: str,
        headlines: Optional[List[Dict[str, str]]] = None,
        max_headlines: int = 20,
    ) -> Dict[str, Any]:
        """Analyze news sentiment for a symbol.

        If headlines are not provided, fetches them automatically
        using PublicDataFetcher.

        Args:
            symbol: Ticker symbol (e.g., "AAPL").
            headlines: Pre-fetched headlines (list of dicts with "title" key).
            max_headlines: Maximum headlines to analyze.

        Returns:
            Dict with:
                sentiment_score: float [-1.0, +1.0]
                sentiment_label: "BULLISH", "BEARISH", or "NEUTRAL"
                confidence: float [0.0, 1.0]
                headlines_analyzed: int
                bullish_count: int
                bearish_count: int
                neutral_count: int
                top_bullish: list of headlines
                top_bearish: list of headlines
                method: str
        """
        # Fetch headlines if not provided
        if headlines is None:
            headlines = self._fetch_headlines(symbol, max_headlines)

        if not headlines:
            return self._empty_result(symbol)

        # Score each headline
        scored: List[Dict[str, Any]] = []
        for h in headlines[:max_headlines]:
            title = h.get("title", "")
            if not title:
                continue

            score = self._score_headline(title)
            scored.append({
                "title": title,
                "score": score,
                "source": h.get("source", ""),
            })

        if not scored:
            return self._empty_result(symbol)

        # Aggregate
        scores = [s["score"] for s in scored]
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores)) if len(scores) > 1 else 0.0

        bullish = [s for s in scored if s["score"] > 0.1]
        bearish = [s for s in scored if s["score"] < -0.1]
        neutral = [s for s in scored if -0.1 <= s["score"] <= 0.1]

        # Confidence: based on agreement and sample size
        agreement = max(0.0, 1.0 - std_score)  # low variance = high agreement
        sample_factor = min(len(scored) / 10, 1.0)  # more headlines = more confident
        confidence = max(0.0, min(1.0, agreement * sample_factor))

        # Label
        if mean_score > 0.1:
            label = "BULLISH"
        elif mean_score < -0.1:
            label = "BEARISH"
        else:
            label = "NEUTRAL"

        result = {
            "symbol": symbol,
            "sentiment_score": round(mean_score, 4),
            "sentiment_label": label,
            "confidence": round(confidence, 4),
            "headlines_analyzed": len(scored),
            "bullish_count": len(bullish),
            "bearish_count": len(bearish),
            "neutral_count": len(neutral),
            "top_bullish": sorted(bullish, key=lambda x: x["score"], reverse=True)[:3],
            "top_bearish": sorted(bearish, key=lambda x: x["score"])[:3],
            "method": self._active_method,
            "analyzed_at": datetime.now().isoformat(),
        }

        logger.info(
            "Sentiment for %s: %.3f (%s) | %d headlines | %d bull, %d bear, %d neutral",
            symbol, mean_score, label, len(scored),
            len(bullish), len(bearish), len(neutral),
        )

        return result

    def analyze_multiple(
        self,
        symbols: List[str],
        max_headlines: int = 15,
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze sentiment for multiple symbols.

        Args:
            symbols: List of ticker symbols.
            max_headlines: Max headlines per symbol.

        Returns:
            Dict mapping symbol to sentiment result.
        """
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.analyze(symbol, max_headlines=max_headlines)
            except Exception as e:
                logger.error("Sentiment analysis failed for %s: %s", symbol, e)
                results[symbol] = self._empty_result(symbol)
        return results

    # ─── Scoring Methods ───

    def _score_headline(self, text: str) -> float:
        """Score a single headline. Returns [-1.0, +1.0]."""
        if self._active_method == "finbert" and self._finbert_pipeline is not None:
            return self._score_finbert(text)
        elif self._active_method == "vader" and self._vader is not None:
            return self._score_vader(text)
        else:
            return self._score_keywords(text)

    def _score_finbert(self, text: str) -> float:
        """Score using FinBERT transformer model."""
        try:
            result = self._finbert_pipeline(text[:512])[0]
            label = result["label"].lower()
            score = result["score"]

            if label == "positive":
                return score
            elif label == "negative":
                return -score
            else:
                return 0.0
        except Exception as e:
            logger.debug("FinBERT scoring failed: %s", e)
            return self._score_keywords(text)

    def _score_vader(self, text: str) -> float:
        """Score using VADER sentiment analyzer."""
        try:
            scores = self._vader.polarity_scores(text)
            return scores["compound"]  # Already in [-1, +1]
        except Exception as e:
            logger.debug("VADER scoring failed: %s", e)
            return self._score_keywords(text)

    @staticmethod
    def _score_keywords(text: str) -> float:
        """Score using keyword matching (zero-dependency fallback)."""
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))

        bull_count = len(words & _BULLISH_KEYWORDS)
        bear_count = len(words & _BEARISH_KEYWORDS)

        # Also check multi-word phrases
        for phrase in _BULLISH_KEYWORDS:
            if " " in phrase and phrase in text_lower:
                bull_count += 1
        for phrase in _BEARISH_KEYWORDS:
            if " " in phrase and phrase in text_lower:
                bear_count += 1

        total = bull_count + bear_count
        if total == 0:
            return 0.0

        score = (bull_count - bear_count) / total
        return max(-1.0, min(1.0, score))

    # ─── Helpers ───

    def _fetch_headlines(self, symbol: str, max_items: int) -> List[Dict[str, str]]:
        """Fetch headlines using PublicDataFetcher."""
        try:
            from shared.data.public_data_fetcher import PublicDataFetcher
            fetcher = PublicDataFetcher(cache_enabled=False)
            return fetcher.fetch_news_headlines(symbol, max_items=max_items)
        except Exception as e:
            logger.warning("Failed to fetch headlines for %s: %s", symbol, e)
            return []

    @staticmethod
    def _empty_result(symbol: str) -> Dict[str, Any]:
        """Return empty/neutral result when no data is available."""
        return {
            "symbol": symbol,
            "sentiment_score": 0.0,
            "sentiment_label": "NEUTRAL",
            "confidence": 0.0,
            "headlines_analyzed": 0,
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
            "top_bullish": [],
            "top_bearish": [],
            "method": "none",
            "analyzed_at": datetime.now().isoformat(),
        }

    def __repr__(self) -> str:
        return f"NewsSentimentAnalyzer(method='{self._active_method}')"
