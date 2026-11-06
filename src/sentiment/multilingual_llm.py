"""Multilingual sentiment scoring engine using LLM JSON schema with lexicon fallback and caching."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests

from src.config import config
from src.sentiment.heuristics import MultilingualLexicon

logger = logging.getLogger(__name__)


class MultilingualSentimentEngine:
    """Evaluates multilingual financial text using LLM structured outputs or local slang heuristics."""

    def __init__(self, cache_file: Optional[Path] = None):
        self.cache_file = cache_file or (config.raw_dir / "sentiment_cache.json")
        self.cache: Dict[str, Dict] = self._load_cache()
        self.api_key = config.openai_api_key or config.openrouter_api_key

    def _load_cache(self) -> Dict[str, Dict]:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self) -> None:
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Could not persist sentiment cache: %s", e)

    def analyze_text(self, text: str, ticker: str = "") -> Tuple[float, str, Optional[str]]:
        """Score a single post/article text.

        Returns:
            Tuple of (sentiment_score [-1.0, 1.0], direction ['bullish', 'bearish', 'neutral'], position_bias)
        """
        text_clean = text.strip()
        if not text_clean:
            return 0.0, "neutral", None

        cache_key = f"{ticker}:{text_clean[:120]}"
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            return entry["score"], entry["direction"], entry.get("position_bias")

        # 1. Try LLM if API key is present
        score: Optional[float] = None
        direction: Optional[str] = None
        pos_bias: Optional[str] = None

        if self.api_key:
            try:
                score, direction, pos_bias = self._call_llm_sentiment(text_clean, ticker)
            except Exception as e:
                logger.debug("LLM sentiment call bypassed/failed (%s); falling back to heuristic lexicon.", e)

        # 2. Fallback to multilingual heuristic lexicon
        if score is None or direction is None:
            score, direction = MultilingualLexicon.score_text(text_clean)
            pos_bias = MultilingualLexicon.infer_position_bias(text_clean)

        # Cache result
        self.cache[cache_key] = {
            "score": score,
            "direction": direction,
            "position_bias": pos_bias
        }
        self._save_cache()

        return score, direction, pos_bias

    def _call_llm_sentiment(self, text: str, ticker: str) -> Tuple[float, str, Optional[str]]:
        """Call LLM with structured JSON output."""
        prompt = (
            f"You are a quant financial sentiment analyst. Analyze the following financial post/article for asset '{ticker}'.\n"
            "Evaluate sentiment on a scale from -1.0 (extremely bearish) to +1.0 (extremely bullish).\n"
            "Classify direction as 'bullish', 'bearish', or 'neutral'.\n"
            "Detect if the author indicates a personal trade position ('long', 'short', or null).\n"
            "Understand English, Cantonese / Traditional Chinese slang (牛軍, 溝淡, 止蝕), and Korean slang (풀매수, 존버, 떡락).\n\n"
            f"Text: \"\"\"{text}\"\"\"\n\n"
            "Respond ONLY with valid JSON in this schema:\n"
            "{\"score\": float, \"direction\": \"bullish\"|\"bearish\"|\"neutral\", \"position_bias\": \"long\"|\"short\"|null}"
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        # Standard OpenAI compatible completion endpoint
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens": 150
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        raw_content = data["choices"][0]["message"]["content"]
        parsed = json.loads(raw_content)

        raw_score = float(parsed.get("score", 0.0))
        score = max(-1.0, min(1.0, raw_score))
        direction = parsed.get("direction", "neutral").lower()
        if direction not in ["bullish", "bearish", "neutral"]:
            direction = "neutral"
        pos_bias = parsed.get("position_bias")
        if pos_bias and pos_bias.lower() in ["long", "short"]:
            pos_bias = pos_bias.lower()
        else:
            pos_bias = None

        return score, direction, pos_bias
