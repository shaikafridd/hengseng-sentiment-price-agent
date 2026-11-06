"""Local storage and caching layer for raw posts, news, and market data."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from src.config import config
from src.sentiment.contracts import RawPost, NewsArticle

logger = logging.getLogger(__name__)


class RawDataStore:
    """Manages persistent caching for scraped social posts and Exa news articles."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or config.raw_dir
        self.posts_dir = self.base_dir / "posts"
        self.news_dir = self.base_dir / "news"
        self.posts_dir.mkdir(parents=True, exist_ok=True)
        self.news_dir.mkdir(parents=True, exist_ok=True)

    def _post_cache_path(self, ticker: str, source: str) -> Path:
        clean_ticker = ticker.replace(".", "_")
        return self.posts_dir / f"{clean_ticker}_{source}.json"

    def _news_cache_path(self, ticker: str) -> Path:
        clean_ticker = ticker.replace(".", "_")
        return self.news_dir / f"{clean_ticker}_news.json"

    def save_posts(self, ticker: str, source: str, posts: List[RawPost]) -> None:
        """Cache raw posts to disk in JSON format."""
        path = self._post_cache_path(ticker, source)
        data = [p.model_dump(mode="json") for p in posts]
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info("Saved %d raw posts for %s (%s) to %s", len(posts), ticker, source, path)
        except Exception as e:
            logger.error("Failed saving posts cache to %s: %s", path, e)

    def load_posts(self, ticker: str, source: str) -> Optional[List[RawPost]]:
        """Load raw posts from disk if cached."""
        path = self._post_cache_path(ticker, source)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            posts = [RawPost(**item) for item in data]
            logger.info("Loaded %d cached posts for %s (%s)", len(posts), ticker, source)
            return posts
        except Exception as e:
            logger.warning("Error reading cached posts from %s: %s", path, e)
            return None

    def save_news(self, ticker: str, articles: List[NewsArticle]) -> None:
        """Cache news articles to disk."""
        path = self._news_cache_path(ticker)
        data = [a.model_dump(mode="json") for a in articles]
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info("Saved %d news articles for %s to %s", len(articles), ticker, path)
        except Exception as e:
            logger.error("Failed saving news cache to %s: %s", path, e)

    def load_news(self, ticker: str) -> Optional[List[NewsArticle]]:
        """Load cached news articles for a ticker."""
        path = self._news_cache_path(ticker)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            articles = [NewsArticle(**item) for item in data]
            logger.info("Loaded %d cached news articles for %s", len(articles), ticker)
            return articles
        except Exception as e:
            logger.warning("Error reading cached news from %s: %s", path, e)
            return None
