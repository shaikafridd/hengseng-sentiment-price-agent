"""Exa AI News Search client for financial news intelligence and sentiment."""

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.config import config, AssetSpec
from src.sentiment.contracts import NewsArticle
from src.ingestion.storage import RawDataStore

logger = logging.getLogger(__name__)


class ExaNewsClient:
    """Client for querying curated financial news using Exa AI's semantic search API."""

    def __init__(self, api_key: Optional[str] = None, data_store: Optional[RawDataStore] = None):
        self.api_key = api_key or config.exa_api_key
        self.endpoint = "https://api.exa.ai/search"
        self.store = data_store or RawDataStore()

    @retry(
        reraise=False,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, ConnectionError))
    )
    def _execute_search(self, payload: dict) -> dict:
        """Execute HTTP request to Exa API with backoff."""
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "User-Agent": "HengSengSentimentAgent/1.0"
        }
        resp = requests.post(self.endpoint, json=payload, headers=headers, timeout=20)
        resp.raise_for_status()
        return resp.json()

    def fetch_news_for_asset(
        self,
        asset: AssetSpec,
        days_back: int = 10,
        num_results: int = 10,
        force_refresh: bool = False
    ) -> List[NewsArticle]:
        """Fetch recent financial news for an asset, cached to local storage."""
        # 1. Check local cache
        if not force_refresh:
            cached = self.store.load_news(asset.ticker)
            if cached is not None:
                return cached

        # 2. Check API key
        if not self.api_key:
            logger.warning("EXA_API_KEY is not configured. Falling back to offline news sample.")
            return []

        # 3. Construct semantic query
        # E.g. "Tencent 0700.HK earnings financial results quarterly report stock"
        start_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%dT00:00:00.000Z")
        query = f"{asset.name} ({asset.ticker}) latest financial news earnings performance stock analysis"

        payload = {
            "query": query,
            "type": "neural",
            "useAutoprompt": True,
            "numResults": num_results,
            "startPublishedDate": start_date,
            "contents": {
                "text": {"maxCharacters": 1500},
                "highlights": {"numSentences": 3}
            }
        }

        articles: List[NewsArticle] = []
        try:
            logger.info("Querying Exa news for %s: '%s'...", asset.ticker, query)
            data = self._execute_search(payload)
            results = data.get("results", [])

            for item in results:
                # Parse published date
                pub_date = None
                raw_date = item.get("publishedDate")
                if raw_date:
                    try:
                        pub_date = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
                    except Exception:
                        pub_date = datetime.now(timezone.utc)

                highlights = item.get("highlights", [])
                text_content = item.get("text", "") or " ".join(highlights)

                articles.append(
                    NewsArticle(
                        ticker=asset.ticker,
                        source="exa",
                        title=item.get("title", f"News for {asset.name}"),
                        url=item.get("url", ""),
                        published_date_utc=pub_date or datetime.now(timezone.utc),
                        text=text_content,
                        highlights=highlights,
                        author=item.get("author") or "Exa Financial News",
                        score=item.get("score")
                    )
                )

            logger.info("Successfully fetched %d Exa news articles for %s", len(articles), asset.ticker)
        except Exception as e:
            logger.error("Failed fetching Exa news for %s: %s", asset.ticker, e)

        # 4. Cache results
        self.store.save_news(asset.ticker, articles)
        return articles
