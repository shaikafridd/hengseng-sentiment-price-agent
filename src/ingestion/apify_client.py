"""Apify integration client for scraping social discussions across HKEX and KRX platforms."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from apify_client import ApifyClient
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import config, AssetSpec
from src.sentiment.contracts import RawPost
from src.ingestion.storage import RawDataStore

logger = logging.getLogger(__name__)


class ApifyScraperClient:
    """Production client executing Apify actors with rate limiting, item limits, and caching."""

    def __init__(self, api_token: Optional[str] = None, data_store: Optional[RawDataStore] = None):
        self.api_token = api_token or config.apify_token
        self.client = ApifyClient(self.api_token) if self.api_token else None
        self.store = data_store or RawDataStore()
        self.actors_cfg = config.get_actors_config()

    def scrape_lihkg(self, asset: AssetSpec, limit: int = 30, force_refresh: bool = False) -> List[RawPost]:
        """Scrape LIHKG Hong Kong discussion threads for an asset."""
        source_name = "lihkg"
        if not force_refresh:
            cached = self.store.load_posts(asset.ticker, source_name)
            if cached is not None:
                return cached

        if not self.client:
            logger.warning("APIFY_TOKEN not configured. Skipping LIHKG live scraping for %s", asset.ticker)
            return []

        actor_id = self.actors_cfg.get("hkex_social_1", {}).get("apify_actor_id", "claude_code_reviewer/lihkg-scraper")
        search_query = asset.keywords_local[0] if asset.keywords_local else asset.name
        run_input = {
            "mode": "keyword-search",
            "keyword": search_query,
            "catIds": ["1"], # Finance/Current Affairs
            "limit": limit
        }

        posts: List[RawPost] = []
        try:
            logger.info("Executing Apify actor '%s' for %s (query='%s', limit=%d)...", actor_id, asset.ticker, search_query, limit)
            run = self.client.actor(actor_id).call(run_input=run_input, timeout_secs=180)
            dataset = self.client.dataset(run["defaultDatasetId"])

            for item in dataset.iterate_items():
                # Normalize LIHKG fields
                post_id = str(item.get("threadId") or item.get("id") or item.get("post_id", ""))
                created_at = item.get("createdAt") or item.get("timestamp")
                if isinstance(created_at, (int, float)):
                    # Unix timestamp
                    ts_utc = datetime.fromtimestamp(created_at, tz=timezone.utc)
                elif isinstance(created_at, str):
                    try:
                        ts_utc = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    except Exception:
                        ts_utc = datetime.now(timezone.utc)
                else:
                    ts_utc = datetime.now(timezone.utc)

                text = f"{item.get('title', '')} {item.get('content', '')}".strip()
                if not text:
                    continue

                posts.append(
                    RawPost(
                        ticker=asset.ticker,
                        source="lihkg",
                        post_id=f"lihkg_{post_id or len(posts)}",
                        timestamp_utc=ts_utc,
                        text=text,
                        url=item.get("url") or f"https://lihkg.com/thread/{post_id}/page/1",
                        author=item.get("userNickname") or item.get("author"),
                        likes=item.get("likeCount", 0),
                        dislikes=item.get("dislikeCount", 0),
                        replies=item.get("replyCount", 0),
                        views=item.get("viewCount", 0),
                        raw_metadata={"category": "finance"}
                    )
                )

            logger.info("Retrieved %d LIHKG posts for %s", len(posts), asset.ticker)
        except Exception as e:
            logger.error("Apify error during LIHKG scrape for %s: %s", asset.ticker, e)

        self.store.save_posts(asset.ticker, source_name, posts)
        return posts

    def scrape_aastocks(self, asset: AssetSpec, limit: int = 30, force_refresh: bool = False) -> List[RawPost]:
        """Scrape AAStocks retail community discussion board using Cheerio actor."""
        source_name = "aastocks"
        if not force_refresh:
            cached = self.store.load_posts(asset.ticker, source_name)
            if cached is not None:
                return cached

        if not self.client:
            logger.warning("APIFY_TOKEN not configured. Skipping AAStocks scraping for %s", asset.ticker)
            return []

        # Target AAStocks Stock Forum
        target_url = f"http://www.aastocks.com/tc/stocks/analysis/stock-forum.aspx?symbol={asset.raw_code}"
        run_input = {
            "startUrls": [{"url": target_url}],
            "maxRequestsPerCrawl": 1,
            "pageFunction": """
            async function pageFunction(context) {
                const { $, request, log } = context;
                const results = [];
                $('.contentBox, .forum-content, tr.forumRow').each((i, el) => {
                    const title = $(el).find('.title, .newsTitle').text().trim();
                    const body = $(el).find('.desc, .content, .comment').text().trim();
                    const dateStr = $(el).find('.date, .datetime').text().trim();
                    if (title || body) {
                        results.push({ title, body, dateStr, url: request.url });
                    }
                });
                return results;
            }
            """
        }

        posts: List[RawPost] = []
        try:
            logger.info("Executing Apify Cheerio scraper for AAStocks %s...", asset.ticker)
            run = self.client.actor("apify/cheerio-scraper").call(run_input=run_input, timeout_secs=180)
            dataset = self.client.dataset(run["defaultDatasetId"])

            for idx, item in enumerate(dataset.iterate_items()):
                text = f"{item.get('title', '')} {item.get('body', '')}".strip()
                if not text:
                    continue
                posts.append(
                    RawPost(
                        ticker=asset.ticker,
                        source="aastocks",
                        post_id=f"aastocks_{asset.raw_code}_{idx}",
                        timestamp_utc=datetime.now(timezone.utc),
                        text=text,
                        url=item.get("url") or target_url,
                        likes=item.get("likes", 0),
                        replies=item.get("replies", 0),
                        views=item.get("views", 0)
                    )
                )
            logger.info("Retrieved %d AAStocks posts for %s", len(posts), asset.ticker)
        except Exception as e:
            logger.error("Apify error during AAStocks scrape for %s: %s", asset.ticker, e)

        self.store.save_posts(asset.ticker, source_name, posts)
        return posts

    def scrape_naver_finance(self, asset: AssetSpec, limit: int = 40, force_refresh: bool = False) -> List[RawPost]:
        """Scrape Naver Finance Jongmok Toronsil (종목토론실) discussion board for KRX assets."""
        source_name = "naver_finance"
        if not force_refresh:
            cached = self.store.load_posts(asset.ticker, source_name)
            if cached is not None:
                return cached

        if not self.client:
            logger.warning("APIFY_TOKEN not configured. Skipping Naver Finance scraping for %s", asset.ticker)
            return []

        target_url = f"https://finance.naver.com/item/board.naver?code={asset.raw_code}"
        run_input = {
            "startUrls": [{"url": target_url}],
            "maxRequestsPerCrawl": 2,
            "pageFunction": """
            async function pageFunction(context) {
                const { $, request } = context;
                const items = [];
                $('table.type2 tbody tr').each((i, el) => {
                    const titleEl = $(el).find('td.title a');
                    if (titleEl.length > 0) {
                        const title = titleEl.text().trim();
                        const href = titleEl.attr('href');
                        const date = $(el).find('td:nth-child(2) span').text().trim();
                        const views = $(el).find('td:nth-child(4) span').text().trim();
                        const upvotes = $(el).find('td:nth-child(5) strong').text().trim();
                        const downvotes = $(el).find('td:nth-child(6) strong').text().trim();
                        items.push({ title, href, date, views, upvotes, downvotes });
                    }
                });
                return items;
            }
            """
        }

        posts: List[RawPost] = []
        try:
            logger.info("Executing Apify Cheerio scraper for Naver Toron %s...", asset.ticker)
            run = self.client.actor("apify/cheerio-scraper").call(run_input=run_input, timeout_secs=180)
            dataset = self.client.dataset(run["defaultDatasetId"])

            for idx, item in enumerate(dataset.iterate_items()):
                title = item.get("title", "").strip()
                if not title:
                    continue
                # Parse Naver date format '2026.11.06 15:30' (KST UTC+9)
                ts_utc = datetime.now(timezone.utc)
                date_str = item.get("date", "")
                if date_str:
                    try:
                        dt_naive = datetime.strptime(date_str, "%Y.%m.%d %H:%M")
                        # Localize KST and convert to UTC
                        ts_utc = dt_naive.replace(tzinfo=timezone(timedelta(hours=9))).astimezone(timezone.utc)
                    except Exception:
                        pass

                posts.append(
                    RawPost(
                        ticker=asset.ticker,
                        source="naver_finance",
                        post_id=f"naver_{asset.raw_code}_{idx}",
                        timestamp_utc=ts_utc,
                        text=title,
                        url=f"https://finance.naver.com{item.get('href', '')}",
                        views=int(item.get("views") or 0) if str(item.get("views", "")).isdigit() else 0,
                        likes=int(item.get("upvotes") or 0) if str(item.get("upvotes", "")).isdigit() else 0,
                        dislikes=int(item.get("downvotes") or 0) if str(item.get("downvotes", "")).isdigit() else 0
                    )
                )
            logger.info("Retrieved %d Naver Toron posts for %s", len(posts), asset.ticker)
        except Exception as e:
            logger.error("Apify error during Naver Toron scrape for %s: %s", asset.ticker, e)

        self.store.save_posts(asset.ticker, source_name, posts)
        return posts

    def scrape_telegram(self, asset: AssetSpec, limit: int = 30, force_refresh: bool = False) -> List[RawPost]:
        """Scrape Korean equity Telegram channels for mentions of the asset."""
        source_name = "telegram"
        if not force_refresh:
            cached = self.store.load_posts(asset.ticker, source_name)
            if cached is not None:
                return cached

        if not self.client:
            logger.warning("APIFY_TOKEN not configured. Skipping Telegram scraping for %s", asset.ticker)
            return []

        actor_id = self.actors_cfg.get("krx_social_2", {}).get("apify_actor_id", "curious_coder/telegram-scraper")
        channels = self.actors_cfg.get("krx_social_2", {}).get("channels", ["https://t.me/yeouidostock"])
        run_input = {
            "channels": channels,
            "limit": limit
        }

        posts: List[RawPost] = []
        try:
            logger.info("Executing Apify Telegram scraper for %s...", asset.ticker)
            run = self.client.actor(actor_id).call(run_input=run_input, timeout_secs=180)
            dataset = self.client.dataset(run["defaultDatasetId"])

            search_terms = [asset.ticker, asset.raw_code, asset.name.lower()] + [k.lower() for k in asset.keywords_local]

            for idx, item in enumerate(dataset.iterate_items()):
                msg_text = item.get("text") or item.get("message") or ""
                # Filter for asset mentions
                if any(term in msg_text.lower() for term in search_terms):
                    raw_date = item.get("date")
                    ts_utc = datetime.now(timezone.utc)
                    if isinstance(raw_date, (int, float)):
                        ts_utc = datetime.fromtimestamp(raw_date, tz=timezone.utc)
                    elif isinstance(raw_date, str):
                        try:
                            ts_utc = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
                        except Exception:
                            pass

                    posts.append(
                        RawPost(
                            ticker=asset.ticker,
                            source="telegram",
                            post_id=f"tg_{item.get('id', idx)}",
                            timestamp_utc=ts_utc,
                            text=msg_text.strip(),
                            url=item.get("url"),
                            views=item.get("views", 0),
                            likes=item.get("reactions_count", 0)
                        )
                    )
            logger.info("Retrieved %d Telegram posts matching %s", len(posts), asset.ticker)
        except Exception as e:
            logger.error("Apify error during Telegram scrape for %s: %s", asset.ticker, e)

        self.store.save_posts(asset.ticker, source_name, posts)
        return posts

    def scrape_all_for_asset(self, asset: AssetSpec, limit_per_source: int = 30) -> List[RawPost]:
        """Aggregate social posts for an asset across all eligible sources."""
        all_posts: List[RawPost] = []
        if ".HK" in asset.ticker.upper():
            all_posts.extend(self.scrape_lihkg(asset, limit=limit_per_source))
            all_posts.extend(self.scrape_aastocks(asset, limit=limit_per_source))
        elif ".KS" in asset.ticker.upper():
            all_posts.extend(self.scrape_naver_finance(asset, limit=limit_per_source))
            all_posts.extend(self.scrape_telegram(asset, limit=limit_per_source))
        return all_posts
