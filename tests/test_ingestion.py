"""Test ingestion clients, caching behavior, and error handling resilience."""

from datetime import datetime, timezone
from pathlib import Path
from src.config import config
from src.sentiment.contracts import RawPost, NewsArticle
from src.ingestion.storage import RawDataStore
from src.ingestion.apify_client import ApifyScraperClient
from src.ingestion.exa_client import ExaNewsClient


def test_storage_roundtrip(tmp_path):
    """Test saving and loading posts and news articles via RawDataStore."""
    store = RawDataStore(base_dir=tmp_path)
    now_utc = datetime.now(timezone.utc)

    # Save and reload posts
    posts = [
        RawPost(
            ticker="1810.HK",
            source="lihkg",
            post_id="p1",
            timestamp_utc=now_utc,
            text="Xiaomi EV order book surge",
            likes=25
        )
    ]
    store.save_posts("1810.HK", "lihkg", posts)
    loaded_posts = store.load_posts("1810.HK", "lihkg")
    assert loaded_posts is not None
    assert len(loaded_posts) == 1
    assert loaded_posts[0].text == "Xiaomi EV order book surge"

    # Save and reload news
    news = [
        NewsArticle(
            ticker="1810.HK",
            title="Xiaomi Q3 revenue beats consensus",
            url="https://example.com/xiaomi",
            published_date_utc=now_utc,
            text="Record shipments reported."
        )
    ]
    store.save_news("1810.HK", news)
    loaded_news = store.load_news("1810.HK")
    assert loaded_news is not None
    assert len(loaded_news) == 1
    assert loaded_news[0].title == "Xiaomi Q3 revenue beats consensus"


def test_apify_client_graceful_missing_token(tmp_path):
    """Verify ApifyScraperClient does not throw errors when API token is None."""
    store = RawDataStore(base_dir=tmp_path)
    client = ApifyScraperClient(api_token=None, data_store=store)

    asset = config.get_asset_by_ticker("0700.HK")
    assert asset is not None

    posts = client.scrape_all_for_asset(asset)
    assert isinstance(posts, list)
    assert len(posts) == 0 # Returns empty list safely without crashing


def test_exa_client_graceful_missing_key(tmp_path):
    """Verify ExaNewsClient does not crash when EXA_API_KEY is not configured."""
    store = RawDataStore(base_dir=tmp_path)
    client = ExaNewsClient(api_key="", data_store=store)

    asset = config.get_asset_by_ticker("005930.KS")
    assert asset is not None

    news = client.fetch_news_for_asset(asset, force_refresh=True)
    assert isinstance(news, list)
    assert len(news) == 0
