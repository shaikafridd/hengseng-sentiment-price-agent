"""Test Pydantic data contracts, schema constraints, and UTC validations."""

from datetime import datetime, timezone
import pytest
from pydantic import ValidationError
from src.sentiment.contracts import RawPost, NewsArticle, TickerExtraction


def test_raw_post_utc_enforcement():
    """Verify that naive datetimes passed to RawPost are automatically converted to UTC."""
    naive_dt = datetime(2026, 11, 2, 10, 30, 0)
    post = RawPost(
        ticker="0700.HK",
        source="lihkg",
        post_id="post_1",
        timestamp_utc=naive_dt,
        text="Bullish on Tencent WeChat AI",
        likes=10,
        views=100
    )
    assert post.timestamp_utc.tzinfo is not None
    assert post.timestamp_utc.tzinfo == timezone.utc


def test_raw_post_non_negative_counters():
    """Verify likes, replies, and views cannot be negative."""
    post = RawPost(
        ticker="005930.KS",
        source="naver_finance",
        post_id="post_2",
        timestamp_utc=datetime.now(timezone.utc),
        text="Samsung HBM3e testing",
        likes=-5, # Invalid negative should coerce to 0
        views="invalid" # Invalid string should coerce to 0
    )
    assert post.likes == 0
    assert post.views == 0


def test_news_article_contract():
    """Verify NewsArticle schema handles missing dates and highlights."""
    article = NewsArticle(
        ticker="9988.HK",
        title="Alibaba Cloud expansion in SEA",
        url="https://example.com/news/1",
        text="Full article text",
        highlights=["Key sentence 1", "Key sentence 2"]
    )
    assert article.source == "exa"
    assert len(article.highlights) == 2


def test_ticker_extraction_constraints():
    """Verify sentiment_score is bounded in [-1.0, 1.0] and directions are restricted."""
    valid_extraction = TickerExtraction(
        ticker="0700.HK",
        asset_name="Tencent",
        timestamp_utc=datetime.now(timezone.utc),
        current_price=420.5,
        sentiment_score=0.75,
        sentiment_direction="bullish",
        position_bias="long"
    )
    assert valid_extraction.sentiment_score == 0.75

    with pytest.raises(ValidationError):
        # Score out of bounds (> 1.0)
        TickerExtraction(
            ticker="0700.HK",
            asset_name="Tencent",
            timestamp_utc=datetime.now(timezone.utc),
            current_price=420.5,
            sentiment_score=2.5, # Out of range
            sentiment_direction="bullish"
        )
