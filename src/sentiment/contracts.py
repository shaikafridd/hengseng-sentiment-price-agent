"""Pydantic data contracts for raw posts, news articles, and extraction payloads."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class RawPost(BaseModel):
    """Normalized social post contract across all platforms (LIHKG, AAStocks, Naver, Telegram)."""
    ticker: str
    source: Literal["lihkg", "aastocks", "naver_finance", "telegram", "mock"]
    post_id: str
    timestamp_utc: datetime
    text: str
    url: Optional[str] = None
    author: Optional[str] = None
    author_followers: Optional[int] = 0
    likes: int = 0
    dislikes: int = 0
    replies: int = 0
    views: int = 0
    raw_metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("timestamp_utc", mode="after")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        """Guarantee all post timestamps are tz-aware UTC."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

    @field_validator("likes", "dislikes", "replies", "views", mode="before")
    @classmethod
    def non_negative_int(cls, v: Any) -> int:
        try:
            val = int(v)
            return max(0, val)
        except (ValueError, TypeError):
            return 0


class NewsArticle(BaseModel):
    """Normalized news article contract from Exa AI Search."""
    ticker: str
    source: str = "exa"
    title: str
    url: str
    published_date_utc: Optional[datetime] = None
    text: str = ""
    highlights: List[str] = Field(default_factory=list)
    author: Optional[str] = None
    score: Optional[float] = None

    @field_validator("published_date_utc", mode="after")
    @classmethod
    def ensure_utc(cls, v: Optional[datetime]) -> Optional[datetime]:
        if v is None:
            return None
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)


class TickerExtraction(BaseModel):
    """Per-ticker structured intelligence extracted from social discussions and price data."""
    ticker: str
    asset_name: str
    timestamp_utc: datetime
    current_price: float
    sentiment_score: float = Field(ge=-1.0, le=1.0, description="Score from -1 (bearish) to +1 (bullish)")
    sentiment_direction: Literal["bullish", "bearish", "neutral"]
    position_bias: Optional[Literal["long", "short", "neutral"]] = None
    support_levels: List[float] = Field(default_factory=list)
    resistance_levels: List[float] = Field(default_factory=list)
    text_extracted_price_levels: List[float] = Field(default_factory=list)
    mention_volume: int = 0
    mention_volume_zscore: float = 0.0
    engagement_weighted_sentiment: float = 0.0
    bull_bear_ratio: float = 1.0
    news_sentiment: float = 0.0
    source_agreement_score: float = 1.0
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.5)
    top_posts: List[Dict[str, Any]] = Field(default_factory=list)
