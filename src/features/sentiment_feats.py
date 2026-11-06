"""Aggregation of social media sentiment and news features strictly up to bar timestamp (zero lookahead)."""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.sentiment.contracts import RawPost, NewsArticle
from src.sentiment.heuristics import MultilingualLexicon
from src.sentiment.multilingual_llm import MultilingualSentimentEngine


def aggregate_sentiment_for_bars(
    price_df: pd.DataFrame,
    posts: List[RawPost],
    news: List[NewsArticle],
    sentiment_engine: Optional[MultilingualSentimentEngine] = None,
    lookback_window_days: float = 3.0
) -> pd.DataFrame:
    """Aggregate social sentiment and news intelligence up to each bar timestamp with strict anti-leakage.

    For a bar at timestamp t, only posts and news strictly timestamped <= t are considered.

    Args:
        price_df: DataFrame with DatetimeIndex in UTC
        posts: List of RawPost instances with UTC timestamps
        news: List of NewsArticle instances with UTC timestamps
        sentiment_engine: Engine to score posts/news
        lookback_window_days: Rolling window to aggregate sentiment over (default 3 days)

    Returns:
        DataFrame aligned with price_df index containing sentiment feature columns.
    """
    engine = sentiment_engine or MultilingualSentimentEngine()

    # Pre-score all posts once
    scored_posts: List[Dict] = []
    for p in posts:
        score, direction, pos_bias = engine.analyze_text(p.text, ticker=p.ticker)
        # Engagement weight: 1 + ln(1 + likes + replies*2 + views*0.05)
        weight = 1.0 + float(np.log1p(max(0.0, float(p.likes + p.replies * 2 + p.views * 0.05))))
        pos_num = 1.0 if pos_bias == "long" else (-1.0 if pos_bias == "short" else 0.0)

        scored_posts.append({
            "timestamp": p.timestamp_utc,
            "score": score,
            "weight": weight,
            "is_bull": 1 if direction == "bullish" else 0,
            "is_bear": 1 if direction == "bearish" else 0,
            "position_num": pos_num
        })

    # Pre-score news
    scored_news: List[Dict] = []
    for n in news:
        n_text = f"{n.title}. {n.text}"
        n_score, _, _ = engine.analyze_text(n_text, ticker=n.ticker)
        scored_news.append({
            "timestamp": n.published_date_utc or datetime.now(timezone.utc),
            "score": n_score
        })

    # Prepare sentiment feature series aligned with price_df index
    records = []
    window_delta = timedelta(days=lookback_window_days)

    for bar_time in price_df.index:
        # Strictly past and concurrent data: timestamp <= bar_time and timestamp >= bar_time - window_delta
        relevant_posts = [
            p for p in scored_posts
            if (bar_time - window_delta) <= p["timestamp"] <= bar_time
        ]

        relevant_news = [
            n for n in scored_news
            if (bar_time - window_delta) <= n["timestamp"] <= bar_time
        ]

        if relevant_posts:
            scores = [p["score"] for p in relevant_posts]
            weights = [p["weight"] for p in relevant_posts]
            pos_nums = [p["position_num"] for p in relevant_posts]
            bull_sum = sum(p["is_bull"] for p in relevant_posts)
            bear_sum = sum(p["is_bear"] for p in relevant_posts)

            sent_mean = float(np.mean(scores))
            sent_weighted = float(np.average(scores, weights=weights))
            mention_cnt = len(relevant_posts)
            bb_ratio = float((bull_sum + 1) / (bear_sum + 1))
            avg_pos = float(np.mean(pos_nums))
        else:
            sent_mean = 0.0
            sent_weighted = 0.0
            mention_cnt = 0
            bb_ratio = 1.0
            avg_pos = 0.0

        if relevant_news:
            news_mean = float(np.mean([n["score"] for n in relevant_news]))
        else:
            news_mean = 0.0

        records.append({
            "sent_mean": sent_mean,
            "sent_weighted": sent_weighted,
            "sent_mentions": float(mention_cnt),
            "sent_bb_ratio": bb_ratio,
            "sent_pos_bias": avg_pos,
            "news_sent": news_mean
        })

    sent_df = pd.DataFrame(records, index=price_df.index)
    return sent_df
