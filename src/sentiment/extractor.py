"""Per-ticker intelligence extraction module computing mathematical S/R, position bias, and sentiment metrics."""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import AssetSpec, config
from src.sentiment.contracts import RawPost, NewsArticle, TickerExtraction
from src.sentiment.heuristics import MultilingualLexicon
from src.sentiment.multilingual_llm import MultilingualSentimentEngine

logger = logging.getLogger(__name__)


class TickerIntelligenceExtractor:
    """Extracts qualitative sentiment, trader position bias, and quantitative technical levels per ticker."""

    def __init__(self, sentiment_engine: Optional[MultilingualSentimentEngine] = None):
        self.engine = sentiment_engine or MultilingualSentimentEngine()

    def compute_technical_levels(self, price_df: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Calculate quantitative Support and Resistance levels from OHLCV price history.
        Uses Classical Floor Pivot Points, Swing Highs/Lows, and Rolling Quantiles.
        """
        if price_df.empty or len(price_df) < 5:
            return [], []

        recent_bars = price_df.tail(20)
        high = float(recent_bars["High"].max())
        low = float(recent_bars["Low"].min())
        close = float(recent_bars["Close"].iloc[-1])

        # 1. Floor Pivot Points
        pivot = (high + low + close) / 3.0
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)

        # 2. Rolling Quantiles
        q10 = float(recent_bars["Low"].quantile(0.10))
        q90 = float(recent_bars["High"].quantile(0.90))

        supports = sorted(list({round(s, 2) for s in [s1, s2, q10, low] if s < close}))
        resistances = sorted(list({round(r, 2) for r in [r1, r2, q90, high] if r > close}))

        return supports, resistances

    def extract_ticker_intelligence(
        self,
        asset: AssetSpec,
        posts: List[RawPost],
        news: List[NewsArticle],
        price_df: pd.DataFrame,
        historical_mention_baseline: float = 10.0,
        historical_mention_std: float = 5.0
    ) -> TickerExtraction:
        """Process all social posts, news, and price history to produce a comprehensive TickerExtraction contract."""
        current_price = float(price_df["Close"].iloc[-1]) if not price_df.empty else 100.0
        now_utc = datetime.now(timezone.utc)

        # 1. Compute Technical Support & Resistance Levels
        supports, resistances = self.compute_technical_levels(price_df)

        # 2. Score individual social posts and accumulate metrics
        post_scores: List[float] = []
        post_weights: List[float] = []
        text_price_levels: List[float] = []
        long_votes = 0
        short_votes = 0
        bull_count = 0
        bear_count = 0
        source_scores: Dict[str, List[float]] = {}
        scored_posts_info: List[Dict] = []

        for p in posts:
            score, direction, pos_bias = self.engine.analyze_text(p.text, ticker=asset.ticker)
            post_scores.append(score)

            # Engagement weight formula: 1 + ln(1 + likes + replies + 0.1 * views)
            engagement_score = float(p.likes + p.replies * 2 + p.views * 0.05)
            weight = 1.0 + float(np.log1p(max(0.0, engagement_score)))
            post_weights.append(weight)

            # Accumulate by source
            if p.source not in source_scores:
                source_scores[p.source] = []
            source_scores[p.source].append(score)

            # Direction counts
            if direction == "bullish":
                bull_count += 1
            elif direction == "bearish":
                bear_count += 1

            # Position bias intent
            if pos_bias == "long":
                long_votes += 1
            elif pos_bias == "short":
                short_votes += 1

            # Price mention extraction
            levels = MultilingualLexicon.extract_price_mentions(p.text, reference_price=current_price)
            text_price_levels.extend(levels)

            scored_posts_info.append({
                "post_id": p.post_id,
                "source": p.source,
                "text": p.text,
                "author": p.author,
                "likes": p.likes,
                "views": p.views,
                "score": score,
                "direction": direction,
                "weight": round(weight, 2),
                "timestamp_utc": p.timestamp_utc.isoformat()
            })

        # 3. Overall Aggregate Social Sentiment
        if post_scores:
            unweighted_sentiment = float(np.mean(post_scores))
            weighted_sentiment = float(np.average(post_scores, weights=post_weights))
        else:
            unweighted_sentiment = 0.0
            weighted_sentiment = 0.0

        # Classify overall direction
        if weighted_sentiment > 0.12:
            sentiment_direction = "bullish"
        elif weighted_sentiment < -0.12:
            sentiment_direction = "bearish"
        else:
            sentiment_direction = "neutral"

        # Position bias inference across all posts
        if long_votes > short_votes:
            position_bias = "long"
        elif short_votes > long_votes:
            position_bias = "short"
        elif long_votes > 0 and long_votes == short_votes:
            position_bias = "neutral"
        else:
            position_bias = None

        # 4. News Sentiment Scoring
        news_scores: List[float] = []
        for n in news:
            n_text = f"{n.title}. {n.text}"
            n_score, _, _ = self.engine.analyze_text(n_text, ticker=asset.ticker)
            news_scores.append(n_score)
        avg_news_sentiment = float(np.mean(news_scores)) if news_scores else 0.0

        # 5. Mention Volume & Z-Score
        mention_vol = len(posts)
        vol_zscore = float((mention_vol - historical_mention_baseline) / max(historical_mention_std, 1.0))

        # 6. Bull/Bear Ratio (Laplace smoothed)
        bull_bear_ratio = float((bull_count + 1) / (bear_count + 1))

        # 7. Source Agreement Score
        # If multiple sources exist, measure concordance (1 - standard deviation between source averages)
        if len(source_scores) > 1:
            source_means = [float(np.mean(scores)) for scores in source_scores.values() if scores]
            source_std = float(np.std(source_means))
            agreement_score = float(max(0.0, min(1.0, 1.0 - source_std)))
        else:
            agreement_score = 1.0

        # 8. Overall Confidence Score
        # Factor in: sample volume (scaled to 1.0 at 25 posts), source diversity, agreement
        vol_factor = min(1.0, mention_vol / 25.0) if mention_vol > 0 else 0.2
        diversity_factor = min(1.0, len(source_scores) / 2.0)
        confidence_score = float(round(0.4 * vol_factor + 0.3 * diversity_factor + 0.3 * agreement_score, 3))

        # 9. Top 3 Most Influential Posts
        scored_posts_info.sort(key=lambda x: (x["likes"] + x["views"] * 0.05), reverse=True)
        top_posts = scored_posts_info[:3]

        unique_text_levels = sorted(list(set(text_price_levels)))

        return TickerExtraction(
            ticker=asset.ticker,
            asset_name=asset.name,
            timestamp_utc=now_utc,
            current_price=current_price,
            sentiment_score=round(weighted_sentiment, 3),
            sentiment_direction=sentiment_direction, # type: ignore
            position_bias=position_bias, # type: ignore
            support_levels=supports,
            resistance_levels=resistances,
            text_extracted_price_levels=unique_text_levels,
            mention_volume=mention_vol,
            mention_volume_zscore=round(vol_zscore, 2),
            engagement_weighted_sentiment=round(weighted_sentiment, 3),
            bull_bear_ratio=round(bull_bear_ratio, 2),
            news_sentiment=round(avg_news_sentiment, 3),
            source_agreement_score=round(agreement_score, 3),
            confidence_score=confidence_score,
            top_posts=top_posts
        )
