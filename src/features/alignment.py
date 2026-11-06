"""Feature matrix builder aligning technical and sentiment features with next-bar return targets."""

from typing import List, Tuple
import pandas as pd
from src.features.technical import compute_technical_indicators
from src.features.sentiment_feats import aggregate_sentiment_for_bars
from src.sentiment.contracts import RawPost, NewsArticle

PRICE_FEATURE_COLS = [
    "ret_1d", "ret_2d", "ret_5d", "ret_10d",
    "vol_5d", "vol_20d",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_position", "natr_14", "volume_zscore"
]

SENTIMENT_FEATURE_COLS = [
    "sent_mean", "sent_weighted", "sent_mentions",
    "sent_bb_ratio", "sent_pos_bias", "news_sent"
]

ALL_FEATURE_COLS = PRICE_FEATURE_COLS + SENTIMENT_FEATURE_COLS


def build_feature_target_matrix(
    price_df: pd.DataFrame,
    posts: List[RawPost],
    news: List[NewsArticle],
    lookback_window_days: float = 3.0
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Build unified dataset with technical features, anti-leakage sentiment features, and next-bar targets.

    Target definition:
        target_return = (Close_{t+1} - Close_t) / Close_t
        target_price  = Close_{t+1}

    Returns:
        Tuple of (clean_dataframe, price_feature_cols, all_feature_cols)
    """
    if price_df.empty or len(price_df) < 25:
        return pd.DataFrame(), PRICE_FEATURE_COLS, ALL_FEATURE_COLS

    # 1. Technical features
    tech_df = compute_technical_indicators(price_df)

    # 2. Sentiment features (strictly <= t)
    sent_df = aggregate_sentiment_for_bars(
        price_df=price_df,
        posts=posts,
        news=news,
        lookback_window_days=lookback_window_days
    )

    # 3. Combine features
    combined = pd.concat([tech_df, sent_df], axis=1)

    # 4. Target definition: Next-bar return R_{t+1}
    combined["current_price"] = combined["Close"]
    combined["target_return"] = combined["Close"].pct_change(1).shift(-1)
    combined["target_price"] = combined["Close"].shift(-1)

    # 5. Drop NaN rows resulting from technical indicator lookbacks and the last unobserved target row
    clean_df = combined.dropna(subset=PRICE_FEATURE_COLS + ["target_return", "target_price"]).copy()

    return clean_df, PRICE_FEATURE_COLS, ALL_FEATURE_COLS
