"""Feature engineering and multi-modal alignment modules."""
from src.features.technical import compute_technical_indicators
from src.features.sentiment_feats import aggregate_sentiment_for_bars
from src.features.alignment import (
    build_feature_target_matrix,
    PRICE_FEATURE_COLS,
    SENTIMENT_FEATURE_COLS,
    ALL_FEATURE_COLS
)

__all__ = [
    "compute_technical_indicators",
    "aggregate_sentiment_for_bars",
    "build_feature_target_matrix",
    "PRICE_FEATURE_COLS",
    "SENTIMENT_FEATURE_COLS",
    "ALL_FEATURE_COLS"
]
