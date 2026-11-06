"""Rigorous unit test verifying zero lookahead data leakage in feature generation."""

from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
from src.features.alignment import build_feature_target_matrix, PRICE_FEATURE_COLS, ALL_FEATURE_COLS
from src.features.technical import compute_technical_indicators
from src.sentiment.contracts import RawPost, NewsArticle


def test_target_is_strictly_next_bar():
    """Verify that target_return is strictly the subsequent bar's percentage change."""
    dates = pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC")
    closes = [100.0, 105.0, 102.0, 110.0, 108.0, 115.0, 112.0, 120.0, 118.0, 125.0]
    df = pd.DataFrame(
        {
            "Open": closes,
            "High": [c * 1.02 for c in closes],
            "Low": [c * 0.98 for c in closes],
            "Close": closes,
            "Volume": [10000] * 10,
        },
        index=dates
    )

    matrix, _, _ = build_feature_target_matrix(df, posts=[], news=[])
    # For every row t in matrix, target_price must equal the actual close at t+1
    for i in range(len(matrix) - 1):
        idx_t = matrix.index[i]
        # In the original df, locate the subsequent timestamp
        pos = df.index.get_loc(idx_t)
        next_close = df["Close"].iloc[pos + 1]
        assert matrix.loc[idx_t, "target_price"] == next_close
        expected_ret = (next_close - matrix.loc[idx_t, "Close"]) / matrix.loc[idx_t, "Close"]
        assert np.isclose(matrix.loc[idx_t, "target_return"], expected_ret)


def test_no_future_leakage_in_features():
    """Verify that modifying future price data at t > t0 has ZERO effect on features at t0."""
    dates = pd.date_range("2026-01-01", periods=30, freq="D", tz="UTC")
    closes = [float(c) for c in range(100, 130)]
    df1 = pd.DataFrame(
        {
            "Open": closes,
            "High": [c + 2 for c in closes],
            "Low": [c - 2 for c in closes],
            "Close": closes,
            "Volume": [50000] * 30,
        },
        index=dates
    )

    tech1 = compute_technical_indicators(df1)

    # df2 is identical up to bar 20, but drastically altered from bar 21 onward
    df2 = df1.copy()
    df2.iloc[20:, df2.columns.get_loc("Close")] = [c * 5.0 for c in df2.iloc[20:]["Close"]]

    tech2 = compute_technical_indicators(df2)

    # Feature values at bar 19 (strictly before modification) must be 100% identical
    for col in PRICE_FEATURE_COLS:
        val1 = tech1.iloc[19][col]
        val2 = tech2.iloc[19][col]
        assert np.isclose(val1, val2, equal_nan=True), f"Leakage detected in {col}: {val1} != {val2}"


def test_sentiment_strict_timestamp_boundary():
    """Verify that posts occurring AFTER bar timestamp t are NEVER included in bar t features."""
    from src.features.sentiment_feats import aggregate_sentiment_for_bars

    t0 = datetime(2026, 11, 2, 16, 0, 0, tzinfo=timezone.utc)
    dates = pd.date_range(end=t0, periods=5, freq="D", tz="UTC")
    df = pd.DataFrame(
        {"Close": [100.0] * 5, "High": [105.0] * 5, "Low": [95.0] * 5, "Open": [100.0] * 5, "Volume": [1000] * 5},
        index=dates
    )

    # Post 1 occurs 2 hours BEFORE bar t0 close
    past_post = RawPost(
        ticker="0700.HK", source="lihkg", post_id="past",
        timestamp_utc=t0 - timedelta(hours=2),
        text="Bullish news before close", likes=10
    )

    # Post 2 occurs 10 minutes AFTER bar t0 close
    future_post = RawPost(
        ticker="0700.HK", source="lihkg", post_id="future",
        timestamp_utc=t0 + timedelta(minutes=10),
        text="Bearish crash after close", likes=999
    )

    sent_df = aggregate_sentiment_for_bars(df, posts=[past_post, future_post], news=[])

    # At bar t0, only the past post should be counted (mentions == 1)
    t0_row = sent_df.loc[t0]
    assert t0_row["sent_mentions"] == 1.0, f"Expected 1 mention, got {t0_row['sent_mentions']}"
    assert t0_row["sent_mean"] >= 0.0, "Future post sentiment must not leak into bar t0"
